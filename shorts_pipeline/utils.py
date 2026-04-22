"""
Utilitários compartilhados por todos os módulos do pipeline.

Fornece:
- Retry com backoff exponencial (classificando erros transitórios vs definitivos)
- Logging estruturado em arquivo + console
- Helpers para ler/salvar JSON
- Gerenciamento de ciclo (pasta por execução)
"""
from __future__ import annotations

import functools
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import requests


# =========================================================================
# Exceções customizadas
# =========================================================================

class PipelineError(Exception):
    """Erro genérico do pipeline."""
    pass


class TransientError(PipelineError):
    """Erro temporário — vale retentar (rede, 5xx, timeout)."""
    pass


class PermanentError(PipelineError):
    """Erro definitivo — não adianta retentar (config errada, 4xx exceto 429)."""
    pass


class StepSkippedError(PipelineError):
    """Etapa pulada por decisão estratégica — não é erro real."""
    pass


# =========================================================================
# Logging
# =========================================================================

def setup_logging(cycle_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configura logging duplo: console + arquivo dentro do cycle_dir.
    """
    log_file = cycle_dir / "pipeline.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    return logging.getLogger(name)


# =========================================================================
# Retry com backoff
# =========================================================================

def classify_http_error(status_code: int) -> Type[PipelineError]:
    """
    Classifica um status code HTTP como transitório ou permanente.
    """
    if status_code == 429:  # rate limit
        return TransientError
    if 500 <= status_code < 600:  # server errors
        return TransientError
    if 400 <= status_code < 500:  # client errors (auth, config, etc.)
        return PermanentError
    return PermanentError


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (TransientError, requests.ConnectionError, requests.Timeout),
) -> Callable:
    """
    Decorator que retenta uma função com backoff exponencial em caso de erro transitório.
    Erros permanentes (PermanentError) são propagados imediatamente sem retry.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            last_exc: Optional[Exception] = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except PermanentError as exc:
                    # Não retenta erro permanente
                    logger.error(f"[{func.__name__}] erro permanente na tentativa {attempt}: {exc}")
                    raise

                except retry_on as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"[{func.__name__}] falhou após {max_attempts} tentativas. Último erro: {exc}"
                        )
                        raise
                    wait = delay
                    if jitter:
                        wait = wait * (0.5 + random.random())
                    wait = min(wait, max_delay)
                    logger.warning(
                        f"[{func.__name__}] tentativa {attempt}/{max_attempts} falhou: {exc}. "
                        f"Aguardando {wait:.1f}s antes de retentar."
                    )
                    time.sleep(wait)
                    delay *= backoff_factor

            # Teoricamente inalcançável
            if last_exc:
                raise last_exc
            raise PipelineError(f"{func.__name__} falhou sem exceção capturada.")

        return wrapper
    return decorator


def http_request_with_retry(
    method: str,
    url: str,
    *,
    max_attempts: int = 4,
    initial_delay: float = 3.0,
    timeout: int = 180,
    **kwargs,
) -> requests.Response:
    """
    Wrapper em cima de requests que:
    - Classifica erros HTTP em transitórios (5xx, 429) e permanentes (4xx outros)
    - Retenta transitórios com backoff exponencial
    - Falha imediatamente em erros permanentes
    """
    logger = get_logger()
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_attempts:
                raise TransientError(f"Falha de conexão após {max_attempts} tentativas: {exc}")
            wait = delay * (0.5 + random.random())
            logger.warning(f"HTTP {method} {url} tentativa {attempt}/{max_attempts}: {exc}. Aguardando {wait:.1f}s")
            time.sleep(wait)
            delay *= 2.0
            continue

        if response.ok:
            return response

        err_class = classify_http_error(response.status_code)
        body_preview = (response.text or "")[:500]

        if err_class is PermanentError:
            raise PermanentError(
                f"HTTP {response.status_code} permanente em {method} {url}. Body: {body_preview}"
            )

        # transitório
        if attempt == max_attempts:
            raise TransientError(
                f"HTTP {response.status_code} após {max_attempts} tentativas em {method} {url}. Body: {body_preview}"
            )
        wait = delay * (0.5 + random.random())
        logger.warning(
            f"HTTP {response.status_code} tentativa {attempt}/{max_attempts} em {method} {url}. "
            f"Aguardando {wait:.1f}s. Body: {body_preview}"
        )
        time.sleep(wait)
        delay *= 2.0

    raise TransientError(f"HTTP {method} {url} falhou sem resposta após {max_attempts} tentativas.")


# =========================================================================
# JSON helpers
# =========================================================================

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return default if default is not None else {}
    return load_json(path)


# =========================================================================
# Cycle management
# =========================================================================

MODULE_DIRS = [
    "trend-detection",
    "editorial-selection",
    "story-generation",
    "narration",
    "visual-selection",
    "video-assembly",
    "publishing",
]


def build_cycle_id() -> str:
    return datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def create_cycle(base_dir: Path, channel_name: str) -> Path:
    """
    Cria pasta do ciclo em base_dir/<channel>/cycles/<cycle_id>/ com todos os subdirs.
    Retorna o caminho da pasta do ciclo.
    """
    channel_root = base_dir / channel_name
    cycles_root = channel_root / "cycles"
    cycles_root.mkdir(parents=True, exist_ok=True)

    cycle_id = build_cycle_id()
    cycle_dir = cycles_root / cycle_id
    while cycle_dir.exists():
        cycle_id = build_cycle_id()
        cycle_dir = cycles_root / cycle_id

    cycle_dir.mkdir(parents=True)
    for sub in MODULE_DIRS:
        (cycle_dir / sub).mkdir(parents=True, exist_ok=True)

    # pasta de controle no nível do canal (recent_trends, recent_profiles, etc.)
    (channel_root / "control").mkdir(parents=True, exist_ok=True)

    return cycle_dir


def get_control_dir(cycle_dir: Path) -> Path:
    """A pasta de controle fica 2 níveis acima do cycle_dir (channel_root/control)."""
    return cycle_dir.parent.parent / "control"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
