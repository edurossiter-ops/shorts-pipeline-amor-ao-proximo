"""
Orquestrador principal do pipeline.

Executa as 6 etapas em sequência, com retry inteligente por etapa.
Escreve um manifest do ciclo com o status final de cada etapa.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

from .config import PipelineConfig
from .utils import (
    PermanentError,
    StepSkippedError,
    TransientError,
    create_cycle,
    get_logger,
    now_iso,
    save_json,
    setup_logging,
)
from . import bible_reflection
from . import story_generation
from . import narration as narration_mod
from . import visual_selection
from . import video_assembly
from . import publishing


# Ordem das etapas e suas dependências lógicas.
# Se uma etapa falhar definitivamente, as próximas são puladas (não adianta tentar).
STEPS: List[Dict[str, Any]] = [
    {"name": "bible_reflection",  "fn": bible_reflection.run,  "required": True},
    {"name": "story_generation",  "fn": story_generation.run,  "required": True},
    {"name": "narration",         "fn": narration_mod.run,     "required": True},
    {"name": "visual_selection",  "fn": visual_selection.run,  "required": True},
    {"name": "video_assembly",    "fn": video_assembly.run,    "required": True},
    {"name": "publishing",        "fn": publishing.run,        "required": True},
]


def _run_step_with_retry(
    step_name: str,
    step_fn: Callable,
    cycle_dir: Path,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """
    Executa uma etapa com retry a nível de etapa (acima do retry intra-HTTP).

    A ideia: mesmo que os retries de HTTP do módulo esgotem, tentamos rodar a
    etapa inteira de novo algumas vezes — útil quando uma API está degradada.
    """
    logger = get_logger()
    max_attempts = config.retry.max_attempts_per_step
    delay = config.retry.initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        logger.info(f"=== ETAPA [{step_name}] tentativa {attempt}/{max_attempts} ===")
        try:
            result = step_fn(cycle_dir, config)
            elapsed = time.time() - t0
            logger.info(f"=== ETAPA [{step_name}] OK em {elapsed:.1f}s ===")
            return {
                "step": step_name,
                "status": "success",
                "attempts": attempt,
                "elapsed_seconds": round(elapsed, 2),
                "completed_at": now_iso(),
            }

        except PermanentError as exc:
            # Erro definitivo — não adianta retentar
            logger.error(f"Etapa [{step_name}] erro permanente: {exc}")
            return {
                "step": step_name,
                "status": "failed",
                "attempts": attempt,
                "elapsed_seconds": round(time.time() - t0, 2),
                "error_type": "permanent",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at": now_iso(),
            }

        except StepSkippedError as exc:
            logger.info(f"Etapa [{step_name}] pulada: {exc}")
            return {
                "step": step_name,
                "status": "skipped",
                "attempts": attempt,
                "reason": str(exc),
                "completed_at": now_iso(),
            }

        except Exception as exc:
            last_exc = exc
            logger.warning(f"Etapa [{step_name}] tentativa {attempt} falhou: {exc}")
            if attempt == max_attempts:
                break
            wait = min(delay, config.retry.max_delay)
            logger.info(f"Aguardando {wait:.1f}s antes de retentar a etapa inteira...")
            time.sleep(wait)
            delay *= 2.0

    # esgotou
    return {
        "step": step_name,
        "status": "failed",
        "attempts": max_attempts,
        "error_type": "transient_exhausted",
        "error_message": str(last_exc) if last_exc else "unknown",
        "traceback": traceback.format_exc() if last_exc else "",
        "completed_at": now_iso(),
    }


def run_pipeline(config: PipelineConfig, base_dir: Path) -> Dict[str, Any]:
    """
    Executa o pipeline completo para um canal.
    Retorna manifest com status de cada etapa.
    """
    cycle_dir = create_cycle(base_dir, config.channel.name)
    logger = setup_logging(cycle_dir)
    logger.info(f"### INICIANDO CICLO {cycle_dir.name} — canal '{config.channel.name}' ###")

    manifest: Dict[str, Any] = {
        "cycle_id": cycle_dir.name,
        "channel": config.channel.name,
        "started_at": now_iso(),
        "steps": [],
        "overall_status": "unknown",
    }

    stopped = False
    for step in STEPS:
        if stopped:
            manifest["steps"].append({
                "step": step["name"],
                "status": "skipped",
                "reason": "previous_step_failed",
            })
            continue

        result = _run_step_with_retry(step["name"], step["fn"], cycle_dir, config)
        manifest["steps"].append(result)

        if result["status"] == "failed":
            if step["required"] and not config.retry.continue_on_step_failure:
                stopped = True

    # Determina status geral
    statuses = [s["status"] for s in manifest["steps"]]
    if all(s == "success" for s in statuses):
        manifest["overall_status"] = "success"
    elif any(s == "failed" for s in statuses):
        manifest["overall_status"] = "partial_failure" if "success" in statuses else "failed"
    else:
        manifest["overall_status"] = "mixed"

    manifest["finished_at"] = now_iso()
    save_json(cycle_dir / "pipeline_manifest.json", manifest)
    logger.info(f"### FIM DO CICLO — status: {manifest['overall_status']} ###")
    return manifest


def main() -> int:
    """CLI entry point: python -m shorts_pipeline.orchestrator --config configs/drama.yml"""
    import argparse
    from .config import load_config

    parser = argparse.ArgumentParser(description="Executa o pipeline de shorts para um canal.")
    parser.add_argument("--config", required=True, help="Caminho para o arquivo YAML de config do canal")
    parser.add_argument("--base-dir", default="data", help="Pasta onde os ciclos são salvos")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    base_dir = Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_pipeline(config, base_dir)

    if manifest["overall_status"] == "success":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
