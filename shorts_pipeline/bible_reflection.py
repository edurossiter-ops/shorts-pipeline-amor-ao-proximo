"""
Etapa 1 (alternativa ao trend_detection): Seleção de versículo + tema + gancho.

Em vez de buscar conteúdo no Reddit/sites externos (que são bloqueados em
IPs de data center), esse módulo escolhe:
- uma combinação versículo+tema de um arquivo JSON local
- um gancho de abertura da lista do YAML do canal

Ambos servem como "seed" pro Claude gerar a reflexão cristã na próxima etapa.

Lógica:
1. Lê configs/versiculos.json (definido em config.bible.versiculos_file)
2. Gera TODAS as combinações possíveis (versículo × tema)
3. Lê histórico de combinações e ganchos usados recentemente
4. Filtra combinações disponíveis (não usadas nos últimos N dias)
5. Filtra ganchos disponíveis (não usados nos últimos M vídeos)
6. Se não sobra nenhuma: fallback na combinação/gancho mais antigo
7. Escolhe aleatoriamente
8. Salva payload compatível com o resto do pipeline
9. Atualiza históricos
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import PipelineConfig
from .utils import (
    PermanentError,
    get_control_dir,
    get_logger,
    load_json_if_exists,
    now_iso,
    save_json,
)


# Quantos dias uma combinação (versículo+tema) fica bloqueada após uso
RECENT_WINDOW_DAYS = 45

# Nomes dos arquivos de histórico (na pasta de controle do canal)
VERSICULOS_HISTORY = "versiculos_usados.json"
GANCHOS_HISTORY = "ganchos_usados.json"

# Quantas entradas manter nos históricos (limite de memória)
HISTORY_MAX_ENTRIES = 500


def _load_versiculos_file(path: Path) -> Dict[str, Any]:
    """Lê o arquivo versiculos.json e valida estrutura mínima."""
    if not path.exists():
        raise PermanentError(
            f"Arquivo de versículos não encontrado: {path}. "
            f"Configure 'bible.versiculos_file' no YAML do canal."
        )
    data = load_json_if_exists(path, None)
    if not data or "versiculos" not in data:
        raise PermanentError(
            f"Arquivo {path} não tem estrutura esperada (campo 'versiculos')."
        )
    versiculos = data["versiculos"]
    if not isinstance(versiculos, list) or not versiculos:
        raise PermanentError(f"Arquivo {path} tem lista de versículos vazia.")
    return data


def _generate_combinations(versiculos: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Gera todas as combinações possíveis (versiculo_id, tema).
    Retorna lista de tuplas: [("joao_3_16", "amor de Deus"), ("joao_3_16", "salvação"), ...]
    """
    combinations = []
    for v in versiculos:
        vid = v.get("id")
        temas = v.get("temas", [])
        if not vid or not temas:
            continue
        for tema in temas:
            combinations.append((vid, tema))
    return combinations


def _combo_key(versiculo_id: str, tema: str) -> str:
    """Chave única pra uma combinação versículo+tema (usada no histórico)."""
    return f"{versiculo_id}::{tema}"


def _parse_iso(iso_str: str) -> datetime:
    """Parse de data ISO com fallback seguro pra datas muito antigas."""
    try:
        # Python 3.11+ aceita 'Z' no fromisoformat, mas pra compatibilidade:
        s = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        # Fallback: assume "muito antigo" (1970)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _filter_available(
    all_combinations: List[Tuple[str, str]],
    history: List[Dict[str, Any]],
    window_days: int,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Retorna (combinações_disponíveis, quantas_foram_bloqueadas).
    Uma combinação está bloqueada se foi usada nos últimos `window_days`.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    # Monta set de combo_keys usadas recentemente
    recently_used_keys = set()
    for entry in history:
        used_at = _parse_iso(entry.get("used_at", ""))
        if used_at >= cutoff:
            key = entry.get("combo_key")
            if key:
                recently_used_keys.add(key)

    available = [
        (vid, tema)
        for vid, tema in all_combinations
        if _combo_key(vid, tema) not in recently_used_keys
    ]
    blocked_count = len(all_combinations) - len(available)
    return available, blocked_count


def _fallback_oldest(
    all_combinations: List[Tuple[str, str]],
    history: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Quando todas as combinações estão bloqueadas, escolhe a mais antiga
    (a que está há mais tempo sem ser usada).
    """
    # Mapa combo_key -> data do último uso
    last_used: Dict[str, datetime] = {}
    for entry in history:
        key = entry.get("combo_key")
        if not key:
            continue
        used_at = _parse_iso(entry.get("used_at", ""))
        # Se a chave já existe, mantém a mais recente
        if key not in last_used or used_at > last_used[key]:
            last_used[key] = used_at

    # Ordena combinações pela data de último uso (mais antiga primeiro)
    # Combinações sem histórico ficam no começo (improvável nesse fallback)
    def sort_key(combo: Tuple[str, str]) -> datetime:
        return last_used.get(_combo_key(combo[0], combo[1]), datetime(1970, 1, 1, tzinfo=timezone.utc))

    sorted_combos = sorted(all_combinations, key=sort_key)
    return sorted_combos[0]


def _choose_hook(
    ganchos: List[str],
    history: List[Dict[str, Any]],
    avoid_last_n: int,
) -> Tuple[str, bool]:
    """
    Escolhe um gancho sorteando aleatoriamente entre os que não foram usados
    nos últimos `avoid_last_n` vídeos.

    Se todos os ganchos disponíveis estiverem bloqueados, usa fallback
    (gancho que está há mais tempo sem ser usado).

    Retorna: (gancho_escolhido, used_fallback).
    """
    if not ganchos:
        raise PermanentError(
            "Lista de ganchos vazia no config. "
            "Adicione ganchos em bible.ganchos do YAML do canal."
        )

    # Pega os últimos N ganchos usados (history já está em ordem: mais recente primeiro)
    recently_used = set()
    for entry in history[:avoid_last_n]:
        gancho = entry.get("gancho")
        if gancho:
            recently_used.add(gancho)

    available = [g for g in ganchos if g not in recently_used]

    if available:
        return random.choice(available), False

    # Fallback: todos os ganchos foram usados recentemente
    # Escolhe o que está há mais tempo sem ser usado
    # (i.e., o que NÃO aparece nos últimos N, ou o que aparece mais pro fim)
    used_order = {}  # gancho -> posição (quanto menor, mais recente)
    for pos, entry in enumerate(history[:avoid_last_n]):
        g = entry.get("gancho")
        if g and g not in used_order:
            used_order[g] = pos

    # Escolhe o gancho com maior posição (mais antigo) — ou não no histórico
    def staleness(gancho: str) -> int:
        # Quanto maior o número, mais "velho" — ganchos sem histórico ficam mais velhos
        return used_order.get(gancho, avoid_last_n + 1)

    sorted_ganchos = sorted(ganchos, key=staleness, reverse=True)
    return sorted_ganchos[0], True


def _build_payload(
    cycle_dir: Path,
    versiculo: Dict[str, Any],
    tema_escolhido: str,
    gancho_escolhido: str,
    gancho_fallback: bool,
    blocked_count: int,
    available_count: int,
    used_fallback: bool,
) -> Dict[str, Any]:
    """Monta o payload que vai pra topic_candidate.json."""
    vid = versiculo["id"]
    return {
        "cycle_id": cycle_dir.name,
        "status": "success",
        "selected_at": now_iso(),
        "topic_source": "bible",
        "topic_source_id": _combo_key(vid, tema_escolhido),
        "versiculo_id": vid,
        "versiculo_ref": versiculo.get("ref", ""),
        "versiculo_texto": versiculo.get("texto", ""),
        "categoria": versiculo.get("categoria", ""),
        "tema_escolhido": tema_escolhido,
        "temas_disponiveis": versiculo.get("temas", []),
        "gancho_escolhido": gancho_escolhido,
        "one_line_summary": f"{versiculo.get('ref', '')} — reflexão sobre {tema_escolhido}",
        # Campos de compatibilidade com o pipeline (antes vinham do Reddit)
        "pillar_hint": "reflexao_crista",
        "theme_hint": tema_escolhido,
        "source_title": versiculo.get("ref", ""),
        "source_text": versiculo.get("texto", ""),
        "topic_descriptor": f"{versiculo.get('ref', '')}: {tema_escolhido}",
        # Metadados de seleção
        "blocked_combinations": blocked_count,
        "available_combinations": available_count,
        "used_fallback": used_fallback,
        "gancho_fallback": gancho_fallback,
    }


def _update_history(
    history: List[Dict[str, Any]],
    cycle_id: str,
    versiculo_id: str,
    tema: str,
) -> List[Dict[str, Any]]:
    """Adiciona nova entrada no topo e corta na capacidade máxima."""
    new_entry = {
        "cycle_id": cycle_id,
        "combo_key": _combo_key(versiculo_id, tema),
        "versiculo_id": versiculo_id,
        "tema": tema,
        "used_at": now_iso(),
    }
    return [new_entry] + history[: HISTORY_MAX_ENTRIES - 1]


def _update_ganchos_history(
    history: List[Dict[str, Any]],
    cycle_id: str,
    gancho: str,
) -> List[Dict[str, Any]]:
    """Adiciona novo gancho no topo do histórico."""
    new_entry = {
        "cycle_id": cycle_id,
        "gancho": gancho,
        "used_at": now_iso(),
    }
    return [new_entry] + history[: HISTORY_MAX_ENTRIES - 1]


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    """
    Executa a seleção de versículo + tema + gancho para o ciclo.
    Retorna o payload salvo em topic_candidate.json.
    """
    logger = get_logger()

    # Pastas de I/O (seguem a mesma convenção que trend_detection antigo)
    output_dir = cycle_dir / "bible-reflection"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "topic_candidate.json"

    control_dir = get_control_dir(cycle_dir)
    versiculos_history_file = control_dir / VERSICULOS_HISTORY
    ganchos_history_file = control_dir / GANCHOS_HISTORY

    # Caminho do arquivo de versículos vem do config do canal
    versiculos_path = Path(config.bible.versiculos_file)
    if not versiculos_path.is_absolute():
        # Se for relativo, resolve a partir da raiz do projeto (diretório de trabalho)
        versiculos_path = Path.cwd() / versiculos_path

    logger.info(f"Lendo versículos de {versiculos_path}")
    data = _load_versiculos_file(versiculos_path)
    versiculos = data["versiculos"]

    # --- PARTE 1: escolher versículo + tema ---
    all_combos = _generate_combinations(versiculos)
    logger.info(
        f"Total de combinações disponíveis: {len(all_combos)} "
        f"({len(versiculos)} versículos)"
    )

    history_data = load_json_if_exists(versiculos_history_file, {"items": []})
    history = history_data.get("items", [])

    available, blocked_count = _filter_available(all_combos, history, RECENT_WINDOW_DAYS)
    logger.info(
        f"Combinações bloqueadas (últimos {RECENT_WINDOW_DAYS} dias): {blocked_count} | "
        f"disponíveis: {len(available)}"
    )

    used_fallback = False
    if available:
        chosen_id, chosen_tema = random.choice(available)
    else:
        logger.warning(
            "Todas as combinações estão bloqueadas. "
            "Usando fallback: combinação mais antiga."
        )
        chosen_id, chosen_tema = _fallback_oldest(all_combos, history)
        used_fallback = True

    versiculo = next((v for v in versiculos if v["id"] == chosen_id), None)
    if versiculo is None:
        raise PermanentError(
            f"Bug: versículo id={chosen_id} não encontrado no arquivo."
        )

    # --- PARTE 2: escolher gancho ---
    ganchos_history_data = load_json_if_exists(ganchos_history_file, {"items": []})
    ganchos_history = ganchos_history_data.get("items", [])

    gancho_escolhido, gancho_fallback = _choose_hook(
        config.bible.ganchos,
        ganchos_history,
        config.bible.recent_hooks_avoid,
    )
    logger.info(
        f"Gancho escolhido: \"{gancho_escolhido}\" "
        f"{'(FALLBACK)' if gancho_fallback else ''}"
    )

    # --- PARTE 3: salvar payload e atualizar históricos ---
    payload = _build_payload(
        cycle_dir=cycle_dir,
        versiculo=versiculo,
        tema_escolhido=chosen_tema,
        gancho_escolhido=gancho_escolhido,
        gancho_fallback=gancho_fallback,
        blocked_count=blocked_count,
        available_count=len(available),
        used_fallback=used_fallback,
    )
    save_json(output_file, payload)
    logger.info(
        f"Seed escolhido: {versiculo.get('ref')} | tema: {chosen_tema} "
        f"{'(FALLBACK)' if used_fallback else ''}"
    )

    # Atualiza os dois históricos
    updated_history = _update_history(history, cycle_dir.name, chosen_id, chosen_tema)
    save_json(versiculos_history_file, {"items": updated_history})

    updated_ganchos = _update_ganchos_history(ganchos_history, cycle_dir.name, gancho_escolhido)
    save_json(ganchos_history_file, {"items": updated_ganchos})

    return payload
