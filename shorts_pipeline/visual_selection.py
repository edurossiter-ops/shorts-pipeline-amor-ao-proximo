"""
Etapa 4: Seleção de vídeos de fundo via Pexels.

Refatorado para:
- Queries configuráveis por canal (drama urbano vs cristão vs motivacional)
- API key via env var
- Lista de bloqueio de palavras configurável
- Retry em erros transitórios
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import PipelineConfig
from .utils import (
    PermanentError,
    get_control_dir,
    get_logger,
    http_request_with_retry,
    load_json,
    load_json_if_exists,
    now_iso,
    save_json,
)


PEXELS_API_URL = "https://api.pexels.com/videos/search"


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s:/.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_headers(config: PipelineConfig) -> Dict[str, str]:
    api_key = config.secrets.get(config.secrets.pexels_api_key_env, required=True)
    return {
        "Authorization": api_key,
        "Accept": "application/json",
        "User-Agent": "shorts-pipeline-visual/1.0",
    }


def _search_videos(config: PipelineConfig, query: str, page: int) -> Dict[str, Any]:
    params = {
        "query": query,
        "per_page": config.visual.per_page,
        "page": page,
        "orientation": "portrait",
    }
    headers = _build_headers(config)
    resp = http_request_with_retry(
        "GET", PEXELS_API_URL, headers=headers, params=params, timeout=60, max_attempts=3
    )
    return resp.json()


def _is_portrait(v: Dict[str, Any]) -> bool:
    w, h = v.get("width"), v.get("height")
    return isinstance(w, int) and isinstance(h, int) and h > w


def _duration_ok(v: Dict[str, Any], min_s: int, max_s: int) -> bool:
    d = v.get("duration")
    return isinstance(d, (int, float)) and min_s <= float(d) <= max_s


def _pick_best_file(v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    files = v.get("video_files") or []
    target = 1080 * 1920
    portraits = [
        f for f in files
        if f.get("link") and isinstance(f.get("width"), int) and isinstance(f.get("height"), int)
        and f["height"] > f["width"]
        and ("mp4" in str(f.get("file_type", "")).lower() or str(f["link"]).lower().endswith(".mp4"))
    ]
    if not portraits:
        return None
    portraits.sort(key=lambda f: (abs(f["width"] * f["height"] - target), -f["width"] * f["height"]))
    return portraits[0]


def _is_safe(v: Dict[str, Any], query: str, blocked_kw: List[str]) -> bool:
    user = v.get("user") or {}
    haystack = " ".join([
        _normalize(str(v.get("title", ""))),
        _normalize(str(v.get("url", ""))),
        _normalize(query),
        _normalize(str(user.get("name", ""))),
    ])
    for kw in blocked_kw:
        if _normalize(kw) in haystack:
            return False
    return True


def _signature(v: Dict[str, Any], query: str) -> str:
    user = v.get("user") or {}
    composite = "|".join([
        query,
        _normalize(str(user.get("name", ""))),
        str(v.get("width", "")),
        str(v.get("height", "")),
        str(int(round(float(v.get("duration", 0))))),
    ])
    return hashlib.sha1(composite.encode("utf-8")).hexdigest()


def _download(url: str, dest: Path) -> None:
    resp = http_request_with_retry("GET", url, timeout=240, stream=True, max_attempts=3)
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    logger = get_logger()
    narration_dir = cycle_dir / "narration"
    visual_dir = cycle_dir / "visual-selection"
    control_dir = get_control_dir(cycle_dir)

    narration_json = narration_dir / "narration_asset.json"
    if not narration_json.exists():
        raise PermanentError(f"narration_asset.json não encontrado: {narration_json}")
    narration = load_json(narration_json)
    audio_duration = float(narration["duration_seconds"])

    # Quanto de vídeo precisamos (considerando o final speed multiplier)
    target_duration = audio_duration * config.video.final_speed_multiplier + 2.0

    recent_file = control_dir / "recent_backgrounds.json"
    recent_items = load_json_if_exists(recent_file, {"items": []}).get("items", [])
    blocked_ids = set()
    for item in recent_items:
        for vid in item.get("video_ids", []):
            if isinstance(vid, int):
                blocked_ids.add(vid)

    queries = list(config.visual.queries)
    random.shuffle(queries)

    chosen: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_sigs = set()
    total = 0.0

    for query in queries:
        if total >= target_duration:
            break
        clips_from_query = 0
        for page in range(1, config.visual.max_pages_per_query + 1):
            if total >= target_duration or clips_from_query >= config.visual.max_clips_per_query:
                break
            try:
                data = _search_videos(config, query, page)
            except Exception as exc:
                logger.warning(f"Erro na busca '{query}' página {page}: {exc}")
                break

            videos = data.get("videos") or []
            if not videos:
                break

            for v in videos:
                if total >= target_duration or clips_from_query >= config.visual.max_clips_per_query:
                    break
                vid_id = v.get("id")
                if not isinstance(vid_id, int) or vid_id in blocked_ids or vid_id in seen_ids:
                    continue
                if not _is_portrait(v) or not _duration_ok(v, config.visual.min_clip_seconds, config.visual.max_clip_seconds):
                    continue
                if not _is_safe(v, query, config.visual.blocked_keywords):
                    continue
                picked = _pick_best_file(v)
                if not picked:
                    continue
                sig = _signature(v, query)
                if sig in seen_sigs:
                    continue

                duration = float(v["duration"])
                clip_idx = len(chosen) + 1
                dest = visual_dir / f"background_clip_{clip_idx:03d}.mp4"
                logger.info(f"Baixando {dest.name} | query='{query}' | {duration:.1f}s")
                try:
                    _download(picked["link"], dest)
                except Exception as exc:
                    logger.warning(f"Falha ao baixar {dest.name}: {exc}")
                    continue

                chosen.append({
                    "clip_index": clip_idx,
                    "clip_file": dest.name,
                    "clip_file_path": str(dest),
                    "video_id": vid_id,
                    "query": query,
                    "duration_seconds": duration,
                    "width": v.get("width"),
                    "height": v.get("height"),
                    "url": v.get("url"),
                })
                seen_ids.add(vid_id)
                seen_sigs.add(sig)
                total += duration
                clips_from_query += 1

    if not chosen:
        raise PermanentError(
            "Nenhum vídeo elegível encontrado no Pexels. "
            "Revise as queries de config.visual.queries ou as listas de bloqueio."
        )

    payload = {
        "cycle_id": cycle_dir.name,
        "narration_duration_seconds": audio_duration,
        "required_source": "pexels",
        "target_total_duration_seconds": round(target_duration, 3),
        "downloaded_total_duration_seconds": round(total, 3),
        "selected_clip_count": len(chosen),
        "queries_used": config.visual.queries,
        "clips": chosen,
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(visual_dir / "background_clip.json", payload)

    # Atualiza recent
    new_item = {
        "cycle_id": cycle_dir.name,
        "video_ids": [c["video_id"] for c in chosen],
        "saved_at": now_iso(),
    }
    updated = [new_item] + recent_items[: config.visual.recent_backgrounds_limit - 1]
    save_json(recent_file, {"items": updated})

    logger.info(f"Selecionados {len(chosen)} clipes totalizando {total:.1f}s")
    return payload
