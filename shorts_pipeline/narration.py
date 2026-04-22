"""
Etapa 3: Narração via ElevenLabs.

Refatorado para:
- voice_id, model_id, speed via config
- API key via env var
- Retry com classificação de erro
- Divisão automática de textos longos em múltiplas chamadas
- Concatenação via ffmpeg
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from mutagen.mp3 import MP3

from .config import PipelineConfig
from .utils import (
    PermanentError,
    get_logger,
    http_request_with_retry,
    now_iso,
    save_json,
)


def _split_text(text: str, max_chars: int) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue
        extra = len(p) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            parts.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += extra if current != [p] else len(p)

    if current:
        parts.append("\n\n".join(current))

    # segurança extra: quebra brutalmente se necessário
    final: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            for start in range(0, len(part), max_chars):
                final.append(part[start:start + max_chars])
    return final


def _call_elevenlabs(config: PipelineConfig, text: str) -> bytes:
    api_key = config.secrets.get(config.secrets.elevenlabs_api_key_env, required=True)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.narration.voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": config.narration.model_id,
        "voice_settings": {"speed": config.narration.speed},
    }

    resp = http_request_with_retry(
        "POST", url, headers=headers, json=payload, timeout=240, max_attempts=4, initial_delay=5.0
    )
    return resp.content


def _concat_mp3(parts: List[Path], final: Path) -> None:
    if len(parts) == 1:
        final.write_bytes(parts[0].read_bytes())
        return

    list_file = final.parent / "concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.name}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(final),
    ]
    subprocess.run(cmd, check=True, cwd=str(final.parent),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        list_file.unlink()
    except FileNotFoundError:
        pass


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    logger = get_logger()
    story_dir = cycle_dir / "story-generation"
    narration_dir = cycle_dir / "narration"

    text_file = story_dir / "story_text_formatted.txt"
    if not text_file.exists():
        raise PermanentError(f"story_text_formatted.txt não encontrado: {text_file}")

    text = text_file.read_text(encoding="utf-8").strip()
    if not text:
        raise PermanentError("story_text_formatted.txt está vazio.")

    parts = _split_text(text, config.narration.max_chars_per_call)
    logger.info(f"Texto de {len(text)} chars dividido em {len(parts)} parte(s).")

    audio_parts: List[Path] = []
    for i, part in enumerate(parts, start=1):
        logger.info(f"Gerando narração parte {i}/{len(parts)} ({len(part)} chars)")
        audio = _call_elevenlabs(config, part)
        p = narration_dir / f"narration_part_{i:02d}.mp3"
        p.write_bytes(audio)
        audio_parts.append(p)

    final_audio = narration_dir / "narration_asset.mp3"
    _concat_mp3(audio_parts, final_audio)

    mp3 = MP3(final_audio)
    duration = float(mp3.info.length)

    payload = {
        "cycle_id": cycle_dir.name,
        "audio_file": "narration_asset.mp3",
        "audio_file_path": str(final_audio),
        "voice_provider": "elevenlabs",
        "voice_id": config.narration.voice_id,
        "model_id": config.narration.model_id,
        "speed": config.narration.speed,
        "duration_seconds": round(duration, 3),
        "duration_ms": int(round(duration * 1000)),
        "file_size_bytes": final_audio.stat().st_size,
        "text_char_count": len(text),
        "text_word_count": len(text.split()),
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(narration_dir / "narration_asset.json", payload)
    logger.info(f"Narração gerada: {duration:.2f}s")
    return payload
