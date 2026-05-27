"""
Etapa 3: Narração via Google Cloud Text-to-Speech (Gemini 2.5 Flash).

Substituição do ElevenLabs por Google TTS gratuito.
Modelo: Gemini 2.5 Flash (pt-BR-Neural2-B voz masculina)
"""
from __future__ import annotations

import base64
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

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Voz masculina neural em português brasileiro
VOICE_NAME = "pt-BR-Neural2-B"
LANGUAGE_CODE = "pt-BR"


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

    final: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            for start in range(0, len(part), max_chars):
                final.append(part[start:start + max_chars])
    return final


def _call_google_tts(api_key: str, text: str) -> bytes:
    """
    Chama Google TTS e retorna bytes do MP3.
    Usa modelo Gemini 2.5 Flash com voz neural pt-BR.
    """
    url = f"{GOOGLE_TTS_URL}?key={api_key}"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": LANGUAGE_CODE,
            "name": VOICE_NAME,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 1.1,   # levemente mais rápido — tom de oração com energia
            "pitch": -2.0,         # voz um pouco mais grave — tom solene e masculino
        },
    }

    resp = http_request_with_retry(
        "POST", url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
        max_attempts=3,
        initial_delay=3.0,
    )

    data = resp.json()

    if "audioContent" not in data:
        raise PermanentError(
            f"Google TTS não retornou audioContent. Resposta: {data}"
        )

    return base64.b64decode(data["audioContent"])


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

    # Lê API key do Google TTS
    import os
    api_key = os.environ.get("GOOGLE_TTS_API_KEY")
    if not api_key:
        raise PermanentError("Variável de ambiente GOOGLE_TTS_API_KEY não definida.")

    # Google TTS aceita até 5000 chars por chamada
    parts = _split_text(text, 4900)
    logger.info(f"Texto de {len(text)} chars dividido em {len(parts)} parte(s).")

    audio_parts: List[Path] = []
    for i, part in enumerate(parts, start=1):
        logger.info(f"Gerando narração parte {i}/{len(parts)} ({len(part)} chars)")
        audio = _call_google_tts(api_key, part)
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
        "voice_provider": "google_tts",
        "voice_name": VOICE_NAME,
        "language_code": LANGUAGE_CODE,
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
