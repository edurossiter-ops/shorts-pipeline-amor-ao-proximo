"""
Worker isolado de Whisper.

Roda em processo filho separado pra evitar contenção de memória/CPU com o
pipeline principal (que tem vários módulos pesados carregados como Anthropic
SDK, ElevenLabs, Pillow, Pexels, FFmpeg, etc.).

Uso (invocado via subprocess pelo video_assembly.py):
    python -m shorts_pipeline._whisper_worker \
        --audio /caminho/audio.mp3 \
        --model small \
        --output /caminho/words.json

Saída: arquivo JSON com lista de palavras timestampadas, formato:
[
  {"word": "Esta", "start": 0.0, "end": 0.42},
  ...
]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper worker (processo isolado)")
    parser.add_argument("--audio", required=True, help="Caminho pro .mp3")
    parser.add_argument("--model", default="small", help="Nome do modelo Whisper")
    parser.add_argument("--output", required=True, help="Caminho pro JSON de saída")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_path = Path(args.output)

    if not audio_path.exists():
        print(f"[worker] ERRO: áudio não encontrado: {audio_path}", file=sys.stderr)
        return 2

    print(f"[worker] Carregando modelo {args.model}...", flush=True)
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"[worker] Transcrevendo {audio_path.name}...", flush=True)
    segments, info = model.transcribe(
        str(audio_path), language="pt", word_timestamps=True, vad_filter=False,
    )

    words = []
    for seg in segments:
        for w in getattr(seg, "words", None) or []:
            if w.start is None or w.end is None:
                continue
            token = (w.word or "").strip()
            if token:
                words.append({
                    "word": token,
                    "start": float(w.start),
                    "end": float(w.end),
                })

    print(f"[worker] {len(words)} palavras extraídas", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)

    print(f"[worker] JSON salvo em {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
