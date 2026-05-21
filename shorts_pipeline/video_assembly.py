"""
Etapa 5: Montagem final do vídeo via FFmpeg.

Abertura e encerramento são tratados como clipes normais de background —
entram no início e fim da lista de clipes, sem processamento especial.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mutagen.mp3 import MP3

from .config import PipelineConfig
from .utils import (
    PermanentError,
    get_logger,
    load_json,
    now_iso,
    save_json,
)


@dataclass
class TimedWord:
    word: str
    start: float
    end: float


def _check_deps() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise PermanentError("ffmpeg/ffprobe não encontrados no PATH.")


def _run_ffmpeg(cmd: List[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise PermanentError(
            f"Comando ffmpeg falhou: {' '.join(cmd[:5])}... stderr={result.stderr[-500:]}"
        )


def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise PermanentError(f"ffprobe falhou em {path}: {result.stderr}")
    return float(result.stdout.strip())


def _sanitize(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _format_ass_time(s: float) -> str:
    cs = int(round(s * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    sec, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _align_words(audio_path: Path, model_name: str) -> List[TimedWord]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path), language="pt", word_timestamps=True, vad_filter=True,
    )
    words: List[TimedWord] = []
    for seg in segments:
        for w in getattr(seg, "words", None) or []:
            if w.start is None or w.end is None:
                continue
            token = (w.word or "").strip()
            if token:
                words.append(TimedWord(token, float(w.start), float(w.end)))
    if not words:
        raise PermanentError("Whisper não retornou palavras com timestamp.")
    return words


def _build_srt(words: List[TimedWord], out: Path, max_per_caption: int) -> None:
    chunks = []
    current: List[TimedWord] = []
    start = None
    for w in words:
        if start is None:
            start = w.start
        current.append(w)
        if len(current) >= max_per_caption or w.word.endswith((".", "!", "?", ",")):
            chunks.append((start, current[-1].end, current[:]))
            current = []
            start = None
    if current:
        chunks.append((start, current[-1].end, current[:]))

    lines = []
    for i, (s, e, ws) in enumerate(chunks, start=1):
        lines.extend([str(i), f"{_format_srt_time(s)} --> {_format_srt_time(e)}",
                      " ".join(w.word for w in ws).strip(), ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def _build_ass(words: List[TimedWord], out: Path, config: PipelineConfig) -> None:
    v = config.video
    # Fonte maior, alinhamento topo-centro (8), margem do topo
    font_size = int(v.font_size * 1.1)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {v.target_width}
PlayResY: {v.target_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Karaoke,{v.font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,60,60,200,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events: List[str] = []
    current: List[TimedWord] = []
    block_start = None

    def flush(block: List[TimedWord], s: float, e: float):
        text = " ".join(_escape_ass(w.word) for w in block).strip()
        if text:
            events.append(
                f"Dialogue: 0,{_format_ass_time(s)},{_format_ass_time(e)},Karaoke,,0,0,0,,{text}"
            )

    for w in words:
        if block_start is None:
            block_start = w.start
        current.append(w)
        if len(current) >= v.max_words_per_caption or w.word.endswith((".", "!", "?", ",")):
            flush(current, block_start, current[-1].end)
            current = []
            block_start = None
    if current:
        flush(current, block_start, current[-1].end)

    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _list_clips(
    background_json: Dict[str, Any],
    visual_dir: Path,
    abertura: Optional[Path] = None,
    encerramento: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Monta a lista de clipes de background.
    Se abertura/encerramento existirem, entram como primeiro e último clipe.
    São tratados exatamente igual aos clipes do Pexels.
    """
    clips = []

    # Abertura entra primeiro
    if abertura and abertura.exists():
        dur = _ffprobe_duration(abertura)
        clips.append({"path": str(abertura), "duration": dur})

    # Clipes do Pexels
    for c in background_json.get("clips", []):
        p = Path(c.get("clip_file_path") or (visual_dir / c["clip_file"]))
        if not p.exists():
            raise PermanentError(f"Clipe não encontrado: {p}")
        clips.append({"path": str(p), "duration": float(c["duration_seconds"])})

    # Encerramento entra por último
    if encerramento and encerramento.exists():
        dur = _ffprobe_duration(encerramento)
        clips.append({"path": str(encerramento), "duration": dur})

    if not clips:
        raise PermanentError("Lista de clipes vazia.")
    return clips


def _make_background(
    audio_duration: float,
    clips: List[Dict[str, Any]],
    out_path: Path,
    config: PipelineConfig,
    abertura_dur: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Monta o background completo: abertura (se houver) + clipes Pexels em loop + encerramento (se houver).
    O background precisa ter exatamente a duração do áudio.
    Abertura e encerramento entram completos; os clipes do meio preenchem o restante.
    """
    v = config.video
    logger = get_logger()

    # Separa abertura, meio e encerramento
    abertura_clip = clips[0] if clips and clips[0].get("is_abertura") else None
    encerramento_clip = clips[-1] if clips and clips[-1].get("is_encerramento") else None

    # Recalcula: usa todos os clipes da lista (abertura já está incluída com duração real)
    # O background total precisa cobrir audio_duration * speed
    target = audio_duration * v.final_speed_multiplier

    plan: List[Dict[str, Any]] = []
    remaining = target
    i = 0
    while remaining > 0.01 and i < len(clips) * 10:
        c = clips[i % len(clips)]
        use = min(c["duration"], remaining)
        plan.append({"path": c["path"], "used_duration": use})
        remaining -= use
        i += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        segments: List[Path] = []

        for idx, item in enumerate(plan, start=1):
            seg = tmpdir / f"seg_{idx:03d}.mp4"
            _run_ffmpeg([
                "ffmpeg", "-y", "-ss", "0", "-i", item["path"],
                "-map", "0:v:0", "-t", f"{item['used_duration']:.3f}",
                "-vf",
                f"scale={v.target_width}:{v.target_height}:force_original_aspect_ratio=increase,"
                f"crop={v.target_width}:{v.target_height},setsar=1,fps={v.fps},format=yuv420p",
                "-an", "-c:v", "libx264", "-preset", "medium",
                "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                "-b:v", "8000k", "-maxrate", "8000k", "-bufsize", "16000k",
                "-r", str(v.fps), seg.as_posix(),
            ])
            segments.append(seg)

        concat_list = tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8"
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list.as_posix(),
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-maxrate", "8000k", "-bufsize", "16000k",
            "-r", str(v.fps), out_path.as_posix(),
        ])

    return plan


def _mux_final(
    background: Path,
    ass: Path,
    audio: Path,
    out: Path,
    config: PipelineConfig,
    musica: Optional[Path] = None,
) -> None:
    """
    Mux final: background + legendas + narração + música de fundo opcional.
    Tudo em um único comando FFmpeg.
    """
    v = config.video
    ass_path = str(ass.resolve()).replace("\\", "/").replace(":", r"\:")
    speed = v.final_speed_multiplier
    vf = f"ass='{ass_path}',setpts=PTS/{speed}" if speed != 1.0 else f"ass='{ass_path}'"
    af_narr = f"atempo={speed}" if speed != 1.0 else "anull"

    if musica and musica.exists():
        # Com música: narração (100%) + música de fundo (8%)
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(background),
            "-i", str(audio),
            "-stream_loop", "-1", "-i", str(musica),
            "-filter:v", vf,
            "-filter_complex",
            f"[1:a]{af_narr}[narr];"
            "[2:a]volume=0.08[music];"
            "[narr][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-maxrate", "8000k", "-bufsize", "16000k",
            "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-shortest", out.as_posix(),
        ])
    else:
        # Sem música: só narração
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(background), "-i", str(audio),
            "-filter:v", vf, "-filter:a", af_narr,
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-maxrate", "8000k", "-bufsize", "16000k",
            "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-shortest", out.as_posix(),
        ])


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    logger = get_logger()
    _check_deps()

    narration_dir = cycle_dir / "narration"
    visual_dir = cycle_dir / "visual-selection"
    assembly_dir = cycle_dir / "video-assembly"

    audio_path = narration_dir / "narration_asset.mp3"
    narration_json_path = narration_dir / "narration_asset.json"
    background_json_path = visual_dir / "background_clip.json"

    for p in (audio_path, narration_json_path, background_json_path):
        if not p.exists():
            raise PermanentError(f"Arquivo necessário não encontrado: {p}")

    narration_meta = load_json(narration_json_path)
    background_data = load_json(background_json_path)

    # Assets opcionais na raiz do repositório
    repo_root = Path.cwd()
    abertura = repo_root / "abertura.mp4"
    encerramento = repo_root / "encerramento.mp4"
    musica = repo_root / "musica_fundo.mp3"

    has_abertura = abertura.exists()
    has_encerramento = encerramento.exists()
    has_musica = musica.exists()
    logger.info(
        f"Assets: abertura={'✓' if has_abertura else '✗'} | "
        f"encerramento={'✓' if has_encerramento else '✗'} | "
        f"música={'✓' if has_musica else '✗'}"
    )

    audio_duration = float(MP3(audio_path).info.length)
    logger.info("Alinhando palavras com Whisper (pode demorar 1-2 min)...")
    words = _align_words(audio_path, config.video.whisper_model)

    srt_path = assembly_dir / "subtitles.srt"
    ass_path = assembly_dir / "subtitles.ass"
    _build_srt(words, srt_path, config.video.max_words_per_caption)
    _build_ass(words, ass_path, config)

    # Monta lista de clipes: abertura + Pexels + encerramento (todos iguais)
    clips = _list_clips(
        background_data, visual_dir,
        abertura=abertura if has_abertura else None,
        encerramento=encerramento if has_encerramento else None,
    )

    background_prep = assembly_dir / "assembled_video_background.mp4"
    plan = _make_background(audio_duration, clips, background_prep, config)
    save_json(assembly_dir / "clip_plan.json", {"cycle_id": cycle_dir.name, "clips": plan})

    # Mux final: background + legendas + narração + música (tudo junto, um comando)
    final_video = assembly_dir / "assembled_video.mp4"
    _mux_final(
        background=background_prep,
        ass=ass_path,
        audio=audio_path,
        out=final_video,
        config=config,
        musica=musica if has_musica else None,
    )

    try:
        background_prep.unlink()
    except FileNotFoundError:
        pass

    final_duration = _ffprobe_duration(final_video)
    payload = {
        "cycle_id": cycle_dir.name,
        "video_file": "assembled_video.mp4",
        "video_file_path": str(final_video),
        "subtitle_srt_file": srt_path.name,
        "subtitle_ass_file": ass_path.name,
        "audio_duration_seconds": narration_meta.get("duration_seconds"),
        "video_duration_seconds": round(final_duration, 3),
        "final_speed_multiplier": config.video.final_speed_multiplier,
        "clip_count_used": len(plan),
        "has_abertura": has_abertura,
        "has_encerramento": has_encerramento,
        "has_musica_fundo": has_musica,
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(assembly_dir / "assembled_video.json", payload)
    logger.info(f"Vídeo final: {final_video} ({final_duration:.1f}s)")
    return payload
