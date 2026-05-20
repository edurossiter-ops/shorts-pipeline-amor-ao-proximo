"""
Etapa 5: Montagem final do vídeo via FFmpeg.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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


def _build_srt(words: List[TimedWord], out: Path, max_per_caption: int,
               time_offset: float = 0.0) -> None:
    """time_offset: segundos a somar em todos os timestamps (pra compensar abertura)."""
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
        lines.extend([
            str(i),
            f"{_format_srt_time(s + time_offset)} --> {_format_srt_time(e + time_offset)}",
            " ".join(w.word for w in ws).strip(), "",
        ])
    out.write_text("\n".join(lines), encoding="utf-8")


def _build_ass(words: List[TimedWord], out: Path, config: PipelineConfig,
               time_offset: float = 0.0) -> None:
    """
    time_offset: segundos a somar em todos os timestamps (pra compensar abertura).
    Legenda centralizada no TOPO (Alignment=8), fonte maior, margem superior.
    """
    v = config.video
    # Fonte 10% maior que o configurado, alinhamento topo-centro (8)
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
    # Alignment=8 = topo centralizado
    # MarginV=200 = 200px de margem do topo

    events: List[str] = []
    current: List[TimedWord] = []
    block_start = None

    def flush(block: List[TimedWord], s: float, e: float):
        text = " ".join(_escape_ass(w.word) for w in block).strip()
        if text:
            ts = _format_ass_time(s + time_offset)
            te = _format_ass_time(e + time_offset)
            events.append(f"Dialogue: 0,{ts},{te},Karaoke,,0,0,0,,{text}")

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


def _list_clips(background_json: Dict[str, Any], visual_dir: Path) -> List[Dict[str, Any]]:
    clips = []
    for c in background_json.get("clips", []):
        p = Path(c.get("clip_file_path") or (visual_dir / c["clip_file"]))
        if not p.exists():
            raise PermanentError(f"Clipe não encontrado: {p}")
        clips.append({"path": str(p), "duration": float(c["duration_seconds"])})
    if not clips:
        raise PermanentError("Lista de clipes vazia em background_clip.json")
    return clips


def _make_background(
    audio_duration: float,
    clips: List[Dict[str, Any]],
    out_path: Path,
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    v = config.video
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
    background: Path, ass: Path, audio: Path, out: Path, config: PipelineConfig
) -> None:
    v = config.video
    ass_path = str(ass.resolve()).replace("\\", "/").replace(":", r"\:")
    speed = v.final_speed_multiplier
    vf = f"ass='{ass_path}',setpts=PTS/{speed}" if speed != 1.0 else f"ass='{ass_path}'"
    af = f"atempo={speed}" if speed != 1.0 else "anull"

    _run_ffmpeg([
        "ffmpeg", "-y", "-i", str(background), "-i", str(audio),
        "-filter:v", vf, "-filter:a", af,
        "-c:v", "libx264", "-preset", "medium",
        "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
        "-b:v", "8000k", "-maxrate", "8000k", "-bufsize", "16000k",
        "-r", str(v.fps),
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-shortest", out.as_posix(),
    ])


def _normalize_clip_for_concat(src: Path, out: Path, config: PipelineConfig) -> None:
    """
    Normaliza abertura/encerramento pra mesma resolução, FPS e codec da reflexão.
    Clipes mudos recebem trilha de silêncio pra o concat funcionar.
    """
    v = config.video

    # Verifica se o clipe tem áudio
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    src_has_audio = "audio" in probe.stdout

    if src_has_audio:
        # Clipe com áudio — normaliza normalmente
        audio_args = ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
    else:
        # Clipe mudo — gera trilha de silêncio sintética
        audio_args = [
            "-f", "lavfi", "-i", "aevalsrc=0:sample_rate=44100:channel_layout=stereo",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        ]

    dur = _ffprobe_duration(src)

    if src_has_audio:
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(src),
            "-vf",
            f"scale={v.target_width}:{v.target_height}:force_original_aspect_ratio=increase,"
            f"crop={v.target_width}:{v.target_height},setsar=1,fps={v.fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-r", str(v.fps),
            *audio_args,
            out.as_posix(),
        ])
    else:
        # Clipe mudo: vídeo do src + silêncio gerado sinteticamente
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(src),
            "-f", "lavfi", "-i", f"aevalsrc=0:sample_rate=44100:channel_layout=stereo:duration={dur}",
            "-map", "0:v", "-map", "1:a",
            "-vf",
            f"scale={v.target_width}:{v.target_height}:force_original_aspect_ratio=increase,"
            f"crop={v.target_width}:{v.target_height},setsar=1,fps={v.fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-shortest",
            out.as_posix(),
        ])


def _add_intro_outro_music(
    reflexao: Path,
    ass_path: Path,
    out: Path,
    config: PipelineConfig,
    words: List[TimedWord],
    assembly_dir: Path,
) -> None:
    """
    1. Detecta duração da abertura (offset das legendas)
    2. Reconstrói ASS com offset pra cobrir abertura + reflexão + encerramento
    3. Normaliza abertura e encerramento com silêncio sintético
    4. Concatena: abertura + reflexão + encerramento
    5. Queima legendas no vídeo concatenado
    6. Mixa musica_fundo.mp3 em todo o vídeo (narração mais alta, música suave)
    """
    logger = get_logger()
    v = config.video

    repo_root = Path.cwd()
    abertura_src = repo_root / "abertura.mp4"
    encerramento_src = repo_root / "encerramento.mp4"
    musica_src = repo_root / "musica_fundo.mp3"

    has_abertura = abertura_src.exists()
    has_encerramento = encerramento_src.exists()
    has_musica = musica_src.exists()

    logger.info(
        f"Assets: abertura={'✓' if has_abertura else '✗'} | "
        f"encerramento={'✓' if has_encerramento else '✗'} | "
        f"música={'✓' if has_musica else '✗'}"
    )

    if not has_abertura and not has_encerramento and not has_musica:
        logger.info("Nenhum asset encontrado — pulando abertura/encerramento/música.")
        return

    # Calcula offset das legendas = duração da abertura
    abertura_dur = _ffprobe_duration(abertura_src) if has_abertura else 0.0
    logger.info(f"Offset das legendas: {abertura_dur:.2f}s (duração da abertura)")

    # Reconstrói ASS com offset pra cobrir todo o vídeo final
    ass_final = assembly_dir / "subtitles_final.ass"
    _build_ass(words, ass_final, config, time_offset=abertura_dur)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        parts_to_concat: List[Path] = []

        # 1. Normaliza abertura (com silêncio sintético)
        if has_abertura:
            abertura_norm = tmpdir / "abertura_norm.mp4"
            logger.info("Normalizando abertura...")
            _normalize_clip_for_concat(abertura_src, abertura_norm, config)
            parts_to_concat.append(abertura_norm)

        # 2. Re-encode reflexão garantindo áudio explícito
        reflexao_norm = tmpdir / "reflexao_norm.mp4"
        logger.info("Normalizando reflexão pra concat...")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(reflexao),
            "-map", "0:v:0", "-map", "0:a:0",   # força mapear vídeo E áudio explicitamente
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            reflexao_norm.as_posix(),
        ])
        parts_to_concat.append(reflexao_norm)

        # 3. Normaliza encerramento (com silêncio sintético)
        if has_encerramento:
            encerramento_norm = tmpdir / "encerramento_norm.mp4"
            logger.info("Normalizando encerramento...")
            _normalize_clip_for_concat(encerramento_src, encerramento_norm, config)
            parts_to_concat.append(encerramento_norm)

        # 4. Concatena tudo (agora todos os clipes têm áudio)
        concat_list = tmpdir / "concat_final.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in parts_to_concat),
            encoding="utf-8",
        )
        concatenado_sem_leg = tmpdir / "concatenado_sem_leg.mp4"
        logger.info(f"Concatenando {len(parts_to_concat)} partes...")
        _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list.as_posix(),
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            concatenado_sem_leg.as_posix(),
        ])

        # 5. Queima legendas no vídeo concatenado (com offset correto)
        ass_final_path = str(ass_final.resolve()).replace("\\", "/").replace(":", r"\:")
        concatenado = tmpdir / "concatenado.mp4"
        logger.info("Queimando legendas com offset...")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", concatenado_sem_leg.as_posix(),
            "-vf", f"ass='{ass_final_path}'",
            "-map", "0:v", "-map", "0:a",
            "-c:v", "libx264", "-preset", "medium",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-b:v", "8000k", "-r", str(v.fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            concatenado.as_posix(),
        ])

        # 6. Mixa música de fundo
        if has_musica:
            logger.info("Mixando música de fundo (narração 100%, música 8%)...")
            video_dur = _ffprobe_duration(concatenado)

            # narração em volume total (1.0), música bem suave (0.08)
            filter_complex = (
                "[0:a]volume=1.0[narr];"
                "[1:a]volume=0.08[music];"
                "[narr][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", concatenado.as_posix(),
                "-stream_loop", "-1", "-i", str(musica_src),
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "medium",
                "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                "-b:v", "8000k", "-r", str(v.fps),
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-t", str(video_dur),
                out.as_posix(),
            ])
        else:
            import shutil as _shutil
            _shutil.copy2(str(concatenado), str(out))

    logger.info(f"Vídeo final com abertura/encerramento/música: {out}")


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
    background = load_json(background_json_path)

    audio_duration = float(MP3(audio_path).info.length)
    logger.info("Alinhando palavras com Whisper (pode demorar 1-2 min)...")
    words = _align_words(audio_path, config.video.whisper_model)

    # ASS inicial (sem offset) pra mux da reflexão
    srt_path = assembly_dir / "subtitles.srt"
    ass_path = assembly_dir / "subtitles.ass"
    _build_srt(words, srt_path, config.video.max_words_per_caption)
    _build_ass(words, ass_path, config)

    clips = _list_clips(background, visual_dir)
    background_prep = assembly_dir / "assembled_video_background.mp4"
    plan = _make_background(audio_duration, clips, background_prep, config)
    save_json(assembly_dir / "clip_plan.json", {"cycle_id": cycle_dir.name, "clips": plan})

    # Vídeo da reflexão com legendas (sem abertura/encerramento)
    reflexao_video = assembly_dir / "assembled_video_reflexao.mp4"
    _mux_final(background_prep, ass_path, audio_path, reflexao_video, config)

    try:
        background_prep.unlink()
    except FileNotFoundError:
        pass

    # Adiciona abertura, encerramento, reconstrói legendas com offset e mixa música
    final_video = assembly_dir / "assembled_video.mp4"
    _add_intro_outro_music(
        reflexao=reflexao_video,
        ass_path=ass_path,
        out=final_video,
        config=config,
        words=words,
        assembly_dir=assembly_dir,
    )

    if not final_video.exists():
        reflexao_video.rename(final_video)
    else:
        try:
            reflexao_video.unlink()
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
        "has_abertura": (Path.cwd() / "abertura.mp4").exists(),
        "has_encerramento": (Path.cwd() / "encerramento.mp4").exists(),
        "has_musica_fundo": (Path.cwd() / "musica_fundo.mp3").exists(),
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(assembly_dir / "assembled_video.json", payload)
    logger.info(f"Vídeo final: {final_video} ({final_duration:.1f}s)")
    return payload
