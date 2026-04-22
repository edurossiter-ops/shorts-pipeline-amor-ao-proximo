"""
Carregamento e validação de configuração do pipeline.

A configuração é lida de um arquivo YAML + variáveis de ambiente para secrets.
NUNCA guarde API keys no YAML — só referências a env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .utils import PermanentError


@dataclass
class Channel:
    name: str
    youtube_channel_id: Optional[str] = None  # opcional; None = canal padrão da conta


@dataclass
class SubredditGroup:
    pillar: str
    theme: str
    subs: List[str]


@dataclass
class TrendConfig:
    subreddits: List[SubredditGroup]
    max_posts_per_subreddit: int = 20
    recent_trends_limit: int = 10
    min_relevance_score: float = 0.0


@dataclass
class BibleConfig:
    """
    Configuração da fonte bíblica (alternativa ao Reddit).
    - versiculos_file: caminho pro JSON com os versículos
    - ganchos: lista de aberturas possíveis (o pipeline sorteia uma)
    - recent_hooks_avoid: quantos ganchos recentes evitar (anti-repetição)
    """
    versiculos_file: str = "configs/versiculos.json"
    ganchos: List[str] = field(default_factory=list)
    recent_hooks_avoid: int = 5


@dataclass
class StoryConfig:
    # Identidade e tom do canal (usado no system prompt)
    language: str = "Português Brasileiro"
    channel_identity: str = ""
    allowed_pillars: List[str] = field(default_factory=list)
    allowed_themes: List[str] = field(default_factory=list)
    forbidden_topics: List[str] = field(default_factory=list)

    # Estrutura da história
    duration_target_seconds: int = 150
    word_count_min: int = 300
    word_count_max: int = 500
    include_hook: bool = True
    cta_text: str = "Se inscreve pra mais histórias assim."

    # Extra user instructions (para canais temáticos tipo cristão)
    extra_instructions: str = ""

    # LLM parameters
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 5000
    temperature: float = 0.9


@dataclass
class NarrationConfig:
    voice_id: str
    model_id: str = "eleven_v3"
    speed: float = 1.0
    max_chars_per_call: int = 4999


@dataclass
class VisualConfig:
    queries: List[str]
    min_clip_seconds: int = 5
    max_clip_seconds: int = 15
    max_clips_per_query: int = 2
    max_pages_per_query: int = 5
    per_page: int = 40
    recent_backgrounds_limit: int = 5
    blocked_keywords: List[str] = field(default_factory=list)


@dataclass
class VideoAssemblyConfig:
    target_width: int = 1080
    target_height: int = 1920
    fps: int = 30
    final_speed_multiplier: float = 1.0
    font_name: str = "Arial"
    font_size: int = 80
    max_words_per_caption: int = 5
    whisper_model: str = "small"


@dataclass
class YouTubeConfig:
    category_id: str = "22"
    privacy_status: str = "public"
    tags: List[str] = field(default_factory=list)
    made_for_kids: bool = False
    default_language: str = "pt"
    default_audio_language: str = "pt"


@dataclass
class SecretsConfig:
    """
    Nomes das variáveis de ambiente que contêm as chaves.
    O valor em si NUNCA entra no YAML.
    """
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    elevenlabs_api_key_env: str = "ELEVENLABS_API_KEY"
    pexels_api_key_env: str = "PEXELS_API_KEY"
    youtube_client_secrets_env: str = "YOUTUBE_CLIENT_SECRETS_JSON"
    youtube_token_env: str = "YOUTUBE_TOKEN_JSON"

    def get(self, env_name: str, required: bool = True) -> Optional[str]:
        value = os.environ.get(env_name)
        if required and not value:
            raise PermanentError(
                f"Variável de ambiente obrigatória '{env_name}' não definida. "
                f"Defina no seu .env local ou nos Secrets do GitHub Actions."
            )
        return value


@dataclass
class RetryConfig:
    max_attempts_per_step: int = 3
    initial_delay: float = 3.0
    max_delay: float = 60.0
    continue_on_step_failure: bool = False  # se True, tenta seguir mesmo se uma etapa falhar


@dataclass
class PipelineConfig:
    channel: Channel
    story: StoryConfig
    narration: NarrationConfig
    visual: VisualConfig
    video: VideoAssemblyConfig
    youtube: YouTubeConfig
    # Fontes de tema — pelo menos uma deve estar presente no YAML.
    # trend: fonte Reddit (legacy, mantido pra compatibilidade com canal "dramas_reais")
    # bible: fonte versículos bíblicos (novo, usado pelo canal "amor_ao_proximo")
    trend: Optional[TrendConfig] = None
    bible: Optional[BibleConfig] = None
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


def _build_trend_config(raw: Dict[str, Any]) -> TrendConfig:
    groups_raw = raw.get("subreddits", [])
    groups = []
    for g in groups_raw:
        groups.append(SubredditGroup(
            pillar=g["pillar"],
            theme=g["theme"],
            subs=list(g["subs"]),
        ))
    return TrendConfig(
        subreddits=groups,
        max_posts_per_subreddit=raw.get("max_posts_per_subreddit", 20),
        recent_trends_limit=raw.get("recent_trends_limit", 10),
        min_relevance_score=raw.get("min_relevance_score", 0.0),
    )


def _build_bible_config(raw: Dict[str, Any]) -> BibleConfig:
    return BibleConfig(
        versiculos_file=raw.get("versiculos_file", "configs/versiculos.json"),
        ganchos=list(raw.get("ganchos", [])),
        recent_hooks_avoid=raw.get("recent_hooks_avoid", 5),
    )


def load_config(path: Path) -> PipelineConfig:
    """
    Carrega e valida um arquivo YAML de configuração.
    Aceita canais com fonte Reddit (trend) ou fonte bíblica (bible).
    """
    if not path.exists():
        raise PermanentError(f"Arquivo de config não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise PermanentError(f"Config vazio: {path}")

    try:
        channel = Channel(**raw["channel"])
        # Fontes de tema são opcionais no schema, mas pelo menos uma precisa existir.
        trend = _build_trend_config(raw["trend"]) if "trend" in raw else None
        bible = _build_bible_config(raw["bible"]) if "bible" in raw else None
        if trend is None and bible is None:
            raise PermanentError(
                f"Config {path} não define fonte de tema. "
                f"Adicione seção 'trend' (Reddit) ou 'bible' (versículos)."
            )

        story = StoryConfig(**raw.get("story", {}))
        narration = NarrationConfig(**raw["narration"])
        visual = VisualConfig(**raw["visual"])
        video = VideoAssemblyConfig(**raw.get("video", {}))
        youtube = YouTubeConfig(**raw.get("youtube", {}))
        secrets = SecretsConfig(**raw.get("secrets", {}))
        retry = RetryConfig(**raw.get("retry", {}))
    except KeyError as e:
        raise PermanentError(f"Campo obrigatório faltando no config: {e}")
    except TypeError as e:
        raise PermanentError(f"Campo inválido no config: {e}")

    return PipelineConfig(
        channel=channel,
        trend=trend,
        bible=bible,
        story=story,
        narration=narration,
        visual=visual,
        video=video,
        youtube=youtube,
        secrets=secrets,
        retry=retry,
    )
