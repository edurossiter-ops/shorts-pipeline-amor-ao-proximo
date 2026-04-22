"""
Etapa 2: Geração de história via Claude API.

Refatorado para:
- System prompt totalmente parametrizado por canal (drama vs cristão vs outros)
- Duração, CTA, gancho, temas proibidos configuráveis
- Retry com classificação de erro (sobrecarga vs auth errada)
- API key vem de env var, NUNCA do código
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

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


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
RECENT_PROFILES_LIMIT = 10


def _build_system_prompt(config: PipelineConfig) -> str:
    """
    Monta o system prompt a partir da config do canal.
    Permite canais muito diferentes (drama / cristão / motivacional) sem mexer no código.
    """
    s = config.story

    pillars_str = ", ".join(s.allowed_pillars) if s.allowed_pillars else "livre"
    themes_str = ", ".join(s.allowed_themes) if s.allowed_themes else "livre"
    forbidden_str = "\n".join(f"- {t}" for t in s.forbidden_topics) if s.forbidden_topics else "- (nenhum)"

    hook_instruction = (
        "- Comece com um hook forte na PRIMEIRA frase que provoque curiosidade imediata."
        if s.include_hook
        else "- Começo natural, sem hook manipulativo."
    )

    return f"""
Você é um redator especializado em roteiros curtos para YouTube Shorts.
Sua tarefa é produzir uma história em {s.language}, já FORMATADA para Text-to-Speech do ElevenLabs.

IDENTIDADE DO CANAL:
{s.channel_identity}

Pilares narrativos permitidos: {pillars_str}
Temas permitidos: {themes_str}

TEMAS PROIBIDOS (nunca mencionar):
{forbidden_str}

REQUISITOS DE ESTRUTURA:
- Duração-alvo da narração: ~{s.duration_target_seconds} segundos
- Contagem de palavras: entre {s.word_count_min} e {s.word_count_max}
{hook_instruction}
- Turning point claro no meio
- CTA como ÚLTIMA frase: "{s.cta_text}"

FORMATAÇÃO PARA ELEVENLABS:
- Audio tags em colchetes em inglês, imediatamente antes do trecho relevante.
- [short pause] para tensão rápida.
- [long pause] no máximo 1 vez.
- Travessões (—) para interrupções e revelações.
- MAIÚSCULAS pontuais para ênfase (máximo 2 por história).

REGRAS DE VARIAÇÃO:
- NÃO repita ou fique próximo demais dos perfis recentes fornecidos.
- Não repita o mesmo pillar do último vídeo.
- Não repita o mesmo conflict_type dos últimos 5 vídeos.

{s.extra_instructions}

FORMATO DE SAÍDA (EXATAMENTE ESTES 3 BLOCOS, NESTA ORDEM):

<editorial_brief_json>
{{...json válido...}}
</editorial_brief_json>

<story_package_json>
{{...json válido...}}
</story_package_json>

<story_text_formatted>
...texto final em {s.language} pronto para ElevenLabs...
</story_text_formatted>

1. editorial_brief_json deve conter:
{{
  "pillar": "...",
  "theme": "...",
  "conflict_type": "...",
  "emotional_pov": "...",
  "hook_direction": "...",
  "target_audience": "...",
  "tone": "...",
  "constraints": ["..."],
  "outline": {{
    "scenario_initial": "...",
    "problem_trigger": "...",
    "worst_moment": "...",
    "turning_point": "...",
    "final_state": "...",
    "target_audience_feeling": "..."
  }}
}}

2. story_package_json deve conter:
{{
  "status": "generated",
  "pillar": "...",
  "theme": "...",
  "conflict_type": "...",
  "emotional_pov": "...",
  "title_hint": "...",
  "cta": "..."
}}

3. story_text_formatted contém APENAS o texto final em {s.language} pronto para narração.
O CTA deve ser a ÚLTIMA frase.
""".strip()


def _build_recent_profiles_text(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "Nenhum perfil recente registrado."
    lines = []
    for idx, item in enumerate(items[:RECENT_PROFILES_LIMIT], start=1):
        lines.append(
            f"{idx}. pillar={item.get('pillar')} | theme={item.get('theme')} | "
            f"conflict_type={item.get('conflict_type')} | title_hint={item.get('title_hint')}"
        )
    return "\n".join(lines)


def _call_claude(
    config: PipelineConfig,
    topic_candidate: Dict[str, Any],
    recent_profiles: List[Dict[str, Any]],
) -> str:
    api_key = config.secrets.get(config.secrets.anthropic_api_key_env, required=True)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    user_prompt = (
        "Abaixo o JSON do agente de trend-detection:\n\n"
        f"{json.dumps(topic_candidate, ensure_ascii=False, indent=2)}\n\n"
        "Perfis recentes a evitar:\n"
        f"{_build_recent_profiles_text(recent_profiles)}"
    )

    payload = {
        "model": config.story.model,
        "max_tokens": config.story.max_tokens,
        "temperature": config.story.temperature,
        "system": _build_system_prompt(config),
        "messages": [{"role": "user", "content": user_prompt}],
    }

    resp = http_request_with_retry(
        "POST", ANTHROPIC_URL, headers=headers, json=payload, timeout=300, max_attempts=4, initial_delay=5.0
    )
    data = resp.json()
    parts = data.get("content", [])
    text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    if not text:
        raise PermanentError("Claude retornou resposta vazia.")
    return text


def _extract_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
    if not m:
        raise PermanentError(f"Bloco <{tag}> não encontrado na resposta do Claude.")
    return m.group(1).strip()


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    logger = get_logger()
    trend_dir = cycle_dir / "trend-detection"
    editorial_dir = cycle_dir / "editorial-selection"
    story_dir = cycle_dir / "story-generation"
    control_dir = get_control_dir(cycle_dir)

    topic_file = trend_dir / "topic_candidate.json"
    if not topic_file.exists():
        raise PermanentError(f"topic_candidate.json não encontrado: {topic_file}")

    topic = load_json(topic_file)
    recent_profiles = load_json_if_exists(
        control_dir / "recent_story_profiles.json", {"items": []}
    ).get("items", [])

    raw = _call_claude(config, topic, recent_profiles)
    (story_dir / "claude_raw_response.txt").write_text(raw, encoding="utf-8")

    editorial_json = _extract_block(raw, "editorial_brief_json")
    package_json = _extract_block(raw, "story_package_json")
    story_text = _extract_block(raw, "story_text_formatted")

    editorial_data = json.loads(editorial_json)
    package_data = json.loads(package_json)

    cta = (package_data.get("cta") or "").strip()
    if not cta:
        last_line = next((ln.strip() for ln in reversed(story_text.splitlines()) if ln.strip()), "")
        cta = last_line or config.story.cta_text

    cycle_id = cycle_dir.name

    editorial_payload = {
        "cycle_id": cycle_id,
        **editorial_data,
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(editorial_dir / "editorial_brief.json", editorial_payload)

    story_payload = {
        "cycle_id": cycle_id,
        "story_body": story_text,
        "target_duration_seconds": config.story.duration_target_seconds,
        "status": "success",
        "pillar": package_data.get("pillar", ""),
        "theme": package_data.get("theme", ""),
        "conflict_type": package_data.get("conflict_type", ""),
        "emotional_pov": package_data.get("emotional_pov", ""),
        "title_hint": package_data.get("title_hint", ""),
        "cta": cta,
        "generated_at": now_iso(),
    }
    save_json(story_dir / "story_text.json", story_payload)
    (story_dir / "story_text_formatted.txt").write_text(story_text, encoding="utf-8")

    # Atualiza recent profiles
    new_profile = {
        "cycle_id": cycle_id,
        "pillar": story_payload["pillar"],
        "theme": story_payload["theme"],
        "conflict_type": story_payload["conflict_type"],
        "emotional_pov": story_payload["emotional_pov"],
        "title_hint": story_payload["title_hint"],
        "saved_at": now_iso(),
    }
    updated = [new_profile] + recent_profiles[: RECENT_PROFILES_LIMIT - 1]
    save_json(control_dir / "recent_story_profiles.json", {"items": updated})

    logger.info(f"História gerada: {story_payload['title_hint']}")
    return story_payload
