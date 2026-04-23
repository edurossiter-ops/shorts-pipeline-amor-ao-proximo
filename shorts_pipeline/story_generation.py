"""
Etapa 2: Geração de REFLEXÃO cristã via Claude API.

Recebe como input um topic_candidate.json com:
- versículo bíblico (seed interno, não citado)
- tema (ângulo da reflexão)
- gancho (frase de abertura sorteada)

Produz uma reflexão em segunda pessoa sobre dilemas da vida real,
ancorada no versículo (sem citá-lo) e que menciona Jesus/Cristo explicitamente.

Formato de saída: mesmo `story_text.json` que o resto do pipeline já espera,
pra não quebrar as etapas seguintes (narração, montagem, upload).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .config import PipelineConfig
from .utils import (
    PermanentError,
    TransientError,
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


def _build_system_prompt(config: PipelineConfig, versiculo_ref: str = "") -> str:
    """
    Monta o system prompt para reflexão cristã a partir da config do canal.
    Recebe versiculo_ref pra substituir o placeholder {versiculo_ref} no cta_text.
    """
    s = config.story
    forbidden_str = (
        "\n".join(f"- {t}" for t in s.forbidden_topics)
        if s.forbidden_topics
        else "- (nenhum)"
    )

    # Substitui placeholder {versiculo_ref} no CTA com a referência real
    cta_text_resolved = s.cta_text.replace("{versiculo_ref}", versiculo_ref)

    header = (
        f"Você é um redator cristão especializado em reflexões curtas para YouTube Shorts.\n"
        f"Sua tarefa é produzir uma REFLEXÃO em {s.language}, já formatada para Text-to-Speech.\n"
        f"\nIDENTIDADE DO CANAL:\n{s.channel_identity}\n"
        f"\nREQUISITOS DE ESTRUTURA:\n"
        f"- Duração-alvo da narração: ~{s.duration_target_seconds} segundos\n"
        f"- Contagem de palavras: ENTRE {s.word_count_min} e {s.word_count_max} palavras.\n"
        f"  Conte antes de fechar. Se passar, corte frases do meio.\n"
        f"- Primeira frase: use LITERAL o \"gancho_escolhido\" fornecido no input.\n"
        f"- Última frase: use o CTA configurado: \"{cta_text_resolved}\"\n"
        f"\nTEMAS PROIBIDOS (nunca mencionar):\n{forbidden_str}\n"
        f"\n{s.extra_instructions}\n"
    )

    footer = """
FORMATO DE SAÍDA (EXATAMENTE ESTES 3 BLOCOS, NESTA ORDEM):

<editorial_brief_json>
{...json válido...}
</editorial_brief_json>

<story_package_json>
{...json válido...}
</story_package_json>

<reflexao_text_formatted>
...texto final pronto para narração...
</reflexao_text_formatted>"""

    return header + footer


def _build_recent_profiles_text(items: List[Dict[str, Any]]) -> str:
    """Texto resumido dos últimos N títulos/temas pra evitar repetição temática."""
    if not items:
        return "Nenhuma reflexão recente registrada."
    lines = []
    for idx, item in enumerate(items[:RECENT_PROFILES_LIMIT], start=1):
        lines.append(
            f"{idx}. tema={item.get('theme')} | emotional_pov={item.get('emotional_pov')} "
            f"| title={item.get('title_hint')}"
        )
    return "\n".join(lines)


def _build_user_prompt(
    topic: Dict[str, Any],
    recent_profiles: List[Dict[str, Any]],
) -> str:
    """
    Constrói o prompt do usuário com versículo + tema + gancho + reflexões recentes.
    Mostra tudo de forma clara e direta pro Claude.
    """
    versiculo_ref = topic.get("versiculo_ref", "")
    versiculo_texto = topic.get("versiculo_texto", "")
    tema = topic.get("tema_escolhido", "")
    gancho = topic.get("gancho_escolhido", "")
    categoria = topic.get("categoria", "")
    recent_profiles_text = _build_recent_profiles_text(recent_profiles)

    return f"""
prompt = f"""
Escreva uma reflexão cristã profundamente conectada ao VERSÍCULO-SEED abaixo, mas sem citar o versículo, sem mencionar a referência e sem parafrasear frases reconhecíveis.

VERSÍCULO-SEED (bússola interna — uso invisível):
Referência: {versiculo_ref}
Texto: "{versiculo_texto}"
Categoria: {categoria}

TEMA/ÂNGULO da reflexão:
{tema}

GANCHO DE ABERTURA (use LITERAL na primeira frase):
"{gancho}"

INSTRUÇÃO CENTRAL:
Use o versículo como matriz invisível da reflexão. Não escreva apenas sobre o mesmo tema geral; escreva sobre a mesma ferida, a mesma tensão espiritual, o mesmo conflito humano e o mesmo tipo de resposta que existem no versículo.

A reflexão deve parecer uma fala pastoral direta para alguém que está vivendo hoje o equivalente emocional e espiritual do que o versículo revela.

OBJETIVO DE CONEXÃO:
Quando alguém ouvir a reflexão e depois ler o versículo original, deve perceber uma conexão direta, clara e inevitável entre os dois — não por repetição de palavras, mas porque ambos tratam da mesma experiência central diante de Deus.

REGRAS DE CONSTRUÇÃO:
- Primeiro, identifique internamente o núcleo do versículo: o que ele expõe no coração humano, que crise ele revela, que movimento ele pede, e como Cristo responde a isso.
- Depois, transforme esse núcleo em linguagem pastoral, concreta e íntima, aplicada à vida real de hoje.
- Descreva cenas internas e situações humanas que correspondam diretamente ao centro do versículo.
- Evite generalidades como “Deus está com você”, “continue firme”, “tenha fé”, a menos que isso surja organicamente da tensão específica do versículo.
- Não produza uma mensagem motivacional genérica.
- Não explique o versículo.
- Não cite o versículo.
- Não use expressões reconhecivelmente próximas do texto bíblico.
- Não entregue uma reflexão apenas temática; entregue uma reflexão estruturalmente alinhada ao versículo.

PERSPECTIVA CRISTOCÊNTRICA:
A reflexão deve carregar a verdade, a compaixão, a confrontação e a esperança de Cristo. Se houver dor, mostre como Cristo a enxerga. Se houver pecado, mostre como Cristo o expõe. Se houver medo, mostre como Cristo o atravessa. Se houver espera, mostre como Cristo a sustenta.

REFLEXÕES RECENTES A EVITAR (não repetir temas, imagens, conflitos ou títulos parecidos):
{recent_profiles_text}

TESTE DE VALIDAÇÃO INTERNA:
Antes de finalizar, verifique silenciosamente:
1. Esta reflexão poderia ter sido usada para muitos outros versículos da mesma categoria?
2. Se sim, ela ainda está genérica e deve ser reescrita.
3. Ela só pode funcionar bem para este versículo-seed e para poucos versículos muito próximos em núcleo?
4. Se sim, o nível de conexão está correto.

⚠️ OBRIGATÓRIO:
A reflexão final (bloco reflexao_text_formatted) deve ter ENTRE 450 E 470 PALAVRAS.
Conte antes de fechar.
Se tiver menos de 450, aprofunde a dor, a luta interna ou a resposta de Cristo.
Se passar de 470, corte excessos.
""".strip()

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

    payload = {
        "model": config.story.model,
        "max_tokens": config.story.max_tokens,
        "temperature": config.story.temperature,
        "system": _build_system_prompt(config, topic_candidate.get("versiculo_ref", "")),
        "messages": [
            {"role": "user", "content": _build_user_prompt(topic_candidate, recent_profiles)}
        ],
    }

    resp = http_request_with_retry(
        "POST",
        ANTHROPIC_URL,
        headers=headers,
        json=payload,
        timeout=300,
        max_attempts=4,
        initial_delay=5.0,
    )
    data = resp.json()
    parts = data.get("content", [])
    text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    if not text:
        raise PermanentError("Claude retornou resposta vazia.")
    return text


def _extract_block(text: str, tag: str) -> str:
    """Extrai conteúdo de uma tag XML-like <tag>...</tag> do texto do Claude."""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
    if not m:
        raise PermanentError(f"Bloco <{tag}> não encontrado na resposta do Claude.")
    return m.group(1).strip()


def _count_words(text: str) -> int:
    """Conta palavras removendo pontuação básica."""
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return len([w for w in cleaned.split() if w.strip()])


def _validate_word_count(text: str, min_words: int, max_words: int) -> None:
    """Valida word count dentro dos limites do canal. Fail se fora."""
    n = _count_words(text)
    if n < min_words:
        raise TransientError(
            f"Reflexão muito curta: {n} palavras (mínimo: {min_words}). "
            f"Retry vai pedir pro Claude gerar de novo."
        )
    if n > max_words:
        raise TransientError(
            f"Reflexão muito longa: {n} palavras (máximo: {max_words}). "
            f"Retry vai pedir pro Claude gerar de novo."
        )


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    """
    Gera reflexão cristã baseada no topic_candidate da etapa anterior.
    """
    logger = get_logger()

    # Input vem de bible-reflection (nova etapa 1)
    input_dir = cycle_dir / "bible-reflection"
    editorial_dir = cycle_dir / "editorial-selection"
    story_dir = cycle_dir / "story-generation"
    control_dir = get_control_dir(cycle_dir)

    editorial_dir.mkdir(parents=True, exist_ok=True)
    story_dir.mkdir(parents=True, exist_ok=True)

    topic_file = input_dir / "topic_candidate.json"
    if not topic_file.exists():
        raise PermanentError(
            f"topic_candidate.json não encontrado: {topic_file}. "
            f"A etapa bible_reflection rodou?"
        )

    topic = load_json(topic_file)
    recent_profiles = load_json_if_exists(
        control_dir / "recent_story_profiles.json", {"items": []}
    ).get("items", [])

    logger.info(
        f"Gerando reflexão | versículo: {topic.get('versiculo_ref')} | "
        f"tema: {topic.get('tema_escolhido')}"
    )

    raw = _call_claude(config, topic, recent_profiles)
    (story_dir / "claude_raw_response.txt").write_text(raw, encoding="utf-8")

    editorial_json = _extract_block(raw, "editorial_brief_json")
    package_json = _extract_block(raw, "story_package_json")
    # Tag renomeada de story_text_formatted -> reflexao_text_formatted pra clareza
    reflexao_text = _extract_block(raw, "reflexao_text_formatted")

    # Validação de word count (fail se fora do limite configurado)
    _validate_word_count(
        reflexao_text, config.story.word_count_min, config.story.word_count_max
    )

    editorial_data = json.loads(editorial_json)
    package_data = json.loads(package_json)

    cta = (package_data.get("cta") or "").strip()
    if not cta:
        last_line = next(
            (ln.strip() for ln in reversed(reflexao_text.splitlines()) if ln.strip()),
            "",
        )
        cta = last_line or config.story.cta_text

    cycle_id = cycle_dir.name

    # Payload editorial (inclui info do versículo/gancho usados)
    editorial_payload = {
        "cycle_id": cycle_id,
        "versiculo_ref": topic.get("versiculo_ref", ""),
        "tema_escolhido": topic.get("tema_escolhido", ""),
        "gancho_usado": topic.get("gancho_escolhido", ""),
        **editorial_data,
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(editorial_dir / "editorial_brief.json", editorial_payload)

    # story_text.json — formato que as etapas seguintes (narração) esperam
    # Mantemos os campos pillar/theme/conflict_type/emotional_pov pra compat,
    # só que agora refletindo a natureza de reflexão (não história).
    story_payload = {
        "cycle_id": cycle_id,
        "story_body": reflexao_text,
        "target_duration_seconds": config.story.duration_target_seconds,
        "status": "success",
        "pillar": "reflexao_crista",
        "theme": topic.get("tema_escolhido", ""),
        "conflict_type": "",  # não se aplica a reflexão
        "emotional_pov": package_data.get("emotional_pov", ""),
        "title_hint": package_data.get("title_hint", ""),
        "cta": cta,
        # Metadados extras específicos de reflexão
        "versiculo_ref": topic.get("versiculo_ref", ""),
        "gancho_usado": topic.get("gancho_escolhido", ""),
        "word_count": _count_words(reflexao_text),
        "generated_at": now_iso(),
    }
    save_json(story_dir / "story_text.json", story_payload)
    (story_dir / "story_text_formatted.txt").write_text(reflexao_text, encoding="utf-8")

    # Atualiza recent_story_profiles (pra o próximo ciclo evitar tema parecido)
    new_profile = {
        "cycle_id": cycle_id,
        "pillar": "reflexao_crista",
        "theme": story_payload["theme"],
        "conflict_type": "",
        "emotional_pov": story_payload["emotional_pov"],
        "title_hint": story_payload["title_hint"],
        "saved_at": now_iso(),
    }
    updated = [new_profile] + recent_profiles[: RECENT_PROFILES_LIMIT - 1]
    save_json(control_dir / "recent_story_profiles.json", {"items": updated})

    logger.info(
        f"Reflexão gerada: \"{story_payload['title_hint']}\" "
        f"({story_payload['word_count']} palavras)"
    )
    return story_payload
