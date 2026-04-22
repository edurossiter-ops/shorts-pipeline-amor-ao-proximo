"""
Etapa 1: Detecção de tendências no Reddit.

Refatorado para:
- Aceitar configuração parametrizada (subreddits por canal/tema)
- Retry/backoff em falhas de rede
- Fallback gracioso quando posts não atendem critérios
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import feedparser

from .config import PipelineConfig
from .utils import (
    PermanentError,
    TransientError,
    get_control_dir,
    get_logger,
    http_request_with_retry,
    load_json_if_exists,
    now_iso,
    save_json,
)


USER_AGENT = "shorts-pipeline-trend-detection/1.0"
HEADERS = {"User-Agent": USER_AGENT}


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> Set[str]:
    return {t for t in _normalize_text(text).split() if len(t) >= 4}


def _get_hot_feed(subreddit: str):
    """
    Baixa o feed RSS do Reddit. Usa requests primeiro (que respeita o User-Agent
    corretamente) e só depois passa o XML pro feedparser. Isso evita o problema
    de o feedparser usar um User-Agent padrão que o Reddit bloqueia.
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.rss"
    try:
        resp = http_request_with_retry("GET", url, headers=HEADERS, timeout=20, max_attempts=3)
        feed = feedparser.parse(resp.content)
        if feed.bozo and not feed.entries:
            get_logger().warning(
                f"Feed r/{subreddit} mal-formado: {getattr(feed, 'bozo_exception', '?')}"
            )
            return []
        return feed.entries or []
    except Exception as exc:
        get_logger().warning(f"Falha ao ler feed r/{subreddit}: {exc}")
        return []


def _fetch_post_json(post_url: str) -> Dict[str, Any]:
    if not post_url.endswith(".json"):
        json_url = post_url.rstrip("/") + "/.json"
    else:
        json_url = post_url
    resp = http_request_with_retry("GET", json_url, headers=HEADERS, timeout=20, max_attempts=3)
    return resp.json()


def _extract_story(json_data: Any) -> Optional[Dict[str, Any]]:
    try:
        post = json_data[0]["data"]["children"][0]["data"]
    except (KeyError, IndexError, TypeError):
        return None

    if not post.get("is_self", False):
        return None

    title = (post.get("title") or "").strip()
    selftext = (post.get("selftext") or "").strip()
    if not title and not selftext:
        return None

    permalink = post.get("permalink") or ""
    return {
        "title": title,
        "text": selftext,
        "score": int(post.get("score") or 0),
        "num_comments": int(post.get("num_comments") or 0),
        "subreddit": post.get("subreddit") or "",
        "post_id": post.get("id") or "",
        "url": f"https://www.reddit.com{permalink}" if permalink else "",
    }


def _build_one_line_summary(story: Dict[str, Any]) -> str:
    base = (story.get("title") or story.get("text") or "")[:200].replace("\n", " ").strip()
    return base[:197] + "..." if len(base) > 200 else base


def _is_too_similar(candidate_summary: str, recent_trends: List[Dict[str, Any]], limit: int) -> Tuple[bool, str]:
    cand_tokens = _tokenize(candidate_summary)
    if not cand_tokens:
        return False, ""
    for item in recent_trends[:limit]:
        recent_summary = item.get("one_line_summary") or item.get("title") or ""
        recent_tokens = _tokenize(recent_summary)
        if recent_tokens and len(cand_tokens & recent_tokens) >= 6:
            return True, f"muito similar ao ciclo {item.get('cycle_id')}"
    return False, ""


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    """
    Executa a detecção de tendências para o ciclo.
    Retorna o payload salvo em topic_candidate.json.
    """
    logger = get_logger()
    trend_dir = cycle_dir / "trend-detection"
    control_dir = get_control_dir(cycle_dir)

    output_file = trend_dir / "topic_candidate.json"
    recent_trends_file = control_dir / "recent_trends.json"

    recent_trends = load_json_if_exists(recent_trends_file, {"items": []}).get("items", [])

    candidates: List[Dict[str, Any]] = []

    for group in config.trend.subreddits:
        for subreddit in group.subs:
            logger.info(f"Coletando r/{subreddit} (pillar={group.pillar}, theme={group.theme})")
            entries = _get_hot_feed(subreddit)
            for entry in entries[:config.trend.max_posts_per_subreddit]:
                post_url = getattr(entry, "link", "")
                if not post_url:
                    continue
                try:
                    json_data = _fetch_post_json(post_url)
                except (TransientError, PermanentError) as exc:
                    logger.warning(f"Skip post {post_url}: {exc}")
                    continue
                story = _extract_story(json_data)
                if not story:
                    continue

                story["word_count"] = len(story["text"].split())
                story["pillar_hint"] = group.pillar
                story["theme_hint"] = group.theme
                story["relevance_score"] = (
                    story["score"] * 1.0
                    + story["num_comments"] * 0.35
                    + story["word_count"] * 0.02
                )
                story["one_line_summary"] = _build_one_line_summary(story)
                candidates.append(story)
                time.sleep(0.5)

    if not candidates:
        raise TransientError(
            "Nenhum post coletado do Reddit. Pode ser bloqueio temporário ou problema de rede."
        )

    # ordena do mais relevante pro menos
    candidates.sort(
        key=lambda s: (s["relevance_score"], s["score"], s["num_comments"], s["word_count"]),
        reverse=True,
    )

    chosen: Optional[Dict[str, Any]] = None
    used_fallback = False
    for story in candidates:
        similar, reason = _is_too_similar(
            story["one_line_summary"], recent_trends, config.trend.recent_trends_limit
        )
        if not similar:
            chosen = story
            break

    if chosen is None:
        logger.warning("Todos os candidatos são similares a histórias recentes; usando fallback (melhor candidato).")
        chosen = candidates[0]
        used_fallback = True

    payload = {
        "cycle_id": cycle_dir.name,
        "status": "success",
        "selected_at": now_iso(),
        "topic_source": "reddit",
        "topic_source_id": chosen["post_id"],
        "source_subreddit": chosen["subreddit"],
        "source_url": chosen["url"],
        "source_title": chosen["title"],
        "source_text": chosen["text"],
        "topic_descriptor": chosen["title"] or chosen["text"][:140],
        "one_line_summary": chosen["one_line_summary"],
        "pillar_hint": chosen["pillar_hint"],
        "theme_hint": chosen["theme_hint"],
        "score": chosen["score"],
        "num_comments": chosen["num_comments"],
        "word_count": chosen["word_count"],
        "relevance_score": round(chosen["relevance_score"], 3),
        "used_fallback": used_fallback,
    }
    save_json(output_file, payload)

    # atualiza recent_trends
    new_item = {
        "cycle_id": cycle_dir.name,
        "topic_source_id": chosen["post_id"],
        "subreddit": chosen["subreddit"],
        "title": chosen["title"],
        "score": chosen["score"],
        "one_line_summary": chosen["one_line_summary"],
        "saved_at": now_iso(),
    }
    updated = [new_item] + recent_trends[: config.trend.recent_trends_limit - 1]
    save_json(recent_trends_file, {"items": updated})

    logger.info(f"Tema escolhido: r/{chosen['subreddit']} — {chosen['one_line_summary'][:80]}")
    return payload
