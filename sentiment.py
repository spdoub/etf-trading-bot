"""Sector-level sentiment scoring via Groq LLM.

Takes the aggregated text from data_sources.collect_all() and sends it to
Groq (llama-3.3-70b-versatile) to score sentiment across 8 major ETF
sectors on a -10 to +10 scale.  Returns a dict of sector scores + reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from groq import Groq, RateLimitError, InternalServerError, APIConnectionError
from dotenv import load_dotenv

from database import insert_daily_sentiment

load_dotenv()
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 2.0
MAX_PROMPT_CHARS = 60_000

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key)
    return _client


# ═══════════════════════════════════════════════════════════════════════════
# Sector definitions
# ═══════════════════════════════════════════════════════════════════════════

SECTORS: dict[str, str] = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "SPY": "Broad Market (S&P 500)",
}

NEUTRAL_RESULT: dict[str, dict] = {
    etf: {"score": 0, "reasoning": "No data available"}
    for etf in SECTORS
}


# ═══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a quantitative financial sentiment analyst at a hedge fund. "
    "You analyze news headlines, government contract data, job market trends, "
    "and international financial news to assess current market sentiment "
    "across major US ETF sectors.\n\n"
    "You MUST respond with ONLY valid JSON — no markdown fences, no "
    "commentary, no text before or after the JSON object."
)

CATEGORY_HEADERS: dict[str, str] = {
    "financial_headlines": "FINANCIAL HEADLINES",
    "local_us_news": "LOCAL US BUSINESS NEWS",
    "government_contracts": "GOVERNMENT CONTRACT AWARDS",
    "job_trends": "JOB MARKET TRENDS",
    "foreign_financial_news": "FOREIGN FINANCIAL NEWS",
}


def _prepare_prompt_text(data: dict[str, list]) -> tuple[str, int]:
    """Convert categorised DataItems into structured text for the LLM.

    If the total character count exceeds MAX_PROMPT_CHARS, each category
    is proportionally trimmed so the most-important items (which come
    first in each list) are preserved.

    Returns (prompt_text, item_count).
    """
    sections: list[str] = []
    total_items = 0

    for cat_key, header in CATEGORY_HEADERS.items():
        items = data.get(cat_key, [])
        if not items:
            continue
        lines = [f"\n--- {header} ---"]
        for item in items:
            lines.append(item.as_text() if hasattr(item, "as_text") else str(item))
            total_items += 1
        sections.append("\n".join(lines))

    full_text = "\n".join(sections)

    if len(full_text) <= MAX_PROMPT_CHARS:
        return full_text, total_items

    # Proportionally trim each category to fit the budget
    ratio = MAX_PROMPT_CHARS / len(full_text)
    sections = []
    total_items = 0
    for cat_key, header in CATEGORY_HEADERS.items():
        items = data.get(cat_key, [])
        if not items:
            continue
        keep = max(3, int(len(items) * ratio))
        lines = [f"\n--- {header} ({keep}/{len(items)} shown) ---"]
        for item in items[:keep]:
            lines.append(item.as_text() if hasattr(item, "as_text") else str(item))
            total_items += 1
        sections.append("\n".join(lines))

    return "\n".join(sections), total_items


def _build_user_prompt(data_text: str, item_count: int) -> str:
    sector_schema = ",\n".join(
        f'  "{etf}": {{"score": "<int from -10 to +10>", '
        f'"reasoning": "<1-2 sentences for {name}>"}}'
        for etf, name in SECTORS.items()
    )

    return (
        f"Below is today's aggregated market intelligence ({item_count} items "
        f"across 5 source categories).  Analyze ALL of this data and score the "
        f"sentiment for each ETF sector on a scale from -10 (extremely bearish) "
        f"to +10 (extremely bullish).\n\n"
        f"Return a JSON object with this EXACT structure:\n"
        f"{{\n{sector_schema}\n}}\n\n"
        f"Scoring guidelines:\n"
        f"- 0 = neutral.  Positive = bullish, negative = bearish.\n"
        f"- Consider both DIRECT sector news AND indirect macro signals.\n"
        f"- Government contracts heavily affect industrials / defense / tech.\n"
        f"- Job data signals broad economic health and consumer spending power.\n"
        f"- Foreign news affects export-heavy and multinational sectors.\n"
        f"- Be specific in reasoning — cite data points that drove each score.\n\n"
        f"DATA:\n{data_text}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Groq API call with retry
# ═══════════════════════════════════════════════════════════════════════════

def _call_groq(messages: list[dict]) -> str:
    """Call Groq with exponential-backoff retry on transient errors.

    Retries on:
        - 429  RateLimitError   (respects Retry-After header)
        - 5xx  InternalServerError
        - Connection failures

    Raises RuntimeError after MAX_RETRIES+1 total attempts.
    """
    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.15,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            wait = _retry_wait(exc, attempt)
            log.warning(
                "Groq rate-limited — waiting %.1fs (attempt %d/%d)",
                wait, attempt + 1, MAX_RETRIES,
            )
            time.sleep(wait)

        except InternalServerError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            wait = INITIAL_BACKOFF_S * (2 ** attempt)
            log.warning(
                "Groq server error (%s) — retrying in %.1fs (attempt %d/%d)",
                exc.status_code, wait, attempt + 1, MAX_RETRIES,
            )
            time.sleep(wait)

        except APIConnectionError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            wait = INITIAL_BACKOFF_S * (2 ** attempt)
            log.warning(
                "Groq connection error — retrying in %.1fs (attempt %d/%d)",
                wait, attempt + 1, MAX_RETRIES,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Groq call failed after {MAX_RETRIES + 1} attempts"
    ) from last_exc


def _retry_wait(exc: RateLimitError, attempt: int) -> float:
    """Extract Retry-After from the response header, or fall back to backoff."""
    default = INITIAL_BACKOFF_S * (2 ** attempt)
    if not hasattr(exc, "response") or exc.response is None:
        return default
    header = exc.response.headers.get("retry-after")
    if header:
        try:
            return max(float(header), 0.5)
        except ValueError:
            pass
    return default


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _validate_scores(parsed: dict) -> dict[str, dict]:
    """Clamp scores to [-10, +10], fill missing sectors with neutral."""
    result: dict[str, dict] = {}
    for etf in SECTORS:
        entry = parsed.get(etf, {})
        if isinstance(entry, (int, float)):
            entry = {"score": entry, "reasoning": ""}

        raw_score = entry.get("score", 0)
        try:
            score = int(round(float(raw_score)))
        except (ValueError, TypeError):
            score = 0
        score = max(-10, min(10, score))

        reasoning = str(entry.get("reasoning", ""))[:500]
        result[etf] = {"score": score, "reasoning": reasoning}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyze(data: dict[str, list] | None = None) -> dict[str, dict]:
    """Score sentiment for 8 ETF sectors from aggregated data sources.

    Args:
        data: Output of ``data_sources.collect_all()`` — a dict mapping
              category names to lists of DataItem objects.

    Returns:
        ``{
            "XLK": {"score": 5, "reasoning": "AI chip demand ..."},
            "XLV": {"score": -2, "reasoning": "Drug pricing pressure ..."},
            ...
        }``

        Scores range from **-10** (extremely bearish) to **+10**
        (extremely bullish).  Divide by 10 to normalise to -1 … +1.
    """
    if not data or all(len(v) == 0 for v in data.values()):
        raise RuntimeError(
            "No data collected from any source — cannot produce fresh sentiment scores"
        )

    data_text, item_count = _prepare_prompt_text(data)
    user_prompt = _build_user_prompt(data_text, item_count)

    log.info(
        "Sending %d items (%d chars) to Groq [%s] for sector scoring",
        item_count, len(data_text), GROQ_MODEL,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = _call_groq(messages)
    parsed = _extract_json(raw)
    result = _validate_scores(parsed)

    source_count = sum(len(v) for v in data.values())
    insert_daily_sentiment(result, source_count)
    log.info("Persisted daily sentiment (%d sources) to DB", source_count)

    for etf, entry in result.items():
        log.info("Sentiment %-3s → %+3d  %s", etf, entry["score"],
                 entry["reasoning"][:100])

    return result
