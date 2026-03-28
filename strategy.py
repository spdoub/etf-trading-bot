"""Three-layer sentiment rotation strategy.

Ranks sector ETFs across three rolling timeframes:
    - Daily   : today's Groq sentiment score
    - Weekly  : 7-day rolling average from the SQLite history
    - Monthly : 30-day rolling average from the SQLite history

A sector is recommended ONLY when it ranks in the top 2 across ALL
three timeframes simultaneously.  If no sector qualifies, the strategy
defaults to holding SPY.

Returns a single Recommendation (ticker + confidence 0–1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from database import insert_strategy_signal, get_sentiment_history

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_ETFS = ["XLK", "XLV", "XLE", "XLI", "XLF", "XLY", "XLU"]
DEFAULT_ETF = "SPY"
ALL_ETFS = SECTOR_ETFS + [DEFAULT_ETF]

TOP_N = 2          # must rank within this across every timeframe
WEEKLY_DAYS = 7
MONTHLY_DAYS = 30


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    etf: str                            # single ticker to hold
    confidence: float                   # 0.0 – 1.0
    daily_scores: dict[str, float]      # {etf: score} for today
    weekly_avgs: dict[str, float]       # {etf: 7-day rolling avg}
    monthly_avgs: dict[str, float]      # {etf: 30-day rolling avg}
    daily_ranks: dict[str, int]         # {etf: rank} (1 = best)
    weekly_ranks: dict[str, int]
    monthly_ranks: dict[str, int]
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_daily_scores(sentiment: dict[str, dict]) -> dict[str, float]:
    """Pull today's -10…+10 score for every ETF from sentiment output."""
    scores: dict[str, float] = {}
    for etf in ALL_ETFS:
        entry = sentiment.get(etf, {})
        if isinstance(entry, dict):
            scores[etf] = float(entry.get("score", 0))
        else:
            scores[etf] = float(entry or 0)
    return scores


def _compute_rolling_averages(
    history: list[dict],
    today_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build 7-day and 30-day rolling averages from DB history + today.

    *history* comes from ``get_sentiment_history()`` and contains one dict
    per day: ``{"date": "2026-03-25", "scores": {"XLK": 5, …}}``.

    Today's score is always included so the averages reflect the freshest
    data even on the very first run (when there is no history yet).
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff_7 = (datetime.now(timezone.utc) - timedelta(days=WEEKLY_DAYS)).strftime("%Y-%m-%d")
    cutoff_30 = (datetime.now(timezone.utc) - timedelta(days=MONTHLY_DAYS)).strftime("%Y-%m-%d")

    weekly: dict[str, float] = {}
    monthly: dict[str, float] = {}

    for etf in ALL_ETFS:
        w_scores = [today_scores.get(etf, 0.0)]
        m_scores = [today_scores.get(etf, 0.0)]

        for day_row in history:
            day_str = day_row["date"]
            if day_str == today_str:
                continue
            score = day_row["scores"].get(etf, 0.0)
            if day_str >= cutoff_7:
                w_scores.append(score)
            if day_str >= cutoff_30:
                m_scores.append(score)

        weekly[etf] = sum(w_scores) / len(w_scores)
        monthly[etf] = sum(m_scores) / len(m_scores)

    return weekly, monthly


def _rank_descending(scores: dict[str, float]) -> dict[str, int]:
    """Rank ETFs 1…N by score (1 = highest)."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    return {etf: pos + 1 for pos, etf in enumerate(ordered)}


def _top_n_set(ranks: dict[str, int], n: int = TOP_N) -> set[str]:
    return {etf for etf, r in ranks.items() if r <= n}


def _confidence(
    etf: str,
    daily: float, weekly: float, monthly: float,
    d_rank: int, w_rank: int, m_rank: int,
) -> float:
    """Map average score + rank consistency into 0–1 confidence.

    60 % weight on the normalised average score (−10…+10 → 0…1),
    40 % weight on rank quality (all #1 → 1.0, all #8 → 0.0).
    Halved for SPY-fallback to reflect lower conviction.
    """
    avg_score = (daily + weekly + monthly) / 3.0
    score_part = max(0.0, (avg_score + 10.0) / 20.0)

    avg_rank = (d_rank + w_rank + m_rank) / 3.0
    rank_part = max(0.0, 1.0 - (avg_rank - 1) / (len(ALL_ETFS) - 1))

    conf = score_part * 0.6 + rank_part * 0.4
    if etf == DEFAULT_ETF:
        conf *= 0.5

    return round(max(0.0, min(1.0, conf)), 3)


def _fmt_top3(scores: dict[str, float], ranks: dict[str, int]) -> str:
    """Pretty-print the top-3 for a log line."""
    top = sorted(ranks, key=ranks.get)[:3]
    return ", ".join(f"#{ranks[e]} {e}({scores[e]:+.1f})" for e in top)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_recommendation(sentiment: dict[str, dict]) -> Recommendation:
    """Produce a single rotation recommendation from three-layer consensus.

    Args:
        sentiment: Output of ``sentiment.analyze()`` —
                   ``{"XLK": {"score": 5, "reasoning": "…"}, …}``

    Returns:
        A :class:`Recommendation` with the ETF to hold and a 0–1
        confidence score.  If no sector dominates all three layers the
        recommendation falls back to SPY with reduced confidence.
    """
    # --- 1. Today's scores --------------------------------------------------
    daily_scores = _extract_daily_scores(sentiment)

    # --- 2. Historical data from SQLite -------------------------------------
    history = get_sentiment_history(days_back=MONTHLY_DAYS)

    # --- 3. Rolling averages ------------------------------------------------
    weekly_avgs, monthly_avgs = _compute_rolling_averages(history, daily_scores)

    # --- 4. Rank each timeframe ---------------------------------------------
    daily_ranks = _rank_descending(daily_scores)
    weekly_ranks = _rank_descending(weekly_avgs)
    monthly_ranks = _rank_descending(monthly_avgs)

    # --- 5. Find sectors in top-N across ALL three --------------------------
    d_top = _top_n_set(daily_ranks)
    w_top = _top_n_set(weekly_ranks)
    m_top = _top_n_set(monthly_ranks)

    # Only consider sector ETFs (not SPY itself)
    aligned = (d_top & w_top & m_top) - {DEFAULT_ETF}

    candidates: list[tuple[str, float]] = []
    for etf in aligned:
        avg = (daily_scores[etf] + weekly_avgs[etf] + monthly_avgs[etf]) / 3.0
        candidates.append((etf, avg))
    candidates.sort(key=lambda c: c[1], reverse=True)

    # --- 6. Pick winner or fall back ----------------------------------------
    if candidates:
        pick = candidates[0][0]
        reasoning = (
            f"{pick} ranks top-{TOP_N} in all three layers: "
            f"daily #{daily_ranks[pick]} ({daily_scores[pick]:+.1f}), "
            f"weekly #{weekly_ranks[pick]} ({weekly_avgs[pick]:+.1f}), "
            f"monthly #{monthly_ranks[pick]} ({monthly_avgs[pick]:+.1f})"
        )
        if len(candidates) > 1:
            runners = ", ".join(c[0] for c in candidates[1:])
            reasoning += f".  Runners-up also aligned: {runners}"
    else:
        pick = DEFAULT_ETF
        reasoning = (
            f"No sector ETF ranks top-{TOP_N} across all three timeframes — "
            f"holding {DEFAULT_ETF} as default"
        )

    conf = _confidence(
        pick,
        daily_scores[pick], weekly_avgs[pick], monthly_avgs[pick],
        daily_ranks[pick], weekly_ranks[pick], monthly_ranks[pick],
    )

    # --- 7. Persist to DB ----------------------------------------------------
    insert_strategy_signal(
        daily_scores=daily_scores,
        weekly_avg={k: round(v, 2) for k, v in weekly_avgs.items()},
        monthly_avg={k: round(v, 2) for k, v in monthly_avgs.items()},
        recommendation=pick,
        confidence=conf,
        reasoning=reasoning,
        meta={
            "daily_ranks": daily_ranks,
            "weekly_ranks": weekly_ranks,
            "monthly_ranks": monthly_ranks,
            "aligned_sectors": [c[0] for c in candidates],
        },
    )

    rec = Recommendation(
        etf=pick,
        confidence=conf,
        daily_scores=daily_scores,
        weekly_avgs={k: round(v, 2) for k, v in weekly_avgs.items()},
        monthly_avgs={k: round(v, 2) for k, v in monthly_avgs.items()},
        daily_ranks=daily_ranks,
        weekly_ranks=weekly_ranks,
        monthly_ranks=monthly_ranks,
        reasoning=reasoning,
    )

    log.info("RECOMMENDATION  ▸ HOLD %s  confidence=%.3f", pick, conf)
    log.info("  Reasoning: %s", reasoning)
    log.info("  Daily  : %s", _fmt_top3(daily_scores, daily_ranks))
    log.info("  Weekly : %s", _fmt_top3(weekly_avgs, weekly_ranks))
    log.info("  Monthly: %s", _fmt_top3(monthly_avgs, monthly_ranks))

    return rec
