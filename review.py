"""Monthly performance review — LLM-powered analysis of trading history.

Pulls the full trade/signal/portfolio history from SQLite, formats it
into a structured prompt, sends it to Groq for analysis, saves the
response as a dated markdown file in ``reviews/``, and sends a summary
to Telegram.

Run manually:  python review.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from groq import (
    Groq,
    APIStatusError,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
)
from dotenv import load_dotenv

from database import (
    init_db,
    get_trade_history,
    get_portfolio_history,
    get_signal_history,
    get_sentiment_history,
)
from telegram_bot import send_alert

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger("review")

_PROJECT_ROOT = Path(__file__).resolve().parent
REVIEWS_DIR = _PROJECT_ROOT / "reviews"
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
ALL_TIME_DAYS = 3650
MAX_DAILY_LOG_ROWS = 180
MAX_PROMPT_CHARS = int(os.getenv("GROQ_REVIEW_MAX_PROMPT_CHARS", "55000"))
GROQ_REVIEW_INPUT_TOKEN_BUDGET = int(
    os.getenv("GROQ_REVIEW_INPUT_TOKEN_BUDGET", "26500")
)
MAX_RETRIES = 3
BACKOFF_S = 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Groq client
# ═══════════════════════════════════════════════════════════════════════════

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=key)
    return _client


def _call_groq(messages: list[dict]) -> str:
    client = _get_client()
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.35,
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError, APIConnectionError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            wait = BACKOFF_S * (2 ** attempt)
            log.warning("Groq error — retry in %.1fs: %s", wait, exc)
            time.sleep(wait)
    raise RuntimeError(
        f"Groq failed after {MAX_RETRIES + 1} attempts"
    ) from last_exc


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a quantitative trading strategy analyst performing a monthly \
performance review of an automated ETF sector-rotation bot.

Bot architecture:
• Data sources: financial news RSS (Reuters, AP, MarketWatch), local US \
business news (~20 cities), government contract awards (SAM.gov), job \
posting trends, foreign financial news (Nikkei, Handelsblatt, Korea Herald).
• Sentiment scoring: all collected text is sent to an LLM which scores \
8 sectors (XLK/Tech, XLV/Healthcare, XLE/Energy, XLI/Industrials, \
XLF/Financials, XLY/Consumer Disc., XLU/Utilities, SPY/S&P 500) on a \
-10 to +10 scale.
• Three-layer strategy: daily score, 7-day rolling average, 30-day \
rolling average.  A sector is selected ONLY when it ranks top-2 in ALL \
three timeframes.  Otherwise the bot holds SPY as default.
• Execution: single-ETF rotation via Alpaca (fractional shares, notional \
market orders).

Provide a thorough, data-backed analysis with specific, actionable \
recommendations.  Format your response in markdown with clear headers.\
"""


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stats(
    portfolio: list[dict],
    trades: list[dict],
    signals: list[dict],
) -> dict:
    port = sorted(portfolio, key=lambda d: d["date"])

    wins = losses = flat = 0
    for d in port:
        r = d.get("daily_return_pct")
        if r is None:
            continue
        if r > 0.001:
            wins += 1
        elif r < -0.001:
            losses += 1
        else:
            flat += 1

    ret_days = [
        (d["date"], d["daily_return_pct"])
        for d in port if d.get("daily_return_pct") is not None
    ]
    best = max(ret_days, key=lambda x: x[1]) if ret_days else None
    worst = min(ret_days, key=lambda x: x[1]) if ret_days else None

    buy_trades = [
        t for t in trades
        if t["action"] == "buy" and t.get("status") != "error"
    ]

    return {
        "first_date": port[0]["date"] if port else "N/A",
        "last_date": port[-1]["date"] if port else "N/A",
        "days": len(port),
        "first_value": port[0]["total_value"] if port else 0,
        "last_value": port[-1]["total_value"] if port else 0,
        "cum_return": port[-1].get("cumulative_return_pct") if port else None,
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate": wins / max(wins + losses + flat, 1) * 100,
        "rotations": len(buy_trades),
        "best": best,
        "worst": worst,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data tables for the prompt
# ═══════════════════════════════════════════════════════════════════════════

def _daily_log(
    portfolio: list[dict],
    signals: list[dict],
    trades: list[dict],
    max_rows: int | None = None,
) -> str:
    """One row per trading day: recommendation, action, return, value."""
    row_cap = max_rows if max_rows is not None else MAX_DAILY_LOG_ROWS
    port_map = {d["date"]: d for d in portfolio}
    sig_map = {d["date"]: d for d in signals}
    trade_map: dict[str, list[dict]] = {}
    for t in trades:
        trade_map.setdefault(t["date"], []).append(t)

    dates = sorted(set(list(port_map) + list(sig_map)))
    if len(dates) > row_cap:
        dates = dates[-row_cap:]

    lines = ["Date       | Rec  | Conf  | Action    | Day Ret  | Value"]
    lines.append("-" * 64)

    for date in dates:
        sig = sig_map.get(date)
        port = port_map.get(date)
        day_t = trade_map.get(date, [])

        rec = sig["recommendation"] if sig else "—"
        conf = f"{sig['confidence']:.0%}" if sig else "—"

        acts = [t["action"] for t in day_t]
        if "sell" in acts and "buy" in acts:
            ticker = next(
                (t["ticker"] for t in day_t if t["action"] == "buy"), "?"
            )
            act = f"rot→{ticker}"
        elif "buy" in acts:
            act = "buy"
        elif "sell" in acts:
            act = "sell"
        else:
            act = "hold"

        dr = (
            f"{port['daily_return_pct']:+.2f}%"
            if port and port.get("daily_return_pct") is not None
            else "—"
        )
        tv = f"${port['total_value']:,.0f}" if port else "—"

        lines.append(
            f"{date} | {rec:<4} | {conf:<5} | {act:<9} | {dr:>8} | {tv:>9}"
        )

    return "\n".join(lines)


def _rotation_log(
    portfolio: list[dict],
    signals: list[dict],
    trades: list[dict],
) -> str:
    """Every rotation event with 5-day forward return."""
    port_dates = sorted(d["date"] for d in portfolio)
    port_map = {d["date"]: d for d in portfolio}
    sig_map = {d["date"]: d for d in signals}

    buys = sorted(
        [t for t in trades if t["action"] == "buy" and t.get("status") != "error"],
        key=lambda t: t["date"],
    )
    if not buys:
        return "(No rotations yet)"

    lines = ["Date       | Ticker | Conf  | 5d Ret  | Reasoning"]
    lines.append("-" * 82)

    for t in buys:
        d = t["date"]
        sig = sig_map.get(d)
        conf = f"{sig['confidence']:.0%}" if sig else "—"

        fwd = "—"
        if d in port_dates:
            i = port_dates.index(d)
            if i + 5 < len(port_dates):
                v0 = port_map[d]["total_value"]
                v5 = port_map[port_dates[i + 5]]["total_value"]
                if v0 > 0:
                    fwd = f"{(v5 - v0) / v0 * 100:+.2f}%"
            else:
                fwd = "pend."

        reason = ""
        if sig and sig.get("reasoning"):
            reason = sig["reasoning"]
            if len(reason) > 65:
                reason = reason[:62] + "…"

        lines.append(
            f"{d} | {t['ticker']:<6} | {conf:<5} | {fwd:>7} | {reason}"
        )

    return "\n".join(lines)


def _sector_accuracy(
    portfolio: list[dict],
    signals: list[dict],
) -> str:
    """How each sector performed when it was the recommendation."""
    port_dates = sorted(d["date"] for d in portfolio)
    port_map = {d["date"]: d for d in portfolio}
    sigs = sorted(signals, key=lambda d: d["date"])

    stats: dict[str, dict] = {}
    for sig in sigs:
        rec = sig["recommendation"]
        d = sig["date"]
        stats.setdefault(rec, {"n": 0, "pos": 0, "neg": 0, "ret": 0.0})
        stats[rec]["n"] += 1

        if d in port_dates:
            i = port_dates.index(d)
            if i + 1 < len(port_dates):
                nxt = port_map.get(port_dates[i + 1])
                if nxt and nxt.get("daily_return_pct") is not None:
                    r = nxt["daily_return_pct"]
                    stats[rec]["ret"] += r
                    if r > 0.001:
                        stats[rec]["pos"] += 1
                    elif r < -0.001:
                        stats[rec]["neg"] += 1

    if not stats:
        return "(No signal data yet)"

    lines = ["Sector | Days Rec'd | Pos Days | Neg Days | Avg Next-Day Ret"]
    lines.append("-" * 62)
    for sec in sorted(stats, key=lambda s: stats[s]["n"], reverse=True):
        s = stats[sec]
        avg = s["ret"] / max(s["n"], 1)
        lines.append(
            f"{sec:<6} | {s['n']:>10} | {s['pos']:>8} | {s['neg']:>8} | {avg:+.4f}%"
        )

    return "\n".join(lines)


def _sentiment_drift(sentiment: list[dict]) -> str:
    """Show how average sector scores trended over time (monthly buckets)."""
    if not sentiment:
        return "(No sentiment data yet)"

    from collections import defaultdict
    monthly: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in sentiment:
        month = row["date"][:7]
        for etf, score in row["scores"].items():
            monthly[month][etf].append(score)

    months = sorted(monthly)
    all_etfs = sorted({e for m in monthly.values() for e in m})

    header = "Month   | " + " | ".join(f"{e:>5}" for e in all_etfs)
    lines = [header, "-" * len(header)]
    for m in months:
        avgs = []
        for e in all_etfs:
            vals = monthly[m].get(e, [])
            avgs.append(f"{sum(vals)/len(vals):+5.1f}" if vals else "    —")
        lines.append(f"{m}  | " + " | ".join(avgs))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt assembly
# ═══════════════════════════════════════════════════════════════════════════

def _rough_review_input_tokens(system: str, user: str) -> int:
    return max(1, (len(system) + len(user)) // 3)


def _build_prompt(
    portfolio: list[dict],
    signals: list[dict],
    trades: list[dict],
    sentiment: list[dict],
    stats: dict,
    *,
    max_prompt_chars: int | None = None,
    max_daily_rows: int | None = None,
) -> str:
    best_s = (
        f"{stats['best'][0]} ({stats['best'][1]:+.2f}%)"
        if stats["best"] else "N/A"
    )
    worst_s = (
        f"{stats['worst'][0]} ({stats['worst'][1]:+.2f}%)"
        if stats["worst"] else "N/A"
    )
    cum_s = (
        f"{stats['cum_return']:+.2f}%"
        if stats["cum_return"] is not None else "N/A"
    )

    overview = (
        "## OVERVIEW\n"
        f"- Period: {stats['first_date']} → {stats['last_date']}"
        f" ({stats['days']} trading days)\n"
        f"- Starting value: ${stats['first_value']:,.2f}\n"
        f"- Current value: ${stats['last_value']:,.2f}\n"
        f"- Cumulative return: {cum_s}\n"
        f"- Win / Loss / Flat: {stats['wins']}W / {stats['losses']}L"
        f" / {stats['flat']}F ({stats['win_rate']:.1f}% win rate)\n"
        f"- Total rotations: {stats['rotations']}\n"
        f"- Best day: {best_s}\n"
        f"- Worst day: {worst_s}"
    )

    mpc = max_prompt_chars if max_prompt_chars is not None else MAX_PROMPT_CHARS
    mdr = max_daily_rows if max_daily_rows is not None else MAX_DAILY_LOG_ROWS

    daily = (
        "## DAILY PERFORMANCE LOG\n```\n"
        + _daily_log(portfolio, signals, trades, max_rows=mdr)
        + "\n```"
    )
    rotations = (
        "## ROTATION EVENTS (with 5-day forward returns)\n```\n"
        + _rotation_log(portfolio, signals, trades)
        + "\n```"
    )
    accuracy = (
        "## SECTOR RECOMMENDATION ACCURACY\n```\n"
        + _sector_accuracy(portfolio, signals)
        + "\n```"
    )
    drift = (
        "## AVERAGE MONTHLY SENTIMENT SCORES\n```\n"
        + _sentiment_drift(sentiment)
        + "\n```"
    )

    questions = (
        "## ANALYSIS REQUESTED\n"
        "Based on the data above, provide a thorough analysis:\n\n"
        "1. **Wrong calls vs right calls** — What patterns distinguish "
        "profitable from unprofitable recommendations?  Are certain "
        "confidence levels more reliable?\n"
        "2. **Data source effectiveness** — Given the five data sources "
        "(financial headlines, local US news, government contracts, job "
        "trends, foreign financial news), which types of signals likely "
        "drove the best vs worst calls?\n"
        "3. **Scoring weight adjustments** — What specific changes to the "
        "-10/+10 scoring or the three-layer (daily / weekly / monthly) "
        "weighting would improve performance?\n"
        "4. **Sector-specific biases** — Is the model systematically "
        "over- or under-scoring any sectors?  Which is it most / least "
        "accurate on?\n"
        "5. **Strategic recommendations** — Any broader changes to "
        "rotation logic, holding period, or risk management?\n\n"
        "Be specific and reference the data.  Provide actionable "
        "recommendations with expected impact."
    )

    sections = [overview, daily, rotations, accuracy, drift, questions]
    full = "\n\n".join(sections)

    if len(full) > mpc:
        trimmed_daily = (
            "## DAILY PERFORMANCE LOG\n"
            f"(Trimmed — {stats['days']} days recorded; see overview)"
        )
        sections[1] = trimmed_daily
        full = "\n\n".join(sections)

    return full


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def _save_markdown(analysis: str, prompt_data: str, date_str: str) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEWS_DIR / f"{date_str}.md"

    content = (
        f"# Monthly Strategy Review — {date_str}\n\n"
        f"{analysis}\n\n"
        f"---\n\n"
        f"<details>\n"
        f"<summary>📊 Raw data sent to analyst</summary>\n\n"
        f"```\n{prompt_data}\n```\n\n"
        f"</details>\n"
    )

    path.write_text(content, encoding="utf-8")
    log.info("Review saved to %s (%d chars)", path, len(content))
    return path


def _send_telegram(analysis: str, stats: dict, date_str: str) -> None:
    cum_s = (
        f"{stats['cum_return']:+.2f}%"
        if stats["cum_return"] is not None else "N/A"
    )

    excerpt = analysis[:2200]
    if len(analysis) > 2200:
        last_break = excerpt.rfind("\n\n")
        if last_break > 600:
            excerpt = excerpt[:last_break]
        excerpt += "\n…"

    safe = (
        excerpt
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    msg = (
        f"📝 <b>Monthly Strategy Review — {date_str}</b>\n\n"
        f"📊 <b>Period Stats</b>\n"
        f"  📅 {stats['first_date']} → {stats['last_date']}"
        f" ({stats['days']} days)\n"
        f"  💰 ${stats['first_value']:,.0f} → ${stats['last_value']:,.0f}"
        f" ({cum_s})\n"
        f"  ✅ {stats['wins']}W  ❌ {stats['losses']}L"
        f"  ({stats['win_rate']:.0f}% win rate)\n"
        f"  🔄 {stats['rotations']} rotations\n\n"
        f"🧠 <b>AI Analysis</b>\n"
        f"<pre>{safe}</pre>\n\n"
        f"📁 Full review: <code>reviews/{date_str}.md</code>"
    )

    if len(msg) > 4090:
        msg = msg[:4085] + "…"

    try:
        send_alert(msg)
        log.info("Review summary sent to Telegram")
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_review() -> bool:
    """Execute the monthly review.  Returns True on success."""
    log.info("=== Monthly review started ===")

    init_db()

    log.info("Pulling full history from SQLite …")
    portfolio = get_portfolio_history(ALL_TIME_DAYS)
    signals = get_signal_history(ALL_TIME_DAYS)
    trades = get_trade_history(ALL_TIME_DAYS)
    sentiment = get_sentiment_history(ALL_TIME_DAYS)

    log.info(
        "Data: %d portfolio days, %d signals, %d trades, %d sentiment days",
        len(portfolio), len(signals), len(trades), len(sentiment),
    )

    if len(portfolio) < 3:
        log.warning("Not enough data for a meaningful review (need >= 3 days)")
        try:
            send_alert(
                "📝 <b>Monthly Review</b>\n\n"
                "⚠️ Not enough trading data yet — need at least 3 days."
            )
        except Exception:
            pass
        return True

    stats = _compute_stats(portfolio, trades, signals)

    max_rows = MAX_DAILY_LOG_ROWS
    max_chars = MAX_PROMPT_CHARS
    analysis: str | None = None
    prompt_data = ""

    for _ in range(22):
        for _inner in range(50):
            prompt_data = _build_prompt(
                portfolio,
                signals,
                trades,
                sentiment,
                stats,
                max_prompt_chars=max_chars,
                max_daily_rows=max_rows,
            )
            if (
                _rough_review_input_tokens(SYSTEM_PROMPT, prompt_data)
                <= GROQ_REVIEW_INPUT_TOKEN_BUDGET
            ):
                break
            max_rows = max(25, int(max_rows * 0.82))
            max_chars = max(12000, int(max_chars * 0.88))
        else:
            raise RuntimeError(
                "GROQ_REVIEW_INPUT_TOKEN_BUDGET is too low — increase it or "
                "reduce GROQ_REVIEW_MAX_PROMPT_CHARS"
            )

        log.info(
            "Review prompt: %d chars (~%d est. input tokens), daily row cap %d",
            len(prompt_data),
            _rough_review_input_tokens(SYSTEM_PROMPT, prompt_data),
            max_rows,
        )

        log.info("Sending to Groq [%s] for analysis …", GROQ_MODEL)
        try:
            analysis = _call_groq([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_data},
            ])
            break
        except APIStatusError as exc:
            if exc.status_code != 413:
                raise
            log.warning(
                "Groq 413 (request too large) — shrinking (rows=%d, char cap=%d)",
                max_rows,
                max_chars,
            )
            max_rows = max(25, int(max_rows * 0.55))
            max_chars = max(12000, int(max_chars * 0.55))

    if analysis is None:
        raise RuntimeError(
            "Monthly review: could not fit prompt under Groq input limits"
        )
    log.info("Received analysis: %d chars", len(analysis))

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _save_markdown(analysis, prompt_data, date_str)
    _send_telegram(analysis, stats, date_str)

    log.info("=== Monthly review complete ===")
    return True


if __name__ == "__main__":
    try:
        success = run_review()
        sys.exit(0 if success else 1)
    except Exception as exc:
        log.error("Review failed: %s", exc, exc_info=True)
        try:
            send_alert(
                f"⚠️ <b>Monthly Review Failed</b>\n\n"
                f"❌ <code>{type(exc).__name__}: {str(exc)[:500]}</code>"
            )
        except Exception:
            pass
        sys.exit(1)
