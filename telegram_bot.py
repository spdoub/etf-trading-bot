"""Daily Telegram report — emoji-rich summary sent after each pipeline run.

Covers: date, action taken, current holdings, portfolio value, daily &
cumulative returns, top-3 sector sentiment with reasoning, strategy
confidence, and a running win/loss record since inception.
"""

from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from telegram import Bot

from database import (
    get_today_trades,
    get_today_signal,
    get_today_sentiment,
    get_latest_portfolio,
    get_portfolio_history,
)

load_dotenv()
log = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SECTOR_NAMES: dict[str, str] = {
    "XLK": "Tech",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLF": "Financials",
    "XLY": "Consumer Disc.",
    "XLU": "Utilities",
    "SPY": "S&P 500",
}

SEP = "━" * 26


# ═══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{'+' if value >= 0 else ''}{value:.2f}%"


def _pct_emoji(value: float | None) -> str:
    if value is None:
        return "➖"
    if value > 0.001:
        return "🟢"
    if value < -0.001:
        return "🔴"
    return "➖"


def _score_emoji(score: float) -> str:
    if score >= 5:
        return "🟢"
    if score >= 2:
        return "🟡"
    if score <= -5:
        return "🔴"
    if score <= -2:
        return "🟠"
    return "⚪"


def _confidence_bar(conf: float) -> str:
    filled = round(conf * 10)
    return "▓" * filled + "░" * (10 - filled)


def _confidence_label(conf: float) -> str:
    if conf >= 0.80:
        return "🔥 Very High"
    if conf >= 0.60:
        return "💪 High"
    if conf >= 0.40:
        return "🟡 Moderate"
    if conf >= 0.20:
        return "⚠️ Low"
    return "🔴 Very Low"


# ═══════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════

def _section_action(trades: list[dict]) -> list[str]:
    """Summarise today's trades into a readable action block."""
    lines: list[str] = [SEP, "🔄 <b>ACTION</b>"]

    if not trades:
        lines.append("  ⏸️ No trades executed")
        return lines

    sells = [t for t in trades if t["action"] == "sell"]
    buys = [t for t in trades if t["action"] == "buy"]
    holds = [t for t in trades if t["action"] == "hold"]

    # Headline summary
    if sells and buys:
        sold = ", ".join(f"<code>{t['ticker']}</code>" for t in sells)
        bought = buys[0]["ticker"]
        lines.append(f"  🔄 Rotated {sold} → <b>{bought}</b>")
    elif buys and not sells:
        lines.append(f"  🟢 Bought <b>{buys[0]['ticker']}</b>")
    elif sells and not buys:
        lines.append(f"  🔴 Sold <b>{sells[0]['ticker']}</b>")
    elif holds:
        h = holds[0]
        if h.get("status") == "market_closed":
            lines.append(f"  ⏸️ Held <b>{h['ticker']}</b>  (market closed)")
        else:
            lines.append(f"  ✊ Held <b>{h['ticker']}</b>  — no rotation needed")

    # Detail sub-lines for real executions
    for t in trades:
        if t["action"] == "hold":
            continue
        side = t["action"].upper()
        emoji = "🟢" if t["action"] == "buy" else "🔴"
        meta = t.get("meta") or {}
        notional = meta.get("notional")
        if notional:
            lines.append(
                f"     {emoji} {side} <code>{t['ticker']}</code>"
                f"  ${notional:,.2f} notional"
            )
        elif t["shares"]:
            price_s = f" @ ${t['price']:,.2f}" if t["price"] else ""
            lines.append(
                f"     {emoji} {side} <code>{t['ticker']}</code>"
                f"  {t['shares']:.4f} sh{price_s}"
            )

    return lines


def _section_portfolio(portfolio: dict | None) -> list[str]:
    lines: list[str] = [SEP, "💼 <b>PORTFOLIO</b>"]

    if not portfolio:
        lines.append("  ⚠️ No portfolio data yet")
        return lines

    tv = portfolio["total_value"]
    cash = portfolio["cash"]
    lines.append(f"  🏦 Total Value:  <b>${tv:,.2f}</b>")
    lines.append(f"  💵 Cash:         ${cash:,.2f}")

    holdings = portfolio.get("holdings", {})
    if holdings:
        for sym, info in holdings.items():
            name = SECTOR_NAMES.get(sym, sym)
            lines.append(
                f"  📦 <code>{sym}</code>  {info['shares']:.4f} sh"
                f"  (${info['value']:,.2f})  <i>{name}</i>"
            )
    else:
        lines.append("  📦 No open positions")

    return lines


def _section_returns(portfolio: dict | None) -> list[str]:
    lines: list[str] = [SEP, "📈 <b>RETURNS</b>"]

    if not portfolio:
        lines.append("  ⏳ Available after first full day")
        return lines

    dr = portfolio.get("daily_return_pct")
    cr = portfolio.get("cumulative_return_pct")

    lines.append(f"  Today:       {_pct_emoji(dr)}  {_pct(dr)}")
    lines.append(f"  Cumulative:  {_pct_emoji(cr)}  {_pct(cr)}")

    return lines


def _section_sentiment(sent: dict | None) -> list[str]:
    lines: list[str] = [SEP, "🧠 <b>TOP 3 SENTIMENT</b>"]

    if not sent:
        lines.append("  ⚠️ No sentiment data")
        return lines

    scores = sent.get("scores", {})

    def _numeric(entry):
        if isinstance(entry, dict):
            return float(entry.get("score", 0))
        return float(entry or 0)

    ranked = sorted(scores.items(), key=lambda kv: _numeric(kv[1]), reverse=True)

    for rank, (etf, entry) in enumerate(ranked[:3], 1):
        if isinstance(entry, dict):
            score = int(entry.get("score", 0))
            reasoning = entry.get("reasoning", "")
        else:
            score = int(float(entry or 0))
            reasoning = ""

        emoji = _score_emoji(score)
        name = SECTOR_NAMES.get(etf, etf)
        lines.append(f"  {rank}. {emoji} <code>{etf}</code> <b>{score:+d}</b>  {name}")
        if reasoning:
            short = (reasoning[:90] + "…") if len(reasoning) > 90 else reasoning
            lines.append(f"      <i>{short}</i>")

    src = sent.get("source_count", 0)
    if src:
        lines.append(f"  📡 <i>{src} sources analysed</i>")

    return lines


def _section_strategy(signal: dict | None) -> list[str]:
    lines: list[str] = [SEP, "🎯 <b>STRATEGY</b>"]

    if not signal:
        lines.append("  ⚠️ No strategy signal")
        return lines

    rec = signal["recommendation"]
    conf = signal["confidence"]
    reasoning = signal.get("reasoning", "")

    lines.append(f"  Holding:     <b>{rec}</b>")
    lines.append(f"  Confidence:  {_confidence_bar(conf)}  {conf:.0%}")
    lines.append(f"               {_confidence_label(conf)}")

    if reasoning:
        short = (reasoning[:120] + "…") if len(reasoning) > 120 else reasoning
        lines.append(f"  <i>{short}</i>")

    return lines


def _section_winloss() -> list[str]:
    lines: list[str] = [SEP, "🏆 <b>WIN / LOSS</b>"]

    history = get_portfolio_history(days_back=3650)
    wins = losses = flat = 0
    streak = 0
    streak_type = ""

    for day in sorted(history, key=lambda d: d["date"]):
        dr = day.get("daily_return_pct")
        if dr is None:
            continue
        if dr > 0.001:
            wins += 1
            streak = streak + 1 if streak_type == "W" else 1
            streak_type = "W"
        elif dr < -0.001:
            losses += 1
            streak = streak + 1 if streak_type == "L" else 1
            streak_type = "L"
        else:
            flat += 1
            streak = 0
            streak_type = ""

    total = wins + losses + flat
    if total == 0:
        lines.append("  📊 Tracking starts after first full day")
        return lines

    win_rate = wins / total * 100

    lines.append(f"  ✅ {wins}W    ❌ {losses}L    ➖ {flat}F")
    lines.append(f"  Win Rate:  <b>{win_rate:.1f}%</b>  ({total} days)")

    if streak >= 2:
        s_emoji = "🔥" if streak_type == "W" else "💧"
        s_word = "winning" if streak_type == "W" else "losing"
        lines.append(f"  {s_emoji} {streak}-day {s_word} streak")

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Report assembly
# ═══════════════════════════════════════════════════════════════════════════

def _build_report() -> str:
    now = datetime.now(timezone.utc)
    date_display = now.strftime("%A, %B %d, %Y")

    trades = get_today_trades()
    portfolio = get_latest_portfolio()
    sent = get_today_sentiment()
    signal = get_today_signal()

    parts: list[str] = [
        f"📊 <b>ETF Bot — Daily Report</b>",
        f"📅 {date_display}",
        "",
    ]

    parts.extend(_section_action(trades))
    parts.append("")
    parts.extend(_section_portfolio(portfolio))
    parts.append("")
    parts.extend(_section_returns(portfolio))
    parts.append("")
    parts.extend(_section_sentiment(sent))
    parts.append("")
    parts.extend(_section_strategy(signal))
    parts.append("")
    parts.extend(_section_winloss())
    parts.append("")
    parts.append(SEP)
    parts.append("🤖 <i>Automated ETF Rotation Bot</i>")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Send
# ═══════════════════════════════════════════════════════════════════════════

async def _send(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Telegram credentials not set — skipping report")
        return
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(
        chat_id=CHAT_ID,
        text=text,
        parse_mode="HTML",
    )
    log.info("Telegram report sent to chat %s", CHAT_ID)


def send_daily_report() -> None:
    """Build and send the daily report.  Safe to call synchronously."""
    report = _build_report()
    log.info("Report preview:\n%s", report)
    asyncio.run(_send(report))


def send_alert(message: str) -> None:
    """Send a short alert (e.g. pipeline failure).  Safe to call synchronously."""
    log.info("Sending alert:\n%s", message)
    asyncio.run(_send(message))
