"""Orchestrates the daily ETF trading pipeline.

Run once:       python main.py
Run on loop:    python main.py --schedule

If Groq sentiment analysis fails for any reason the pipeline aborts
immediately — no strategy, no trades, no guessing.  The failure is
logged to the ``errors`` table, an alert is sent to Telegram, and the
process exits with code 1 so GitHub Actions marks the run as failed.
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone

import schedule
import time

from database import init_db, insert_error
from data_sources import collect_all as collect_data_sources
from sentiment import analyze as analyze_sentiment
from strategy import generate_recommendation
from trader import execute_rotation, snapshot_portfolio
from telegram_bot import send_daily_report, send_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger("main")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline abort
# ═══════════════════════════════════════════════════════════════════════════

def _abort_pipeline(step: str, exc: Exception) -> None:
    """Log error to DB + send Telegram alert.  Never raises."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    error_msg = f"{type(exc).__name__}: {exc}"

    log.error("PIPELINE ABORT at step [%s]: %s", step, error_msg)
    log.error("Traceback:\n%s", traceback.format_exc())

    try:
        insert_error(step, error_msg)
    except Exception as db_exc:
        log.error("Failed to log error to DB: %s", db_exc)

    alert = (
        f"⚠️ <b>ETF Bot — Pipeline Failed</b>\n"
        f"\n"
        f"🚫 Groq API failed — no trades executed today\n"
        f"\n"
        f"📍 Step: <code>{step}</code>\n"
        f"❌ Error: <code>{error_msg[:500]}</code>\n"
        f"🕐 Time: {ts}\n"
        f"\n"
        f"<i>No strategy, trading, or portfolio actions were taken.\n"
        f"The bot will retry on the next scheduled run.</i>"
    )
    try:
        send_alert(alert)
    except Exception as tg_exc:
        log.error("Failed to send Telegram alert: %s", tg_exc)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> bool:
    """Single execution of the full pipeline.

    Returns ``True`` on success, ``False`` if the pipeline was aborted.
    """
    log.info("=== Pipeline started at %s ===", datetime.now(timezone.utc).isoformat())

    # ── Step 1: Collect data from all sources ────────────────────────────
    log.info("Step 1/6 — Collecting data from all sources …")
    raw_data = collect_data_sources()
    total = sum(len(v) for v in raw_data.values())
    log.info("Collected %d items across %d categories", total, len(raw_data))

    # ── Step 2: Score sentiment via Groq — HARD STOP on failure ──────────
    log.info("Step 2/6 — Analyzing sector sentiment …")
    try:
        sentiment = analyze_sentiment(raw_data)
    except Exception as exc:
        _abort_pipeline("sentiment", exc)
        log.info("=== Pipeline aborted — no trades executed ===")
        return False

    # ── Step 3: Three-layer strategy ─────────────────────────────────────
    log.info("Step 3/6 — Running three-layer strategy …")
    recommendation = generate_recommendation(sentiment)

    # ── Step 4: Execute rotation ─────────────────────────────────────────
    log.info("Step 4/6 — Executing rotation …")
    results = execute_rotation(recommendation)
    log.info("Trade results: %s", results)

    # ── Step 5: Portfolio snapshot ────────────────────────────────────────
    log.info("Step 5/6 — Snapshotting portfolio …")
    snapshot_portfolio()

    # ── Step 6: Telegram daily report ────────────────────────────────────
    log.info("Step 6/6 — Sending Telegram report …")
    send_daily_report()

    log.info("=== Pipeline complete ===")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trading Bot")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule (default: 09:35 ET) instead of once",
    )
    parser.add_argument(
        "--time", default="13:35",
        help="UTC time to run when using --schedule (HH:MM, default 13:35 ≈ 09:35 ET)",
    )
    args = parser.parse_args()

    init_db()

    if args.schedule:
        log.info("Scheduling daily run at %s UTC", args.time)
        schedule.every().day.at(args.time).do(run_pipeline)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        success = run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
