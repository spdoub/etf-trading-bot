"""Interactive Telegram bot listener.

Text "trade" (or use /trade) to trigger the full trading pipeline on demand.
Only messages from the configured TELEGRAM_CHAT_ID are honoured.

Run with:
    python bot_listener.py

Keep this process alive (tmux, screen, systemd, etc.) so the bot stays
responsive.  It uses long-polling — no webhook setup required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger("bot_listener")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

from telegram import Update  # noqa: E402 — after env/logging setup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from database import init_db  # noqa: E402
from main import run_pipeline  # noqa: E402


def _authorized(update: Update) -> bool:
    """Only allow the configured chat to trigger actions."""
    return str(update.effective_chat.id) == CHAT_ID


async def _handle_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the full trading pipeline when the user sends 'trade' or /trade."""
    if not _authorized(update):
        log.warning("Unauthorized trigger from chat %s — ignored", update.effective_chat.id)
        return

    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    await update.message.reply_text(
        f"⚙️ <b>Pipeline starting…</b>  ({ts})\n"
        f"<i>This takes 1–2 minutes. You'll get the full report when it's done.</i>",
        parse_mode="HTML",
    )

    log.info("Trade triggered via Telegram by chat %s", CHAT_ID)

    loop = asyncio.get_event_loop()
    try:
        success = await loop.run_in_executor(None, run_pipeline)
        if not success:
            await update.message.reply_text(
                "⚠️ <b>Pipeline aborted</b> — sentiment analysis failed.\n"
                "<i>No trades were executed. Check bot.log for details.</i>",
                parse_mode="HTML",
            )
    except Exception as exc:
        log.exception("Unexpected error during pipeline: %s", exc)
        await update.message.reply_text(
            f"❌ <b>Unexpected error</b>\n<code>{type(exc).__name__}: {exc}</code>",
            parse_mode="HTML",
        )


async def _handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    await update.message.reply_text(
        "🤖 <b>ETF Bot — Commands</b>\n\n"
        "  <code>trade</code>  or  /trade — run the full pipeline now\n"
        "  /help  — show this message",
        parse_mode="HTML",
    )


def main() -> None:
    if not BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is not set — exiting")
        sys.exit(1)
    if not CHAT_ID:
        log.error("TELEGRAM_CHAT_ID is not set — exiting")
        sys.exit(1)

    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    # /trade command
    app.add_handler(CommandHandler("trade", _handle_trade))
    # /help command
    app.add_handler(CommandHandler("help", _handle_help))
    # Plain text "trade" (case-insensitive)
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.Regex(r"(?i)^\s*trade\s*$"),
            _handle_trade,
        )
    )

    log.info("Bot listener started — waiting for messages (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
