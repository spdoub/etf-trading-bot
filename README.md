# ETF Trading Bot

Automated ETF trading bot that combines news sentiment (Groq LLM), alternative data feeds, and multi-timeframe technical signals to generate and execute trades on Alpaca Markets. Sends a daily summary via Telegram.

## Architecture

```
main.py              → orchestrates the daily pipeline
├── sentiment.py     → RSS headlines → Groq LLM → sentiment score
├── data_sources.py  → Fear/Greed, VIX, insider trades, Reddit
├── strategy.py      → blends all inputs into weighted signals
├── trader.py        → places/adjusts orders on Alpaca
├── telegram_bot.py  → formats & sends the daily report
└── database.py      → SQLite persistence for trades, signals, sentiment
```

## Quick Start

```bash
# 1. Clone & enter
cd etf-trading-bot

# 2. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env with your API keys

# 5. Run once
python main.py

# 6. Or run on a schedule (daily at 09:35 ET)
python main.py --schedule
```

## API Keys Needed

| Service | Key | Purpose |
|---------|-----|---------|
| Groq | `GROQ_API_KEY` | Sentiment analysis via LLM |
| Alpaca | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Trade execution + market data |
| Telegram | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Daily reports |
| FRED | `FRED_API_KEY` | VIX data (optional) |
| Finnhub | `FINNHUB_API_KEY` | Insider sentiment (optional) |

## Disclaimer

This is for educational purposes. Use paper trading first (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`). You are responsible for any financial decisions.
