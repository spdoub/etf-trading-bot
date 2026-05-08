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
├── database.py      → SQLite persistence for trades, signals, sentiment
├── dashboard.py     → Streamlit data UI (reads SQLite; used on Railway too)
```

## Deploy on Railway (scheduler + dashboard, no GitHub Actions)

### 1. Authenticate (required for MCP and CLI)

Cursor’s **Railway MCP** and the **`railway`** CLI use the same Railway account. If either says the token is invalid or expired:

1. In a terminal: `railway login` (or `railway login --browserless` if you’re on a headless box).
2. In Cursor: open **Settings → MCP**, open your **Railway** server, and refresh or re-enter credentials if the integration keeps its own token.

After `railway whoami` succeeds, you can run the MCP **deploy** / **set-variables** tools again, or use the CLI commands below.

The Docker image runs **two processes**: the **daily trading loop** (`main.py --schedule` in the background) and the **Streamlit dashboard** on Railway’s **`PORT`** (HTTPS URL they give you). SQLite must live on a **volume** so redeploys do not wipe history.

### One-time setup

1. **New project → Deploy from GitHub** (this repo).
2. **Add a volume**: in the service → **Settings → Volumes** → mount path **`/data`** (any size you like; a few GB is plenty for SQLite).
3. **Variables** (Railway **Variables** tab) — set at least:

| Variable | Example | Purpose |
|----------|---------|---------|
| `DB_PATH` | `/data/etf_bot.db` | SQLite file **on the volume** |
| `GROQ_API_KEY` | (secret) | Sentiment |
| `ALPACA_API_KEY` | (secret) | Trading |
| `ALPACA_SECRET_KEY` | (secret) | Trading |
| `ALPACA_PAPER` | `true` | Use paper until you are sure |
| `TELEGRAM_BOT_TOKEN` | (secret) | Alerts + daily summary |
| `TELEGRAM_CHAT_ID` | (secret) | Chat target |

Optional:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCHEDULE_UTC_TIME` | `13:35` | Daily run time (**container clock is UTC** on Railway; `13:35` ≈ 09:35 ET) |
| `ENABLE_SCHEDULER` | `true` | Set `false` to run **dashboard only** (no trades) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

4. **Deploy**. Open the **public URL** — that is the dashboard. Build uses `Dockerfile` and `railway.toml` (health check: `/_stcore/health`).

### CLI alternative (after `railway login`)

From the repo root, you can provision and ship without the MCP, for example:

```bash
cd etf-trading-bot
railway init -n etf-trading-bot    # new project, or: railway link
railway volume add -m /data        # persistent disk mount path
railway variable set DB_PATH=/data/etf_bot.db
# add secrets: railway variable set GROQ_API_KEY=... (etc.)
railway up                         # build Dockerfile and deploy
railway domain                     # public URL for the service
```

### Operations notes

- **One replica only.** If you scale this service horizontally, every instance would run its own `main.py --schedule` loop and duplicate trades. Keep replicas at **1** unless you split the worker and web app (not covered here).
- **No GitHub schedule required** — the bot’s clock is the in-container `schedule` loop, not Actions.
- If **`DB_PATH`** is not under `/data`, the database may be **lost on redeploy** (ephemeral disk).
- **`review.py`** is still manual or a separate cron job if you want monthly LLM reviews.

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
