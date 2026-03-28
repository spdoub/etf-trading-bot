"""SQLite persistence — four tables tracking the full trading lifecycle.

Tables
------
daily_sentiment    one row per day · all sector scores as JSON · source count
trades             one row per trade action (buy / sell / hold)
daily_portfolio    end-of-day snapshot with holdings, cash, returns
strategy_signals   three-layer strategy output per day

The database defaults to ``data/etf_bot.db`` inside the repository so
it persists across GitHub Actions runs when committed back to the repo.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", str(_PROJECT_ROOT / "data" / "etf_bot.db"))


# ═══════════════════════════════════════════════════════════════════════════
# Connection helpers
# ═══════════════════════════════════════════════════════════════════════════

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """Create tables if they don't already exist.  Safe to call on every run."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL UNIQUE,
            scores_json     TEXT    NOT NULL,
            source_count    INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL,
            action          TEXT    NOT NULL,
            ticker          TEXT    NOT NULL,
            shares          REAL    NOT NULL DEFAULT 0,
            price           REAL    NOT NULL DEFAULT 0,
            portfolio_value REAL    NOT NULL DEFAULT 0,
            order_id        TEXT,
            status          TEXT,
            meta            TEXT,
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_portfolio (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            date                    TEXT    NOT NULL UNIQUE,
            holdings_json           TEXT    NOT NULL,
            cash                    REAL    NOT NULL DEFAULT 0,
            total_value             REAL    NOT NULL DEFAULT 0,
            daily_return_pct        REAL,
            cumulative_return_pct   REAL,
            created_at              TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL UNIQUE,
            daily_scores    TEXT    NOT NULL,
            weekly_avg      TEXT    NOT NULL,
            monthly_avg     TEXT    NOT NULL,
            recommendation  TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            reasoning       TEXT,
            meta            TEXT,
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS errors (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL,
            step_name       TEXT    NOT NULL,
            error_message   TEXT    NOT NULL,
            created_at      TEXT    NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# daily_sentiment
# ═══════════════════════════════════════════════════════════════════════════

def insert_daily_sentiment(scores: dict, source_count: int = 0) -> int:
    """Upsert today's sector scores.

    *scores* is the full ``sentiment.analyze()`` output —
    ``{"XLK": {"score": 5, "reasoning": "…"}, …}``.
    """
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO daily_sentiment (date, scores_json, source_count, created_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(date) DO UPDATE SET "
        "  scores_json  = excluded.scores_json,"
        "  source_count = excluded.source_count,"
        "  created_at   = excluded.created_at",
        (_today(), json.dumps(scores), source_count, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_today_sentiment() -> dict | None:
    """Return today's row with *scores* already parsed, or ``None``."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM daily_sentiment WHERE date = ?", (_today(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["scores"] = json.loads(d.pop("scores_json"))
    return d


def get_sentiment_history(days_back: int = 30) -> list[dict]:
    """Rows with **flat numeric scores** (reasoning stripped) for rolling averages.

    Returns ``[{"date": "2026-03-27", "scores": {"XLK": 5, …}}, …]``.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = _connect()
    rows = conn.execute(
        "SELECT date, scores_json FROM daily_sentiment "
        "WHERE date >= ? ORDER BY date DESC",
        (cutoff,),
    ).fetchall()
    conn.close()

    results: list[dict] = []
    for row in rows:
        raw = json.loads(row["scores_json"])
        numeric: dict[str, float] = {}
        for etf, entry in raw.items():
            if isinstance(entry, dict):
                numeric[etf] = float(entry.get("score", 0))
            else:
                numeric[etf] = float(entry or 0)
        results.append({"date": row["date"], "scores": numeric})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# trades
# ═══════════════════════════════════════════════════════════════════════════

def insert_trade(
    action: str,
    ticker: str,
    shares: float = 0,
    price: float = 0,
    portfolio_value: float = 0,
    order_id: str | None = None,
    status: str | None = None,
    meta: dict | None = None,
) -> int:
    """Append a trade row.  ``action`` is ``buy``, ``sell``, or ``hold``."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO trades "
        "(date, action, ticker, shares, price, portfolio_value, "
        " order_id, status, meta, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (_today(), action, ticker, shares, price, portfolio_value,
         order_id, status, json.dumps(meta) if meta else None, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_today_trades() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM trades WHERE date = ? ORDER BY created_at",
        (_today(),),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("meta"):
            d["meta"] = json.loads(d["meta"])
        results.append(d)
    return results


def get_trade_history(days_back: int = 30) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM trades WHERE date >= ? ORDER BY date DESC, created_at DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("meta"):
            d["meta"] = json.loads(d["meta"])
        results.append(d)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# daily_portfolio
# ═══════════════════════════════════════════════════════════════════════════

def insert_daily_portfolio(
    holdings: dict,
    cash: float,
    total_value: float,
) -> int:
    """Upsert today's portfolio snapshot.  Returns auto-compute daily + cumulative %."""
    today = _today()

    daily_return_pct: float | None = None
    prev = _prev_portfolio(today)
    if prev and prev["total_value"] > 0:
        daily_return_pct = round(
            (total_value - prev["total_value"]) / prev["total_value"] * 100, 4,
        )

    cumulative_return_pct: float | None = None
    first = _first_portfolio()
    if first and first["total_value"] > 0 and first["date"] != today:
        cumulative_return_pct = round(
            (total_value - first["total_value"]) / first["total_value"] * 100, 4,
        )

    conn = _connect()
    cur = conn.execute(
        "INSERT INTO daily_portfolio "
        "(date, holdings_json, cash, total_value, "
        " daily_return_pct, cumulative_return_pct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(date) DO UPDATE SET "
        "  holdings_json        = excluded.holdings_json,"
        "  cash                 = excluded.cash,"
        "  total_value          = excluded.total_value,"
        "  daily_return_pct     = excluded.daily_return_pct,"
        "  cumulative_return_pct= excluded.cumulative_return_pct,"
        "  created_at           = excluded.created_at",
        (today, json.dumps(holdings), cash, total_value,
         daily_return_pct, cumulative_return_pct, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def _prev_portfolio(today: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT total_value, date FROM daily_portfolio "
        "WHERE date < ? ORDER BY date DESC LIMIT 1",
        (today,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _first_portfolio() -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT total_value, date FROM daily_portfolio ORDER BY date ASC LIMIT 1",
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_portfolio() -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM daily_portfolio ORDER BY date DESC LIMIT 1",
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["holdings"] = json.loads(d.pop("holdings_json"))
    return d


def get_portfolio_history(days_back: int = 30) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM daily_portfolio WHERE date >= ? ORDER BY date DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    results: list[dict] = []
    for r in rows:
        d = dict(r)
        d["holdings"] = json.loads(d.pop("holdings_json"))
        results.append(d)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# strategy_signals
# ═══════════════════════════════════════════════════════════════════════════

def insert_strategy_signal(
    daily_scores: dict,
    weekly_avg: dict,
    monthly_avg: dict,
    recommendation: str,
    confidence: float,
    reasoning: str | None = None,
    meta: dict | None = None,
) -> int:
    """Upsert today's three-layer strategy output."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO strategy_signals "
        "(date, daily_scores, weekly_avg, monthly_avg, "
        " recommendation, confidence, reasoning, meta, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(date) DO UPDATE SET "
        "  daily_scores   = excluded.daily_scores,"
        "  weekly_avg     = excluded.weekly_avg,"
        "  monthly_avg    = excluded.monthly_avg,"
        "  recommendation = excluded.recommendation,"
        "  confidence     = excluded.confidence,"
        "  reasoning      = excluded.reasoning,"
        "  meta           = excluded.meta,"
        "  created_at     = excluded.created_at",
        (_today(), json.dumps(daily_scores), json.dumps(weekly_avg),
         json.dumps(monthly_avg), recommendation, confidence,
         reasoning, json.dumps(meta) if meta else None, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_today_signal() -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM strategy_signals WHERE date = ?", (_today(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["daily_scores"] = json.loads(d["daily_scores"])
    d["weekly_avg"] = json.loads(d["weekly_avg"])
    d["monthly_avg"] = json.loads(d["monthly_avg"])
    if d.get("meta"):
        d["meta"] = json.loads(d["meta"])
    return d


def get_signal_history(days_back: int = 30) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM strategy_signals WHERE date >= ? ORDER BY date DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    results: list[dict] = []
    for r in rows:
        d = dict(r)
        d["daily_scores"] = json.loads(d["daily_scores"])
        d["weekly_avg"] = json.loads(d["weekly_avg"])
        d["monthly_avg"] = json.loads(d["monthly_avg"])
        if d.get("meta"):
            d["meta"] = json.loads(d["meta"])
        results.append(d)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# errors
# ═══════════════════════════════════════════════════════════════════════════

def insert_error(step_name: str, error_message: str) -> int:
    """Log a pipeline failure to the errors table."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO errors (date, step_name, error_message, created_at) "
        "VALUES (?, ?, ?, ?)",
        (_today(), step_name, error_message[:2000], _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid
