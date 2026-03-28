"""Trade execution via the alpaca-trade-api SDK — single-ETF rotation.

Checks current holdings against the strategy recommendation and
rotates if they differ.  Uses notional (dollar-amount) market orders
so fractional shares are handled automatically.

Paper vs. live mode is controlled by the ``ALPACA_PAPER`` env var
(default: ``true``).  Every action — buy, sell, or hold — is logged
to the SQLite ``trades`` table with ticker, quantity, price, and
timestamp.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from database import insert_trade, insert_daily_portfolio
from strategy import Recommendation

load_dotenv()
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

PAPER_TRADING = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

_api: tradeapi.REST | None = None


def _get_api() -> tradeapi.REST:
    """Lazy-initialised Alpaca REST client."""
    global _api
    if _api is None:
        base_url = os.getenv("ALPACA_BASE_URL") or (PAPER_URL if PAPER_TRADING else LIVE_URL)
        _api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=base_url,
        )
        mode = "PAPER" if PAPER_TRADING else "*** LIVE ***"
        log.info("Alpaca client ready — %s mode → %s", mode, base_url)
    return _api


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _list_holdings() -> list[dict]:
    """Return every open position as ``{symbol, qty, market_value, price}``."""
    try:
        positions = _get_api().list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "price": float(p.current_price),
            }
            for p in positions
        ]
    except Exception as exc:
        log.error("Failed to list positions: %s", exc)
        return []


def _get_cash() -> float:
    """Available cash in the account."""
    try:
        return float(_get_api().get_account().cash)
    except Exception as exc:
        log.error("Failed to fetch account cash: %s", exc)
        return 0.0


def _market_open() -> bool:
    """Check whether the US equity market is currently open."""
    try:
        return _get_api().get_clock().is_open
    except Exception as exc:
        log.warning("Clock check failed (assuming closed): %s", exc)
        return False


def _latest_price(symbol: str) -> float:
    """Best-effort price for logging.  Returns 0 on failure."""
    try:
        return float(_get_api().get_latest_trade(symbol).price)
    except Exception:
        return 0.0


def _portfolio_value() -> float:
    """Total portfolio value (positions + cash).  Returns 0 on failure."""
    try:
        return float(_get_api().get_account().portfolio_value)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Sell / Buy primitives
# ═══════════════════════════════════════════════════════════════════════════

def _sell_position(symbol: str, qty: float, reason: str) -> dict:
    """Close the entire position in *symbol* via a market sell.

    Returns a result dict (always logged to the DB regardless of outcome).
    """
    api = _get_api()
    pv = _portfolio_value()
    try:
        order = api.close_position(symbol)
        oid = str(order.id)
        status = str(order.status)
        price = (
            float(order.filled_avg_price)
            if getattr(order, "filled_avg_price", None)
            else _latest_price(symbol)
        )
        insert_trade(
            action="sell", ticker=symbol, shares=qty, price=price,
            portfolio_value=pv, order_id=oid, status=status,
            meta={"reason": reason},
        )
        log.info("SELL  %s  %.4f shares @ ~$%.2f  [%s]  order=%s",
                 symbol, qty, price, status, oid)
        return {
            "action": "sell", "etf": symbol, "qty": qty,
            "price": price, "order_id": oid, "status": status,
        }
    except Exception as exc:
        insert_trade(
            action="sell", ticker=symbol, shares=qty, price=0,
            portfolio_value=pv, status="error",
            meta={"error": str(exc)},
        )
        log.error("SELL FAILED  %s: %s", symbol, exc)
        return {"action": "sell", "etf": symbol, "qty": qty, "error": str(exc)}


def _buy_notional(symbol: str, dollars: float, meta_extra: dict) -> dict:
    """Buy *dollars* worth of *symbol* via a notional market order.

    Fractional shares are enabled automatically when using notional.
    """
    api = _get_api()
    pv = _portfolio_value()
    try:
        order = api.submit_order(
            symbol=symbol,
            notional=round(dollars, 2),
            side="buy",
            type="market",
            time_in_force="day",
        )
        oid = str(order.id)
        status = str(order.status)
        filled_qty = float(order.qty) if order.qty else 0.0
        price = (
            float(order.filled_avg_price)
            if getattr(order, "filled_avg_price", None)
            else _latest_price(symbol)
        )
        insert_trade(
            action="buy", ticker=symbol, shares=filled_qty, price=price,
            portfolio_value=pv, order_id=oid, status=status,
            meta={"notional": round(dollars, 2), **meta_extra},
        )
        log.info("BUY   %s  $%.2f notional @ ~$%.2f/sh  [%s]  order=%s",
                 symbol, dollars, price, status, oid)
        return {
            "action": "buy", "etf": symbol, "notional": round(dollars, 2),
            "qty": filled_qty, "price": price,
            "order_id": oid, "status": status,
        }
    except Exception as exc:
        insert_trade(
            action="buy", ticker=symbol, shares=0, price=0,
            portfolio_value=pv, status="error",
            meta={"error": str(exc), "attempted_notional": round(dollars, 2),
                  **meta_extra},
        )
        log.error("BUY FAILED  %s ($%.2f): %s", symbol, dollars, exc)
        return {"action": "buy", "etf": symbol, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def execute_rotation(rec: Recommendation) -> list[dict]:
    """Compare the recommendation against current holdings and act.

    Decision matrix:

    ============== ===============================================
    Holdings       Action
    ============== ===============================================
    Only rec.etf   **HOLD** — do nothing, log the hold.
    Different ETF  **SELL** it, then **BUY** rec.etf with full cash.
    Multiple ETFs  **SELL** all except rec.etf, **BUY** if needed.
    Nothing        **BUY** rec.etf with full cash.
    ============== ===============================================

    Every action (including a hold) is written to the ``trades`` table.
    """
    results: list[dict] = []

    # ── Pre-flight: market must be open for market orders ────────────────
    if not _market_open():
        log.warning("Market is closed — skipping trade execution")
        insert_trade(
            action="hold", ticker=rec.etf, shares=0, price=0,
            portfolio_value=_portfolio_value(), status="market_closed",
            meta={"reason": "market closed", "confidence": rec.confidence},
        )
        results.append({"action": "hold", "etf": rec.etf, "reason": "market_closed"})
        return results

    # ── Inspect current portfolio ────────────────────────────────────────
    holdings = _list_holdings()
    held_symbols = {h["symbol"] for h in holdings}

    log.info(
        "Portfolio: %s",
        ", ".join(f"{h['symbol']} ({h['qty']:.4f} sh, ${h['market_value']:.2f})"
                  for h in holdings) or "(empty)",
    )

    # ── HOLD: already positioned correctly ───────────────────────────────
    if held_symbols == {rec.etf}:
        h = holdings[0]
        insert_trade(
            action="hold", ticker=rec.etf, shares=h["qty"], price=h["price"],
            portfolio_value=_portfolio_value(), status="no_action",
            meta={
                "market_value": h["market_value"],
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
            },
        )
        log.info(
            "HOLD  %s  %.4f shares ($%.2f) — no rotation needed",
            rec.etf, h["qty"], h["market_value"],
        )
        results.append({
            "action": "hold", "etf": rec.etf,
            "qty": h["qty"], "price": h["price"],
            "market_value": h["market_value"],
        })
        return results

    # ── SELL: close positions that aren't the recommendation ─────────────
    sell_failed = False
    already_holds_target = rec.etf in held_symbols

    for h in holdings:
        if h["symbol"] == rec.etf:
            continue
        result = _sell_position(
            h["symbol"], h["qty"],
            reason=f"rotating from {h['symbol']} into {rec.etf}",
        )
        results.append(result)
        if "error" in result:
            sell_failed = True

    if sell_failed:
        log.error("One or more sells failed — aborting buy to avoid margin risk")
        return results

    # ── BUY: acquire the recommended ETF if not already held ─────────────
    if already_holds_target:
        log.info("Already holding %s alongside others — sells done, no buy needed", rec.etf)
        return results

    cash = _get_cash()
    if cash < 1.0:
        log.warning("Cash too low ($%.2f) — cannot buy %s", cash, rec.etf)
        insert_trade(
            action="buy", ticker=rec.etf, shares=0, price=0,
            portfolio_value=_portfolio_value(), status="insufficient_cash",
            meta={"cash": cash},
        )
        results.append({"action": "buy", "etf": rec.etf, "error": "insufficient_cash"})
        return results

    buy_result = _buy_notional(
        rec.etf, cash,
        meta_extra={"confidence": rec.confidence, "reasoning": rec.reasoning},
    )
    results.append(buy_result)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio snapshot
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_portfolio() -> dict | None:
    """Query Alpaca for current state and persist to ``daily_portfolio``.

    Returns the holdings dict on success, ``None`` on failure.
    """
    try:
        api = _get_api()
        account = api.get_account()
        positions = api.list_positions()

        holdings: dict[str, dict] = {}
        for pos in positions:
            holdings[pos.symbol] = {
                "shares": float(pos.qty),
                "value": float(pos.market_value),
                "price": float(pos.current_price),
            }

        cash = float(account.cash)
        total_value = float(account.portfolio_value)

        insert_daily_portfolio(holdings, cash, total_value)
        log.info(
            "Portfolio snapshot saved — %d positions, $%.2f cash, $%.2f total",
            len(holdings), cash, total_value,
        )
        return holdings
    except Exception as exc:
        log.error("Portfolio snapshot failed: %s", exc)
        return None
