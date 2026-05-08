"""SQLite-backed dashboard (local or Railway).

Run locally::

    streamlit run dashboard.py

Railway runs this via ``scripts/railway_start.sh`` on ``$PORT`` with ``DB_PATH`` set
(e.g. a mounted volume at ``/data/etf_bot.db``). No keys required for the UI.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import (
    DB_PATH,
    get_error_history,
    get_latest_portfolio,
    get_portfolio_history,
    get_sentiment_history,
    get_sentiment_history_detail,
    get_signal_history,
    get_trade_history,
    init_db,
)

load_dotenv()
init_db()


def _safe_json(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, default=str)
    return str(obj)


st.set_page_config(
    page_title="ETF Bot · Data",
    layout="wide",
)

st.title("ETF bot · data")
st.caption(f"SQLite · `{DB_PATH}`")

with st.sidebar:
    days = st.slider("History window (days)", min_value=7, max_value=365, value=90, step=1)
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=30)
def _load(
    days_back: int,
) -> tuple[
    list[dict],
    dict | None,
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    return (
        get_portfolio_history(days_back=days_back),
        get_latest_portfolio(),
        get_sentiment_history(days_back=days_back),
        get_sentiment_history_detail(days_back=days_back),
        get_signal_history(days_back=days_back),
        get_trade_history(days_back=days_back),
        get_error_history(days_back=max(days_back, 90)),
    )


portfolio_hist, latest_pf, sent_hist, sent_detail, signals, trades, errors = _load(days)

# ── Overview ────────────────────────────────────────────────────────────────
if not portfolio_hist and not sent_hist and not trades:
    st.info(
        "No rows in the selected window yet. Run **`python main.py`** after the bot "
        "has recorded portfolio, sentiment, or trades."
    )
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    if latest_pf:
        c1.metric("Portfolio value", f"${latest_pf['total_value']:,.2f}")
        c2.metric("Cash", f"${latest_pf['cash']:,.2f}")
        dr = latest_pf.get("daily_return_pct")
        cr = latest_pf.get("cumulative_return_pct")
        c3.metric("Daily return", "—" if dr is None else f"{dr:+.4f}%")
        c4.metric("Cumulative return", "—" if cr is None else f"{cr:+.4f}%")
    else:
        c1.metric("Portfolio value", "—")
        c2.metric("Cash", "—")
        c3.metric("Daily return", "—")
        c4.metric("Cumulative return", "—")

    latest_sig = signals[0] if signals else None
    rec = (latest_sig.get("recommendation") or "").strip() if latest_sig else ""
    rec_label = (rec[:24] + "…") if len(rec) > 24 else (rec or "—")
    c5.metric(
        "Latest recommendation",
        rec_label,
        delta=f"conf {latest_sig['confidence']:.2f}" if latest_sig else None,
    )

st.divider()

# ── Portfolio ───────────────────────────────────────────────────────────────
st.subheader("Portfolio")
col_a, col_b = st.columns((2, 1))

with col_a:
    if portfolio_hist:
        pf_df = pd.DataFrame(portfolio_hist)
        pf_df = pf_df.sort_values("date")
        chart_df = pf_df.set_index("date")[["total_value"]]
        st.line_chart(chart_df)
    else:
        st.caption("No portfolio history in this window.")

with col_b:
    if latest_pf and latest_pf.get("holdings"):
        h_rows = []
        for sym, info in latest_pf["holdings"].items():
            row = {"symbol": sym, **info} if isinstance(info, dict) else {"symbol": sym, "raw": info}
            h_rows.append(row)
        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No holdings snapshot available.")

st.divider()

# ── Sentiment ───────────────────────────────────────────────────────────────
st.subheader("Sector sentiment (LLM)")
if sent_hist:
    sent_df = pd.DataFrame(
        [{"date": r["date"], **r["scores"]} for r in reversed(sent_hist)]
    ).set_index("date")
    st.line_chart(sent_df)
else:
    st.caption("No sentiment rows in this window.")

if sent_detail:
    with st.expander("Latest sentiment — scores & reasoning", expanded=False):
        row = sent_detail[0]
        st.write(f"**Date:** {row['date']} · **Sources counted:** {row['source_count']}")
        for etf, payload in sorted(row["scores"].items()):
            if isinstance(payload, dict):
                st.markdown(f"**{etf}** — score **{payload.get('score', '—')}**")
                st.caption(payload.get("reasoning", ""))
            else:
                st.markdown(f"**{etf}** — {payload}")

st.divider()

# ── Strategy ───────────────────────────────────────────────────────────────
st.subheader("Strategy signals")
if signals:
    sig_rows = []
    for s in signals:
        sig_rows.append(
            {
                "date": s["date"],
                "recommendation": s.get("recommendation", ""),
                "confidence": s.get("confidence"),
                "reasoning": (s.get("reasoning") or "")[:200]
                + ("…" if s.get("reasoning") and len(s["reasoning"]) > 200 else ""),
            }
        )
    st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
    with st.expander("Raw JSON — newest day"):
        st.code(_safe_json(signals[0]), language="json")
else:
    st.caption("No strategy signals in this window.")

st.divider()

# ── Trades ──────────────────────────────────────────────────────────────────
st.subheader("Trades")
if trades:
    t_rows = []
    for t in trades:
        t_rows.append(
            {
                "date": t.get("date"),
                "action": t.get("action"),
                "ticker": t.get("ticker"),
                "shares": t.get("shares"),
                "price": t.get("price"),
                "portfolio_value": t.get("portfolio_value"),
                "status": t.get("status"),
                "order_id": t.get("order_id"),
                "meta": _safe_json(t.get("meta")) if t.get("meta") else "",
            }
        )
    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No trades in this window.")

st.divider()

# ── Errors ──────────────────────────────────────────────────────────────────
st.subheader("Pipeline errors")
if errors:
    st.dataframe(
        pd.DataFrame(errors).sort_values("created_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No logged errors in the error history window.")
