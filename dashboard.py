"""SQLite-backed mission control (local or Railway).

Optional live broker strip: set ``ALPACA_*`` env vars (same as the bot) for
read-only account + clock — no orders from this UI.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
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

# ── Plotly shared layout (dark / neon) ───────────────────────────────────
_PLOT_BG = "rgba(8,8,18,0.85)"
_PAPER = "rgba(0,0,0,0)"
_GRID = "rgba(0,245,255,0.12)"
_TEXT = "#c8d4ff"
_ACCENT = "#00f5ff"
_MAGENTA = "#ff2d6a"


def _chart_layout(title: str | None = None, height: int = 420) -> dict:
    d: dict[str, Any] = dict(
        height=height,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_TEXT, family="ui-sans-serif, system-ui, sans-serif", size=12),
        margin=dict(l=48, r=24, t=48 if title else 24, b=48),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=_GRID, zeroline=False, linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zeroline=False, linecolor=_GRID),
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=18, color="#ffffff"))
    return d


def _inject_mission_css() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  .block-container { padding-top: 1.2rem !important; max-width: 1400px !important; }
  h1.mission-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, #00f5ff 0%, #a855f7 45%, #ff2d6a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.15rem;
  }
  p.mission-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: rgba(200,212,255,0.65);
    margin-top: 0;
  }
  div.metric-blast {
    background: linear-gradient(145deg, rgba(16,16,35,0.95), rgba(8,8,20,0.9));
    border: 1px solid rgba(0,245,255,0.25);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 0 28px rgba(0,245,255,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  div.metric-blast h4 { font-family: 'Orbitron', sans-serif; font-size: 0.72rem; color: rgba(0,245,255,0.85); margin: 0 0 6px 0; letter-spacing: 0.12em; }
  div.metric-blast .big { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 500; color: #f0f4ff; }
  div.live-strip {
    background: linear-gradient(90deg, rgba(168,85,247,0.15), rgba(0,245,255,0.12), rgba(255,45,106,0.12));
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 8px 0 20px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #dde4ff;
  }
  .stExpander { border: 1px solid rgba(0,245,255,0.2) !important; border-radius: 12px !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _safe_json(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, default=str)
    return str(obj)


@st.cache_data(ttl=45)
def _alpaca_live_snapshot() -> dict | None:
    """Read-only Alpaca account + market clock (optional)."""
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        return None
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")
    base = os.getenv("ALPACA_BASE_URL") or (
        "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    )
    try:
        api = tradeapi.REST(key, secret, base_url=base)
        acct = api.get_account()
        clock = api.get_clock()
        return {
            "mode": "PAPER" if paper else "LIVE",
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "market_open": bool(clock.is_open),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }
    except Exception:
        return None


st.set_page_config(
    page_title="ETF · Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_inject_mission_css()
st.markdown('<h1 class="mission-title">ETF ROTATION · MISSION CONTROL</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="mission-sub">SQLite <code>{DB_PATH}</code> · Groq sentiment · Alpaca execution</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Window")
    days = st.slider("History (days)", min_value=7, max_value=365, value=90, step=1)
    st.markdown("### Display")
    crazy_heat = st.toggle("Turbo heatmap (loud)", value=True)
    if st.button("Reload data", type="primary"):
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
live = _alpaca_live_snapshot()

# ── Live Alpaca strip (connector) ─────────────────────────────────────────
if live:
    mo = "MARKET OPEN" if live["market_open"] else "MARKET CLOSED"
    st.markdown(
        f'<div class="live-strip"><b>ALPACA {live["mode"]}</b> · {mo} · '
        f'Equity <b>${live["equity"]:,.2f}</b> · Cash <b>${live["cash"]:,.2f}</b> · '
        f'Buying power <b>${live["buying_power"]:,.2f}</b></div>',
        unsafe_allow_html=True,
    )

# ── Header metrics ────────────────────────────────────────────────────────
def _metric_card(col: Any, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="metric-blast"><h4>{label}</h4><div class="big">{value}</div>'
        + (f'<div style="margin-top:6px;font-size:0.75rem;opacity:0.75">{sub}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


if not portfolio_hist and not sent_hist and not trades:
    st.warning(
        "No database rows in this window yet. Run the pipeline (or wait for the scheduler)."
    )
else:
    m1, m2, m3, m4, m5 = st.columns(5)
    if latest_pf:
        dr = latest_pf.get("daily_return_pct")
        cr = latest_pf.get("cumulative_return_pct")
        dr_s = "—" if dr is None else f"{dr:+.4f}%"
        cr_s = "—" if cr is None else f"{cr:+.4f}%"
        _metric_card(m1, "NAV", f"${latest_pf['total_value']:,.2f}", "Portfolio value")
        _metric_card(m2, "CASH", f"${latest_pf['cash']:,.2f}", "Dry powder")
        _metric_card(m3, "DAY", dr_s, "Daily return")
        _metric_card(m4, "RUN", cr_s, "Cumulative")
    else:
        for c, lbl in zip(
            (m1, m2, m3, m4), ("NAV", "CASH", "DAY", "RUN"), strict=False
        ):
            _metric_card(c, lbl, "—", "")

    latest_sig = signals[0] if signals else None
    rec = (latest_sig.get("recommendation") or "").strip() if latest_sig else "—"
    conf = latest_sig["confidence"] if latest_sig else None
    sub = f"confidence {conf:.0%}" if conf is not None else ""
    _metric_card(m5, "SIGNAL", rec[:12] + ("…" if len(rec) > 12 else ""), sub)

st.divider()

# ── Portfolio charts ─────────────────────────────────────────────────────
c_left, c_right = st.columns((1.55, 1.0))
with c_left:
    st.subheader("Equity curve")
    if portfolio_hist:
        pdf = pd.DataFrame(portfolio_hist).sort_values("date")
        fig_p = go.Figure(
            go.Scatter(
                x=pdf["date"],
                y=pdf["total_value"],
                fill="tozeroy",
                fillcolor="rgba(0,245,255,0.18)",
                line=dict(color=_ACCENT, width=2),
                name="Total value",
            )
        )
        fig_p.update_layout(**_chart_layout(height=380))
        fig_p.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.caption("No portfolio history.")

with c_right:
    st.subheader("Allocation")
    if latest_pf and latest_pf.get("holdings"):
        rows = []
        for sym, info in latest_pf["holdings"].items():
            if isinstance(info, dict):
                rows.append({"sym": sym, "value": float(info.get("value", 0))})
        if rows:
            dfa = pd.DataFrame(rows)
            fig_d = go.Figure(
                go.Pie(
                    labels=dfa["sym"],
                    values=dfa["value"],
                    hole=0.62,
                    marker=dict(
                        colors=["#00f5ff", "#a855f7", "#ff2d6a", "#22c55e", "#eab308", "#3b82f6"],
                        line=dict(color="rgba(0,0,0,0.5)", width=1),
                    ),
                    textinfo="label+percent",
                    textfont=dict(color="#fff", size=12),
                )
            )
            fig_d.update_layout(**_chart_layout(title=None, height=380))
            fig_d.update_layout(showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={"sym": "Ticker", "value": st.column_config.NumberColumn("Value $", format="$%.2f")},
        )
    else:
        st.caption("No holdings snapshot.")

st.divider()

# ── Sentiment: heatmap + radar ────────────────────────────────────────────
st.subheader("Sector sentiment")
h1, h2 = st.columns((1.2, 1.0))
with h1:
    if sent_hist and len(sent_hist) >= 1:
        df = pd.DataFrame([{"date": r["date"], **r["scores"]} for r in reversed(sent_hist)])
        df = df.set_index("date")
        cs = "Turbo" if crazy_heat else "RdYlGn"
        fig_h = go.Figure(
            data=go.Heatmap(
                z=df.T.values,
                x=[str(i) for i in df.index],
                y=list(df.columns),
                colorscale=cs,
                zmid=0,
                colorbar=dict(title="Score"),
            )
        )
        fig_h.update_layout(**_chart_layout(title="Score intensity (rows = ETFs)", height=460))
        fig_h.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.caption("No sentiment history.")

with h2:
    if sent_detail:
        row = sent_detail[0]
        etfs: list[str] = []
        vals: list[float] = []
        for etf, payload in sorted(row["scores"].items()):
            if isinstance(payload, dict):
                vals.append(float(payload.get("score", 0)))
            else:
                vals.append(float(payload or 0))
            etfs.append(etf)
        if len(etfs) >= 3:
            fig_r = go.Figure(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=etfs + [etfs[0]],
                    fill="toself",
                    fillcolor="rgba(168,85,247,0.35)",
                    line=dict(color=_MAGENTA, width=2),
                    name="Latest",
                )
            )
            fig_r.update_layout(**_chart_layout(title=f"Latest pulse · {row['date']}", height=460))
            fig_r.update_layout(
                polar=dict(
                    bgcolor=_PLOT_BG,
                    radialaxis=dict(gridcolor=_GRID, visible=True, range=[-10, 10]),
                    angularaxis=dict(gridcolor=_GRID, linecolor=_GRID),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)
        with st.expander("LLM reasoning (latest day)"):
            st.caption(f"Sources counted: **{row['source_count']}**")
            for etf, payload in sorted(row["scores"].items()):
                if isinstance(payload, dict):
                    st.markdown(f"**{etf}** `{payload.get('score', '—'):+}`")
                    st.caption(payload.get("reasoning", ""))
    else:
        st.caption("No sentiment detail.")

st.divider()

# ── Strategy timeline ─────────────────────────────────────────────────────
st.subheader("Strategy signals")
if signals:
    sdf = pd.DataFrame(
        {
            "date": [s["date"] for s in signals],
            "rec": [s.get("recommendation", "") for s in signals],
            "conf": [float(s.get("confidence", 0) or 0) for s in signals],
        }
    ).sort_values("date")
    fig_s = go.Figure()
    fig_s.add_trace(
        go.Scatter(
            x=sdf["date"],
            y=sdf["conf"],
            mode="lines+markers",
            line=dict(color=_ACCENT, width=2),
            marker=dict(size=7, color=_MAGENTA, line=dict(width=1, color="#fff")),
            name="Confidence",
            text=sdf["rec"],
            hovertemplate="%{x}<br>conf %{y:.0%}<br>%{text}<extra></extra>",
        )
    )
    fig_s.update_layout(**_chart_layout(title="Recommendation confidence over time", height=300))
    fig_s.update_yaxes(tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(fig_s, use_container_width=True)

    tbl = []
    for s in signals:
        tbl.append(
            {
                "date": s["date"],
                "recommendation": s.get("recommendation", ""),
                "confidence": s.get("confidence"),
                "reasoning": (s.get("reasoning") or "")[:160]
                + ("…" if s.get("reasoning") and len(s["reasoning"]) > 160 else ""),
            }
        )
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    with st.expander("Raw JSON — newest signal"):
        st.code(_safe_json(signals[0]), language="json")
else:
    st.caption("No signals in window.")

st.divider()

# ── Trades ────────────────────────────────────────────────────────────────
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
                "order_id": (str(t.get("order_id") or ""))[:16],
            }
        )
    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No trades.")

st.divider()

# ── Errors ─────────────────────────────────────────────────────────────────
st.subheader("Pipeline errors")
if errors:
    edf = pd.DataFrame(errors).sort_values("created_at", ascending=False)
    st.dataframe(edf, use_container_width=True, hide_index=True)
else:
    st.success("No logged errors in this window.")
