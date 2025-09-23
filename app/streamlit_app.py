#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Crypto Trends & Volatility", layout="wide")
st.title("📈 Crypto Market Trends & Volatility — ML Visualization")
st.caption("Compare price, realized volatility, and model forecasts (GARCH vs ML).")

BASE = Path(__file__).resolve().parents[1]
PARQUET = BASE / "data/processed/merged.parquet"
OUT_DIR  = BASE / "outputs"

# ---------- Helpers ----------
@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalize column names from earlier runs
    df = df.rename(columns={
        "timestamp": "date",
        "coin": "coin_id",
        "volume": "total_volume",
        "ret_log": "log_ret",
        "ewmvol_h14": "ewma_vol_14",
    })
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("coin_id", "vs_currency"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

@st.cache_data
def load_forecasts(out_dir: Path) -> pd.DataFrame:
    """Load any CSV in outputs/ and coerce to: date, coin_id, vs_currency, model, yhat_vol."""
    req = ["date", "coin_id", "vs_currency", "model", "yhat_vol"]
    if not out_dir.exists():
        return pd.DataFrame(columns=req)

    frames = []
    for p in sorted(out_dir.glob("*.csv")):
        try:
            d = pd.read_csv(p)

            # Lenient renames
            d = d.rename(columns={
                "Date": "date",
                "timestamp": "date",
                "coin": "coin_id",
                "symbol": "coin_id",
                "vol_forecast": "yhat_vol",
                "sigma": "yhat_vol",
                "yhat": "yhat_vol",
            })

            # Infer IDs from filename when missing
            stem = p.stem.lower()
            if "coin_id" not in d.columns:
                if "eth" in stem or "ethereum" in stem:
                    d["coin_id"] = "ethereum"
                elif "btc" in stem or "bitcoin" in stem:
                    d["coin_id"] = "bitcoin"
                else:
                    d["coin_id"] = "bitcoin"
            if "vs_currency" not in d.columns:
                d["vs_currency"] = "usd"
            if "model" not in d.columns:
                # e.g., garch_btc -> garch
                d["model"] = stem.split("_")[0] if "_" in stem else stem

            # Types & cleaning
            if "date" in d.columns:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
            for c in ("coin_id", "vs_currency", "model"):
                d[c] = d[c].astype(str).str.strip().str.lower()

            if "yhat_vol" not in d.columns:
                continue

            d = d.dropna(subset=["date", "yhat_vol"])
            frames.append(d[req])
        except Exception:
            # ignore unreadable files and keep going
            continue

    if not frames:
        return pd.DataFrame(columns=req)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["coin_id", "vs_currency", "model", "date"])

# ---------- Guard ----------
if not PARQUET.exists():
    st.warning("Processed dataset not found. Run `python scripts/prepare_data.py` first.")
    st.stop()

df    = load_panel(PARQUET)
preds = load_forecasts(OUT_DIR)

# ---------- Filters ----------
coins = sorted(df.get("coin_id", pd.Series([], dtype=str)).dropna().unique().tolist())
if "unknown" in coins and len(coins) > 1:
    coins = [c for c in coins if c != "unknown"]
if not coins:
    coins = ["unknown"]

vses = sorted(df.get("vs_currency", pd.Series(["usd"])).dropna().unique().tolist()) or ["usd"]

st.sidebar.header("Filters")
coin    = st.sidebar.selectbox("Coin", coins, index=0)
vs      = st.sidebar.selectbox("vs_currency", vses, index=vses.index("usd") if "usd" in vses else 0)
horizon = st.sidebar.selectbox("Vol window", [7, 30, 90], index=0)

# Slice
g = df[(df["coin_id"] == coin) & (df["vs_currency"] == vs)].sort_values("date").copy()
if g.empty:
    st.info("No rows for this coin/currency. Re-run Day 2 with live data.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Price", "Volatility", "Forecasts"])

# ---------- Price ----------
with tab1:
    st.subheader(f"{coin.capitalize()}/{vs.upper()} — Price")
    fig = px.line(g, x="date", y="price", title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily last observed price.")

# ---------- Volatility ----------
with tab2:
    rv_col = f"rv_{horizon}"
    ew_col = f"ewma_vol_{horizon}"
    existing = [c for c in (rv_col, ew_col) if c in g.columns]
    st.subheader(f"Realized vs EWMA Volatility (window={horizon})")
    if existing:
        fig2 = px.line(g, x="date", y=existing, title=None)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Volatility columns not found. Re-run `prepare_data.py` (Day 2).")

# ---------- Forecasts ----------
with tab3:
    st.subheader("Forecast Overlays (GARCH vs RF)")
    sel = preds[(preds["coin_id"] == coin) & (preds["vs_currency"] == vs)].copy()

    if sel.empty:
        st.info("No forecasts available yet. Train models first (Day 3).")
    else:
        # Truth (prefer rv_7; fallback to |log_ret|)
        if "rv_7" in g.columns:
            truth = g[["date", "rv_7"]].rename(columns={"rv_7": "ytrue_vol"})
        else:
            truth = g[["date"]].assign(
                ytrue_vol=(g["log_ret"].abs() if "log_ret" in g.columns else np.nan)
            )
        base = px.line(truth.dropna(), x="date", y="ytrue_vol", title="True daily vol (proxy)")
        st.plotly_chart(base, use_container_width=True)

        models = sorted(sel["model"].unique().tolist())
        pick = st.multiselect("Models to overlay", models, default=models)
        for m in pick:
            mdf = sel[sel["model"] == m].sort_values("date")
            figm = px.line(mdf, x="date", y="yhat_vol", title=f"Forecast: {m}")
            st.plotly_chart(figm, use_container_width=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear cache"):
    load_panel.clear()
    load_forecasts.clear()
    st.rerun()

