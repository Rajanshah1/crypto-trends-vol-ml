#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ---------- Paths (resolve relative to repo root) ----------
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root
PARQUET  = BASE_DIR / "data/processed/merged.parquet"
OUT_DIR  = BASE_DIR / "outputs"

st.set_page_config(page_title="Crypto Trends & Volatility", layout="wide")
st.title("📈 Crypto Market Trends & Volatility — ML Visualization")
st.caption("Compare price, realized volatility, and model forecasts (GARCH vs ML).")

# ---- Cache tools
if st.sidebar.button("🔄 Clear cache"):
    st.cache_data.clear()
    st.rerun()

# ---------- Helpers ----------
@st.cache_data
def load_panel(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)

    # normalize types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # older names -> canonical
    rename_map = {
        "coin": "coin_id",
        "timestamp": "date",
        "volume": "total_volume",
        "ret_log": "log_ret",
    }

    # realized vol like rv_30d -> rv_30
    for c in list(df.columns):
        if c.startswith("rv_") and c.endswith("d"):
            core = c[3:-1]  # between 'rv_' and trailing 'd'
            if core.isdigit():
                rename_map[c] = f"rv_{core}"

    # ewma like ewmvol_h14 -> ewma_vol_14
    for c in list(df.columns):
        if c.startswith("ewmvol_h"):
            n = c.split("h", 1)[1]
            if n.isdigit():
                rename_map[c] = f"ewma_vol_{n}"

    df = df.rename(columns=rename_map)

    # normalize ids
    for c in ("coin_id", "vs_currency"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    return df

@st.cache_data
def load_forecasts(out_dir: Path) -> pd.DataFrame:
    cols = ["date", "coin_id", "vs_currency", "model", "yhat_vol"]
    frames = []
    for p in out_dir.glob("*.csv"):
        try:
            d = pd.read_csv(p)

            # normalize column names
            if "Date" in d.columns and "date" not in d.columns:
                d = d.rename(columns={"Date": "date"})
            if "vol_forecast" in d.columns and "yhat_vol" not in d.columns:
                d = d.rename(columns={"vol_forecast": "yhat_vol"})

            # fill required id fields if missing
            d["coin_id"]     = (d.get("coin_id") or "bitcoin")
            d["vs_currency"] = (d.get("vs_currency") or "usd")
            d["model"]       = d.get("model") if "model" in d else p.stem

            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            for c in ("coin_id", "vs_currency", "model"):
                d[c] = d[c].astype(str).str.lower()

            keep = d[cols].dropna(subset=["date", "yhat_vol"])
            frames.append(keep)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["coin_id", "vs_currency", "model", "date"])

# ---------- Guard: data present ----------
if not PARQUET.exists():
    st.warning("Processed dataset not found. Run `python scripts/prepare_data.py` first.")
    st.stop()

df    = load_panel(PARQUET)
preds = load_forecasts(OUT_DIR)

# ----- Coins / currencies
coins = sorted(set(df.get("coin_id", pd.Series([], dtype=str)).dropna().unique()))
if "unknown" in coins and len(coins) > 1:
    coins = [c for c in coins if c != "unknown"]
if not coins:
    coins = ["unknown"]

vses = sorted(set(df.get("vs_currency", pd.Series(["usd"])).dropna().unique())) or ["usd"]

PREFERRED = ["bitcoin", "ethereum"]
default_coin = next((c for c in PREFERRED if c in coins), coins[0])
default_vs   = "usd" if "usd" in vses else vses[0]

# ----- Available vol windows (auto-detect)
rv_windows   = sorted({int(c.split("_")[1]) for c in df.columns
                       if c.startswith("rv_") and c.split("_")[1].isdigit()})
ewma_windows = sorted({int(c.split("_")[-1]) for c in df.columns
                       if c.startswith("ewma_vol_") and c.split("_")[-1].isdigit()})
available_windows = sorted(set(rv_windows) | set(ewma_windows)) or [30]

# ----- Sidebar filters
st.sidebar.header("Filters")
coin    = st.sidebar.selectbox("Coin", coins, index=coins.index(default_coin))
vs      = st.sidebar.selectbox("vs_currency", vses, index=vses.index(default_vs))
horizon = st.sidebar.selectbox("Vol window", available_windows, index=0)

# slice data
g = df[(df["coin_id"] == coin) & (df["vs_currency"] == vs)].sort_values("date").copy()
if g.empty:
    st.info("No rows for this coin/currency. Re-run Day 2 with live data.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Price", "Volatility", "Forecasts"])

# ---------- Price ----------
with tab1:
    st.subheader(f"{coin.capitalize()} / {vs.upper()} — Price")
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
    if "liq_tier" in g.columns:
        st.caption(f"Liquidity tiers present ({g['liq_tier'].nunique()} levels).")

# ---------- Forecasts ----------
with tab3:
    st.subheader("Forecast Overlays (GARCH vs RF)")
    sel = preds[(preds["coin_id"] == coin) & (preds["vs_currency"] == vs)].copy()
    if sel.empty:
        st.info("No forecasts available yet. Train models first (Day 3).")
    else:
        # Choose truth series: prefer smallest available rv_*, else |log_ret|
        rv_candidates = sorted([w for w in rv_windows if f"rv_{w}" in g.columns])
        truth = None
        if rv_candidates:
            truth_col = f"rv_{rv_candidates[0]}"
            truth = g[["date", truth_col]].rename(columns={truth_col: "ytrue_vol"})
        elif "log_ret" in g.columns:
            truth = pd.DataFrame({"date": g["date"], "ytrue_vol": np.abs(g["log_ret"])})
        # plot truth (if present)
        if truth is not None:
            base_fig = px.line(truth, x="date", y="ytrue_vol", labels={"ytrue_vol": "vol"}, title=None)
            st.plotly_chart(base_fig, use_container_width=True)
        # plot each model
        models = sorted(sel["model"].unique().tolist())
        pick = st.multiselect("Models", models, default=models)
        for m in pick:
            mdf = sel[sel["model"] == m].sort_values("date")
            figm = px.line(mdf, x="date", y="yhat_vol", title=f"Forecast: {m}")
            st.plotly_chart(figm, use_container_width=True)

st.divider()
st.caption("Data: processed features from Day 2; forecasts from Day 3 outputs. Use sidebar to switch coin/currency.")

