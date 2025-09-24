#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Crypto Market Dashboards", layout="wide")
st.title("📊 Crypto Market Dashboards + 📈 Price / Vol / Forecasts")
st.caption("Market map, rankings, momentum, dominance, breadth, risk, returns, volatility, compare-to-100, liquidity — plus detailed coin price/volatility and model forecasts (GARCH vs ML).")

# Project-root defaults (works if file lives at src/cryptoml_vol/app/streamlit_app.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "merged.parquet"
DEFAULT_OUTS = PROJECT_ROOT / "outputs"

# Canonical column names
DATE_COL = "date"
COIN_COL = "coin_id"
PRICE_COL = "price"
MKT_CAP_COL = "market_cap"
VOL24_COL = "volume_24h"
VS_CURR_COL = "vs_currency"

# Optional %-change horizon columns
H1_COL = "pct_change_1h"
D1_COL = "pct_change_24h"
D7_COL = "pct_change_7d"
D30_COL = "pct_change_30d"

# Other optional columns (computed if missing)
DOM_COL = "dominance_share"
TIER_COL = "liquidity_tier"
RET_COL = "ret_d1"
ROLLVOL_COL = "rolling_vol_14"

# =========================================
# HELPERS (shared)
# =========================================
@st.cache_data(show_spinner=False)
def load_panel(path_like: Path | str) -> pd.DataFrame:
    """
    Load the main panel dataset (parquet/csv or a directory), normalize columns,
    and compute missing derived fields (returns, rolling vol, dominance, tiers, momentum).
    """
    p = Path(path_like).expanduser()

    # Allow a directory: pick a sensible file
    if p.is_dir():
        candidates = [
            p / "data" / "processed" / "merged.parquet",
            p / "merged.parquet",
            p / "data" / "processed" / "merged.csv",
            p / "merged.csv",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if not found:
            globs = list(p.glob("**/*.parquet")) + list(p.glob("**/*.csv"))
            found = globs[0] if globs else None
        if not found:
            return pd.DataFrame()
        p = found

    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "timestamp": DATE_COL,
        "date": DATE_COL,
        "coin": COIN_COL,
        "symbol": COIN_COL,
        "name": COIN_COL,
        "total_volume": VOL24_COL,
        "volume_24h": VOL24_COL,
        "volume": VOL24_COL,
        "vol_24h": VOL24_COL,
        "marketcap": MKT_CAP_COL,
        "market_cap": MKT_CAP_COL,
        "price_usd": PRICE_COL,
        "close": PRICE_COL,
        "vs_currency": VS_CURR_COL,
        "ret_log": "log_ret",
        "ewmvol_h14": "ewma_vol_14",
        "liq_tier": TIER_COL,  # honor precomputed tiers if present
    })

    # Types & cleaning
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL])

    for c in (COIN_COL, VS_CURR_COL):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Basic guards
    need = [DATE_COL, COIN_COL, PRICE_COL, MKT_CAP_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return pd.DataFrame()  # caller shows warning

    # Sort for time ops
    df = df.sort_values([COIN_COL, DATE_COL])

    # Daily returns if absent
    if RET_COL not in df.columns:
        df[RET_COL] = df.groupby(COIN_COL)[PRICE_COL].pct_change()

    # Rolling vol if absent
    if ROLLVOL_COL not in df.columns:
        df[ROLLVOL_COL] = (
            df.groupby(COIN_COL)[RET_COL]
              .rolling(window=14, min_periods=7)
              .std()
              .reset_index(level=0, drop=True)
        )

    # Dominance % if absent
    if DOM_COL not in df.columns and MKT_CAP_COL in df.columns:
        total_mcap = df.groupby(DATE_COL)[MKT_CAP_COL].transform("sum")
        df[DOM_COL] = np.where(total_mcap > 0, (df[MKT_CAP_COL] / total_mcap) * 100.0, np.nan)

    # Liquidity tiers if absent
    if TIER_COL not in df.columns and MKT_CAP_COL in df.columns:
        latest_date = df[DATE_COL].max()
        snap = df[df[DATE_COL] == latest_date][[COIN_COL, MKT_CAP_COL]].dropna()
        if len(snap) >= 10:
            q66 = snap[MKT_CAP_COL].quantile(2/3)
            q33 = snap[MKT_CAP_COL].quantile(1/3)
            def tier(m):
                if m >= q66: return "High"
                if m <= q33: return "Low"
                return "Mid"
            df[TIER_COL] = df[COIN_COL].map(dict(zip(snap[COIN_COL], snap[MKT_CAP_COL].apply(tier)))).fillna("Mid")
        else:
            df[TIER_COL] = "Mid"

    # ---- Momentum horizons: auto-compute % changes (in percent) ----
    # Detect median step; compute steps for 1h/24h/7d/30d; fallback to daily.
    try:
        med_step = (
            df.groupby(COIN_COL)[DATE_COL]
              .apply(lambda s: s.sort_values().diff().median())
              .dropna()
              .median()
        )
        if pd.isna(med_step) or med_step == pd.Timedelta(0):
            med_step = df[DATE_COL].sort_values().diff().median()

        if not pd.isna(med_step) and med_step != pd.Timedelta(0):
            steps_per_hour = max(1, int(round(pd.Timedelta(hours=1) / med_step)))
            steps_per_day  = max(1, int(round(pd.Timedelta(days=1)  / med_step)))
        else:
            steps_per_hour = None
            steps_per_day  = None

        def ensure_pct(colname: str, periods: int | None):
            if colname not in df.columns and periods and periods > 0:
                df[colname] = df.groupby(COIN_COL)[PRICE_COL].pct_change(periods=periods).mul(100.0)

        if steps_per_day:
            # 1h only meaningful for sub-daily (avoid duplicating 24h for daily)
            if steps_per_hour and steps_per_day > 1:
                ensure_pct(H1_COL, steps_per_hour)
            ensure_pct(D1_COL, steps_per_day)
            ensure_pct(D7_COL, 7 * steps_per_day)
            ensure_pct(D30_COL, 30 * steps_per_day)
        else:
            # Fallback assume daily
            ensure_pct(D1_COL, 1)
            ensure_pct(D7_COL, 7)
            ensure_pct(D30_COL, 30)
    except Exception:
        if D1_COL not in df.columns:
            df[D1_COL] = df.groupby(COIN_COL)[PRICE_COL].pct_change().mul(100.0)

    return df


@st.cache_data(show_spinner=False)
def load_forecasts(out_dir: Path) -> pd.DataFrame:
    """Load any CSV in outputs/ and coerce to: date, coin_id, vs_currency, model, yhat_vol."""
    req = [DATE_COL, COIN_COL, VS_CURR_COL, "model", "yhat_vol"]
    if not out_dir.exists():
        return pd.DataFrame(columns=req)

    frames = []
    for p in sorted(out_dir.glob("*.csv")):
        try:
            d = pd.read_csv(p).rename(columns={
                "Date": DATE_COL, "timestamp": DATE_COL, "date": DATE_COL,
                "coin": COIN_COL, "symbol": COIN_COL,
                "vol_forecast": "yhat_vol", "sigma": "yhat_vol", "yhat": "yhat_vol",
                "vs_currency": VS_CURR_COL, "vs": VS_CURR_COL,
            })
            stem = p.stem.lower()
            if COIN_COL not in d.columns:
                d[COIN_COL] = "ethereum" if any(x in stem for x in ["eth","ethereum"]) else "bitcoin"
            if VS_CURR_COL not in d.columns:
                d[VS_CURR_COL] = "usd"
            if "model" not in d.columns:
                d["model"] = stem.split("_")[0] if "_" in stem else stem

            if DATE_COL in d.columns:
                d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
            for c in (COIN_COL, VS_CURR_COL, "model"):
                d[c] = d[c].astype(str).str.strip().str.lower()

            if "yhat_vol" not in d.columns:
                continue

            frames.append(d.dropna(subset=[DATE_COL, "yhat_vol"])[req])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=req)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values([COIN_COL, VS_CURR_COL, "model", DATE_COL])


def normalize_to_100(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    first = s.iloc[0]
    return s * np.nan if (pd.isna(first) or first == 0) else (s / first) * 100.0


# =========================================
# SIDEBAR: DATA PATHS & FILTERS
# =========================================
st.sidebar.header("Data")
data_path = st.sidebar.text_input("Path to data (.parquet or .csv, or a folder)", str(DEFAULT_DATA))
outs_path = st.sidebar.text_input("Path to forecasts folder (outputs/)", str(DEFAULT_OUTS))

df = load_panel(data_path)
preds = load_forecasts(Path(outs_path))

# Guard rails / schema hints
need = [DATE_COL, COIN_COL, PRICE_COL, MKT_CAP_COL]
missing = [c for c in need if c not in df.columns] if not df.empty else need
if df.empty or missing:
    st.warning(
        "Processed dataset not found or missing required columns.\n\n"
        f"Expected: {need}\n"
        f"Missing: {missing}\n\n"
        "Run your data prep (e.g., `python scripts/prepare_data.py`) or adjust the path in the sidebar."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Controls")

all_coins = sorted(df[COIN_COL].dropna().unique().tolist())
min_date, max_date = df[DATE_COL].min().date(), df[DATE_COL].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter by date (global)
start_d = pd.to_datetime(date_range[0])
end_d = pd.to_datetime(date_range[-1])
dff = df[(df[DATE_COL] >= start_d) & (df[DATE_COL] <= end_d)].copy()

# Optional filter: subset of coins for the dashboards
subset_coins = st.sidebar.multiselect("Coins (optional filter for dashboards)", options=all_coins, default=[])
if subset_coins:
    dff = dff[dff[COIN_COL].isin(subset_coins)]

top_n = st.sidebar.slider("Top-N (Ranking Bars)", min_value=5, max_value=50, value=10, step=5)
heatmap_horizons = st.sidebar.multiselect(
    "Momentum heatmap horizons",
    options=[H1_COL, D1_COL, D7_COL, D30_COL],
    default=[D1_COL, D7_COL, D30_COL]
)
cmp_coins = st.sidebar.multiselect(
    "Compare-to-100: pick coins",
    options=all_coins[:100],
    default=all_coins[:5] if len(all_coins) >= 5 else all_coins
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear cache"):
    load_panel.clear()
    load_forecasts.clear()
    st.rerun()

# =========================================
# TABS
# =========================================
tabs = st.tabs([
    "Coin Detail: Price · Vol · Forecasts",
    "Market Map", "Ranking Bars", "Momentum Heatmap", "Dominance Trend",
    "Breadth Gauge", "Risk Scatter", "Return Distribution",
    "Volatility Trend", "Compare-to-100", "Liquidity Breakdown"
])

# =========================================
# TAB 0 — Coin Detail
# =========================================
with tabs[0]:
    st.subheader("Coin Detail — Price, Realized/EWMA Volatility, and Forecast Overlays")
    coin_opts = sorted(df.get(COIN_COL, pd.Series([], dtype=str)).dropna().unique().tolist()) or ["unknown"]
    vses = sorted(df.get(VS_CURR_COL, pd.Series(["usd"])).dropna().unique().tolist()) or ["usd"]

    c1, c2, c3 = st.columns([1,1,1])
    coin = c1.selectbox("Coin", coin_opts, index=0)
    vs   = c2.selectbox("vs_currency", vses, index=(vses.index("usd") if "usd" in vses else 0))
    horizon = c3.selectbox("Vol window", [7, 30, 90], index=0)

    g = df[(df[COIN_COL] == coin) & (df.get(VS_CURR_COL, "usd") == vs)].sort_values(DATE_COL).copy()
    if g.empty:
        st.info("No rows for this coin/currency.")
    else:
        tab1, tab2, tab3 = st.tabs(["Price", "Volatility", "Forecasts"])

        with tab1:
            st.subheader(f"{coin.capitalize()}/{vs.upper()} — Price")
            st.plotly_chart(px.line(g, x=DATE_COL, y=PRICE_COL, title=None, labels={PRICE_COL: "Price"}), use_container_width=True)
            st.caption("Daily last observed price.")

        with tab2:
            rv_col, ew_col = f"rv_{horizon}", f"ewma_vol_{horizon}"
            existing = [c for c in (rv_col, ew_col) if c in g.columns]
            st.subheader(f"Realized vs EWMA Volatility (window={horizon})")
            if existing:
                st.plotly_chart(px.line(g, x=DATE_COL, y=existing, title=None), use_container_width=True)
            else:
                st.info("Volatility columns not found. Add rv_* / ewma_vol_* in preprocessing.")

        with tab3:
            st.subheader("Forecast Overlays (GARCH vs ML)")
            sel = preds[(preds[COIN_COL] == coin) & (preds[VS_CURR_COL] == vs)].copy()
            if sel.empty:
                st.info("No forecasts available yet. Export CSVs to outputs/.")
            else:
                if "rv_7" in g.columns:
                    truth = g[[DATE_COL, "rv_7"]].rename(columns={"rv_7": "ytrue_vol"})
                else:
                    base = pd.to_numeric(g.get("log_ret", pd.Series(index=g.index, dtype="float64")), errors="coerce").abs()
                    truth = g[[DATE_COL]].assign(ytrue_vol=base)
                st.plotly_chart(px.line(truth.dropna(), x=DATE_COL, y="ytrue_vol", title="True daily vol (proxy)"),
                                use_container_width=True)

                models = sorted(sel["model"].unique().tolist())
                pick = st.multiselect("Models to overlay", models, default=models)
                for m in pick:
                    mdf = sel[sel["model"] == m].sort_values(DATE_COL)
                    st.plotly_chart(px.line(mdf, x=DATE_COL, y="yhat_vol", title=f"Forecast: {m}"),
                                    use_container_width=True)

# =========================================
# TAB 1 — Market Map
# =========================================
with tabs[1]:
    st.subheader("Market Map — Treemap (Size = MarketCap, Color = % Change 24h)")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    if D1_COL not in snap.columns and PRICE_COL in df.columns:
        tmp = df[[DATE_COL, COIN_COL, PRICE_COL]].copy()
        tmp[D1_COL] = tmp.groupby(COIN_COL)[PRICE_COL].pct_change().mul(100)
        snap = snap.merge(tmp[tmp[DATE_COL] == tmp[DATE_COL].max()][[COIN_COL, D1_COL]], on=COIN_COL, how="left")
    fig = px.treemap(
        snap.dropna(subset=[MKT_CAP_COL]),
        path=[COIN_COL], values=MKT_CAP_COL,
        color=D1_COL if D1_COL in snap.columns else MKT_CAP_COL,
        color_continuous_scale="RdYlGn",
        hover_data={MKT_CAP_COL:":,.0f", **({D1_COL:".2f"} if D1_COL in snap.columns else {})}
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 2 — Ranking Bars
# =========================================
with tabs[2]:
    st.subheader(f"Ranking Bars — Top {top_n} by MarketCap")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    top = snap.sort_values(MKT_CAP_COL, ascending=False).head(top_n)
    fig = px.bar(top, x=COIN_COL, y=MKT_CAP_COL, text=top[MKT_CAP_COL].map(lambda v: f"${v:,.0f}"))
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Market Cap (USD)", uniformtext_minsize=8, uniformtext_mode='show')
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 3 — Momentum Heatmap
# =========================================
with tabs[3]:
    st.subheader("Momentum Heatmap — Rows: Coins, Cols: Horizon, Color: % change")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    cols = [c for c in heatmap_horizons if c in snap.columns]
    if not cols:
        st.info("Selected horizon columns not present in data. Add pct_change_* features.")
    else:
        mat = snap[[COIN_COL] + cols].set_index(COIN_COL)
        fig = px.imshow(mat, aspect="auto", color_continuous_scale="RdYlGn", labels=dict(color="% Change"))
        st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 4 — Dominance Trend
# =========================================
with tabs[4]:
    st.subheader("Dominance Trend — Dominance Share over time")
    latest = dff[dff[DATE_COL] == dff[DATE_COL].max()]
    top_dom = latest.sort_values(DOM_COL, ascending=False).head(8)[COIN_COL].tolist()
    sub = dff[dff[COIN_COL].isin(top_dom)]
    fig = px.line(sub, x=DATE_COL, y=DOM_COL, color=COIN_COL, labels={DOM_COL: "Dominance (%)"})
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 5 — Breadth Gauge (robust)
# =========================================
with tabs[5]:
    st.subheader("Breadth Gauge — Advancers vs Decliners per day")
    if dff.empty or PRICE_COL not in dff.columns:
        st.info("Not enough data to compute breadth.")
    else:
        tmp = dff[[DATE_COL, COIN_COL, PRICE_COL]].copy()
        tmp["ret_d1"] = tmp.groupby(COIN_COL)[PRICE_COL].pct_change()
        sign = np.sign(tmp["ret_d1"].fillna(0))
        daily = (
            tmp.assign(sign=sign)
               .groupby(DATE_COL, as_index=False)
               .agg(
                   advancers=("sign", lambda s: int((s > 0).sum())),
                   decliners=("sign", lambda s: int((s < 0).sum())),
                   unchanged=("sign", lambda s: int((s == 0).sum())),
                   breadth=("sign", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0),
               )
        )
        if daily.empty:
            st.info("Breadth could not be computed for the selected range.")
        else:
            latest_row = daily.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Advancers (latest)", f"{int(latest_row.get('advancers', 0))}")
            c2.metric("Decliners (latest)", f"{int(latest_row.get('decliners', 0))}")
            c3.metric("Breadth % (latest)", f"{latest_row.get('breadth', 0.0):.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Advancers", x=daily[DATE_COL], y=daily["advancers"]))
            fig.add_trace(go.Bar(name="Decliners", x=daily[DATE_COL], y=daily["decliners"]))
            fig.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 6 — Risk Scatter
# =========================================
with tabs[6]:
    st.subheader("Risk Scatter — Price vs Volume24h (log scale)")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    if VOL24_COL not in snap.columns:
        st.info("Missing volume_24h column; cannot plot risk scatter.")
    else:
        fig = px.scatter(
            snap, x=PRICE_COL, y=VOL24_COL, size=MKT_CAP_COL, color=TIER_COL,
            hover_name=COIN_COL, size_max=45, log_x=True, log_y=True
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 7 — Return Distribution
# =========================================
with tabs[7]:
    st.subheader("Return Distribution — Histogram of daily returns")
    sub = dff.dropna(subset=[RET_COL])
    fig = px.histogram(sub, x=RET_COL, nbins=60, marginal="box")
    fig.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 8 — Volatility Trend
# =========================================
with tabs[8]:
    st.subheader("Volatility Trend — Rolling Volatility (14d)")
    latest = dff[dff[DATE_COL] == dff[DATE_COL].max()]
    top_mc = latest.sort_values(MKT_CAP_COL, ascending=False).head(8)[COIN_COL].tolist()
    sub = dff[dff[COIN_COL].isin(top_mc)]
    fig = px.line(sub, x=DATE_COL, y=ROLLVOL_COL, color=COIN_COL, labels={ROLLVOL_COL: "Rolling Vol (σ)"})
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 9 — Compare-to-100
# =========================================
with tabs[9]:
    st.subheader("Compare-to-100 — Normalize price series to 100 at start date")
    if cmp_coins:
        sub = dff[dff[COIN_COL].isin(cmp_coins)][[DATE_COL, COIN_COL, PRICE_COL]].copy()
        sub = sub.sort_values([COIN_COL, DATE_COL])
        sub["idx100"] = sub.groupby(COIN_COL)[PRICE_COL].transform(normalize_to_100)
        fig = px.line(sub, x=DATE_COL, y="idx100", color=COIN_COL, labels={"idx100":"Index (100=start)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick at least one coin in the sidebar to show the compare-to-100 chart.")

# =========================================
# TAB 10 — Liquidity Breakdown
# =========================================
with tabs[10]:
    st.subheader("Liquidity Breakdown — Pie/Bar by tier")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    if TIER_COL not in snap.columns:
        st.info("Liquidity tiers are missing; add a 'liquidity_tier' column or let the app derive it from market cap.")
    else:
        agg = snap.groupby(TIER_COL)[MKT_CAP_COL].sum().reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(agg, names=TIER_COL, values=MKT_CAP_COL, hole=0.35), use_container_width=True)
        with c2:
            fig = px.bar(agg, x=TIER_COL, y=MKT_CAP_COL, text=agg[MKT_CAP_COL].map(lambda v: f"${v:,.0f}"))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Tip: Use the sidebar to change date range, Top-N, horizons, and comparison sets. The first tab also lets you pick coin / vs / window for detailed forecasts.")

