#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Crypto Market Dashboards", layout="wide")

# ---- Fix title clipping (padding + responsive title) --------------------------
st.markdown("""
<style>
/* Give the main content some breathing room from the very top */
div.block-container { padding-top: 2.75rem !important; }

/* In very small viewports (Cloud mobile), add a bit more */
@media (max-width: 900px){
  div.block-container { padding-top: 3.25rem !important; }
}

/* Make sure long titles wrap and never clip */
.app-title{
  margin: .25rem 0 .4rem 0;     /* small top margin keeps it off the top edge */
  font-weight: 800;
  font-size: clamp(22px, 3.0vw, 40px);
  line-height: 1.15;
  white-space: normal;
  overflow-wrap: anywhere;
}
.app-title .muted{
  font-weight: 700;
  opacity: .9;
}
.app-subtitle{
  margin: 0 0 1rem 0;
  color: #4B5563;              /* slate-600 */
  font-size: clamp(12px, 1.2vw, 14px);
}
@media (prefers-color-scheme: dark){
  .app-subtitle{ color: #9CA3AF; }  /* slate-400 */
}

/* Ensure the app itself can scroll past the header without hiding content */
.stApp { overflow: visible !important; }
</style>
""", unsafe_allow_html=True)

# ---- Title (short & unclipped) ------------------------------------------------
st.markdown(
    "<h1 class='app-title'>📊 Crypto Dashboards <span class='muted'>· 📈 Price · Vol · Forecasts</span></h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Market map, rankings, momentum, dominance, breadth, risk, returns, volatility, "
    "compare-to-100, liquidity — plus detailed coin price/volatility and model forecasts (GARCH vs ML)."
)

# ---- Plotly theme base (product) ---------------------------------------------
px.defaults.template = "plotly_white"
pio.templates["product"] = pio.templates["plotly_white"]
pio.templates["product"].layout.update(
    margin=dict(l=24, r=24, t=48, b=24),
    font=dict(family="Inter, ui-sans-serif, system-ui", size=13, color="#111827"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12),
    colorway=["#6C5CE7","#00C2A8","#F66D44","#3E92CC","#FFC857",
              "#2BB0ED","#9B59B6","#10B981","#F59E0B","#EF4444"],
)
px.defaults.template = "product"

# ---- Theming helpers (dark mode) ---------------------------------------------
LIGHT_TEMPLATE = "product"
DARK_TEMPLATE  = "plotly_dark"
LIGHT_SEQ     = ["#6C5CE7"]
DARK_SEQ      = ["#A29BFE"]

def apply_theme(dark: bool):
    if dark:
        px.defaults.template = DARK_TEMPLATE
        px.defaults.color_discrete_sequence = DARK_SEQ
        st.markdown("""
        <style>
          .stApp { background:#0f1221; color:#e7e9f1; }
          .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
          /* cards / boxes */
          .kpi, .card { background:#141833 !important; border:1px solid #1f2748 !important; }
          .kpi h4 { color:#aab1c6 !important; }
          .kpi .value { color:#e7e9f1 !important; }
          section[data-testid="stSidebar"] { background:#0b1020; }
          section[data-testid="stSidebar"] label { color:#cbd5e1 !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        px.defaults.template = LIGHT_TEMPLATE
        px.defaults.color_discrete_sequence = LIGHT_SEQ

# ---- Modern CSS (cards, KPIs, tabs) ------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] { font-family:'Inter', ui-sans-serif, system-ui !important; }
  .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
  .kpi { border-radius:16px; padding:14px 16px; background:#fff;
         border:1px solid #E5E7EB; box-shadow:0 1px 2px rgba(16,24,40,0.04); }
  .kpi h4 { font-size:12px; font-weight:600; color:#667085; margin:0 0 6px 0;}
  .kpi .value { font-size:20px; font-weight:700; color:#111827;}
  .stTabs [role="tab"][aria-selected="true"] { border-bottom:2px solid #6C5CE7; }
  section[data-testid="stSidebar"] label { font-weight:600; color:#344054; }
  .card { border-radius:16px; padding:12px; background:#fff;
          border:1px solid #E5E7EB; box-shadow:0 1px 3px rgba(16,24,40,0.06); }
</style>
""", unsafe_allow_html=True)

# Small helper to display charts inside a "card"
def card_chart(fig):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d","select2d"]}
    )
    st.markdown('</div>', unsafe_allow_html=True)

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
def _normalize_panel_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Same renames/typing used by load_panel, for uploaded files."""
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={
        "timestamp": DATE_COL, "date": DATE_COL,
        "coin": COIN_COL, "symbol": COIN_COL, "name": COIN_COL,
        "total_volume": VOL24_COL, "volume_24h": VOL24_COL, "volume": VOL24_COL, "vol_24h": VOL24_COL,
        "marketcap": MKT_CAP_COL, "market_cap": MKT_CAP_COL,
        "price_usd": PRICE_COL, "close": PRICE_COL,
        "vs_currency": VS_CURR_COL,
        "ret_log": "log_ret", "ewmvol_h14": "ewma_vol_14", "liq_tier": TIER_COL,
    })
    if DATE_COL in d.columns:
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        d = d.dropna(subset=[DATE_COL])
    for c in (COIN_COL, VS_CURR_COL):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower()
    return d

@st.cache_data(show_spinner=False)
def load_panel(path_like: Path | str) -> pd.DataFrame:
    p = Path(path_like).expanduser()
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
    df = _normalize_panel_from_df(df)

    need = [DATE_COL, COIN_COL, PRICE_COL, MKT_CAP_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.sort_values([COIN_COL, DATE_COL])

    if RET_COL not in df.columns:
        df[RET_COL] = df.groupby(COIN_COL)[PRICE_COL].pct_change()

    if ROLLVOL_COL not in df.columns:
        df[ROLLVOL_COL] = (
            df.groupby(COIN_COL)[RET_COL]
              .rolling(window=14, min_periods=7)
              .std()
              .reset_index(level=0, drop=True)
        )

    if DOM_COL not in df.columns and MKT_CAP_COL in df.columns:
        total_mcap = df.groupby(DATE_COL)[MKT_CAP_COL].transform("sum")
        df[DOM_COL] = np.where(total_mcap > 0, (df[MKT_CAP_COL] / total_mcap) * 100.0, np.nan)

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
            if steps_per_hour and steps_per_day > 1:
                ensure_pct(H1_COL, steps_per_hour)
            ensure_pct(D1_COL, steps_per_day)
            ensure_pct(D7_COL, 7 * steps_per_day)
            ensure_pct(D30_COL, 30 * steps_per_day)
        else:
            ensure_pct(D1_COL, 1)
            ensure_pct(D7_COL, 7)
            ensure_pct(D30_COL, 30)
    except Exception:
        if D1_COL not in df.columns:
            df[D1_COL] = df.groupby(COIN_COL)[PRICE_COL].pct_change().mul(100.0)
    return df

@st.cache_data(show_spinner=False)
def load_forecasts(out_dir: Path) -> pd.DataFrame:
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
# SIDEBAR: APPEARANCE + DATA
# =========================================
st.sidebar.header("Appearance")
dark_default = st.session_state.get("dark_mode", False)
dark_mode = st.sidebar.toggle("�� Dark mode", value=dark_default)
st.session_state["dark_mode"] = dark_mode
apply_theme(dark_mode)

st.sidebar.markdown("---")
st.sidebar.header("Data source")

uploaded = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv","parquet"])
st.sidebar.caption("Tip: Leave the path blank to auto-detect from the repo on Streamlit Cloud.")

data_path = st.sidebar.text_input("Path to data (.parquet or .csv, or a folder)", value="")
outs_path  = st.sidebar.text_input("Path to forecasts folder (outputs/)", str(DEFAULT_OUTS))

# Resolve data
if uploaded is not None:
    if uploaded.name.lower().endswith(".parquet"):
        df = _normalize_panel_from_df(pd.read_parquet(uploaded))
    else:
        df = _normalize_panel_from_df(pd.read_csv(uploaded))
    resolved_label = f"Uploaded file: **{uploaded.name}**"
else:
    if not data_path.strip():
        data_path = "."
    df = load_panel(data_path)
    try:
        resolved_label = f"Resolved data path → `{Path(data_path).resolve()}`"
    except Exception:
        resolved_label = f"Data path: {data_path}"

preds = load_forecasts(Path(outs_path))
st.sidebar.caption(resolved_label)

# Guard rails
need = [DATE_COL, COIN_COL, PRICE_COL, MKT_CAP_COL]
missing = [c for c in need if c not in df.columns] if not df.empty else need
if df.empty or missing:
    st.warning(
        "Processed dataset not found or missing required columns.\n\n"
        f"Expected: {need}\n"
        f"Missing: {missing}\n\n"
        "Run your data prep (e.g., `python scripts/prepare_data.py`), upload a file, or adjust the path in the sidebar."
    )
    st.stop()

# =========================================
# SIDEBAR: DASHBOARD CONTROLS (+ live coin search)
# =========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Controls")

all_coins = sorted(df[COIN_COL].dropna().unique().tolist())
min_date, max_date = df[DATE_COL].min().date(), df[DATE_COL].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# live search for coin filtering
query = st.sidebar.text_input("🔎 Filter coins (optional)", "")
filtered = [c for c in all_coins if query.lower() in c.lower()] if query else all_coins

subset_coins = st.sidebar.multiselect("Coins (applies to dashboards)", options=filtered, default=[])
top_n = st.sidebar.slider("Top-N (Ranking Bars)", 5, 50, 10, 5)
heatmap_horizons = st.sidebar.multiselect("Momentum heatmap horizons",
                                          [H1_COL, D1_COL, D7_COL, D30_COL],
                                          default=[D1_COL, D7_COL, D30_COL])

cmp_query = st.sidebar.text_input("🔎 Compare-to-100 coin search", "")
cmp_pool = [c for c in all_coins if cmp_query.lower() in c.lower()] if cmp_query else all_coins
cmp_coins = st.sidebar.multiselect("Compare-to-100: pick coins", options=cmp_pool[:200],
                                   default=cmp_pool[:5] if len(cmp_pool) >= 5 else cmp_pool)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear cache"):
    load_panel.clear(); load_forecasts.clear(); st.rerun()

# Apply date/coin filters
start_d = pd.to_datetime(date_range[0]); end_d = pd.to_datetime(date_range[-1])
dff = df[(df[DATE_COL] >= start_d) & (df[DATE_COL] <= end_d)].copy()
if subset_coins:
    dff = dff[dff[COIN_COL].isin(subset_coins)]

# ---------- Global KPI strip ---------------------------------------------------
def _fmt_money(v): 
    try: return f"${v:,.0f}"
    except: return "—"
def _fmt_pct(v):
    try: return f"{v:+.1f}%"
    except: return "—"

latest_snap = dff[dff[DATE_COL] == dff[DATE_COL].max()]
total_mc = latest_snap[MKT_CAP_COL].sum() if MKT_CAP_COL in latest_snap else np.nan

tmp = dff[[DATE_COL, COIN_COL, PRICE_COL]].copy()
tmp["ret_d1"] = tmp.groupby(COIN_COL)[PRICE_COL].pct_change()
sign = np.sign(tmp["ret_d1"].fillna(0))
daily = tmp.assign(sign=sign).groupby(DATE_COL, as_index=False).agg(
    advancers=("sign", lambda s: int((s > 0).sum())),
    decliners=("sign", lambda s: int((s < 0).sum())),
    breadth=("sign", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0),
)
adv = int(daily.iloc[-1]["advancers"]) if not daily.empty else 0
dec = int(daily.iloc[-1]["decliners"]) if not daily.empty else 0
brd = float(daily.iloc[-1]["breadth"]) if not daily.empty else 0.0

c1, c2, c3, c4 = st.columns([1.2,1,1,1])
with c1: st.markdown(f"""<div class="kpi"><h4>Total Market Cap</h4><div class="value">{_fmt_money(total_mc)}</div></div>""", unsafe_allow_html=True)
with c2: st.markdown(f"""<div class="kpi"><h4>Advancers</h4><div class="value">{adv}</div></div>""", unsafe_allow_html=True)
with c3: st.markdown(f"""<div class="kpi"><h4>Decliners</h4><div class="value">{dec}</div></div>""", unsafe_allow_html=True)
with c4: st.markdown(f"""<div class="kpi"><h4>Breadth</h4><div class="value">{_fmt_pct(brd)}</div></div>""", unsafe_allow_html=True)
st.markdown("")

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
            card_chart(px.line(g, x=DATE_COL, y=PRICE_COL, title=None, labels={PRICE_COL: "Price"}))
            st.caption("Daily last observed price.")

        with tab2:
            rv_col, ew_col = f"rv_{horizon}", f"ewma_vol_{horizon}"
            existing = [c for c in (rv_col, ew_col) if c in g.columns]
            st.subheader(f"Realized vs EWMA Volatility (window={horizon})")
            if existing:
                card_chart(px.line(g, x=DATE_COL, y=existing, title=None))
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
                card_chart(px.line(truth.dropna(), x=DATE_COL, y="ytrue_vol", title="True daily vol (proxy)"))
                models = sorted(sel["model"].unique().tolist())
                pick = st.multiselect("Models to overlay", models, default=models)
                for m in pick:
                    mdf = sel[sel["model"] == m].sort_values(DATE_COL)
                    card_chart(px.line(mdf, x=DATE_COL, y="yhat_vol", title=f"Forecast: {m}"))

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
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Market Cap: %{value:$,.0f}"
                                   + ("<br>%Δ 24h: %{color:.2f}%" if D1_COL in snap.columns else "")
                                   + "<extra></extra>")
    card_chart(fig)

# =========================================
# TAB 2 — Ranking Bars
# =========================================
with tabs[2]:
    st.subheader(f"Ranking Bars — Top {top_n} by MarketCap")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    top = snap.sort_values(MKT_CAP_COL, ascending=False).head(top_n)

    fig = px.bar(
        top,
        x=COIN_COL,
        y=MKT_CAP_COL,
        text=top[MKT_CAP_COL].map(lambda v: f"${v:,.0f}"),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>Market Cap: %{y:$,.0f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Market Cap (USD)",
        uniformtext_minsize=8,
        uniformtext_mode="show",
    )
    card_chart(fig)

    # Download Top-N CSV
    csv = top[[COIN_COL, MKT_CAP_COL]].to_csv(index=False).encode("utf-8")
    st.download_button("Download Top-N CSV", csv, "top_marketcap.csv", "text/csv")

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
        fig = px.imshow(mat, aspect="auto", color_continuous_scale="RdYlGn", labels=dict(color="% Change"), origin="lower")
        fig.update_coloraxes(cmid=0)
        card_chart(fig)

# =========================================
# TAB 4 — Dominance Trend
# =========================================
with tabs[4]:
    st.subheader("Dominance Trend — Dominance Share over time")
    latest = dff[dff[DATE_COL] == dff[DATE_COL].max()]
    top_dom = latest.sort_values(DOM_COL, ascending=False).head(8)[COIN_COL].tolist()
    sub = dff[dff[COIN_COL].isin(top_dom)]
    card_chart(px.line(sub, x=DATE_COL, y=DOM_COL, color=COIN_COL, labels={DOM_COL: "Dominance (%)"}))

# =========================================
# TAB 5 — Breadth Gauge
# =========================================
with tabs[5]:
    st.subheader("Breadth Gauge — Advancers vs Decliners per day")
    if dff.empty or PRICE_COL not in dff.columns:
        st.info("Not enough data to compute breadth.")
    else:
        tmp = dff[[DATE_COL, COIN_COL, PRICE_COL]].copy()
        tmp["ret_d1"] = tmp.groupby(COIN_COL)[PRICE_COL].pct_change()
        sign = np.sign(tmp["ret_d1"].fillna(0))
        daily = (tmp.assign(sign=sign)
                   .groupby(DATE_COL, as_index=False)
                   .agg(advancers=("sign", lambda s: int((s > 0).sum())),
                        decliners=("sign", lambda s: int((s < 0).sum())),
                        unchanged=("sign", lambda s: int((s == 0).sum())),
                        breadth=("sign", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0)))
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
            card_chart(fig)

# =========================================
# TAB 6 — Risk Scatter
# =========================================
with tabs[6]:
    st.subheader("Risk Scatter — Price vs Volume24h (log scale)")
    snap = dff.loc[dff[DATE_COL] == dff[DATE_COL].max()].copy()
    if VOL24_COL not in snap.columns:
        st.info("Missing volume_24h column; cannot plot risk scatter.")
    else:
        card_chart(px.scatter(snap, x=PRICE_COL, y=VOL24_COL, size=MKT_CAP_COL, color=TIER_COL,
                              hover_name=COIN_COL, size_max=45, log_x=True, log_y=True))

# =========================================
# TAB 7 — Return Distribution
# =========================================
with tabs[7]:
    st.subheader("Return Distribution — Histogram of daily returns")
    sub = dff.dropna(subset=[RET_COL])
    fig = px.histogram(sub, x=RET_COL, nbins=60, marginal="box")
    fig.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency")
    card_chart(fig)

# =========================================
# TAB 8 — Volatility Trend
# =========================================
with tabs[8]:
    st.subheader("Volatility Trend — Rolling Volatility (14d)")
    latest = dff[dff[DATE_COL] == dff[DATE_COL].max()]
    top_mc = latest.sort_values(MKT_CAP_COL, ascending=False).head(8)[COIN_COL].tolist()
    sub = dff[dff[COIN_COL].isin(top_mc)]
    card_chart(px.line(sub, x=DATE_COL, y=ROLLVOL_COL, color=COIN_COL, labels={ROLLVOL_COL: "Rolling Vol (σ)"}))

# =========================================
# TAB 9 — Compare-to-100
# =========================================
with tabs[9]:
    st.subheader("Compare-to-100 — Normalize price series to 100 at start date")
    if cmp_coins:
        sub = dff[dff[COIN_COL].isin(cmp_coins)][[DATE_COL, COIN_COL, PRICE_COL]].copy()
        sub = sub.sort_values([COIN_COL, DATE_COL])
        sub["idx100"] = sub.groupby(COIN_COL)[PRICE_COL].transform(normalize_to_100)
        card_chart(px.line(sub, x=DATE_COL, y="idx100", color=COIN_COL, labels={"idx100":"Index (100=start)"}))
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
            card_chart(px.pie(agg, names=TIER_COL, values=MKT_CAP_COL, hole=0.35))
        with c2:
            fig = px.bar(agg, x=TIER_COL, y=MKT_CAP_COL, text=agg[MKT_CAP_COL].map(lambda v: f"${v:,.0f}"))
            fig.update_traces(textposition="outside")
            card_chart(fig)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Tip: Toggle dark mode, upload a CSV/Parquet to explore new data, and use the coin search boxes to focus the dashboards.")

