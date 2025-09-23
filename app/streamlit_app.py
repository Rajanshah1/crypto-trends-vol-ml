import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Crypto Trends & Volatility", layout="wide")
st.title("📈 Crypto Market Trends & Volatility — ML Visualization")
st.caption("Compare price, realized volatility, and model forecasts (GARCH vs ML).")

data_path = Path("data/processed/merged.parquet")
if not data_path.exists():
    st.warning("Processed dataset not found. Run `python scripts/prepare_data.py` first.")
else:
    df = pd.read_parquet(data_path)
    coins = sorted(df["coin"].unique().tolist())
    coin = st.selectbox("Coin", coins, index=0)
    horizon = st.selectbox("Vol window", [7,30,90], index=0)

    sdf = df[df["coin"]==coin].sort_values("timestamp")
    garch_csv = Path("outputs/garch_forecasts.csv")
    rf_csv = Path("outputs/rf_forecasts.csv")

    tab1, tab2, tab3 = st.tabs(["Price","Volatility","Forecasts"])

    with tab1:
        fig = px.line(sdf, x="timestamp", y="price", title=f"{coin} Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cols = [f"rv_{horizon}", f"ewma_vol_{horizon}"]
        existing = [c for c in cols if c in sdf.columns]
        if existing:
            fig2 = px.line(sdf, x="timestamp", y=existing, title=f"{coin} Realized vs EWMA Vol (window={horizon})")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Volatility columns not found. Re-run prepare_data.")

    with tab3:
        if rf_csv.exists():
            rf = pd.read_csv(rf_csv)
            if "coin" in rf.columns:
                rf = rf[rf["coin"]==coin]
            if not rf.empty:
                fig3 = px.line(rf, x="timestamp", y="rf_vol_pred", title="RandomForest Volatility Forecast")
                st.plotly_chart(fig3, use_container_width=True)
        if garch_csv.exists():
            garch = pd.read_csv(garch_csv)
            if "coin" in garch.columns and coin in garch['coin'].unique():
                st.write("Latest GARCH(1,1) σ forecasts (%):")
                st.dataframe(garch[garch["coin"]==coin].sort_values("h"))
        if (not rf_csv.exists()) and (not garch_csv.exists()):
            st.info("No forecasts available yet. Train models first.")
