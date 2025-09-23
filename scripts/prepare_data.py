#!/usr/bin/env python3
import argparse, sys, math, warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATE_CANDIDATES = ["date","Date","timestamp","Timestamp","time","Time","open_time","datetime","Datetime"]
PRICE_CANDIDATES = ["price","Price","close","Close","adj_close","Adj Close","price_usd","Close Price USD"]
MCAP_CANDIDATES  = ["market_cap","MarketCap","mktcap","MktCap","marketcap"]
VOL_CANDIDATES   = ["total_volume","TotalVolume","volume","Volume","Volume24h","volume_24h","volumeusd","volume_usd"]
COIN_CANDIDATES  = ["coin_id","CoinID","name","Name","symbol","Symbol","Ticker","ticker"]
VS_CANDIDATES    = ["vs_currency","vs","currency","Currency","quote","Quote"]

def first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def standardize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = list(df.columns)
    # date
    dcol = first_present(cols, DATE_CANDIDATES)
    if dcol is None:
        # cannot time-index → skip
        return None
    df = df.copy()
    df.rename(columns={dcol: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[~df["date"].isna()]
    # price
    pcol = first_present(cols, PRICE_CANDIDATES)
    if pcol: df.rename(columns={pcol: "price"}, inplace=True)
    # market cap
    mcol = first_present(cols, MCAP_CANDIDATES)
    if mcol: df.rename(columns={mcol: "market_cap"}, inplace=True)
    # volume (prefer USD if available)
    vcol = first_present(cols, VOL_CANDIDATES)
    if vcol: df.rename(columns={vcol: "total_volume"}, inplace=True)
    # coin id
    ccol = first_present(cols, COIN_CANDIDATES)
    if ccol:
        df.rename(columns={ccol: "coin_id"}, inplace=True)
    else:
        df["coin_id"] = "unknown"
    # vs currency
    vscol = first_present(cols, VS_CANDIDATES)
    if vscol:
        df.rename(columns={vscol: "vs_currency"}, inplace=True)
    else:
        df["vs_currency"] = "usd"

    # canonical subset
    keep = ["date","coin_id","vs_currency","price","market_cap","total_volume"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    # clean coin_id
    df["coin_id"] = df["coin_id"].astype(str).str.strip().str.lower()
    df["vs_currency"] = df["vs_currency"].astype(str).str.strip().str.lower()
    # ensure numeric
    for k in ["price","market_cap","total_volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    # coerce to date (not datetime) for daily index
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[keep]

def load_any(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            return None
        out = standardize(df)
        return out
    except Exception as e:
        print(f"⚠️  Skipping {path.name}: {e}", file=sys.stderr)
        return None

def daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    # For duplicate dates per coin, pick last observation for price/mcap/volume
    agg = (df
        .groupby(["coin_id","vs_currency","date"], as_index=False)
        .agg({
            "price":"last",
            "market_cap":"last",
            "total_volume":"last"
        })
    )
    return agg.sort_values(["coin_id","vs_currency","date"])

def add_features(df: pd.DataFrame, rolling_days: int, ewm_halflife: float) -> pd.DataFrame:
    df = df.sort_values(["coin_id","vs_currency","date"]).copy()
    # convert date back to datetime index per group to ease rolling
    df["date"] = pd.to_datetime(df["date"])
    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("date").sort_index()
        # Forward fill price (optional) to avoid holes before returns? We compute returns only on available days.
        g["log_price"] = np.log(g["price"])
        g["log_ret"] = g["log_price"].diff()
        # Realized volatility (rolling std of log returns)
        win = max(2, int(rolling_days))
        g[f"rv_{rolling_days}d"] = g["log_ret"].rolling(window=win, min_periods=max(2, int(win*0.5))).std()
        # EWMA volatility on log returns
        g[f"ewmvol_h{int(ewm_halflife)}"] = g["log_ret"].ewm(halflife=ewm_halflife, adjust=False, min_periods=5).std()
        # 30d average dollar volume proxy (if total_volume is USD; else this is just a volume proxy)
        g["avgvol_30d"] = g["total_volume"].rolling(30, min_periods=5).mean()
        # back to columns
        return g.reset_index()

    df = (df.groupby(["coin_id","vs_currency"], group_keys=False).apply(_per_group))

    # Liquidity tiers by global quantiles of avgvol_30d
    # If a lot of NaNs, fill with 0 for tiering only
    qsrc = df["avgvol_30d"].fillna(0.0)
    try:
        df["liquidity_tier"] = pd.qcut(qsrc, q=3, labels=["Low","Mid","High"])
    except ValueError:
        # not enough distinct values; fall back to binary
        try:
            df["liquidity_tier"] = pd.qcut(qsrc, q=2, labels=["Low","High"])
        except ValueError:
            df["liquidity_tier"] = "Low"

    # drop helper
    df.drop(columns=["log_price"], inplace=True, errors="ignore")
    # Final ordering
    cols = ["date","coin_id","vs_currency","price","market_cap","total_volume",
            "log_ret", f"rv_{rolling_days}d", f"ewmvol_h{int(ewm_halflife)}",
            "avgvol_30d","liquidity_tier"]
    # ensure date as date (not Timestamp) for compact parquet
    df["date"] = df["date"].dt.date
    return df[cols]

def main():
    ap = argparse.ArgumentParser(description="Prepare tidy daily panel with returns/vol features.")
    ap.add_argument("--input", required=True, help="Input file or directory (CSV/Parquet). If dir, loads all *.csv/*.parquet")
    ap.add_argument("--output", required=True, help="Output Parquet path, e.g., data/processed/merged.parquet")
    ap.add_argument("--rolling-days", type=int, default=30, help="Window for realized volatility (default 30)")
    ap.add_argument("--ewm-halflife", type=float, default=14.0, help="Halflife for EWMA vol (default 14)")
    args = ap.parse_args()

    in_path = Path(args.input)
    files: List[Path] = []
    if in_path.is_dir():
        files = sorted(list(in_path.glob("*.csv")) + list(in_path.glob("*.parquet")))
    else:
        files = [in_path]

    frames: List[pd.DataFrame] = []
    for f in files:
        out = load_any(f)
        if out is None:
            print(f"⚠️  Skipping (no usable date/price): {f.name}", file=sys.stderr)
            continue
        frames.append(out)

    if not frames:
        print("No usable input files found.", file=sys.stderr)
        sys.exit(2)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["date"]).copy()

    daily = daily_agg(raw)
    # drop rows with no price for returns
    daily = daily[~daily["price"].isna()].copy()

    feat = add_features(daily, rolling_days=args.rolling_days, ewm_halflife=args.ewm_halflife)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(outp, index=False)
    print(f"✅ Wrote {len(feat):,} rows → {outp.resolve()}")

if __name__ == "__main__":
    main()

