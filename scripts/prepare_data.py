#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Candidate headers
DATE_CANDIDATES = ["date","Date","timestamp","Timestamp","time","Time","datetime","Datetime"]
PRICE_CANDIDATES = ["price","Price","close","Close","adj_close","Adj Close"]
MCAP_CANDIDATES  = ["market_cap","MarketCap","marketcap","MktCap"]
VOL_CANDIDATES   = ["total_volume","TotalVolume","volume","Volume","Volume24h","volume_24h","volumeusd","volume_usd"]
COIN_CANDIDATES  = ["coin_id","CoinID","coin","Coin","name","Name","symbol","Symbol"]
VS_CANDIDATES    = ["vs_currency","vs","currency","Currency","quote","Quote"]

def first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols: return c
    return None

def standardize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = list(df.columns)
    # date
    dcol = first_present(cols, DATE_CANDIDATES)
    if dcol is None: return None
    df = df.copy()
    df.rename(columns={dcol: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[~df["date"].isna()]
    # price / mcap / volume
    pcol = first_present(cols, PRICE_CANDIDATES)
    if pcol: df.rename(columns={pcol: "price"}, inplace=True)
    mcol = first_present(cols, MCAP_CANDIDATES)
    if mcol: df.rename(columns={mcol: "market_cap"}, inplace=True)
    vcol = first_present(cols, VOL_CANDIDATES)
    if vcol: df.rename(columns={vcol: "total_volume"}, inplace=True)
    # coin id
    ccol = first_present(cols, COIN_CANDIDATES)
    if ccol: df.rename(columns={ccol: "coin_id"}, inplace=True)
    else: df["coin_id"] = "unknown"
    # vs currency
    vscol = first_present(cols, VS_CANDIDATES)
    if vscol: df.rename(columns={vscol: "vs_currency"}, inplace=True)
    else: df["vs_currency"] = "usd"

    keep = ["date","coin_id","vs_currency","price","market_cap","total_volume"]
    for k in keep:
        if k not in df.columns: df[k] = np.nan

    df["coin_id"] = df["coin_id"].astype(str).str.strip().str.lower()
    df["vs_currency"] = df["vs_currency"].astype(str).str.strip().str.lower()
    for k in ["price","market_cap","total_volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[keep]

def infer_from_filename(out: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Fill coin_id/vs_currency from filename when missing or 'unknown'."""
    stem = path.stem.lower()        # e.g. bitcoin_usd_market_chart_365d
    need_coin = (
        ("coin_id" not in out.columns) or
        out["coin_id"].isna().all() or
        out["coin_id"].astype(str).str.lower().eq("unknown").all() or
        (out["coin_id"].astype(str).str.len().fillna(0) == 0).all()
    )
    if need_coin:
        first = stem.split("_")[0]
        out["coin_id"] = first if first not in {"data","coingecko","merged","market","currencies"} else "unknown"

    need_vs = ("vs_currency" not in out.columns) or out["vs_currency"].isna().all()
    if need_vs:
        guess = None
        for code in ["usd","eur","gbp","inr","jpy","cny","aud","cad"]:
            if f"_{code}_" in stem or stem.endswith(f"_{code}") or stem.startswith(f"{code}_"):
                guess = code; break
        out["vs_currency"] = guess or "usd"

    out["coin_id"] = out["coin_id"].astype(str).str.strip().str.lower()
    out["vs_currency"] = out["vs_currency"].astype(str).str.strip().str.lower()
    return out

def load_any(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            # coingecko CSVs are lower-case ['price','coin','market_cap','volume','timestamp']
            # map to our candidates so standardize() can see them
            if "timestamp" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"timestamp": "date"})
        elif path.suffix.lower() in [".parquet",".pq"]:
            df = pd.read_parquet(path)
        else:
            return None
        out = standardize(df)
        if out is None: return None
        out = infer_from_filename(out, path)
        return out
    except Exception as e:
        print(f"⚠️  Skipping {path.name}: {e}", file=sys.stderr)
        return None

def daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["coin_id","vs_currency","date"], as_index=False)
             .agg({"price":"last","market_cap":"last","total_volume":"last"}))
    return agg.sort_values(["coin_id","vs_currency","date"])

def compute_features(
    df: pd.DataFrame,
    resample_rule: str = "1D",
    vol_windows: Tuple[int,...] = (7,30,90),
    annualize: bool = True
) -> pd.DataFrame:
    df = df.sort_values(["coin_id","vs_currency","date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    scale = np.sqrt(365.0) if annualize else 1.0

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("date").sort_index()
        g = g[["price","market_cap","total_volume"]].resample(resample_rule).last()
        g = g.dropna(subset=["price"])
        g["log_ret"] = np.log(g["price"]).diff()
        for w in vol_windows:
            g[f"rv_{w}"] = g["log_ret"].rolling(window=w, min_periods=max(2,int(w*0.5))).std() * scale
            g[f"ewma_vol_{w}"] = g["log_ret"].ewm(span=w, adjust=False, min_periods=max(2,int(w*0.5))).std() * scale
        g["avgvol_30d"] = g["total_volume"].rolling(30, min_periods=5).mean()
        # tiers within group (fallback if few distinct)
        try:
            g["liq_tier"] = pd.qcut(g["avgvol_30d"].fillna(0.0), q=3, labels=["Low","Mid","High"])
        except ValueError:
            try:
                g["liq_tier"] = pd.qcut(g["avgvol_30d"].fillna(0.0), q=2, labels=["Low","High"])
            except ValueError:
                g["liq_tier"] = "Low"
        return g.reset_index()

    out = (df.groupby(["coin_id","vs_currency"], group_keys=False)
             .apply(lambda g: _per_group(g).assign(coin_id=g.name[0], vs_currency=g.name[1])))

    cols = (["date","coin_id","vs_currency","price","market_cap","total_volume","log_ret"]
            + [f"rv_{w}" for w in vol_windows] + [f"ewma_vol_{w}" for w in vol_windows]
            + ["avgvol_30d","liq_tier"])
    out["date"] = out["date"].dt.date
    return out[[c for c in cols if c in out.columns]].copy()

def main():
    ap = argparse.ArgumentParser(description="Tidy daily crypto panel + features.")
    ap.add_argument("--input", required=True, help="File or directory (CSV/Parquet)")
    ap.add_argument("--output", required=True, help="Output Parquet path")
    ap.add_argument("--resample", default="1D")
    ap.add_argument("--vol_windows", nargs="+", type=int, default=[7,30,90])
    ap.add_argument("--no_annualize", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    files = (sorted(list(in_path.glob("*.csv")) + list(in_path.glob("*.parquet")))
             if in_path.is_dir() else [in_path])

    frames: List[pd.DataFrame] = []
    for f in files:
        out = load_any(f)
        if out is None:
            print(f"⚠️  Skipping (no usable date/price): {f.name}", file=sys.stderr)
            continue
        frames.append(out)

    if not frames:
        sys.exit("No compatible files found to merge.")

    raw = pd.concat(frames, ignore_index=True).dropna(subset=["date"]).copy()
    daily = daily_panel(raw)
    daily = daily[~daily["price"].isna()].copy()

    tidy = compute_features(
        daily,
        resample_rule=args.resample,
        vol_windows=tuple(args.vol_windows),
        annualize=not args.no_annualize
    )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(outp, index=False)
    print(f"✅ Wrote {outp} with {len(tidy):,} rows.")

if __name__ == "__main__":
    main()

