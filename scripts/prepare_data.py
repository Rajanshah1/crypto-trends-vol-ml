#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def compute_features(df, resample="1D", vol_windows=(7,30,90)):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = []
    for coin, g in df.groupby("coin"):
        g = g.sort_values("timestamp").set_index("timestamp")
        g = g[["price","market_cap","volume"]].resample(resample).last().dropna()
        g["ret_log"] = np.log(g["price"]).diff()
        for w in vol_windows:
            g[f"rv_{w}"] = g["ret_log"].rolling(w).std() * np.sqrt(365)
            g[f"ewma_vol_{w}"] = g["ret_log"].ewm(span=w).std() * np.sqrt(365)
        g["liq_tier"] = pd.qcut(g["volume"].rank(pct=True), q=3, labels=["Low","Mid","High"])
        g["coin"] = coin
        g = g.reset_index().rename(columns={"index":"timestamp"})
        feats.append(g)
    return pd.concat(feats, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resample", default="1D")
    ap.add_argument("--vol_windows", nargs="+", type=int, default=[7,30,90])
    args = ap.parse_args()

    inp = Path(args.input)
    files = list(inp.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found in {inp}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if {"timestamp","coin","price"}.issubset(df.columns):
            frames.append(df[["timestamp","coin","price","market_cap","volume"]])
        else:
            cols = {c.lower():c for c in df.columns}
            date_col = cols.get("date") or cols.get("timestamp")
            name_col = cols.get("name") or cols.get("coin") or cols.get("symbol")
            price_col = cols.get("close") or cols.get("price")
            cap_col = cols.get("marketcap") or cols.get("market_cap")
            vol_col = cols.get("volume") or cols.get("volume24h")
            if date_col and name_col and price_col:
                tmp = pd.DataFrame({
                    "timestamp": pd.to_datetime(df[date_col], errors="coerce"),
                    "coin": df[name_col].astype(str).str.lower(),
                    "price": pd.to_numeric(df[price_col], errors="coerce"),
                    "market_cap": pd.to_numeric(df[cap_col], errors="coerce") if cap_col else np.nan,
                    "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan
                }).dropna(subset=["timestamp","coin","price"])
                frames.append(tmp)

    if not frames:
        raise SystemExit("No compatible files found to merge.")

    raw = pd.concat(frames, ignore_index=True)
    tidy = compute_features(raw, resample=args.resample, vol_windows=tuple(args.vol_windows))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(args.output, index=False)
    print(f"Wrote {args.output} with {len(tidy):,} rows.")

if __name__ == "__main__":
    main()
