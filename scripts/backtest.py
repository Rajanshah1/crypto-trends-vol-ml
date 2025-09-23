#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--metric", default="rmse", choices=["rmse","mae","mape"])
    ap.add_argument("--coin", default="bitcoin")
    args = ap.parse_args()

    truth = pd.read_parquet(args.truth)
    truth = truth[truth["coin"]==args.coin][["timestamp","rv_7"]].rename(columns={"rv_7":"truth"})
    pred = pd.read_csv(args.pred)

    if "timestamp" not in pred.columns:
        print("WARN: No timestamp in pred; cannot align exactly."); return

    df = truth.merge(pred, on="timestamp", how="inner")
    pred_cols = [c for c in df.columns if c not in ["timestamp","truth","coin_x","coin_y"]]
    if not pred_cols:
        raise SystemExit("No prediction column found.")
    pcol = pred_cols[-1]

    e = df[pcol] - df["truth"]
    rmse = float(np.sqrt((e**2).mean()))
    mae  = float(np.abs(e).mean())
    mape = float((np.abs(e/df["truth"]).replace([np.inf,-np.inf], np.nan).dropna()).mean() * 100)
    print(f"Samples: {len(df)}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\nMAPE: {mape:.3f}%")

if __name__ == "__main__":
    main()
