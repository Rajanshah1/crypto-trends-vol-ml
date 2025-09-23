#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

def build_features(df, coin):
    x = df[df["coin"]==coin].sort_values("timestamp").copy()
    x["ret_log"] = np.log(x["price"]).diff()
    x["target"] = x["rv_7"].shift(-1)
    for lag in [1,2,3,5,7,14,21]:
        x[f"ret_lag_{lag}"] = x["ret_log"].shift(lag)
        x[f"rv7_lag_{lag}"] = x["rv_7"].shift(lag)
        x[f"ewma7_lag_{lag}"] = x["ewma_vol_7"].shift(lag)
    x = x.dropna()
    feats = [c for c in x.columns if c.startswith(("ret_lag_","rv7_lag_","ewma7_lag_"))]
    return x, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--coin", required=True)
    ap.add_argument("--out", default="outputs/rf_forecasts.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    x, feats = build_features(df, args.coin)
    y = x["target"].values; X = x[feats].values

    tscv = TimeSeriesSplit(n_splits=5)
    preds = pd.Series(index=x.index, dtype=float)
    for tr, te in tscv.split(X):
        mdl = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        mdl.fit(X[tr], y[tr])
        preds.iloc[te] = mdl.predict(X[te])

    out = x.loc[preds.index, ["timestamp"]].copy()
    out["coin"] = args.coin
    out["rf_vol_pred"] = preds.values
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
