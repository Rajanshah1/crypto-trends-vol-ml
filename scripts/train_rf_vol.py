#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def build_panel(df: pd.DataFrame, coin: str, vs: str) -> pd.DataFrame:
    g = df[(df["coin_id"] == coin) & (df["vs_currency"] == vs)].copy()
    if g.empty:
        raise SystemExit(f"No rows for coin_id={coin}, vs={vs}.")
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").reset_index(drop=True)
    return g

def make_features(g: pd.DataFrame, lags=(1,2,3,5,7,14)):
    # Base features available from prepare_data.py
    base_cols = [c for c in [
        "rv_7","rv_30","rv_90",
        "ewma_vol_7","ewma_vol_30","ewma_vol_90",
        "avgvol_30d"
    ] if c in g.columns]

    x = g[["date"] + base_cols].copy()

    # Lagged |log_ret| features
    if "log_ret" in g.columns:
        x["abs_ret"] = g["log_ret"].abs()
        for L in lags:
            x[f"abs_ret_l{L}"] = x["abs_ret"].shift(L)

    # Target = next-day realized vol (rv_7) as a proxy
    y = g["rv_7"].shift(-1) if "rv_7" in g.columns else g["log_ret"].abs().shift(-1)

    # Drop rows with any NaNs in features/target
    data = x.copy()
    data["y"] = y
    data = data.dropna().reset_index(drop=True)

    X_cols = [c for c in data.columns if c not in ("date","y")]
    return data[["date"] + X_cols + ["y"]], X_cols

def walk_forward_predict(
    data: pd.DataFrame,
    X_cols,
    min_train=60,
    rf_kwargs=None
):
    rf_kwargs = rf_kwargs or dict(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf = RandomForestRegressor(**rf_kwargs)

    preds = []
    # one-step-ahead expanding window
    for t in range(min_train, len(data)-1):
        train = data.iloc[:t]
        test_next = data.iloc[t:t+1]  # we’ll predict the y for the next row’s date

        rf.fit(train[X_cols], train["y"])
        yhat = rf.predict(test_next[X_cols])[0]
        # the forecast date is the *next* date (the label row already shifted)
        preds.append((data.iloc[t]["date"], yhat))

    out = pd.DataFrame(preds, columns=["date","yhat_vol"])
    return out

def main():
    ap = argparse.ArgumentParser(description="RandomForest baseline: next-day vol forecasts (walk-forward).")
    ap.add_argument("--input", required=True, help="Processed parquet from prepare_data.py")
    ap.add_argument("--coin", required=True, help="coin_id (e.g., bitcoin)")
    ap.add_argument("--vs", default="usd", help="vs_currency (default: usd)")
    ap.add_argument("--min-train", type=int, default=60, help="minimum training points before first prediction")
    ap.add_argument("--out", required=True, help="Output CSV")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    g = build_panel(df, coin=args.coin, vs=args.vs)
    data, X_cols = make_features(g)

    if len(data) <= args.min_train + 1:
        raise SystemExit(f"Not enough rows after alignment (got {len(data)}).")

    preds = walk_forward_predict(data, X_cols, min_train=args.min_train)

    preds["coin_id"] = args.coin
    preds["vs_currency"] = args.vs
    preds["model"] = "rf"

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    preds[["date","coin_id","vs_currency","model","yhat_vol"]].to_csv(outp, index=False)
    print(f"✅ Wrote RF vol forecasts → {outp.resolve()} (rows={len(preds)})")

if __name__ == "__main__":
    main()

