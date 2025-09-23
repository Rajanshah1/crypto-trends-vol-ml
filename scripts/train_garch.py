#!/usr/bin/env python3
import argparse, sys, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd

def _die(msg: str, code: int = 2):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)

def main():
    ap = argparse.ArgumentParser(description="Walk-forward 1-step GARCH(1,1) volatility forecasts for a coin.")
    ap.add_argument("--input", required=True, help="Parquet from Day 2 (merged.parquet)")
    ap.add_argument("--coin", required=True, help="coin_id (e.g., bitcoin)")
    ap.add_argument("--vs", default="usd", help="vs_currency (default: usd)")
    ap.add_argument("--min-train-size", type=int, default=150, help="minimum observations to start walk-forward")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    try:
        from arch import arch_model  # type: ignore
    except Exception:
        _die("arch package not found. Try: `pip install arch` (or use Python 3.11/3.12).")

    df = pd.read_parquet(args.input)
    df = df[(df["coin_id"] == args.coin) & (df["vs_currency"] == args.vs)].copy()
    if df.empty:
        _die(f"No rows for coin_id={args.coin}, vs={args.vs} in {args.input}")
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    # Use log returns from Day 2
    r = df["log_ret"].dropna().reset_index(drop=True)
    r_idx = df.loc[df["log_ret"].notna(), "date"].reset_index(drop=True)

    if len(r) < args.min_train_size + 5:
        _die(f"Not enough data for walk-forward (have {len(r)}, need >= {args.min_train_size}+).")

    rows = []
    for t in range(args.min_train_size, len(r) - 1):
        y = r.iloc[:t]  # returns up to day t-1
        try:
            am = arch_model(y, mean="zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            fcast = res.forecast(horizon=1)
            sigma = float(np.sqrt(fcast.variance.values[-1, 0]))
        except Exception as e:
            # fallback if optimizer hiccups
            sigma = np.nan

        # predict next day (t -> t+1)
        target_date = r_idx.iloc[t + 1]
        rows.append({
            "date": pd.to_datetime(target_date).date(),
            "coin_id": args.coin,
            "model": "garch11",
            "yhat_vol": sigma
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ Wrote GARCH forecasts → {out.resolve()} (rows={len(rows)})")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

