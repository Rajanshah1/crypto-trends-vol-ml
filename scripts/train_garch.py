#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from arch import arch_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--coin", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--out", default="outputs/garch_forecasts.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df = df[df["coin"]==args.coin].sort_values("timestamp")
    if df.empty: raise SystemExit(f"No rows for coin={args.coin}")

    r = np.log(df["price"]).diff().dropna()
    am = arch_model(r*100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=args.horizon)
    f = fc.variance.dropna().iloc[-1] ** 0.5
    out = pd.DataFrame({
        "coin": args.coin,
        "h": list(range(1, args.horizon+1)),
        "garch_sigma_pct": f.values
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
