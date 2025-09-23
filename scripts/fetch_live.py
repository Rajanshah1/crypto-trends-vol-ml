#!/usr/bin/env python
import argparse, time
from pathlib import Path
import requests
import pandas as pd

API_BASE = "https://api.coingecko.com/api/v3"

def fetch_market_chart(coin_id, vs_currency="usd", days=365):
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(f"{API_BASE}/coins/{coin_id}/market_chart", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["ts_ms","price"]).assign(coin=coin_id)
    mktcap = pd.DataFrame(data["market_caps"], columns=["ts_ms","market_cap"]).assign(coin=coin_id)
    vol = pd.DataFrame(data["total_volumes"], columns=["ts_ms","volume"]).assign(coin=coin_id)
    df = prices.merge(mktcap, on=["ts_ms","coin"]).merge(vol, on=["ts_ms","coin"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms")
    return df.drop(columns=["ts_ms"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["bitcoin","ethereum"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("-o","--output", default="data/raw")
    ap.add_argument("--vs", default="usd")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in args.coins:
        df = fetch_market_chart(c, vs_currency=args.vs, days=args.days)
        out_csv = out / f"{c}_{args.vs}_market_chart_{args.days}d.csv"
        df.to_csv(out_csv, index=False); frames.append(df)
        print(f"Wrote {out_csv} ({len(df)} rows)")
        time.sleep(1)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(out / "coingecko_merged.csv", index=False)
        print(f"Wrote merged -> {out/'coingecko_merged.csv'} ({len(all_df)} rows)")

if __name__ == "__main__":
    main()
