#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Verify schema & write Day 2 report.")
    ap.add_argument("--input", default="data/processed/merged.parquet", help="Parquet input")
    ap.add_argument("--output", default="reports/day2.md", help="Markdown report output")
    ap.add_argument("--rolling-days", type=int, default=30)
    ap.add_argument("--ewm-halflife", type=float, default=14.0)
    args = ap.parse_args()

    p = Path(args.input)
    df = pd.read_parquet(p)

    # Basic profile
    nrows = len(df)
    coins = sorted(df["coin_id"].dropna().unique().tolist())
    vs = sorted(df["vs_currency"].dropna().unique().tolist())
    dmin = pd.to_datetime(df["date"]).min()
    dmax = pd.to_datetime(df["date"]).max()

    # Nulls on key cols
    key_cols = ["price","market_cap","total_volume","log_ret",
                f"rv_{args.rolling_days}d", f"ewmvol_h{int(args.ewm_halflife)}","avgvol_30d","liquidity_tier"]
    null_lines = []
    for c in key_cols:
        if c in df.columns:
            null_lines.append(f"- `{c}`: {int(df[c].isna().sum())} nulls")
    null_text = "\n".join(null_lines)

    # Liquidity tier distribution
    tier_counts = df["liquidity_tier"].value_counts(dropna=False)
    tier_md = tier_counts.to_frame("rows").to_markdown()

    # Schema
    schema_lines = [f"- `{c}`: {df[c].dtype}" for c in df.columns]
    schema_text = "\n".join(schema_lines)

    md = f"""# Day 2 — Features & Tidy Data

**Input parquet:** `{p}`  
**Rows:** {nrows:,}  
**Coins:** {len(coins)} → {', '.join(coins[:15])}{'…' if len(coins)>15 else ''}  
**VS currencies:** {', '.join(vs)}  
**Date range:** {dmin.date()} → {dmax.date()}

## Feature decisions
- Daily aggregation: last observation per day per coin_id/vs_currency.
- Returns: `log_ret = ln(price).diff()`.
- Realized volatility: rolling std of `log_ret` over **{args.rolling_days}** days (`rv_{args.rolling_days}d`).
- EWMA volatility: std of `log_ret` with **halflife={args.ewm_halflife}** (`ewmvol_h{int(args.ewm_halflife)}`).
- Liquidity: 30-day rolling mean of `total_volume` (`avgvol_30d`), bucketed by global quantiles into `liquidity_tier` (Low/Mid/High; fallback to Low/High if insufficient variance).
- Missing handling: numeric parsing with coercion; rows without `price` removed before return calc.

## Null counts (key columns)
{null_text}

## Liquidity tiers (row counts)
{tier_md}

## Schema
{schema_text}
"""
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md, encoding="utf-8")
    print(f"✅ Wrote report → {outp.resolve()}")

if __name__ == "__main__":
    main()

