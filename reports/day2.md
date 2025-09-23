# Day 2 — Features & Tidy Data

**Input parquet:** `data/processed/merged.parquet`  
**Rows:** 379  
**Coins:** 1 → unknown  
**VS currencies:** usd  
**Date range:** 2024-01-01 → 2025-09-23

## Feature decisions
- Daily aggregation: last observation per day per coin_id/vs_currency.
- Returns: `log_ret = ln(price).diff()`.
- Realized volatility: rolling std of `log_ret` over **30** days (`rv_30d`).
- EWMA volatility: std of `log_ret` with **halflife=14.0** (`ewmvol_h14`).
- Liquidity: 30-day rolling mean of `total_volume` (`avgvol_30d`), bucketed by global quantiles into `liquidity_tier` (Low/Mid/High; fallback to Low/High if insufficient variance).
- Missing handling: numeric parsing with coercion; rows without `price` removed before return calc.

## Null counts (key columns)
- `price`: 0 nulls
- `market_cap`: 0 nulls
- `total_volume`: 0 nulls
- `log_ret`: 1 nulls
- `rv_30d`: 15 nulls
- `ewmvol_h14`: 5 nulls
- `avgvol_30d`: 4 nulls
- `liquidity_tier`: 0 nulls

## Liquidity tiers (row counts)
| liquidity_tier   |   rows |
|:-----------------|-------:|
| Low              |    127 |
| High             |    127 |
| Mid              |    125 |

## Schema
- `date`: object
- `coin_id`: object
- `vs_currency`: object
- `price`: float64
- `market_cap`: float64
- `total_volume`: float64
- `log_ret`: float64
- `rv_30d`: float64
- `ewmvol_h14`: float64
- `avgvol_30d`: float64
- `liquidity_tier`: category
