.PHONY: setup kaggle live prepare garch rf backtest app
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
kaggle:
	python scripts/fetch_kaggle.py --dataset harshalhonde/coinmarketcap-cryptocurrency-dataset-2023 -o data/raw
live:
	python scripts/fetch_live.py --coins bitcoin ethereum --days 365 -o data/raw
prepare:
	python scripts/prepare_data.py --input data/raw --output data/processed/merged.parquet
garch:
	python scripts/train_garch.py --input data/processed/merged.parquet --coin bitcoin --horizon 7 --out outputs/garch_forecasts.csv
rf:
	python scripts/train_rf_vol.py --input data/processed/merged.parquet --coin bitcoin --out outputs/rf_forecasts.csv
backtest:
	python scripts/backtest.py --truth data/processed/merged.parquet --pred outputs/garch_forecasts.csv --metric rmse
app:
	streamlit run app/streamlit_app.py
