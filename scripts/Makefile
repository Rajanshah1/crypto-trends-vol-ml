# ===== Makefile (Crypto Vol ML) =====
# Quick usage:
#   make help
#   make live
#   make prepare
#   make garch COIN=bitcoin
#   make rf COIN=bitcoin
#   make backtest
#   make run-all
#   make app

PY          ?= python
RAW         ?= data/raw
PROC        ?= data/processed
OUT         ?= outputs
APP_DIR     ?= app

COINS       ?= bitcoin ethereum
DAYS        ?= 365

COIN        ?= bitcoin
VS          ?= usd
MIN_TRAIN   ?= 5

PORT        ?= 8501
ADDRESS     ?= 0.0.0.0

.DEFAULT_GOAL := help
.PHONY: help live prepare garch rf backtest app run-all all-no-app clean clean-data clean-outputs

help:
	@echo "Targets:"
	@echo "  live            Fetch live data → $(RAW) (COINS=\"$(COINS)\" DAYS=$(DAYS))"
	@echo "  prepare         Build merged parquet → $(PROC)/merged.parquet"
	@echo "  garch           Train GARCH → $(OUT)/garch_forecasts.csv (COIN=$(COIN), MIN_TRAIN=$(MIN_TRAIN))"
	@echo "  rf              Train RandomForest vol → $(OUT)/rf_forecasts.csv (COIN=$(COIN))"
	@echo "  backtest        Evaluate forecasts (RMSE)"
	@echo "  app             Run Streamlit on :$(PORT)"
	@echo "  run-all         live → prepare → garch → rf → backtest → app"
	@echo "  all-no-app      live → prepare → garch → rf → backtest (no app)"
	@echo "  clean-*         Remove generated artifacts"

live:
	@mkdir -p "$(RAW)"
	$(PY) scripts/fetch_live.py --coins $(COINS) --days $(DAYS) -o $(RAW)

prepare:
	@mkdir -p "$(PROC)"
	$(PY) scripts/prepare_data.py --input $(RAW) --output $(PROC)/merged.parquet

garch: prepare
	@mkdir -p "$(OUT)"
	$(PY) scripts/train_garch.py --input $(PROC)/merged.parquet --coin $(COIN) --vs $(VS) --min-train-size $(MIN_TRAIN) --out $(OUT)/garch_forecasts.csv

rf: prepare
	@mkdir -p "$(OUT)"
	$(PY) scripts/train_rf_vol.py --input $(PROC)/merged.parquet --coin $(COIN) --out $(OUT)/rf_forecasts.csv

backtest: garch rf
	$(PY) scripts/backtest.py --truth $(PROC)/merged.parquet --pred $(OUT)/garch_forecasts.csv --metric rmse
	$(PY) scripts/backtest.py --truth $(PROC)/merged.parquet --pred $(OUT)/rf_forecasts.csv   --metric rmse

app:
	streamlit run $(APP_DIR)/streamlit_app.py --server.port $(PORT) --server.address $(ADDRESS)

run-all: live prepare garch rf backtest app
all-no-app: live prepare garch rf backtest

clean:
	@$(MAKE) clean-data
	@$(MAKE) clean-outputs

clean-data:
	@rm -f $(PROC)/merged.parquet

clean-outputs:
	@rm -f $(OUT)/*.csv

