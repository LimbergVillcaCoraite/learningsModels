-- Databricks notebook source
-- MAGIC %md
-- MAGIC # NVDA Serving SQL Queries
-- MAGIC
-- MAGIC Queries listas para Databricks SQL, dashboards o consumo desde FastAPI.

-- COMMAND ----------

-- Prediccion mas reciente
SELECT
  symbol,
  next_trading_day,
  predicted_close,
  predicted_low_80,
  predicted_high_80,
  model_name,
  cv_mae,
  exposed_ts
FROM workspace.serving.vw_nvda_forecast_latest;

-- COMMAND ----------

-- Historico completo de forecasts
SELECT
  symbol,
  forecast_date,
  horizon_day,
  pred_close,
  pred_low_80,
  pred_high_80,
  model_name,
  cv_mae,
  forecast_ts
FROM workspace.serving.nvda_forecast_history
ORDER BY forecast_date ASC;

-- COMMAND ----------

-- Metricas del ultimo forecast
SELECT
  symbol,
  selected_model,
  best_cv_mae,
  pred_std,
  train_rows
FROM workspace.serving.nvda_forecast_metrics;
