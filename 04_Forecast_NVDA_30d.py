# Databricks notebook source
# MAGIC %pip install scikit-learn xgboost pandas_market_calendars

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas_market_calendars as mcal

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("symbol", "NVDA")
dbutils.widgets.text("forecast_horizon", "30")
dbutils.widgets.text("min_std_threshold", "0.0001")

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
symbol = dbutils.widgets.get("symbol")
forecast_horizon = int(dbutils.widgets.get("forecast_horizon"))
min_std_threshold = float(dbutils.widgets.get("min_std_threshold"))

gold_daily_table = f"{catalog}.{gold_schema}.equity_prices_daily_features"
forecast_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast"
metrics_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast_metrics"

raw_df = (
    spark.table(gold_daily_table)
    .filter(f"symbol = '{symbol}'")
    .select("date", "close", "volume")
    .orderBy("date")
    .toPandas()
)

if raw_df.empty:
    raise ValueError(f"Sin datos para {symbol} en {gold_daily_table}")

raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df = raw_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["target"] = x["close"].shift(-1)
    for lag in [1, 2, 3, 5, 10, 20]:
        x[f"lag_close_{lag}"] = x["close"].shift(lag)
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_5"] = x["close"].pct_change(5)
    for w in [5, 10, 20, 50]:
        x[f"roll_mean_{w}"] = x["close"].rolling(w).mean().shift(1)
        x[f"roll_std_{w}"] = x["close"].rolling(w).std().shift(1)
    x["dow"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["log_volume"] = np.log1p(x["volume"])
    return x

feat_df = build_features(raw_df)
feature_cols = [c for c in feat_df.columns if c not in ["date", "target", "close", "volume"]]
train_df = feat_df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
X = train_df[feature_cols]
y = train_df["target"]

if len(train_df) < 250:
    raise ValueError("Insuficiente historia para entrenar")

models = {
    "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=42, max_iter=5000),
    "random_forest": RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}
if HAS_XGB:
    models["xgboost"] = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)

tscv = TimeSeriesSplit(n_splits=5)
best_model_name, best_model, best_cv_mae, best_residuals = None, None, 1e18, None
for name, model in models.items():
    maes, residuals = [], []
    for tr, va in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        maes.append(mean_absolute_error(y.iloc[va], pred))
        residuals.extend((y.iloc[va] - pred).tolist())
    cv_mae = float(np.mean(maes))
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_model_name = name
        best_model = model
        best_residuals = np.array(residuals)

best_model.fit(X, y)
q_low, q_high = np.quantile(best_residuals, [0.1, 0.9]) if len(best_residuals) >= 30 else (-best_cv_mae, best_cv_mae)

nyse = mcal.get_calendar("NYSE")
last_date = raw_df["date"].max()
future_sched = nyse.schedule(start_date=last_date + pd.Timedelta(days=1), end_date=last_date + pd.Timedelta(days=120))
future_days = pd.to_datetime(future_sched.index).tz_localize(None)[:forecast_horizon]

hist_df = raw_df.copy()
rows = []
for i, next_dt in enumerate(future_days, start=1):
    hf = build_features(hist_df)
    hf_valid = hf.dropna(subset=feature_cols)
    x_next = hf_valid.iloc[[-1]][feature_cols]
    pred_close = float(max(best_model.predict(x_next)[0], 0.01))
    rows.append({
        "symbol": symbol,
        "forecast_date": next_dt,
        "horizon_day": i,
        "pred_close": pred_close,
        "pred_low_80": float(min(pred_close + q_low, pred_close + q_high)),
        "pred_high_80": float(max(pred_close + q_low, pred_close + q_high)),
        "model_name": best_model_name,
        "cv_mae": float(best_cv_mae)
    })
    hist_df = pd.concat([hist_df, pd.DataFrame({"date": [next_dt], "close": [pred_close], "volume": [hist_df["volume"].iloc[-1]]})], ignore_index=True)

forecast_df = pd.DataFrame(rows)
std_pred = float(forecast_df["pred_close"].std())
if std_pred < min_std_threshold:
    raise ValueError(f"Forecast plano detectado. std={std_pred:.8f} < {min_std_threshold}")

spark.createDataFrame(forecast_df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(forecast_table)
metrics_df = pd.DataFrame([{"symbol": symbol, "selected_model": best_model_name, "best_cv_mae": float(best_cv_mae), "pred_std": std_pred, "train_rows": int(len(train_df))}])
spark.createDataFrame(metrics_df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(metrics_table)

print(f"Forecast guardado: {forecast_table}")
print(f"Metricas guardadas: {metrics_table}")
display(spark.table(forecast_table).orderBy("horizon_day").limit(30))
