# Databricks notebook source
# MAGIC %pip install scikit-learn xgboost pandas_market_calendars

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
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
dbutils.widgets.text("mlflow_experiment", "/Shared/NVDA_Medallion_Experiment")

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
symbol = dbutils.widgets.get("symbol")
forecast_horizon = int(dbutils.widgets.get("forecast_horizon"))
min_std_threshold = float(dbutils.widgets.get("min_std_threshold"))
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")

gold_daily_table = f"{catalog}.{gold_schema}.equity_prices_daily_features"
forecast_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast"
metrics_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast_metrics"

raw_df = (
    spark.table(gold_daily_table)
    .filter(f"symbol = '{symbol}'")
    .select("date", "close", "adj_close", "volume")
    .orderBy("date")
    .toPandas()
)

if raw_df.empty:
    raise ValueError(f"Sin datos para {symbol} en {gold_daily_table}")

raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df = raw_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

# adj_close corrige el split 10:1 de NVDA (junio 2024); si no existe en Gold
# se usa close como fallback para no romper el pipeline
if "adj_close" not in raw_df.columns or raw_df["adj_close"].isna().all():
    raw_df["adj_close"] = raw_df["close"]
    print("WARN: adj_close no disponible en Gold — usando close como fallback")
else:
    raw_df["adj_close"] = raw_df["adj_close"].fillna(raw_df["close"])

required_cols = ["date", "close", "adj_close", "volume"]
for col in required_cols:
    if raw_df[col].isna().any():
        raise ValueError(f"Datos invalidos: columna {col} contiene nulos")

if (raw_df["close"] <= 0).any():
    raise ValueError("Datos invalidos: close debe ser > 0")

if (raw_df["volume"] < 0).any():
    raise ValueError("Datos invalidos: volume debe ser >= 0")


def resolve_pipeline_run_id() -> str:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        if ctx.jobRunId().isDefined():
            return str(ctx.jobRunId().get())
    except Exception:
        pass
    return "interactive"


pipeline_run_id = resolve_pipeline_run_id()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # Target: siguiente cierre sin ajustar (lo que reporta el mercado)
    x["target"] = x["close"].shift(-1)
    # Features derivadas de adj_close para corregir discontinuidades por splits
    # (ej: split 10:1 de NVDA en junio 2024 aparece como outlier extremo en close)
    p = x["adj_close"]
    for lag in [1, 2, 3, 5, 10, 20]:
        x[f"lag_close_{lag}"] = p.shift(lag)
    x["ret_1"] = p.pct_change(1)
    x["ret_5"] = p.pct_change(5)
    for w in [5, 10, 20, 50]:
        x[f"roll_mean_{w}"] = p.rolling(w).mean().shift(1)
        x[f"roll_std_{w}"] = p.rolling(w).std().shift(1)
    x["dow"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["log_volume"] = np.log1p(x["volume"])
    return x

feat_df = build_features(raw_df)
feature_cols = [c for c in feat_df.columns if c not in ["date", "target", "close", "adj_close", "volume"]]
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
mlflow.set_experiment(mlflow_experiment)
with mlflow.start_run(run_name=f"{symbol}_forecast_30d"):
    mlflow.log_params({
        "symbol": symbol,
        "forecast_horizon": forecast_horizon,
        "catalog": catalog,
        "gold_schema": gold_schema,
        "min_std_threshold": min_std_threshold,
        "pipeline_run_id": pipeline_run_id,
        "train_rows": int(len(train_df)),
        "feature_count": int(len(feature_cols)),
    })

    for name, model in models.items():
        maes, residuals = [], []
        for tr, va in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[va])
            maes.append(mean_absolute_error(y.iloc[va], pred))
            residuals.extend((y.iloc[va] - pred).tolist())
        cv_mae = float(np.mean(maes))
        mlflow.log_metric(f"cv_mae_{name}", cv_mae)
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
            "cv_mae": float(best_cv_mae),
            "pipeline_run_id": pipeline_run_id,
        })
        # Para días futuros adj_close == close (no hay splits conocidos por adelantado)
        hist_df = pd.concat([hist_df, pd.DataFrame({"date": [next_dt], "close": [pred_close], "adj_close": [pred_close], "volume": [hist_df["volume"].iloc[-1]]})], ignore_index=True)

    forecast_df = pd.DataFrame(rows)
    std_pred = float(forecast_df["pred_close"].std())
    if std_pred < min_std_threshold:
        raise ValueError(f"Forecast plano detectado. std={std_pred:.8f} < {min_std_threshold}")

    mlflow.log_metrics({
        "best_cv_mae": float(best_cv_mae),
        "pred_std": std_pred,
        "forecast_rows": int(len(forecast_df)),
    })
    mlflow.set_tag("selected_model", best_model_name)

    spark.createDataFrame(forecast_df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(forecast_table)
    metrics_df = pd.DataFrame([{
        "symbol": symbol,
        "selected_model": best_model_name,
        "best_cv_mae": float(best_cv_mae),
        "pred_std": std_pred,
        "train_rows": int(len(train_df)),
        "pipeline_run_id": pipeline_run_id,
    }])
    spark.createDataFrame(metrics_df).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(metrics_table)

print(f"Forecast guardado: {forecast_table}")
print(f"Metricas guardadas: {metrics_table}")
display(spark.table(forecast_table).orderBy("horizon_day").limit(30))

