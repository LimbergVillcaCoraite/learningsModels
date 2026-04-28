# Databricks notebook source
# MAGIC %md
# MAGIC # Expose NVDA Forecast Data (Serving Layer)
# MAGIC
# MAGIC Objetivo:
# MAGIC - Publicar una capa de consumo estable en Databricks
# MAGIC - Exponer la ultima prediccion y el historial de pronosticos
# MAGIC - Dejar vistas/tablas listas para Databricks SQL, dashboards o consumers externos

# COMMAND ----------

from pyspark.sql import functions as F
from delta.tables import DeltaTable

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("serving_schema", "serving")
dbutils.widgets.text("symbol", "NVDA")
dbutils.widgets.text("pipeline_run_id", "")

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
serving_schema = dbutils.widgets.get("serving_schema")
symbol = dbutils.widgets.get("symbol")
pipeline_run_id = dbutils.widgets.get("pipeline_run_id")


def resolve_pipeline_run_id() -> str:
    if pipeline_run_id:
        return pipeline_run_id
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        if ctx.jobRunId().isDefined():
            return str(ctx.jobRunId().get())
    except Exception:
        pass
    return "interactive"


pipeline_run_id = resolve_pipeline_run_id()

forecast_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast"
metrics_table = f"{catalog}.{gold_schema}.equity_prices_30d_forecast_metrics"
serving_latest_table = f"{catalog}.{serving_schema}.nvda_forecast_latest"
serving_history_table = f"{catalog}.{serving_schema}.nvda_forecast_history"
serving_metrics_table = f"{catalog}.{serving_schema}.nvda_forecast_metrics"
serving_view = f"{catalog}.{serving_schema}.vw_nvda_forecast_latest"
serving_status_table = f"{catalog}.{serving_schema}.nvda_serving_status"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{serving_schema}")

# COMMAND ----------

# Cargar predicciones y metricas desde Gold
forecast_df = (
    spark.table(forecast_table)
    .filter(F.col("symbol") == F.lit(symbol))
    .withColumn("forecast_ts", F.current_timestamp())
)

metrics_df = (
    spark.table(metrics_table)
    .filter(F.col("symbol") == F.lit(symbol))
    .withColumn("loaded_ts", F.current_timestamp())
)

if forecast_df.limit(1).count() == 0:
    raise ValueError(f"No hay predicciones disponibles para {symbol} en {forecast_table}")

invalid_rows = forecast_df.filter(
    F.col("pred_close").isNull() |
    F.col("pred_low_80").isNull() |
    F.col("pred_high_80").isNull() |
    (F.col("pred_close") <= 0) |
    (F.col("pred_low_80") > F.col("pred_high_80"))
).limit(1).count()

if invalid_rows > 0:
    raise ValueError("Data quality check failed en forecast_df: valores nulos o rangos invalidos")

# COMMAND ----------

# Tabla historica de pronosticos (idempotente con MERGE)
history_df = forecast_df.select(
    "symbol",
    "forecast_date",
    "horizon_day",
    "pred_close",
    "pred_low_80",
    "pred_high_80",
    "model_name",
    "cv_mae",
    "forecast_ts",
).withColumn("pipeline_run_id", F.lit(pipeline_run_id))

if spark.catalog.tableExists(serving_history_table):
    history_target = DeltaTable.forName(spark, serving_history_table)
    (
        history_target.alias("t")
        .merge(
            history_df.alias("s"),
            "t.symbol = s.symbol AND t.forecast_date = s.forecast_date AND t.horizon_day = s.horizon_day",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    history_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(serving_history_table)

# Tabla con el ultimo forecast por simbolo
latest_df = (
    forecast_df
    .orderBy(F.col("horizon_day").asc())
    .limit(1)
    .select(
        "symbol",
        F.col("forecast_date").alias("next_trading_day"),
        F.col("pred_close").alias("predicted_close"),
        F.col("pred_low_80").alias("predicted_low_80"),
        F.col("pred_high_80").alias("predicted_high_80"),
        F.col("model_name"),
        F.col("cv_mae"),
        F.current_timestamp().alias("exposed_ts"),
        F.lit(pipeline_run_id).alias("pipeline_run_id"),
    )
)

latest_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(serving_latest_table)

metrics_out_df = metrics_df.withColumn("pipeline_run_id", F.lit(pipeline_run_id))
metrics_out_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(serving_metrics_table)

status_df = spark.createDataFrame([
    {
        "symbol": symbol,
        "status": "OK",
        "pipeline_run_id": pipeline_run_id,
        "history_rows": int(history_df.count()),
        "metrics_rows": int(metrics_out_df.count()),
    }
]).withColumn("updated_ts", F.current_timestamp())

if spark.catalog.tableExists(serving_status_table):
    status_target = DeltaTable.forName(spark, serving_status_table)
    (
        status_target.alias("t")
        .merge(status_df.alias("s"), "t.symbol = s.symbol")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    status_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(serving_status_table)

# View estable para consumidores SQL
spark.sql(f"DROP VIEW IF EXISTS {serving_view}")
spark.sql(f"CREATE VIEW {serving_view} AS SELECT * FROM {serving_latest_table}")

# COMMAND ----------

# Optimizar tablas de serving
spark.sql(f"OPTIMIZE {serving_history_table} ZORDER BY (symbol, forecast_date)")
spark.sql(f"OPTIMIZE {serving_latest_table} ZORDER BY (symbol)")

print(f"Serving latest: {serving_latest_table}")
print(f"Serving history: {serving_history_table}")
print(f"Serving metrics: {serving_metrics_table}")
print(f"Serving view: {serving_view}")
print(f"Serving status: {serving_status_table}")

display(spark.table(serving_latest_table))
display(spark.table(serving_history_table).orderBy(F.col("forecast_date").asc()).limit(30))
display(spark.table(serving_metrics_table))
display(spark.table(serving_status_table))
