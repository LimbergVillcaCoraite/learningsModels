# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality & Serving Report
# MAGIC
# MAGIC Objetivo:
# MAGIC - Validar freshness y consistencia minima de tablas de serving
# MAGIC - Publicar un resumen operacional para monitoreo rapido

# COMMAND ----------

from pyspark.sql import functions as F

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("serving_schema", "serving")
dbutils.widgets.text("symbol", "NVDA")
dbutils.widgets.text("pipeline_run_id", "")

catalog = dbutils.widgets.get("catalog")
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

serving_latest_table = f"{catalog}.{serving_schema}.nvda_forecast_latest"
serving_history_table = f"{catalog}.{serving_schema}.nvda_forecast_history"
serving_metrics_table = f"{catalog}.{serving_schema}.nvda_forecast_metrics"
serving_status_table = f"{catalog}.{serving_schema}.nvda_serving_status"
quality_report_table = f"{catalog}.{serving_schema}.nvda_quality_report"

required_tables = [
    serving_latest_table,
    serving_history_table,
    serving_metrics_table,
    serving_status_table,
]

for table_name in required_tables:
    if not spark.catalog.tableExists(table_name):
        raise ValueError(f"Tabla requerida no encontrada: {table_name}")

latest_df = spark.table(serving_latest_table).filter(F.col("symbol") == F.lit(symbol))
history_df = spark.table(serving_history_table).filter(F.col("symbol") == F.lit(symbol))
metrics_df = spark.table(serving_metrics_table).filter(F.col("symbol") == F.lit(symbol))
status_df = spark.table(serving_status_table).filter(F.col("symbol") == F.lit(symbol))

latest_count = latest_df.count()
history_count = history_df.count()
metrics_count = metrics_df.count()
status_count = status_df.count()

if latest_count == 0:
    raise ValueError(f"Sin datos en {serving_latest_table} para symbol={symbol}")
if history_count == 0:
    raise ValueError(f"Sin datos en {serving_history_table} para symbol={symbol}")
if metrics_count == 0:
    raise ValueError(f"Sin datos en {serving_metrics_table} para symbol={symbol}")
if status_count == 0:
    raise ValueError(f"Sin datos en {serving_status_table} para symbol={symbol}")

freshness_row = latest_df.select(F.max("exposed_ts").alias("latest_exposed_ts")).collect()[0]
latest_exposed_ts = freshness_row["latest_exposed_ts"]
if latest_exposed_ts is None:
    raise ValueError("latest_exposed_ts es nulo en serving_latest")

invalid_history = history_df.filter(
    F.col("pred_close").isNull()
    | F.col("pred_low_80").isNull()
    | F.col("pred_high_80").isNull()
    | (F.col("pred_close") <= 0)
    | (F.col("pred_low_80") > F.col("pred_high_80"))
).limit(1).count()
if invalid_history > 0:
    raise ValueError("Data quality failed: se detectaron filas invalidas en serving_history")

report_df = spark.createDataFrame([
    {
        "symbol": symbol,
        "pipeline_run_id": pipeline_run_id,
        "latest_rows": int(latest_count),
        "history_rows": int(history_count),
        "metrics_rows": int(metrics_count),
        "status_rows": int(status_count),
        "quality_status": "PASS",
    }
]).withColumn("report_ts", F.current_timestamp())

report_df.write.format("delta").mode("append").saveAsTable(quality_report_table)

print(f"Quality report table: {quality_report_table}")
print(f"Pipeline run id: {pipeline_run_id}")
print(f"Latest exposed ts: {latest_exposed_ts}")

display(report_df)
display(spark.table(quality_report_table).orderBy(F.col("report_ts").desc()).limit(20))

