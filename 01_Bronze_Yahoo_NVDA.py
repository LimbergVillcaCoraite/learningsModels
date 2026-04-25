# Databricks notebook source
# MAGIC %pip install yfinance

# COMMAND ----------

from pyspark.sql import functions as F
import yfinance as yf
import uuid

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("symbol", "NVDA")
dbutils.widgets.text("source_interval", "1d")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
symbol = dbutils.widgets.get("symbol")
source_interval = dbutils.widgets.get("source_interval")

source_system = "yahoo_finance"
bronze_table = f"{catalog}.{bronze_schema}.yahoo_finance_prices_raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{bronze_schema}")

# COMMAND ----------

pdf = yf.download(
    symbol,
    period="max",
    interval=source_interval,
    auto_adjust=False,
    progress=False,
)

if pdf.empty:
    raise ValueError(f"No se recibieron datos para {symbol} desde Yahoo Finance")

# Aplanar columnas MultiIndex de yfinance y normalizar nombres
pdf = pdf.reset_index()
pdf.columns = [c[0] if isinstance(c, tuple) else c for c in pdf.columns]
pdf.columns = [str(c).strip().lower().replace(" ", "_") for c in pdf.columns]

# Metadatos tecnicos de ingesta
pdf["source_symbol"] = symbol
pdf["source_system"] = source_system
pdf["source_interval"] = source_interval
pdf["ingestion_run_id"] = str(uuid.uuid4())

sdf = spark.createDataFrame(pdf)
sdf = sdf.withColumn("ingestion_ts", F.current_timestamp()).withColumn("ingestion_date", F.to_date(F.col("ingestion_ts")))

# COMMAND ----------

(
    sdf.write
       .format("delta")
       .mode("append")
       .partitionBy("ingestion_date")
       .saveAsTable(bronze_table)
)

spark.sql(f"OPTIMIZE {bronze_table} ZORDER BY (source_symbol)")
print(f"Bronze actualizado: {bronze_table}")
display(spark.table(bronze_table).orderBy(F.col("ingestion_ts").desc()).limit(20))
