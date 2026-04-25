# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")

bronze_table = f"{catalog}.{bronze_schema}.yahoo_finance_prices_raw"
silver_table = f"{catalog}.{silver_schema}.equity_prices_daily"
rejects_table = f"{catalog}.{silver_schema}.equity_prices_daily_rejects"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{silver_schema}")

# COMMAND ----------

bronze_df = spark.table(bronze_table)
bronze_cols = set(bronze_df.columns)

def pick_col(candidates):
    for name in candidates:
        if name in bronze_cols:
            return F.col(f"`{name}`")
    return F.lit(None)

base_df = (
    bronze_df
    .select(
        pick_col(["source_symbol"]).cast("string").alias("symbol"),
        F.to_date(pick_col(["Date", "date"]).cast("timestamp")).alias("date"),
        pick_col(["Open", "open"]).cast("double").alias("open"),
        pick_col(["High", "high"]).cast("double").alias("high"),
        pick_col(["Low", "low"]).cast("double").alias("low"),
        pick_col(["Close", "close"]).cast("double").alias("close"),
        pick_col(["Adj Close", "adj_close", "adj close"]).cast("double").alias("adj_close"),
        pick_col(["Volume", "volume"]).cast("bigint").alias("volume"),
        pick_col(["source_system"]).cast("string").alias("source_system"),
        pick_col(["source_interval"]).cast("string").alias("source_interval"),
        pick_col(["ingestion_ts"]).cast("timestamp").alias("ingestion_ts"),
        pick_col(["ingestion_date"]).cast("date").alias("ingestion_date"),
        pick_col(["ingestion_run_id"]).cast("string").alias("ingestion_run_id")
    )
)

validated_df = (
    base_df
    .withColumn(
        "quality_issue",
        F.when(F.col("symbol").isNull(), F.lit("symbol_null"))
         .when(F.col("date").isNull(), F.lit("date_null"))
         .when(F.col("open").isNull(), F.lit("open_null"))
         .when(F.col("high").isNull(), F.lit("high_null"))
         .when(F.col("low").isNull(), F.lit("low_null"))
         .when(F.col("close").isNull(), F.lit("close_null"))
         .when(F.col("volume").isNull(), F.lit("volume_null"))
         .when(F.col("high") < F.col("low"), F.lit("high_lt_low"))
         .when(F.col("volume") < 0, F.lit("volume_negative"))
         .otherwise(F.lit(None))
    )
    .withColumn("is_valid", F.col("quality_issue").isNull())
)

valid_df = validated_df.filter(F.col("is_valid") == True).drop("quality_issue", "is_valid")
rejects_df = validated_df.filter(F.col("is_valid") == False)

w = Window.partitionBy("symbol", "date").orderBy(F.col("ingestion_ts").desc_nulls_last())
latest_valid_df = valid_df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
latest_valid_df.createOrReplaceTempView("silver_upsert_source")

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {silver_table} (
  symbol STRING,
  date DATE,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  adj_close DOUBLE,
  volume BIGINT,
  source_system STRING,
  source_interval STRING,
  ingestion_ts TIMESTAMP,
  ingestion_date DATE,
  ingestion_run_id STRING
)
USING DELTA
PARTITIONED BY (symbol)
""")

spark.sql(f"""
MERGE INTO {silver_table} t
USING silver_upsert_source s
ON t.symbol = s.symbol AND t.date = s.date
WHEN MATCHED AND s.ingestion_ts >= t.ingestion_ts THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

rejects_df.write.format("delta").mode("append").saveAsTable(rejects_table)

spark.sql(f"OPTIMIZE {silver_table} ZORDER BY (date)")
spark.sql(f"OPTIMIZE {rejects_table} ZORDER BY (quality_issue, ingestion_date)")

print(f"Silver actualizado: {silver_table}")
print(f"Rejects actualizado: {rejects_table}")
print(f"Total: {base_df.count():,} | Validos: {valid_df.count():,} | Rechazados: {rejects_df.count():,}")

display(spark.table(silver_table).orderBy(F.col("date").desc()).limit(20))
