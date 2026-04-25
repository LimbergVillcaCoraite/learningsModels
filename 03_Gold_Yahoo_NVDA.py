# Databricks notebook source
﻿# Databricks notebook source
%md
# Gold - Serving Layer (Best Practices)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Parametros (widgets)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")

catalog = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
gold_schema = dbutils.widgets.get("gold_schema")

silver_table = f"{catalog}.{silver_schema}.equity_prices_daily"
gold_daily_table = f"{catalog}.{gold_schema}.equity_prices_daily_features"
gold_monthly_table = f"{catalog}.{gold_schema}.equity_prices_monthly"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{gold_schema}")

# COMMAND ----------

silver_df = spark.table(silver_table)
if silver_df.rdd.isEmpty():
    raise ValueError(f"Silver vacia: {silver_table}")

w_symbol_date = Window.partitionBy("symbol").orderBy("date")
w_20 = w_symbol_date.rowsBetween(-19, 0)
w_50 = w_symbol_date.rowsBetween(-49, 0)

gold_daily_df = (
    silver_df
    .withColumn("daily_return_pct", (F.col("close") / F.lag("close", 1).over(w_symbol_date) - 1.0) * 100.0)
    .withColumn("sma_20", F.avg("close").over(w_20))
    .withColumn("sma_50", F.avg("close").over(w_50))
    .withColumn("avg_volume_20", F.avg("volume").over(w_20))
    .withColumn("volatility_20d_pct", F.stddev("daily_return_pct").over(w_20))
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date"))
)

gold_daily_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(gold_daily_table)

monthly_df = (
    silver_df
    .groupBy(F.col("symbol"), F.date_trunc("month", F.col("date")).alias("month_start"))
    .agg(
        F.first("open", ignorenulls=True).alias("month_open"),
        F.max("high").alias("month_high"),
        F.min("low").alias("month_low"),
        F.last("close", ignorenulls=True).alias("month_close"),
        F.sum("volume").alias("month_volume")
    )
    .withColumn("month_return_pct", (F.col("month_close") / F.col("month_open") - 1.0) * 100.0)
)

monthly_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(gold_monthly_table)

spark.sql(f"OPTIMIZE {gold_daily_table} ZORDER BY (symbol, date)")
spark.sql(f"OPTIMIZE {gold_monthly_table} ZORDER BY (symbol, month_start)")

print(f"Gold diario: {gold_daily_table}")
print(f"Gold mensual: {gold_monthly_table}")
display(spark.table(gold_daily_table).orderBy(F.col("date").desc()).limit(20))
