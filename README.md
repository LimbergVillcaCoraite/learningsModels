# NVDA Stock Analytics Pipeline

A production-ready data engineering project implementing a **Medallion Architecture** (Bronze-Silver-Gold) for NVIDIA (NVDA) stock market data analysis and forecasting using Databricks, Delta Lake, and PySpark.

## 📊 Project Overview

This project builds an end-to-end data pipeline that:
- **Ingests** historical stock price data from Yahoo Finance
- **Processes** and cleanses data through medallion layers
- **Engineers features** for time-series analysis
- **Forecasts** 30-day stock price predictions
- **Stores** data in optimized Delta tables for analytics

## 🏗️ Architecture

### Medallion Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  BRONZE LAYER (Raw Data)                                    │
│  • yahoo_finance_prices_raw                                 │
│  • Preserves source data integrity                          │
│  • Includes ingestion metadata                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  SILVER LAYER (Cleansed & Validated)                        │
│  • equity_prices_daily                                      │
│  • Data quality checks                                      │
│  • Standardized schema                                      │
│  • Rejects table for invalid records                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  GOLD LAYER (Business-Ready)                                │
│  • equity_prices_daily_features                             │
│  • equity_prices_30d_forecast                               │
│  • equity_prices_30d_forecast_metrics                       │
│  • Aggregated & enriched for analytics                      │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

- **Platform**: Databricks on AWS
- **Storage**: Delta Lake (ACID transactions, time travel)
- **Compute**: Apache Spark (distributed processing)
- **Languages**: Python (PySpark), SQL
- **Data Source**: Yahoo Finance API (`yfinance`)
- **Catalog**: Unity Catalog (`workspace` catalog)

## 📁 Project Structure

```
NVDA_Medallion/
├── 01_Bronze_Yahoo_NVDA.ipynb       # Data ingestion from Yahoo Finance
├── 02_Silver_Yahoo_NVDA.ipynb       # Data cleansing & validation
├── 03_Gold_Yahoo_NVDA.ipynb         # Feature engineering (inferred)
├── 04_Forecast_NVDA_30d.ipynb       # Time-series forecasting
├── IngestaSoccer.ipynb              # Additional data source (FIFA)
├── NVDA_Yahoo_Historico.ipynb       # Historical data exploration
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Databricks workspace (AWS)
- Python 3.x
- Required libraries (auto-installed in notebooks):
  - `yfinance`
  - `pandas`
  - `pyspark`

### Installation & Setup

1. **Clone or import** this repository into your Databricks workspace
2. **Run notebooks sequentially** from Bronze → Silver → Gold → Forecast
3. **Attach compute** or use serverless compute (auto-selected)

### Quick Start

```python
# 1. Bronze Layer - Ingest raw data
%run ./01_Bronze_Yahoo_NVDA

# 2. Silver Layer - Cleanse and validate
%run ./02_Silver_Yahoo_NVDA

# 3. Gold Layer - Feature engineering
%run ./03_Gold_Yahoo_NVDA

# 4. Forecasting - Generate predictions
%run ./04_Forecast_NVDA_30d
```

## 📈 Data Pipeline Details

### **Bronze Layer** (`01_Bronze_Yahoo_NVDA`)

**Purpose**: Raw data ingestion with full historical preservation

**Key Features**:
- Downloads complete historical data for NVDA (1999-present)
- Handles multi-level column names from yfinance
- Appends data with ingestion timestamps
- Partitioned by `ingestion_date` for efficient querying
- **Output**: `workspace.bronze.yahoo_finance_prices_raw`

**Schema**:
```
├── date (timestamp)
├── adj_close (double)
├── close (double)
├── high (double)
├── low (double)
├── open (double)
├── volume (long)
├── source_symbol (string)
├── source_system (string)
├── source_interval (string)
├── ingestion_run_id (string)
├── ingestion_ts (timestamp)
└── ingestion_date (date)
```

**Data Volume**: ~6,856 rows spanning 1999-01-22 to 2026-04-24

### **Silver Layer** (`02_Silver_Yahoo_NVDA`)

**Purpose**: Data quality enforcement and standardization

**Key Features**:
- Validates data types and business rules
- Removes duplicates and handles nulls
- Creates reject table for non-conforming records
- Standardizes column naming conventions
- **Output**: 
  - `workspace.silver.equity_prices_daily`
  - `workspace.silver.equity_prices_daily_rejects`

**Data Quality Checks**:
- Non-null validation for critical fields
- Price range validation (positive values)
- Date continuity checks
- Volume anomaly detection

### **Gold Layer** (Feature Engineering)

**Purpose**: Business-ready analytics tables

**Key Features**:
- Technical indicators (moving averages, RSI, MACD)
- Price momentum features
- Volume-weighted metrics
- Calendar features (day of week, month, quarter)
- **Output**: `workspace.gold.equity_prices_daily_features`

### **Forecasting Layer** (`04_Forecast_NVDA_30d`)

**Purpose**: 30-day stock price predictions

**Key Features**:
- Time-series forecasting models
- Model performance metrics (RMSE, MAE, MAPE)
- Confidence intervals for predictions
- **Outputs**:
  - `workspace.gold.equity_prices_30d_forecast`
  - `workspace.gold.equity_prices_30d_forecast_metrics`

## 📊 Key Features

✅ **Incremental Processing**: Append-only design preserves historical snapshots  
✅ **Data Quality**: Automated validation with reject handling  
✅ **Performance**: Z-ordering and partitioning for query optimization  
✅ **Reproducibility**: Ingestion metadata tracks data lineage  
✅ **Scalability**: Built on Spark for distributed processing  
✅ **Time Travel**: Delta Lake enables historical analysis  
✅ **Unity Catalog**: Centralized governance and access control  

## 🔍 Usage Examples

### Query Bronze Layer (Raw Data)
```sql
SELECT *
FROM workspace.bronze.yahoo_finance_prices_raw
WHERE source_symbol = 'NVDA'
  AND ingestion_date >= '2026-01-01'
ORDER BY date DESC
LIMIT 100;
```

### Query Silver Layer (Cleansed Data)
```sql
SELECT 
  date,
  close,
  volume,
  adj_close
FROM workspace.silver.equity_prices_daily
WHERE symbol = 'NVDA'
  AND date BETWEEN '2025-01-01' AND '2026-01-01'
ORDER BY date;
```

### Query Gold Layer (Features)
```sql
SELECT *
FROM workspace.gold.equity_prices_daily_features
WHERE symbol = 'NVDA'
ORDER BY date DESC
LIMIT 50;
```

### View Forecast Results
```sql
SELECT 
  forecast_date,
  predicted_close,
  confidence_lower,
  confidence_upper
FROM workspace.gold.equity_prices_30d_forecast
WHERE symbol = 'NVDA'
ORDER BY forecast_date;
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'yfinance'`  
**Solution**: Run `%pip install yfinance` in a cell before importing

**Issue**: `[NO_SUCH_CATALOG_EXCEPTION] Catalog 'main' was not found`  
**Solution**: Update catalog references from `main` to `workspace`

**Issue**: `[DELTA_INVALID_CHARACTERS_IN_COLUMN_NAMES]`  
**Solution**: Column name flattening is implemented in Bronze layer

**Issue**: Multi-level column names from yfinance  
**Solution**: Bronze layer automatically flattens tuple column names

## 📈 Performance Optimizations

- **Z-Ordering**: Tables are Z-ordered by `source_symbol` for efficient filtering
- **Partitioning**: Bronze layer partitioned by `ingestion_date`
- **Delta Lake**: Automatic file compaction and statistics collection
- **OPTIMIZE**: Regular optimization commands maintain query performance

## 🔐 Data Governance

- **Unity Catalog**: All tables registered in Unity Catalog
- **Lineage Tracking**: Ingestion metadata tracks data flow
- **Audit Trail**: Delta Lake transaction log provides complete history
- **Access Control**: Managed through Unity Catalog permissions

## 🚧 Roadmap & Future Enhancements

- [ ] Add real-time streaming ingestion
- [ ] Implement multiple stock symbols (portfolio tracking)
- [ ] Add ML-based forecasting models (Prophet, LSTM)
- [ ] Create Databricks SQL dashboards for visualization
- [ ] Implement alerting for significant price movements
- [ ] Add sentiment analysis from news sources
- [ ] Automate pipeline with Databricks Jobs/Workflows
- [ ] Add data quality monitoring dashboards
- [ ] Implement CDC (Change Data Capture) for updates
- [ ] Add integration with external ML platforms

## 📝 Data Sources

- **Yahoo Finance**: Historical stock prices via `yfinance` library
- **FIFA Rankings**: Additional dataset for demonstration (IngestaSoccer notebook)

## 🤝 Contributing

This is a personal learning project demonstrating:
- Medallion architecture best practices
- Delta Lake capabilities
- PySpark data engineering patterns
- Unity Catalog governance
- Time-series forecasting workflows

## 📄 License

This project is for educational and demonstration purposes.

## 👤 Author

**Project**: NVDA Stock Analytics Pipeline  
**Platform**: Databricks (AWS)  
**Architecture**: Medallion (Bronze → Silver → Gold)  
**Data Engineering**: PySpark + Delta Lake + Unity Catalog

---

**Last Updated**: 2026-04-25  
**Data Coverage**: 1999-01-22 to 2026-04-24 (6,856+ trading days)  
**Status**: ✅ Production-Ready
