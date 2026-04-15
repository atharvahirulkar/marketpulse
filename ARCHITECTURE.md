# SignalStack - System Architecture Diagram

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNALSTACK ARCHITECTURE                            │
│                    Real-Time ML Systems for Equity Markets                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Live Data Ingestion Path

```
                          LIVE DATA INGESTION
                          ═══════════════════

   Polygon.io WebSocket
   (wss://socket.polygon.io/stocks)
           │
           │  async, exponential backoff reconnection
           │  auth: POLYGON_API_KEY
           ▼
   ingestion/polygon_ws.py
   ┌─────────────────────────────┐
   │ • Parse trade events        │
   │ • Normalize schema          │
   │ • ~1000 ticks/sec target    │
   │ • Metrics collection        │
   └─────────────────────────────┘
           │
           │  canonical trade schema (JSON)
           │  {symbol, price, size, timestamp_ms, ...}
           ▼
   Apache Kafka (6 partitions)
   ┌─────────────────────────────┐
   │ Topic: market.trades        │
   │ • Compression: lz4          │
   │ • Retention: 7 days         │
   │ • Acks: all                 │
   │ • Idempotent producer       │
   └─────────────────────────────┘
```

---

## 2. Stream Processing Path

```
                      STREAM PROCESSING
                      ═════════════════

   Kafka: market.trades
           │
           ▼
   PySpark Structured Streaming
   ┌─────────────────────────────────────────────────────┐
   │ streaming/spark_consumer.py                         │
   │                                                     │
   │  • Parse JSON, validate schema                      │
   │  • Watermark on event_time (10s)                    │
   │  • Deduplication: (symbol, timestamp_ms)            │
   │                                                     │
   │          ┌──────────────────┐                       │
   │          │                  │                       │
   │    1-min OHLCV          5-min Features              │
   │  (tumbling window)    (pandas UDF)                  │
   │          │                  │                       │
   │  • open, high         • vwap_5m/15m                 │
   │  • low, close         • price_momentum              │
   │  • volume             • realized_vol                │
   │  • vwap               • volume_ratio                │
   │          │                  │                       │
   │          └──────────────────┘                       │
   └─────────────────────────────────────────────────────┘
           │                      │
           │                      │
    ┌──────▼──────┐        ┌──────▼──────┐
    │ ohlcv_1m    │        │  features   │
    │ (TimescaleDB)        │ (TimescaleDB)
    └─────────────┘        └──────┬──────┘
                                  │
                           market.features
                           (Kafka topic)
```

---

## 3. Storage Layer (TimescaleDB + PostgreSQL 16)

```
                      TIMESCALEDB HYPERTABLES
                      ═══════════════════════

   ┌──────────────────────────────────────────────────┐
   │ timescaledb:5432 (port 5433 from host)           │
   │                                                  │
   │  ┌─ trades ──────────────────────────────┐       │
   │  │ Raw tick data                         │       │
   │  │ • Chunk: 1h    • Compress: 2h         │       │
   │  │ • Retention: 7d                       │       │
   │  └───────────────────────────────────────┘       │
   │                                                  │
   │  ┌─ ohlcv_1m ────────────────────────────┐       │
   │  │ 1-minute bars                         │       │
   │  │ • Chunk: 1d    • Compress: 1d         │       │
   │  │ • Retention: 90d                      │       │
   │  └───────────────────────────────────────┘       │
   │                                                  │
   │  ┌─ features ───────────────────────────────┐    │
   │  │ ML feature store (10 engineered features)│    │
   │  │ • Chunk: 1d    • Compress: 3d            │    │
   │  │ • Retention: 365d                        │    │
   │  └──────────────────────────────────────────┘    │
   │                                                  │
   │  ┌─ anomalies ────────────────────────────┐      │
   │  │ Model predictions + anomaly flags      │      │
   │  │ • Chunk: 1d    • Retention: 365d       │      │
   │  └────────────────────────────────────────┘      │
   │                                                  │
   │  ┌─ model_registry ───────────────────────┐      │
   │  │ Trained model metadata                 │      │
   │  │ • name, version, artifact_path         │      │
   │  │ • train_dates, val_metric, params      │      │
   │  │ • status: staging→production→archived  │      │
   │  └────────────────────────────────────────┘      │
   │                                                  │
   │  ┌─ Continuous Aggregates ───────────────────┐   │
   │  │ • ohlcv_5m   (5-min buckets, auto-refresh)│   │
   │  │ • ohlcv_1h   (1-hour buckets)             │   │
   │  └───────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────┘
            ▲                           ▲
            │                           │
       Spark writes              API + Grafana reads
```

---

## 4. ML Training Path

```
                        ML TRAINING PIPELINE
                        ════════════════════

   Data Sources:
   ┌─────────────────┐         ┌──────────────────────┐
   │  yfinance       │         │  TimescaleDB         │
   │  (free, dev)    │         │  (production, live)  │
   └────────┬────────┘         └──────────┬───────────┘
            │                             │
            └─────────────┬───────────────┘
                          ▼
            training/data_loader.py
            ┌──────────────────────────┐
            │ • Fetch OHLCV bars       │
            │ • Compute 10 features    │
            │ • Point-in-time correct  │
            │   (no lookahead bias)    │
            └──────────┬───────────────┘
                       ▼
            training/dataset.py
            ┌──────────────────────────┐
            │ • 70/15/15 temporal split│
            │ • Label: direction (5m)  │
            │ • Feature vectors        │
            └──────────┬───────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    LSTMModel    XGBoostRegime  AnomalyDetector
    (PyTorch)    (LightGBM)     (scikit-learn)
    • 2-layer    • 500 trees    • Isolation Forest
    • n=128      • depth=6      • contamination=0.01
    • seq_len=24 • n_jobs=1     
         │             │             │
         └─────────────┼─────────────┘
                       ▼
            MLflow Experiment Tracking
            ┌──────────────────────────┐
            │ • Run metadata           │
            │ • Metrics (AUC, F1, etc) │
            │ • Feature importances    │
            │ • Model artifacts        │
            │ • PSI drift baseline     │
            └──────────┬───────────────┘
                       ▼
            artifacts/ directory
            ┌──────────────────────────┐
            │ • lstm_model.pt          │
            │ • xgboost_model.pkl      │
            │ • anomaly_model.pkl      │
            │ • scaler.pkl             │
            └──────────────────────────┘
```

---

## 5. Inference & Serving Path

```
                     INFERENCE & SERVING
                     ═══════════════════

   artifacts/ (trained models)
           │
           ▼
   serving/inference_api.py
   ┌─────────────────────────────────────────────┐
   │         FastAPI Inference Server            │
   │  (uvicorn, 0.0.0.0:8000, 2 workers)         │
   │                                             │
   │  ┌─ ModelRegistry (in-memory) ──────────┐   │
   │  │ • Load 3 models at startup           │   │
   │  │ • Per-symbol LSTM sequence buffer    │   │
   │  │ • DriftMonitor (PSI tracking)        │   │
   │  └──────────────────────────────────────┘   │
   │                                             │
   │  ┌─ Endpoints ────────────────────────────┐ │
   │  │ POST /predict/direction  ← LSTM        │ │
   │  │ POST /predict/regime     ← LightGBM    │ │
   │  │ POST /predict/anomaly    ← IsoForest   │ │
   │  │ WS   /ws/predictions     ← streaming   │ │
   │  │ GET  /health             ← status      │ │
   │  │ GET  /metrics            ← Prometheus  │ │
   │  │ GET  /drift              ← PSI scores  │ │
   │  └────────────────────────────────────────┘ │
   │                                             │
   │  ┌─ Prometheus Metrics ───────────────────┐ │
   │  │ • inference_requests_total             │ │
   │  │ • inference_latency_seconds (histogram)│ │
   │  │ • anomalies_detected_total             │ │
   │  │ • feature_psi (drift histograms)       │ │
   │  └────────────────────────────────────────┘ │
   └─────────────────────────────────────────────┘
           │                 │
           │                 └─ Prometheus scrape
           │                    (every 15s)
           ▼
   Client Applications
   ┌────────────────────────┐
   │ • Trading algos        │
   │ • Risk dashboards      │
   │ • REST clients         │
   │ • WebSocket subscribers│
   └────────────────────────┘
```

---

## 6. Monitoring & Dashboards

```
                    MONITORING & OBSERVABILITY
                    ═════════════════════════

   TimescaleDB                    Prometheus
   (metrics queries)              (metrics collection)
           │                           ▲
           │                           │
           ▼                           │
   serving/metrics_server.py           │
   ┌────────────────────────────────┐  │
   │ Background metrics collector   │  │
   │ • ticks_per_sec                │  │
   │ • kafka_lag                    │  │
   │ • feature_psi (drift detection)│  │
   │ • anomaly_rate                 │  │
   │ • db_write_latency             │  │
   │ • prediction_distribution      │  │
   └────────────────────────────────┘  │
                                       │
                                       ▼
                                   Grafana (3000)
                                   ┌──────────────────────┐
                                   │ Dashboards           │
                                   │ • Live tick feed     │
                                   │ • OHLCV 1m close     │
                                   │ • Feature heatmap    │
                                   │ • Anomaly overlay    │
                                   │ • Model drift monitor│
                                   │ • Inference latency  │
                                   │ (refresh: 5s)        │
                                   └──────────────────────┘
```

---

## 7. Infrastructure & Containerization

```
                      DOCKER COMPOSE STACK
                      ════════════════════

   ┌───────────────────────────────────────────────────┐
   │                  docker-compose.yml               │
   │                                                   │
   │  ┌─ zookeeper ───────────────────────────────┐    │
   │  │ confluentinc/cp-zookeeper:7.6.1           │    │
   │  │ (Kafka coordination)                      │    │
   │  └───────────────────────────────────────────┘    │
   │                                                   │
   │  ┌─ kafka ───────────────────────────────────┐    │
   │  │ confluentinc/cp-kafka:7.6.1               │    │
   │  │ • Port: 9092 (internal), 29092 (docker)   │    │
   │  │ • 6 partitions, lz4, 7d retention         │    │
   │  └───────────────────────────────────────────┘    │
   │                                                   │
   │  ┌─ kafka-ui ────────────────────────────────┐    │
   │  │ provectuslabs/kafka-ui:latest             │    │
   │  │ • Port: 8080 (Kafka browser)              │    │
   │  └───────────────────────────────────────────┘    │
   │                                                   │
   │  ┌─ timescaledb ──────────────────────────────┐   │
   │  │ timescale/timescaledb:latest-pg16          │   │
   │  │ • Port: 5433 (host), 5432 (container)      │   │
   │  │ • Auto-init: schema.sql                    │   │
   │  │ • User/pass from .env                      │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  ┌─ grafana ──────────────────────────────────┐   │
   │  │ grafana/grafana:11.0.0                     │   │
   │  │ • Port: 3000                               │   │
   │  │ • Auto-provisioned datasource + dashboard  │   │
   │  │ • Admin / admin (default credentials)      │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  ┌─ mlflow ───────────────────────────────────┐   │
   │  │ mlflow:v2.13.0 + psycopg2                  │   │
   │  │ • Port: 5001                               │   │
   │  │ • Tracking URI: postgresql://...           │   │
   │  │ • TimescaleDB as backend store             │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  ┌─ ingestion (--profile app) ────────────────┐   │
   │  │ Dockerfile → ingestion/polygon_ws.py       │   │
   │  │ • WebSocket consumer                       │   │
   │  │ • Kafka producer                           │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  ┌─ streaming (--profile app) ────────────────┐   │
   │  │ Dockerfile → streaming/spark_consumer.py   │   │
   │  │ • PySpark 3.5 + Kafka + TimescaleDB        │   │
   │  │ • Java/JRE included in stage               │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   │  ┌─ inference (--profile app) ────────────────┐   │
   │  │ Dockerfile → serving/inference_api.py      │   │
   │  │ • Port: 8000                               │   │
   │  │ • FastAPI + Uvicorn                        │   │
   │  │ • Models loaded from artifacts/            │   │
   │  └────────────────────────────────────────────┘   │
   │                                                   │
   └───────────────────────────────────────────────────┘

   Dockerfile Stages:
   ┌─ base ─────────────────────────────────┐
   │ python:3.11-slim                       │
   │ + gcc, libpq, curl, netcat             │
   │ + liblz4-dev (for Kafka compression)   │
   │ + PyTorch CPU, requirements.txt        │
   │                                        │
   ├─ ingestion (FROM base)                 │
   ├─ streaming (FROM base + openjdk:21)    │
   ├─ backfill (FROM base)                  │
   └─ inference (FROM base)                 │
```

---

## 8. Data Flow Summary (End-to-End)

```
Step 1: INGESTION
   Polygon.io WebSocket  ──[JSON ticks]──>  polygon_ws.py

Step 2: MESSAGE BUS
   polygon_ws.py  ──[Kafka produce]──>  market.trades (6 partitions)

Step 3: STREAM PROCESSING
   market.trades  ──[Spark reads]──>  spark_consumer.py
                                      ├──[1m OHLCV]──>  ohlcv_1m table
                                      └──[features]──>  features table
                                                    +
                                                    └──[Kafka produce]──>  market.features

Step 4: STORAGE
   ohlcv_1m, features tables  ──[continuous aggregates]──>  ohlcv_5m, ohlcv_1h

Step 5: TRAINING (offline)
   features table  ──[point-in-time load]──>  training/dataset.py
                                             ├──[train LSTM]──>  artifacts/lstm_model.pt
                                             ├──[train LightGBM]──>  artifacts/xgboost_model.pkl
                                             └──[train IsoForest]──>  artifacts/anomaly_model.pkl
                                                    +
                                                    └──[MLflow track]──>  mlflow (5001)

Step 6: SERVING (inference)
   artifacts/  ──[load at startup]──>  inference_api.py
                                       ├──[POST /predict/direction]
                                       ├──[POST /predict/regime]
                                       ├──[POST /predict/anomaly]
                                       └──[WS /ws/predictions]
                                              +
                                              └──[Prometheus metrics]──>  Grafana (3000)

Step 7: MONITORING
   TimescaleDB  ──[metrics_server.py queries]──>  Prometheus
                                                 └──[scrape]──>  Grafana dashboards
```

---

## 9. Performance Targets

| Component | Target | Status |
|-----------|--------|--------|
| **Ingestion** | >1,000 ticks/sec | ✓ Designed for throughput |
| **End-to-End Latency** | <100ms (WebSocket → DB) | 📊 TBD (pipeline running) |
| **Kafka Consumer Lag** | <500ms | 📊 TBD (pipeline running) |
| **Inference API p99** | <10ms | 📊 TBD (benchmarks needed) |
| **Grafana Refresh** | 5s | ✓ Configured |

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **TimescaleDB** | PostgreSQL + time-series → continuous aggregates auto-materialize views; compression 10–20×; point-in-time queries prevent lookahead bias |
| **PySpark Streaming** | Distributed, handles 1000+ ticks/sec; pandas UDF for feature compute; watermarking for late data |
| **Dual Data Source** | yfinance (free, dev) + Polygon.io (production); feature parity via window rescaling |
| **LSTM Sequence Buffer** | Per-symbol, in-memory; stateful but fast; lossy on restart (consider Redis in prod) |
| **Drift Monitoring (PSI)** | Population Stability Index detects feature distribution shift; thresholds: <0.10 (stable), 0.10–0.20 (warning), >0.20 (alert) |
| **LightGBM not XGBoost** | XGBoost 3.2.0 segfaults on Python 3.14 + asyncio; LightGBM same API, no threading issues |
| **Prometheus + Grafana** | Standard observability; low-overhead metrics; live dashboards for model health |

---

## 11. Project Structure Reference

```
signalstack/
├── .github/
│   └── workflows/ci.yml           # GitHub Actions: lint → test → Docker build
├── ingestion/
│   ├── polygon_ws.py             # WebSocket → Kafka producer
│   └── producer.py               # Kafka utilities
├── streaming/
│   ├── spark_consumer.py          # PySpark → OHLCV + features
│   ├── metrics.py                 # Feature computation
│   └── watermark.py               # Windowing config
├── storage/
│   ├── schema.sql                 # TimescaleDB init
│   └── writer.py                  # Async batch writer
├── training/
│   ├── data_loader.py             # yfinance + TimescaleDB loader
│   ├── dataset.py                 # Feature loading (point-in-time correct)
│   ├── models.py                  # LSTM, LightGBM, IsoForest
│   ├── train.py                   # Training pipeline + MLflow
│   └── drift.py                   # PSI drift detection
├── serving/
│   ├── inference_api.py           # FastAPI server
│   └── metrics_server.py          # Prometheus collector
├── backfill/
│   ├── historical.py              # Polygon REST paginator
│   └── scheduler.py               # Gap detection daemon
├── dashboard/
│   └── grafana/
│       ├── datasource.yaml        # Auto-provisioned datasource
│       ├── dashboard.yaml         # Provider config
│       └── dashboards/
│           └── signalstack.json   # Main dashboard
├── docker-compose.yml             # Full stack orchestration
├── Dockerfile                     # Multi-stage build
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Configuration template
└── README.md
```

---

*Last updated: 2026-04-15*
*Architecture reflects production-grade design: throughput >1000/sec, latency <100ms, containerized, monitored.*
