# Logistics AI Intelligence Platform — Technical & Operations Guide

**Current release:** V2.2 with local Ollama hybrid RAG
**Application type:** FastAPI backend, Vue 3 frontend, governed scikit-learn predictive models, local Ollama reasoning and semantic reranking
**Intended use:** Historical logistics analysis and decision support—not live dispatch, routing, pricing, or safety control.

---

## 1. What the platform does

The platform combines four logistics datasets into two deliberately separate capabilities:

1. **Predictive intelligence** uses validated, deployment-gated scikit-learn models for shipment delay risk, delay hours, US shipment cost, and US transit days.
2. **Ask Logistics** uses retrieval-augmented generation (RAG) over the unified historical dataset. It retrieves records, reranks them locally with qwen embeddings, and asks Gemma to answer only from that evidence.

The separation is a core governance rule: **Ollama never replaces, recalculates, or overrules the validated sklearn predictions.**

```text
                                      Browser
                                        │
                          Vue 3 / Vite frontend :5173
                                        │
                                        ▼
                           FastAPI application :8000
                  ┌─────────────────────┴─────────────────────┐
                  │                                           │
                  ▼                                           ▼
      Governed sklearn prediction APIs                  Ask Logistics RAG
      ├─ Tracking delay risk/hours                     ├─ TF-IDF candidate recall
      ├─ US cost/transit prediction                    ├─ qwen3-embedding:4b reranking
      ├─ Carrier recommendation                         └─ gemma4:12b grounded answer
      └─ US data-quality anomalies                                  │
                  │                                                  ▼
                  └──────────────► generated artifacts       Local Ollama :11434
```

## 2. Datasets and source-aware roles

| Source | Project role | Important constraint |
| --- | --- | --- |
| Transportation and Logistics Tracking | Tracking delay classification and delay-hours regression | Only clean, source-specific tracking features are used for these models. |
| Logistics Operations Database | Operational analytics and retrieval context | Its weaker experimental predictive tasks remain research-only unless they pass the gate. |
| Logistics and Supply Chain | Supply-chain conditions and retrieval context | Source fields are retained but not treated as universal targets. |
| US Logistics Performance | Cost/transit prediction, carrier analytics, carrier ranking, anomaly monitor | Historical reliability is observed and smoothed—not a learned exception prediction. |

The canonical cross-source schema is declared in `backend/app/data/schema.py`. Every adapter normalizes its source into that schema for analytics and retrieval, while predictive models train against source-specific frames to prevent feature/target mismatch.

## 3. Repository map

```text
backend/app/
├─ main.py                    FastAPI app, request validation, API routes
├─ core/config.py             paths and environment configuration
├─ data/
│  ├─ schema.py               unified retrieval/analytics column contract
│  ├─ adapters.py             four Kaggle-source normalizers
│  └─ demo.py                 synthetic demo fixture generator
├─ ml/
│  ├─ training.py             TF-IDF corpus index and candidate retrieval
│  ├─ source_training.py      tracking/operations/supply source-aware training
│  └─ us_performance.py       US cost/transit training and quality preparation
└─ services/
   ├─ decision_engine.py      governed inference, ranking, and anomaly rules
   ├─ insights.py             Ask Logistics orchestration and route scoring
   └─ ollama.py               local Ollama HTTP client and grounded prompting

frontend/src/
├─ App.vue                    dashboard orchestration and startup data loading
├─ services/api.ts            common backend HTTP helpers
└─ components/                individual operational panels

scripts/
├─ download_kaggle.py         downloads the four source datasets
├─ train.py                   source-aware retraining entry point
├─ build_ollama_index.py      fast semantic-RAG readiness validation
├─ run_backend.ps1            starts FastAPI with the project virtual environment
└─ run_frontend.ps1           installs/runs the Vue frontend

artifacts/
├─ models/                    generated sklearn model files (not committed)
├─ metrics/                   generated evaluation/governance metrics (not committed)
├─ figures/                   generated EDA and model figures (not committed)
└─ index/                     generated TF-IDF and local-RAG readiness files (not committed)
```

## 4. Prerequisites

- Windows PowerShell
- Python 3.12 or compatible Python available as `python`
- Node.js and npm
- Ollama with these local models installed:
  - `gemma4:12b`
  - `qwen3-embedding:4b`

Do not use a global Anaconda installation to run the project. Saved artifacts require NumPy 2.x and the project’s `.venv` provides a reproducible, non-admin environment.

## 5. Initial setup

Run the following in the source root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

The requirements intentionally pin NumPy to major version 2 because existing serialized artifacts refer to the NumPy 2 module path `numpy._core`.

### Environment configuration

`.env` is local-only and is ignored by Git. These are the active defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_MODE` | `real` | Requires real Kaggle input data for retraining. |
| `RANDOM_STATE` | `42` | Reproducible split/model seed. |
| `CORS_ORIGINS` | `http://localhost:5173` | Browser origin allowed to call FastAPI. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend URL used by the Vue app. |
| `OLLAMA_ENABLED` | `true` | Enables local Ollama requests. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL. |
| `OLLAMA_CHAT_MODEL` | `gemma4:12b` | Grounded explanation model. |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:4b` | Semantic reranking model. |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Per-request Ollama timeout. |
| `OLLAMA_RERANK_CANDIDATES` | `16` | Number of TF-IDF candidates reranked per Ask Logistics request. |

## 6. Run the complete application

Use three PowerShell terminals and leave each running.

### Terminal 1 — Ollama

```powershell
ollama serve
```

If this reports that port `11434` is already in use, Ollama is already running; do not start another instance.

### Terminal 2 — API

```powershell
cd H:\MAETRIX_AI_26\Logistics\Logistics_AI_Intelligence_Platform\Logistics_AI_Intelligence_Platform
.\scripts\run_backend.ps1
```

FastAPI is available at `http://localhost:8000`. Its interactive API documentation is at `http://localhost:8000/docs`.

### Terminal 3 — frontend

```powershell
cd H:\MAETRIX_AI_26\Logistics\Logistics_AI_Intelligence_Platform\Logistics_AI_Intelligence_Platform\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Validate local RAG once after setup

```powershell
cd H:\MAETRIX_AI_26\Logistics\Logistics_AI_Intelligence_Platform\Logistics_AI_Intelligence_Platform
.\.venv\Scripts\python.exe scripts\build_ollama_index.py
Invoke-RestMethod http://localhost:8000/health
```

The health response should show `reachable_with_configured_models: true` and `semantic_index_ready: true`.

## 7. Training and generated artifacts

### Download source data

```powershell
.\.venv\Scripts\python.exe scripts\download_kaggle.py
```

This uses `kagglehub`, downloads the four configured sources, and places each beneath `data/raw/`.

### Retrain from real source data

```powershell
.\.venv\Scripts\python.exe scripts\train.py
```

Training recreates processed frames, TF-IDF assets, model artifacts, metrics, and figures. Review `artifacts/metrics/training_metrics_v2.json` after every retrain. Generated raw data and model artifacts are intentionally excluded from source control.

## 8. API reference

| Endpoint | Purpose | Authority |
| --- | --- | --- |
| `GET /health` | Runtime, data, model, and Ollama readiness | Operational status only |
| `POST /train` | Retrain from real source data | Local training job |
| `GET /analytics/summary` | Unified-dataset aggregate statistics | Historical dataset |
| `POST /ask` | Retrieval-grounded logistics question answering | Retrieved records + local Ollama, with deterministic fallback |
| `POST /tracking/predict` | Delay risk and optional delay-hours estimate | Governed tracking sklearn models |
| `POST /us-performance/predict` | US cost and transit estimates | Governed US sklearn models |
| `GET /carriers/analytics` | Historical carrier statistics | US historical dataset |
| `POST /carriers/recommend` | Rank carriers for an objective | Cost/transit models + historical reliability |
| `GET /us-performance/anomalies` | Data-quality flags | Deterministic rules |
| `POST /routes/recommend` | Legacy supplied-candidate route ranking | Supplied fields only |
| `GET /models/metadata` | Model metrics and deployment recommendations | Training metrics |
| `GET /figures/list` | Available generated figures | Local artifacts |

### Example requests

```powershell
# Ask Logistics
@{ question = 'Compare delay patterns for the most relevant shipments.'; top_k = 8 } |
  ConvertTo-Json |
  Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType 'application/json'

# Tracking prediction
@{
  distance_km = 900; minimum_km_per_day = 350; planned_transit_hours = 48
  booking_to_start_hours = 12; origin_code = 'MIA'; destination_code = 'DTW'
  vehicle_type = '40 FT 3XL Trailer 35MT'; market_regular = 'Regular'
  material_shipped = 'General Freight'
} | ConvertTo-Json | Invoke-RestMethod -Method Post -Uri http://localhost:8000/tracking/predict -ContentType 'application/json'

# Carrier recommendation
@{
  origin_warehouse = 'Warehouse_MIA'; destination = 'Detroit'; shipment_date = '2026-08-22'
  weight_kg = 30; distance_miles = 1200; objective = 'balanced'
} | ConvertTo-Json | Invoke-RestMethod -Method Post -Uri http://localhost:8000/carriers/recommend -ContentType 'application/json'
```

## 9. Model governance and validity boundary

Production classification requires balanced accuracy, macro-F1, and ROC-AUC of at least `0.60` for binary tasks. Production regression requires at least a 5% improvement over the median dummy MAE and `R² >= 0.10`.

The API checks `deployment_recommended` before serving governed predictions. Models that do not pass remain reported in the metrics as research artifacts only. The application makes no claim of live traffic, weather, construction, GPS, guaranteed pricing, ETA certainty, safety dispatch authority, or fraud detection.

Carrier reliability is an observed historical exception rate with empirical-Bayes-style smoothing for sparse carrier/lane observations. It is not a classifier prediction.

## 10. Local Ollama RAG design

`POST /ask` follows this sequence:

1. `backend/app/ml/training.py` uses the persisted TF-IDF word/bigram index to retrieve the strongest lexical candidates.
2. `backend/app/services/insights.py` serializes only that small candidate set and obtains qwen embeddings for the question plus candidate records.
3. Cosine similarity reranks those candidates semantically.
4. `backend/app/services/ollama.py` sends Gemma the question and selected evidence with a temperature of `0` and strict instructions not to invent facts or contradict sklearn predictions.
5. The API returns the answer **and** evidence rows. If Ollama is unavailable, a deterministic dataset-grounded summary is returned instead.

This is a hybrid RAG implementation. It intentionally does not generate a full 126k-record, 2,560-dimension embedding matrix because that would be slow to build and consume substantial storage and memory.

## 11. Code walkthrough: important blocks and line anchors

The following map documents the important executable blocks. Line numbers refer to the current checked-in layout; when code changes, use the named function/class as the stable anchor.

### Application, validation, and configuration

| File and anchor | What the block does |
| --- | --- |
| `backend/app/core/config.py:9-13` | Loads `.env` without overriding environment variables already set by the operating system. |
| `backend/app/core/config.py:16-40` | Defines all project/data/artifact paths and application/Ollama configuration in the immutable `Settings` object. |
| `backend/app/core/config.py:44-51` | Creates required artifact and data directories on startup. |
| `backend/app/main.py:18-31` | Constructs FastAPI and permits the configured Vue origin through CORS. |
| `backend/app/main.py:34-80` | Pydantic request models; these validate Ask, tracking, US prediction, and carrier-ranking inputs before any service code runs. |
| `backend/app/main.py:84-102` | `/health`; reports data/metrics state plus configured-model reachability and semantic-RAG readiness. |
| `backend/app/main.py:105-188` | Route handlers. Each delegates to a narrow service and converts operational exceptions to `503` or training errors to `400`. |

### Data contract and ingestion

| File and anchor | What the block does |
| --- | --- |
| `backend/app/data/schema.py:3-22` | `UNIFIED_COLUMNS`, the canonical retrieval/analytics schema shared by all source adapters. |
| `backend/app/data/schema.py:24-38` | Explicit numeric/categorical groups used for safe normalization and processing. |
| `backend/app/data/adapters.py:14-53` | Shared helpers normalize column names, pick alternate source fields, parse booleans/coordinates, and create record IDs. |
| `backend/app/data/adapters.py:53-67` | `_finalize` fills missing canonical columns, coerces types, adds source/record identifiers, and returns the controlled schema. |
| `backend/app/data/adapters.py:70-117` | `DynamicSupplyChainAdapter`; maps source-specific supply-chain fields and retains unstructured source text for retrieval. |
| `backend/app/data/adapters.py:119-178` | `TransportationTrackingAdapter`; derives clean delayed/on-time labels, delay hours, planned transit, and lane IDs without silently resolving contradictory labels. |
| `backend/app/data/adapters.py:180-260` | `LogisticsOperationsAdapter`; joins relational files, aggregates fuel/maintenance, and derives service deviation/detention fields. |
| `backend/app/data/adapters.py:262-311` | `USLogisticsPerformanceAdapter`; maps cost/transit/carrier fields and marks date-quality anomalies. |
| `backend/app/data/adapters.py:312-349` | `discover_and_load`; finds expected source files and loads each safely into the unified corpus. |

### Retrieval and source-aware model training

| File and anchor | What the block does |
| --- | --- |
| `backend/app/ml/training.py:13-37` | Builds text documents from unified columns, including source-specific `free_text` where present. |
| `backend/app/ml/training.py:39-56` | Fits and saves TF-IDF word/bigram vectorizer and sparse matrix. |
| `backend/app/ml/training.py:57-74` | Writes unified data and retrieval metadata for the simple retrieval-only training path. |
| `backend/app/ml/training.py:76-99` | `retrieve`; loads TF-IDF artifacts, performs cosine similarity, and returns only approved evidence fields plus scores. |
| `backend/app/ml/source_training.py:112-137` | Builds preprocessing pipelines: median imputation/scaling for numeric inputs and imputation/one-hot encoding for categoricals. |
| `backend/app/ml/source_training.py:139-166` | Produces classification and regression governance metrics. |
| `backend/app/ml/source_training.py:167-273` | Benchmarks dummy, logistic, random-forest, and extra-trees classifiers; persists the selected research artifact and computes classification deployment approval. |
| `backend/app/ml/source_training.py:274-351` | Benchmarks dummy, ridge, random-forest, and extra-trees regressors; requires MAE improvement and R² before deployment approval. |
| `backend/app/ml/source_training.py:352-480` | Prepares tracking, operations, and supply frames using source-specific features/targets. |
| `backend/app/ml/source_training.py:535-664` | Orchestrates source models, US models, unified retrieval data, figures, and `training_metrics_v2.json`. |
| `backend/app/ml/us_performance.py:64-150` | Prepares the US source and builds safe inference features from request payloads. |
| `backend/app/ml/us_performance.py:199-279` | Cost/transit model benchmark and deployment gate; only deployable artifacts are written. |
| `backend/app/ml/us_performance.py:280-359` | Research-only status classifier evaluation; these artifacts are deliberately not exposed for production prediction. |
| `backend/app/ml/us_performance.py:403-427` | Trains US tasks, writes diagnostics/metrics, and produces US figures. |

### Predictive decision services

| File and anchor | What the block does |
| --- | --- |
| `backend/app/services/decision_engine.py:36-64` | Loads metrics/model metadata, verifies an artifact exists, and returns the selected-model quality record. |
| `backend/app/services/decision_engine.py:66-105` | Derives a valid one-row tracking feature frame, including date-derived fields and safe categorical null handling. |
| `backend/app/services/decision_engine.py:107-140` | Runs tracking inference only if the classifier has passed its deployment gate; optionally includes gated delay-hours output. |
| `backend/app/services/decision_engine.py:141-169` | Runs cost/transit inference only for individually approved US models. |
| `backend/app/services/decision_engine.py:177-212` | Produces carrier history, cost, transit, and observed reliability analytics. |
| `backend/app/services/decision_engine.py:224-313` | Carrier recommendation engine: exact-lane fallback, historical exception smoothing, objective-specific normalization, and recommendation ranking. |
| `backend/app/services/decision_engine.py:314-364` | Anomaly rules for impossible transit dates, date discrepancies, extreme cost, and missing values. |

### Ask Logistics and local Ollama integration

| File and anchor | What the block does |
| --- | --- |
| `backend/app/services/ollama.py:18-35` | Defines a clear unavailable-model error and centralizes timeout-protected local HTTP requests. |
| `backend/app/services/ollama.py:38-47` | Calls `/api/embed` and validates that Ollama returned one vector per input record. |
| `backend/app/services/ollama.py:49-61` | Fast health check against `/api/tags`, verifying both configured model names. |
| `backend/app/services/ollama.py:63-74` | Builds the constrained Gemma prompt and calls `/api/generate` with `temperature: 0`. |
| `backend/app/services/insights.py:12-35` | Loads unified data and creates aggregate analytics statistics. |
| `backend/app/services/insights.py:38-51` | Converts selected rows into clean API evidence objects. |
| `backend/app/services/insights.py:53-76` | `_semantic_rerank`; qwen embeds the question/candidates, normalizes vectors, and ranks by cosine similarity. |
| `backend/app/services/insights.py:78-145` | `answer`; combines TF-IDF recall, qwen reranking, deterministic fallback, Gemma grounding, and transparent response metadata. |
| `backend/app/services/insights.py:146-189` | Legacy generic route ranking; scores only supplied fields and never claims live-condition data. |
| `scripts/build_ollama_index.py:21-34` | Performs one qwen readiness embedding and writes a small manifest—not a corpus-wide vector index. |

### Frontend

| File and anchor | What the block does |
| --- | --- |
| `frontend/src/services/api.ts:1-4` | Selects the configured backend URL and exposes common JSON GET/POST functions that raise backend errors. |
| `frontend/src/App.vue:10-18` | Dashboard reactive state and computed KPI counts. |
| `frontend/src/App.vue:20-44` | Loads health/analytics/metadata/carrier state on mount; the training action invokes `/train` then refreshes. |
| `frontend/src/App.vue:47-101` | Dashboard composition, readiness status, KPI strip, governed-model view, and panels. |
| `frontend/src/components/ChatPanel.vue:5-16` | Ask Logistics question state and `POST /ask` action. |
| `frontend/src/components/ChatPanel.vue:19-46` | Renders answer, grounding/retrieval metadata, and auditable evidence rows. |
| `frontend/src/components/TrackingPanel.vue:5-31` | Tracking input defaults and prediction request lifecycle. |
| `frontend/src/components/TrackingPanel.vue:33-72` | Delay-risk and delay-hours result display. |
| `frontend/src/components/CarrierDecisionPanel.vue:5-41` | Loads available lanes/carriers, maintains form state, and posts ranking requests. |
| `frontend/src/components/CarrierDecisionPanel.vue:43-107` | Renders ranked recommendations and their historical-reliability context. |
| `frontend/src/components/AnomalyPanel.vue:5-15` | Fetches/reloads deterministic anomaly flags. |
| `frontend/src/components/AnomalyPanel.vue:17-43` | Presents severity, reasons, and affected-shipment context. |

### Scripts and tests

| File and anchor | What the block does |
| --- | --- |
| `scripts/download_kaggle.py:10-28` | Dataset slug registry and reproducible download destination layout. |
| `scripts/train.py:1-11` | Blocks demo mode for the real V2.2 training workflow and calls `train_v2`. |
| `scripts/run_backend.ps1:1-9` | Uses `.venv` explicitly, sets `PYTHONPATH`, and starts Uvicorn reload mode. |
| `scripts/run_frontend.ps1:1-3` | Switches to the frontend, installs JavaScript packages, and starts Vite. |
| `tests/test_pipeline.py` | Tests demo data ingestion, TF-IDF retrieval, and generic route ranking. |
| `tests/test_v22_us_performance.py` | Tests US source normalization, quality flags, and inference-feature construction. |
| `tests/test_ollama_service.py` | Tests embedding payload construction, constrained prompt options, and qwen reranking without needing a live model. |

## 12. Troubleshooting

| Symptom | Cause and action |
| --- | --- |
| `WinError 10061` from `build_ollama_index.py` | Ollama is not listening. Start `ollama serve`, then retry. |
| Ollama port already in use | The server is already running. Do not start a second instance. |
| `numpy._core.numeric` missing | A global/old NumPy is being used. Run through `.venv\\Scripts\\python.exe` and reinstall requirements. |
| `ModuleNotFoundError: fastapi` | Dependencies were installed into the wrong interpreter or installation failed. Run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. |
| `semantic_index_ready` is false | Run `.\.venv\Scripts\python.exe scripts\build_ollama_index.py`; it performs a fast model validation. |
| First Ask Logistics call is slow | Local qwen and Gemma must load into memory. Subsequent requests are normally faster while models remain loaded. |
| `/favicon.ico` returns 404 | Harmless browser icon request; it does not affect API functionality. |

## 13. GitHub handoff checklist

Before committing, verify the application and keep generated/private files out of the repository:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
Invoke-RestMethod http://localhost:8000/health
git status
```

Commit source code, documentation, `.env.example`, and tests. Do not commit `.env`, `.venv`, `node_modules`, raw datasets, generated model artifacts, metrics, figures, TF-IDF assets, or the local semantic-RAG readiness manifest.
