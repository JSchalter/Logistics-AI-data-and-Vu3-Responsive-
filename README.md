# Logistics AI Intelligence Platform V2.2

A CodeSearchNet-inspired, source-aware logistics ML/RAG application that integrates four Kaggle datasets into governed predictive intelligence, a unified retrieval corpus, carrier decisioning, anomaly monitoring, figures, FastAPI services, and a Vue 3 operational frontend.

For the current end-to-end setup, architecture, operations guide, API reference, local Ollama RAG design, and line-referenced code walkthrough, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Source datasets

1. Nicole Machado — Transportation and Logistics Tracking Dataset
   https://www.kaggle.com/datasets/nicolemachado/transportation-and-logistics-tracking-dataset
2. Yogape Rodriguez — Logistics Operations Database
   https://www.kaggle.com/datasets/yogape/logistics-operations-database
3. DatasetEngineer — Logistics and Supply Chain Dataset
   https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset
4. Shahriar Kabir — US Logistics Performance Dataset
   https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset

## V2.2 architecture

```text
                LOGISTICS AI INTELLIGENCE PLATFORM
                              │
         ┌────────────────────┴─────────────────────┐
         │                                          │
   Predictive Intelligence                    Knowledge / RAG
         │                                          │
┌────────┴──────────┐                       ~126,355 records*
│                   │                              │
Tracking           US Performance                Ask Logistics
│                   │                              │
Delay Risk         Cost Prediction                Search records
Delay Hours        Transit Prediction             Explain trends
                   Carrier Ranking                Compare carriers
                   Anomaly Detection              Route intelligence
│                   │
└──────────┬────────┘
           │
     Decision Engine
           │
┌──────────┼──────────────┐
│          │              │
Cheapest  Fastest   Most Reliable
           │
        Balanced
```

`*` Expected after indexing the observed 124,355 three-source corpus plus 2,000 US shipment records. The runtime metrics JSON is authoritative.

## What is production-exposed

V2.2 uses explicit deployment gates instead of exposing every experiment:

- **Tracking delay classification** — exposed only when held-out balanced accuracy, macro-F1, and ROC-AUC meet the deployment thresholds.
- **Tracking delay-hours regression** — exposed when it beats the median dummy baseline by at least 5% MAE and reaches R² ≥ 0.10.
- **US shipment cost regression** — same regression gate; intended for shipment-cost estimation.
- **US transit-days regression** — same regression gate; intended for transit-time estimation.
- **Operations/supply-chain/US delay-exception experiments** remain in metrics for research and governance when they fail the gate, but prediction endpoints do not use them.

## Carrier decision engine

For an origin, destination, shipment date, weight, and distance, the engine evaluates historically available carriers using:

- trained predicted shipment cost
- trained predicted transit days
- smoothed historical carrier/lane exception rate
- exact-lane carrier coverage when available

Objectives:

- `cheapest`
- `fastest`
- `most_reliable`
- `balanced` = 35% cost + 30% transit + 35% historical exception risk

Reliability is explicitly **historical**, not a failed exception-classifier prediction.

## Anomaly monitor

The US dataset quality layer preserves and flags source anomalies rather than silently rewriting them:

- delivery dates earlier than shipment dates
- large recorded-vs-calculated transit discrepancies
- extreme shipment-cost outliers using a robust 3-IQR rule
- missing cost
- missing delivery date

These are data-quality flags, not fraud determinations.

## Unified RAG / Ask Logistics

All four adapters feed the canonical retrieval corpus. The default local-AI path uses a two-stage local RAG flow: TF-IDF retrieves a small candidate set, `qwen3-embedding:4b` semantically reranks those records, and `gemma4:12b` writes a concise answer constrained to the selected evidence. This avoids a slow, multi-gigabyte corpus-wide embedding index. When Ollama is unavailable, the existing TF-IDF retrieval and deterministic dataset summary are used instead.

This layer does not make, override, or recalculate predictions. The validated sklearn models remain authoritative for delay, cost, transit, and carrier-decision prediction endpoints.

## Frequently asked questions

### What is this project?

It is a full-stack logistics AI application. A Vue 3 dashboard calls a FastAPI backend that provides governed historical-model predictions, carrier decision support, data-quality monitoring, and retrieval-grounded logistics answers.

### What questions can Ask Logistics answer?

It can answer questions grounded in the retrieved historical records, such as:

- What delay patterns appear for comparable shipments?
- Which carriers are most represented in the relevant evidence?
- What are the observed cost, transit, risk, or detention patterns?
- What lanes or shipment statuses are most common in the retrieved data?

The response includes evidence rows so the answer can be reviewed. It will state when the available historical evidence is insufficient.

For a copy-ready prompt library, see [ASK_LOGISTICS_EXAMPLE_QUESTIONS.txt](ASK_LOGISTICS_EXAMPLE_QUESTIONS.txt).

### Does Ask Logistics use live traffic, GPS, weather, prices, or carrier feeds?

No. It is a historical, dataset-grounded capability. It does not claim live operational conditions or real-time routing information.

### Which models make predictions?

Validated scikit-learn models are the authoritative predictors:

- tracking delay classification
- tracking delay-hours regression, when deployment-approved
- US shipment-cost regression
- US transit-days regression

Only models that pass the project’s held-out performance gates are exposed by prediction endpoints.

### What do the local Ollama models do?

`qwen3-embedding:4b` semantically reranks a small set of retrieved logistics records. `gemma4:12b` converts the selected evidence into a concise answer. Neither model generates prediction values or overrides the sklearn models.

### Does this require a massive vector database or cloud API?

No. The RAG workflow is local and hybrid: TF-IDF first finds candidate records, then qwen embeddings rerank those candidates. This avoids creating and storing a full-corpus embedding matrix. No paid cloud LLM API is required for the local configuration.

### What hardware is required?

The backend and frontend run on ordinary development hardware, but the local Ollama models are resource-intensive—especially Gemma 12B. Use a machine with sufficient memory and preferably adequate GPU VRAM. Do not expect the full local LLM workflow to run well on an underpowered system.

### Which datasets are used?

The project is designed around four Kaggle datasets: transportation tracking, logistics operations, supply-chain logistics, and US logistics performance. Dataset files are downloaded locally and are intentionally excluded from Git.

### Is my local data or trained model output uploaded to GitHub?

No. `.gitignore` excludes raw/processed data, model artifacts, figures, metrics, local RAG files, `.env`, virtual environments, and frontend dependencies. The repository contains source code, tests, documentation, configuration templates, and the notebook.

### How do I start the complete project?

Keep three terminals open:

```powershell
# Terminal 1: Ollama
ollama serve

# Terminal 2: API (from the project root)
.\scripts\run_backend.ps1

# Terminal 3: frontend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

### What if Ollama says port 11434 is already in use?

Ollama is already running. Leave that existing service running; do not start a second instance.

### How do I verify that the platform is ready?

Open `http://localhost:8000/health`. A ready local-AI setup reports `data_ready`, `reachable_with_configured_models`, and `semantic_index_ready` as `true`.

### How do I enable or validate the local RAG workflow?

After Ollama is running, execute this once from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\build_ollama_index.py
```

Despite its compatibility-focused name, this command performs a fast readiness check; it does not create a full-corpus vector index.

### How do I retrain the models?

Download the four source datasets, then run:

```powershell
.\.venv\Scripts\python.exe scripts\download_kaggle.py
.\.venv\Scripts\python.exe scripts\train.py
```

Review `artifacts/metrics/training_metrics_v2.json` after training. Generated artifacts stay local and are not committed.

### Why are some models shown as research-only?

The application retains evaluated models and their metrics for transparency, but it does not expose a model for production prediction unless it meets the defined accuracy and baseline-improvement thresholds. Research-only does not mean unusable data—it means the model did not meet the project’s deployment standard.

### How are carrier recommendations calculated?

The engine compares deployable predicted cost, deployable predicted transit time, and smoothed historical carrier/lane exception rates. The balanced objective weights cost at 35%, transit at 30%, and historical exception risk at 35%.

### Are anomaly flags fraud determinations?

No. They are deterministic data-quality flags for issues such as impossible date order, large transit-date discrepancy, extreme cost, missing cost, and missing delivery date.

### What happens if Ollama is unavailable?

The backend remains available. Ask Logistics falls back to TF-IDF retrieval and a deterministic evidence-based summary, and reports that fallback in its response metadata.

### What does a 503 response from `/ask` mean?

It usually indicates that the backend could not access the expected retrieval data or local model service. Check `/health`, confirm Ollama is running, and restart the backend after configuration changes.

### Why might the first Ask Logistics response take longer?

The first request may need to load qwen and Gemma into memory. Subsequent requests are normally faster while Ollama keeps the models loaded.

### Where is the complete technical documentation?

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for architecture, environment variables, API examples, model governance, source-code walkthrough, training operations, and the GitHub handoff checklist.

## Quick start

```powershell
cd Logistics_AI_Intelligence_Platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_kaggle.py
$env:PYTHONPATH=(Get-Location).Path
python scripts\train.py
python scripts\build_ollama_index.py
python -m uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The backend launch script intentionally uses `.venv\\Scripts\\python.exe`, not a global Anaconda installation. The saved data artifacts require NumPy 2.x; use the fresh project environment above rather than trying to repair a system-wide Python installation.

## API

- `GET /health`
- `POST /train`
- `GET /analytics/summary`
- `POST /ask`
- `POST /tracking/predict`
- `POST /us-performance/predict`
- `GET /carriers/analytics`
- `POST /carriers/recommend`
- `GET /us-performance/anomalies`
- `POST /routes/recommend` — legacy generic route scoring
- `GET /models/metadata`
- `GET /figures/list`
- `GET /figures/{file}.png`

### Local Ollama configuration

Copy `.env.example` to `.env` (or set equivalent environment variables) and retain the defaults for the locally installed models:

```text
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma4:12b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
OLLAMA_TIMEOUT_SECONDS=120
```

Start `ollama serve` before running the semantic-RAG readiness check or starting the backend. `python scripts\\build_ollama_index.py` validates qwen3 and writes a small readiness manifest; it does not embed every historical record. `GET /health` reports whether the configured models are reachable and whether the manifest exists.

## Project structure

```text
backend/app/data/                 canonical schema + four source adapters
backend/app/ml/source_training.py governed multi-source training
backend/app/ml/us_performance.py  US cost/transit research + production models
backend/app/ml/training.py        unified TF-IDF retrieval index
backend/app/services/             grounded insights + decision engine
backend/app/main.py               FastAPI API
frontend/                         Vue 3 operational UI
scripts/                          downloads, training, launch helpers
tests/                            pipeline and adapter tests
data/raw/                         four Kaggle sources (gitignored)
data/processed/                   source frames + unified corpus
artifacts/models/                 trained model artifacts
artifacts/index/                  retrieval vectorizer/matrix
artifacts/figures/                EDA/model/data-quality figures
artifacts/metrics/                governance/evaluation metrics
```

## Validity boundary

This system is historical/dataset-grounded. It is not turn-by-turn navigation, a guaranteed ETA/pricing system, a safety dispatch authority, or a live traffic/weather product. Live geospatial routing and current-condition feeds can be integrated later without changing the source-aware model-governance design.
