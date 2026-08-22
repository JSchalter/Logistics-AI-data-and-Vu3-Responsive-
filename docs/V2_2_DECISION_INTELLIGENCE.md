# V2.2 Decision Intelligence Upgrade

V2.2 adds the fourth Kaggle source `shahriarkabir/us-logistics-performance-dataset` and turns the project into a governed predictive + retrieval + carrier-decision platform.

## New files

- `backend/app/ml/us_performance.py` — preprocessing, model benchmarking, deployable cost/transit artifacts, research-only status classifiers, figures.
- `backend/app/services/decision_engine.py` — tracking inference, US prediction, carrier analytics/ranking, anomaly monitor.
- `frontend/src/components/TrackingPanel.vue`
- `frontend/src/components/CarrierDecisionPanel.vue`
- `frontend/src/components/AnomalyPanel.vue`

## Expected source count

After retraining against the user's observed data, the retrieval corpus should grow from 124,355 to approximately 126,355 rows. The emitted `training_metrics_v2.json` is authoritative.

## Recommended run sequence

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m compileall -q backend scripts
python scripts\train.py
Get-Content .\artifacts\metrics\training_metrics_v2.json
python -m uvicorn backend.app.main:app --reload --port 8000
```

Then in another terminal:

```powershell
cd frontend
npm install
npm run dev
```
