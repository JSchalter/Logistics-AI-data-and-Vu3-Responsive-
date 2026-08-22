# Validation Report

Validation was executed with `DATA_MODE=demo` because the Kaggle raw files are not present in this runtime. The demo fixture is deterministic and explicitly labeled `SYNTHETIC_DEMO_FIXTURE`; it exists only to prove that the complete application pipeline executes.

Validated components:

- canonical ingestion adapter
- feature preprocessing
- delay classification model comparison
- risk classification model comparison
- delay-hours regression
- TF-IDF semantic index
- dataset-grounded retrieval
- smart-route scoring
- analytical/model figure generation
- FastAPI health, summary, model metadata, ask, route-rank, and figure endpoints
- Python compilation
- unit/integration tests

Latest demo validation metrics:

- Delay Logistic Regression: accuracy 0.8000; macro-F1 0.7999; ROC-AUC 0.8598
- Delay Random Forest: accuracy 0.8000; macro-F1 0.7989; ROC-AUC 0.8499
- Risk Logistic Regression: accuracy 0.9500; macro-F1 0.9378
- Risk Random Forest: accuracy 1.0000; macro-F1 1.0000
- Delay regression: MAE 0.8525 hours; R² 0.4890
- Tests: 2 passed

These numbers are **not Kaggle results** and must not be cited as project findings. Real metrics are generated only after `scripts/download_kaggle.py` and real-mode training.
