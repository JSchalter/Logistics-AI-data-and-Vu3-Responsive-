# Logistics AI V2 — Source-Aware Training Patch

## Why V2 exists
The first training pass concatenated three different Kaggle datasets and trained universal supervised models over the combined frame. Real-data diagnostics showed that the datasets have different record grains and different target semantics. V2 keeps the unified frame for semantic retrieval/chat, but trains supervised models source-specifically.

## Source-specific tasks

### Transportation tracking
- Clean binary label from `delay == R` versus `ontime == G`.
- Contradictory rows (`R` and `G`) and fully unlabeled rows are excluded from the primary classification target.
- Delay-hours regression uses `actual_eta - Planned_ETA` as the outcome.
- Outcome fields are excluded from prediction-time features.

### Logistics operations database
- Filters `delivery_events.csv` to `event_type == Delivery`.
- Joins delivery events to loads, trips, routes, drivers, trucks, trailers, and customers.
- Classifies the stored `on_time_flag`.
- Regresses signed service deviation (`actual_datetime - scheduled_datetime`).
- Reports agreement between the source flag and an inferred ±2-hour service window instead of silently redefining the source target.

### Dynamic supply chain
- Treats `order_fulfillment_status` and `cargo_condition_status` as continuous scores.
- Trains a leakage-safe risk classifier that excludes direct outcome/risk-score fields.
- Separately predicts disruption likelihood, delay probability, and delivery-time deviation.
- Reports whether the supplied `risk_classification` matches 0.3/0.7 thresholds on `disruption_likelihood_score`.

## Baselines and deployment gating
Every task includes a dummy baseline. V2 reports whether the selected non-dummy model actually beats that baseline. A model that fails to beat the baseline is kept as an experimental artifact but is marked `deployment_recommended: false` in the metrics JSON.

## Retrieval/chat
All three sources are still normalized into `data/processed/unified_logistics.pkl` and indexed with TF-IDF word/bigram retrieval. This layer is separate from supervised model training.

## Outputs
- `artifacts/metrics/training_metrics_v2.json`
- `artifacts/metrics/training_metrics.json` (API compatibility copy)
- source-specific `.joblib` models under `artifacts/models/`
- corrected unified retrieval index under `artifacts/index/`
- processed source model frames under `data/processed/`
- source-aware figures under `artifacts/figures/`
