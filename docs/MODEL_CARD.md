# Model Card — V2.2

## Production-capable tasks

1. **Tracking delay classification** — delayed vs on-time shipment class, subject to classification deployment gate.
2. **Tracking delay-hours regression** — expected delay magnitude, subject to regression deployment gate.
3. **US shipment cost regression** — pre-shipment cost estimate from origin, destination, carrier, weight, distance, and calendar features.
4. **US transit-days regression** — pre-shipment transit estimate from the same safe feature set.
5. **Semantic retrieval** — TF-IDF word/bigram index across all four source adapters.
6. **Carrier decision ranking** — predicted cost + predicted transit + smoothed historical reliability.

## Research-only tasks

Operations on-time/deviation, leakage-safe supply risk/disruption/delay, and US delay/exception classification remain in metrics even when they fail deployment thresholds. They are not used by production prediction endpoints merely because they beat a weak dummy metric.

## Deployment gates

Classification: balanced accuracy ≥ 0.60, macro-F1 ≥ 0.60, and ROC-AUC ≥ 0.60 for binary tasks. Regression: MAE beats median dummy by ≥5% and R² ≥ 0.10.

## Reliability semantics

Carrier reliability is an observed historical exception rate, smoothed for sparse carrier/lane samples. It is not represented as a learned exception probability because the evaluated exception classifiers did not demonstrate useful discrimination.

## Validity

Predictions are estimates from historical/synthetic Kaggle data. They are not guaranteed prices, guaranteed ETAs, safety dispatch decisions, fraud determinations, or live routing instructions. Operational deployment requires current provider data, monitoring, calibration, persistence, access control, and human oversight.
