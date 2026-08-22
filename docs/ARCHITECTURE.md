# V2.2 Architecture

```text
                LOGISTICS AI INTELLIGENCE PLATFORM
                              │
         ┌────────────────────┴─────────────────────┐
         │                                          │
   Predictive Intelligence                    Knowledge / RAG
         │                                          │
┌────────┴──────────┐                         Unified 4-source corpus
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

## Four-source ingestion

- Nicole Machado tracking → shipment delay intelligence.
- Yogape operations database → relational operational analytics/retrieval; weak predictive tasks remain research-only when gated out.
- DatasetEngineer supply-chain conditions → risk/disruption context and grounded retrieval; leakage-safe weak models remain research-only when gated out.
- US Logistics Performance → cost/transit prediction, carrier analytics, decision ranking, and anomaly monitoring.

## Governance contract

Classification production gate: balanced accuracy ≥ 0.60, macro-F1 ≥ 0.60, and ROC-AUC ≥ 0.60 for binary tasks. Regression production gate: held-out MAE beats the median dummy by at least 5% and R² ≥ 0.10.

Research artifacts/metrics may exist for models that fail. Prediction endpoints must check `deployment_recommended` and do not use failed models.

## Decision-engine contract

Carrier ranking combines only deployable cost/transit models with smoothed historical reliability. Sparse exact-lane observations shrink toward overall carrier performance. No live conditions are claimed.

## Grounding contract

`/ask` retrieves from the unified corpus and returns evidence rows. It does not silently fill missing source facts or claim live traffic/weather/construction/GPS context.
