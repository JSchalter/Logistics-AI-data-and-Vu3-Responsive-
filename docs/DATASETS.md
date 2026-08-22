# Dataset Integration Notes

## 1. Transportation and Logistics Tracking Dataset — Nicole Machado
Shipment/GPS tracking source used for clean delayed-vs-on-time labels, delay-hours regression, lane/vehicle/customer/supplier context, and retrieval. Contradictory `delay=R` plus `ontime=G` rows are not silently reconciled into the primary target.

## 2. Logistics Operations Database — Yogape Rodriguez
Relational trucking database. Delivery outcomes are joined through loads/trips/routes/drivers/trucks/trailers/customers instead of flat-concatenating unrelated tables. Predictive experiments remain research-only when governance metrics do not show useful discrimination.

## 3. Logistics and Supply Chain Dataset — DatasetEngineer
Dynamic supply-chain context including traffic, weather, supplier reliability, route risk, disruption likelihood, delay probability, and delivery deviation. `risk_classification` is strongly tied to `disruption_likelihood_score`; leakage-safe evaluation excludes direct outcome/score fields where appropriate.

## 4. US Logistics Performance Dataset — Shahriar Kabir
2,000 US shipment records with origin warehouse, destination, carrier, dates, weight, cost, status, distance, and transit days. Used for:

- shipment-cost regression
- transit-days regression
- carrier analytics and ranking
- data-quality/anomaly monitoring
- grounded retrieval

`Delivery_Date` is not used as a pre-shipment transit predictor. Source date/transit disagreements are preserved as quality flags rather than overwriting `Transit_Days`.

## Current/live limitation

None of the four historical datasets constitutes a live road-network/traffic/weather feed. V2.2 can recommend among historical carrier choices but must not represent results as turn-by-turn navigation or live-condition routing.
