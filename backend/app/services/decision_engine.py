from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.ml.us_performance import build_us_feature_row

TRACKING_NUMERIC = [
    "distance_km",
    "minimum_km_per_day",
    "planned_transit_hours",
    "booking_to_start_hours",
    "start_month",
    "start_dayofweek",
    "start_hour",
]
TRACKING_CATEGORICAL = [
    "market_regular",
    "vehicle_type",
    "origin_code",
    "destination_code",
    "route_id",
    "customer_id",
    "supplier_id",
    "material_shipped",
    "gps_provider",
]


def _metrics() -> dict:
    path = settings.metrics_dir / "training_metrics_v2.json"
    if not path.exists():
        path = settings.metrics_dir / "training_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_meta(key: str) -> dict:
    return _metrics().get("models", {}).get(key, {})


def _require_artifact(name: str) -> Path:
    path = settings.models_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required trained model is missing: {path.name}. Run V2.2 training first.")
    return path


def _selected_quality(meta: dict) -> dict:
    selected = meta.get("selected_model")
    rec = meta.get("models", {}).get(selected, {}) if selected else {}
    return {
        "selected_model": selected,
        "deployment_recommended": bool(meta.get("deployment_recommended")),
        "metrics": {k: v for k, v in rec.items() if k != "classification_report"},
    }


def _tracking_row(payload: dict[str, Any]) -> pd.DataFrame:
    trip_start = pd.to_datetime(payload.get("trip_start_date"), errors="coerce")
    planned_eta = pd.to_datetime(payload.get("planned_eta"), errors="coerce")
    booking_date = pd.to_datetime(payload.get("booking_date"), errors="coerce")

    planned_transit = payload.get("planned_transit_hours")
    if planned_transit is None and pd.notna(trip_start) and pd.notna(planned_eta):
        planned_transit = (planned_eta - trip_start).total_seconds() / 3600.0

    booking_to_start = payload.get("booking_to_start_hours")
    if booking_to_start is None and pd.notna(trip_start) and pd.notna(booking_date):
        booking_to_start = (trip_start - booking_date).total_seconds() / 3600.0

    origin = str(payload.get("origin_code") or "").strip()
    destination = str(payload.get("destination_code") or "").strip()
    route_id = str(payload.get("route_id") or f"{origin}->{destination}").strip()

    row = {
        "distance_km": payload.get("distance_km"),
        "minimum_km_per_day": payload.get("minimum_km_per_day"),
        "planned_transit_hours": planned_transit,
        "booking_to_start_hours": booking_to_start,
        "start_month": int(trip_start.month) if pd.notna(trip_start) else payload.get("start_month"),
        "start_dayofweek": int(trip_start.dayofweek) if pd.notna(trip_start) else payload.get("start_dayofweek"),
        "start_hour": int(trip_start.hour) if pd.notna(trip_start) else payload.get("start_hour"),
        "market_regular": payload.get("market_regular"),
        "vehicle_type": payload.get("vehicle_type"),
        "origin_code": origin,
        "destination_code": destination,
        "route_id": route_id,
        "customer_id": payload.get("customer_id"),
        "supplier_id": payload.get("supplier_id"),
        "material_shipped": payload.get("material_shipped"),
        "gps_provider": payload.get("gps_provider"),
    }
    for c in TRACKING_CATEGORICAL:
        v = row.get(c)
        row[c] = np.nan if v is None or str(v).strip() == "" else str(v).strip()
    return pd.DataFrame([row], columns=TRACKING_NUMERIC + TRACKING_CATEGORICAL)


def tracking_predict(payload: dict[str, Any]) -> dict:
    class_meta = _model_meta("tracking_delay_classification")
    reg_meta = _model_meta("tracking_delay_hours_regression")
    if not class_meta.get("deployment_recommended"):
        raise RuntimeError("Tracking delay classifier did not pass the V2.2 deployment gate.")

    X = _tracking_row(payload)
    classifier = joblib.load(_require_artifact("tracking_delay_classifier.joblib"))
    pred = int(classifier.predict(X)[0])
    prob = None
    if hasattr(classifier, "predict_proba"):
        classes = list(classifier.classes_)
        probs = classifier.predict_proba(X)[0]
        if 1 in classes:
            prob = float(probs[classes.index(1)])
        else:
            prob = float(np.max(probs))

    delay_hours = None
    if reg_meta.get("deployment_recommended") and (settings.models_dir / "tracking_delay_hours_regressor.joblib").exists():
        reg = joblib.load(settings.models_dir / "tracking_delay_hours_regressor.joblib")
        delay_hours = float(reg.predict(X)[0])

    return {
        "predicted_delayed": bool(pred),
        "delay_probability": prob,
        "predicted_delay_hours": delay_hours,
        "classifier": _selected_quality(class_meta),
        "delay_hours_model": _selected_quality(reg_meta),
        "grounding": "tracking Kaggle model",
        "live_conditions_used": False,
    }


def us_predict(payload: dict[str, Any]) -> dict:
    cost_meta = _model_meta("us_cost_regression")
    transit_meta = _model_meta("us_transit_days_regression")
    X = build_us_feature_row(payload)
    out: dict[str, Any] = {
        "carrier": payload.get("carrier"),
        "grounding": "US Logistics Performance trained models",
        "live_conditions_used": False,
    }

    if cost_meta.get("deployment_recommended"):
        cost_model = joblib.load(_require_artifact("us_cost_regressor.joblib"))
        out["predicted_cost_usd"] = float(cost_model.predict(X)[0])
        out["cost_model"] = _selected_quality(cost_meta)
    else:
        out["predicted_cost_usd"] = None
        out["cost_model"] = _selected_quality(cost_meta)

    if transit_meta.get("deployment_recommended"):
        transit_model = joblib.load(_require_artifact("us_transit_days_regressor.joblib"))
        out["predicted_transit_days"] = max(0.0, float(transit_model.predict(X)[0]))
        out["transit_model"] = _selected_quality(transit_meta)
    else:
        out["predicted_transit_days"] = None
        out["transit_model"] = _selected_quality(transit_meta)

    return out


def _us_frame() -> pd.DataFrame:
    path = settings.processed_dir / "us_performance_model_frame.pkl"
    if not path.exists():
        raise FileNotFoundError("US performance model frame is missing. Run V2.2 training first.")
    return pd.read_pickle(path)


def carrier_analytics() -> dict:
    df = _us_frame().copy()
    terminal = df[df["Status"] != "In Transit"].copy()
    terminal["is_exception_hist"] = terminal["Status"].isin(["Delayed", "Lost", "Returned"]).astype(int)

    carrier = df.groupby("Carrier").agg(
        shipments=("Shipment_ID", "count"),
        avg_cost_usd=("Cost", "mean"),
        median_cost_usd=("Cost", "median"),
        avg_transit_days=("Transit_Days", "mean"),
        transit_std_days=("Transit_Days", "std"),
        avg_cost_per_mile=("cost_per_mile", "mean"),
        avg_cost_per_kg=("cost_per_kg", "mean"),
    )
    exc = terminal.groupby("Carrier")["is_exception_hist"].mean().rename("exception_rate")
    carrier = carrier.join(exc)
    carrier["historical_reliability"] = 1.0 - carrier["exception_rate"]
    carrier = carrier.sort_values(["historical_reliability", "avg_cost_usd"], ascending=[False, True])

    records = []
    for name, row in carrier.iterrows():
        rec = {"carrier": str(name)}
        for k, v in row.items():
            rec[k] = None if pd.isna(v) else float(v) if isinstance(v, (float, np.floating)) else int(v)
        records.append(rec)
    return {
        "carriers": records,
        "carrier_count": int(df["Carrier"].nunique()),
        "lane_count": int(df["lane"].nunique()),
        "origins": sorted(df["Origin_Warehouse"].dropna().astype(str).unique().tolist()),
        "destinations": sorted(df["Destination"].dropna().astype(str).unique().tolist()),
        "historical_reliability_is_predictive": False,
        "note": "Reliability is historical observed performance, smoothed for sparse lane/carrier combinations in recommendations; it is not a trained exception-probability model.",
    }


def _norm(values: list[float]) -> list[float]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return []
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [0.5] * len(a)
    return [float((x - lo) / (hi - lo)) for x in a]


def recommend_carriers(payload: dict[str, Any]) -> dict:
    df = _us_frame().copy()
    origin = str(payload.get("origin_warehouse", "")).strip()
    destination = str(payload.get("destination", "")).strip()
    objective = str(payload.get("objective", "balanced")).strip().lower().replace(" ", "_")
    objective = {"reliable": "most_reliable", "reliability": "most_reliable"}.get(objective, objective)
    if objective not in {"cheapest", "fastest", "most_reliable", "balanced"}:
        raise ValueError("objective must be cheapest, fastest, most_reliable, or balanced")

    exact_lane = df[(df["Origin_Warehouse"] == origin) & (df["Destination"] == destination)]
    carriers = sorted(exact_lane["Carrier"].dropna().astype(str).unique().tolist())
    lane_coverage_used = True
    if not carriers:
        carriers = sorted(df["Carrier"].dropna().astype(str).unique().tolist())
        lane_coverage_used = False

    terminal = df[df["Status"] != "In Transit"].copy()
    terminal["is_exception_hist"] = terminal["Status"].isin(["Delayed", "Lost", "Returned"]).astype(int)
    global_rate = float(terminal["is_exception_hist"].mean())

    candidates = []
    for carrier in carriers:
        model_payload = {**payload, "carrier": carrier}
        pred = us_predict(model_payload)

        carrier_hist = terminal[terminal["Carrier"] == carrier]
        carrier_rate = float(carrier_hist["is_exception_hist"].mean()) if len(carrier_hist) else global_rate
        lane_hist = carrier_hist[
            (carrier_hist["Origin_Warehouse"] == origin) & (carrier_hist["Destination"] == destination)
        ]
        lane_n = int(len(lane_hist))
        lane_exc = int(lane_hist["is_exception_hist"].sum()) if lane_n else 0
        # Empirical-Bayes style smoothing: sparse lane evidence shrinks to carrier history.
        prior_strength = 10.0
        smoothed_exception = float((lane_exc + prior_strength * carrier_rate) / (lane_n + prior_strength))
        reliability = 1.0 - smoothed_exception

        candidates.append(
            {
                "carrier": carrier,
                "predicted_cost_usd": pred.get("predicted_cost_usd"),
                "predicted_transit_days": pred.get("predicted_transit_days"),
                "historical_reliability": float(reliability),
                "historical_exception_rate": float(smoothed_exception),
                "lane_history_rows": lane_n,
                "carrier_history_rows": int(len(carrier_hist)),
            }
        )

    if not candidates:
        return {"objective": objective, "recommendations": []}

    cost_n = _norm([c["predicted_cost_usd"] for c in candidates])
    transit_n = _norm([c["predicted_transit_days"] for c in candidates])
    risk_n = _norm([c["historical_exception_rate"] for c in candidates])

    for i, c in enumerate(candidates):
        if objective == "cheapest":
            score = cost_n[i]
        elif objective == "fastest":
            score = transit_n[i]
        elif objective == "most_reliable":
            score = risk_n[i]
        else:
            score = 0.35 * cost_n[i] + 0.30 * transit_n[i] + 0.35 * risk_n[i]
        c["decision_score"] = float(score)
        c["objective"] = objective

    candidates.sort(key=lambda x: x["decision_score"])
    for i, c in enumerate(candidates, start=1):
        c["rank"] = i
        c["recommended"] = i == 1

    return {
        "objective": objective,
        "origin_warehouse": origin,
        "destination": destination,
        "lane_coverage_used": lane_coverage_used,
        "recommendations": candidates,
        "decision_method": {
            "cheapest": "minimum normalized predicted cost",
            "fastest": "minimum normalized predicted transit time",
            "most_reliable": "minimum smoothed historical exception rate",
            "balanced": "35% predicted cost + 30% predicted transit + 35% smoothed historical exception rate",
        }[objective],
        "reliability_note": "Reliability is historical and smoothed; the delay/exception classifiers failed deployment gates and are not used as predictions.",
        "live_conditions_used": False,
    }


def us_anomalies(limit: int = 25) -> dict:
    df = _us_frame().copy()
    anomalies: list[dict[str, Any]] = []

    q1 = float(df["Cost"].quantile(0.25))
    q3 = float(df["Cost"].quantile(0.75))
    iqr = q3 - q1
    extreme_cost_threshold = q3 + 3.0 * iqr

    def add(row: pd.Series, anomaly_type: str, severity: str, reason: str):
        anomalies.append(
            {
                "shipment_id": row.get("Shipment_ID"),
                "carrier": row.get("Carrier"),
                "origin": row.get("Origin_Warehouse"),
                "destination": row.get("Destination"),
                "status": row.get("Status"),
                "anomaly_type": anomaly_type,
                "severity": severity,
                "reason": reason,
                "cost_usd": None if pd.isna(row.get("Cost")) else float(row.get("Cost")),
                "recorded_transit_days": None if pd.isna(row.get("Transit_Days")) else float(row.get("Transit_Days")),
                "calculated_transit_days": None if pd.isna(row.get("calculated_transit_days")) else float(row.get("calculated_transit_days")),
                "transit_discrepancy_days": None if pd.isna(row.get("transit_discrepancy")) else float(row.get("transit_discrepancy")),
            }
        )

    for _, row in df[df["calculated_transit_days"].lt(0)].iterrows():
        add(row, "impossible_negative_transit", "critical", "Delivery date precedes shipment date.")
    for _, row in df[df["transit_discrepancy"].abs().ge(7) & df["calculated_transit_days"].ge(0)].iterrows():
        add(row, "large_transit_date_mismatch", "high", "Recorded transit days differ from shipment/delivery dates by at least 7 days.")
    for _, row in df[df["Cost"].gt(extreme_cost_threshold)].iterrows():
        add(row, "extreme_cost", "high", f"Cost exceeds the robust 3-IQR threshold (${extreme_cost_threshold:,.2f}).")
    for _, row in df[df["Cost"].isna()].iterrows():
        add(row, "missing_cost", "medium", "Shipment cost is missing.")
    for _, row in df[df["Delivery_Date"].isna()].iterrows():
        add(row, "missing_delivery_date", "medium", "Delivery date is missing.")

    sev = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    anomalies.sort(key=lambda x: (sev.get(x["severity"], 9), x["shipment_id"] or ""))
    counts: dict[str, int] = {}
    for a in anomalies:
        counts[a["anomaly_type"]] = counts.get(a["anomaly_type"], 0) + 1

    return {
        "total_flags": len(anomalies),
        "counts_by_type": counts,
        "extreme_cost_threshold_usd": float(extreme_cost_threshold),
        "items": anomalies[: max(1, min(int(limit), 200))],
        "note": "Flags are data-quality/anomaly rules, not fraud determinations.",
    }
