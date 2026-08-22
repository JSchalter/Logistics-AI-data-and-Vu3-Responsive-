from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from .schema import UNIFIED_COLUMNS, NUMERIC_COLUMNS


def _snake(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return re.sub(r"_+", "_", value)


def _first(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        n = _snake(n)
        if n in df.columns:
            return df[n]
    return pd.Series([np.nan] * len(df), index=df.index)


def _boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapped = series.astype(str).str.strip().str.lower().map({
        "1": True, "true": True, "yes": True, "y": True, "on time": True, "ontime": True,
        "g": True,
        "0": False, "false": False, "no": False, "n": False, "late": False, "delayed": False,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(mapped.notna(), numeric.map({1.0: True, 0.0: False}))
    return mapped.astype("boolean")


def _record_ids(df: pd.DataFrame, source: str) -> pd.Series:
    vals = []
    for i, row in df.iterrows():
        raw = f"{source}|{i}|{row.get('shipment_id','')}|{row.get('route_id','')}"
        vals.append(hashlib.sha1(raw.encode()).hexdigest()[:16])
    return pd.Series(vals, index=df.index)


def _parse_lat_lon(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.extract(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*\)?")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def _finalize(out: pd.DataFrame, source: str) -> pd.DataFrame:
    out = out.copy()
    out["source_dataset"] = source
    for c in UNIFIED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["record_id"] = _record_ids(out, source)
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("timestamp", "planned_eta", "actual_eta"):
        out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    for c in ("delayed", "on_time"):
        out[c] = _boolish(out[c])
    return out[UNIFIED_COLUMNS]


@dataclass
class DynamicSupplyChainAdapter:
    source_name: str = "datasetengineer/logistics-and-supply-chain-dataset"

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [_snake(c) for c in df.columns]
        out = pd.DataFrame(index=df.index)
        mapping = {
            "timestamp": ("timestamp",),
            "current_lat": ("vehicle_gps_latitude",),
            "current_lon": ("vehicle_gps_longitude",),
            "fuel_consumption_rate": ("fuel_consumption_rate",),
            "eta_variation_hours": ("eta_variation_hours",),
            "delivery_time_deviation": ("delivery_time_deviation",),
            # Keep a delay-like value for retrieval/analytics, but do not use the unified field as a universal ML target.
            "delay_hours": ("delivery_time_deviation",),
            "traffic_level": ("traffic_congestion_level",),
            "inventory_level": ("warehouse_inventory_level",),
            "loading_time_hours": ("loading_unloading_time",),
            "handling_equipment_availability": ("handling_equipment_availability",),
            "order_fulfillment_score": ("order_fulfillment_status",),
            "weather_severity": ("weather_condition_severity",),
            "port_congestion": ("port_congestion_level",),
            "shipping_cost_usd": ("shipping_costs",),
            "supplier_reliability": ("supplier_reliability_score",),
            "lead_time_days": ("lead_time_days",),
            "historical_demand": ("historical_demand",),
            "iot_temperature": ("iot_temperature",),
            "cargo_condition": ("cargo_condition_status",),
            "route_risk": ("route_risk_level",),
            "customs_clearance_time": ("customs_clearance_time",),
            "driver_behavior_score": ("driver_behavior_score",),
            "fatigue_score": ("fatigue_monitoring_score",),
            "disruption_likelihood": ("disruption_likelihood_score",),
            "delay_probability": ("delay_probability",),
            "risk_class": ("risk_classification",),
        }
        for dst, srcs in mapping.items():
            out[dst] = _first(df, *srcs)
        # This dataset has no categorical on-time field; order_fulfillment_status is a continuous 0-1 score.
        out["on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out["delayed"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out["free_text"] = df.astype(str).apply(
            lambda r: " | ".join(f"{c}={r[c]}" for c in df.columns if r[c] not in ("nan", "None")), axis=1
        )
        return _finalize(out, self.source_name)


@dataclass
class TransportationTrackingAdapter:
    source_name: str = "nicolemachado/transportation-and-logistics-tracking-dataset"

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        df.columns = [_snake(c) for c in df.columns]
        out = pd.DataFrame(index=df.index)

        out["timestamp"] = _first(df, "data_ping_time", "bookingid_date")
        out["shipment_id"] = _first(df, "bookingid", "booking_id")
        out["origin"] = _first(df, "origin_location")
        out["destination"] = _first(df, "destination_location")
        out["current_lat"] = _first(df, "curr_lat")
        out["current_lon"] = _first(df, "curr_lon")
        out["vehicle_type"] = _first(df, "vehicletype", "vehicle_type")
        out["planned_eta"] = _first(df, "planned_eta")
        out["actual_eta"] = _first(df, "actual_eta")
        out["distance_km"] = _first(df, "transportation_distance_in_km")
        out["minimum_km_per_day"] = _first(df, "minimum_kms_to_be_covered_in_a_day")
        out["driver_id"] = _first(df, "driver_name")
        out["customer_id"] = _first(df, "customerid")

        org_lat, org_lon = _parse_lat_lon(_first(df, "org_lat_lon"))
        des_lat, des_lon = _parse_lat_lon(_first(df, "des_lat_lon"))
        out["origin_lat"], out["origin_lon"] = org_lat, org_lon
        out["destination_lat"], out["destination_lon"] = des_lat, des_lon

        delay_raw = _first(df, "delay").astype(str).str.strip().str.upper()
        ontime_raw = _first(df, "ontime").astype(str).str.strip().str.upper()
        delay_is_r = delay_raw.eq("R")
        ontime_is_g = ontime_raw.eq("G")
        # Primary clean label: 4,318 delayed + 2,524 on-time in the observed Kaggle release.
        # Contradictory (R and G) and unlabeled rows remain NA rather than being silently reconciled.
        clean_delayed = delay_is_r & ~ontime_is_g
        clean_ontime = ontime_is_g & ~delay_is_r
        out["delayed"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out["on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out.loc[clean_delayed, "delayed"] = True
        out.loc[clean_delayed, "on_time"] = False
        out.loc[clean_ontime, "delayed"] = False
        out.loc[clean_ontime, "on_time"] = True

        planned = pd.to_datetime(out["planned_eta"], errors="coerce")
        actual = pd.to_datetime(out["actual_eta"], errors="coerce")
        out["delay_hours"] = (actual - planned).dt.total_seconds() / 3600.0
        trip_start = pd.to_datetime(_first(df, "trip_start_date"), errors="coerce")
        out["planned_transit_hours"] = (planned - trip_start).dt.total_seconds() / 3600.0

        origin_code = _first(df, "originlocation_code").astype("string")
        dest_code = _first(df, "destinationlocation_code").astype("string")
        route_codes = origin_code.fillna("") + "->" + dest_code.fillna("")
        route_names = out["origin"].astype("string").fillna("") + "->" + out["destination"].astype("string").fillna("")
        out["route_id"] = route_codes.where(origin_code.notna() & dest_code.notna(), route_names)

        out["free_text"] = df.astype(str).apply(
            lambda r: " | ".join(f"{c}={r[c]}" for c in df.columns if r[c] not in ("nan", "None")), axis=1
        )
        return _finalize(out, self.source_name)


@dataclass
class LogisticsOperationsAdapter:
    source_name: str = "yogape/logistics-operations-database"

    def load(self, directory: Path) -> pd.DataFrame:
        def csv(name: str) -> pd.DataFrame:
            p = directory / name
            if not p.exists():
                return pd.DataFrame()
            d = pd.read_csv(p)
            d.columns = [_snake(c) for c in d.columns]
            return d

        loads = csv("loads.csv")
        trips = csv("trips.csv")
        routes = csv("routes.csv")
        events = csv("delivery_events.csv")
        trailers = csv("trailers.csv")
        fuel = csv("fuel_purchases.csv")
        maintenance = csv("maintenance_records.csv")

        if loads.empty:
            raise FileNotFoundError(f"Expected loads.csv under {directory}")

        base = loads.copy()
        if not trips.empty:
            base = base.merge(trips, on="load_id", how="left", suffixes=("", "_trip"))
        if not routes.empty:
            base = base.merge(routes, on="route_id", how="left", suffixes=("", "_route"))
        if not trailers.empty and "trailer_id" in base.columns:
            tr = trailers[[c for c in trailers.columns if c in {"trailer_id", "trailer_type", "length_feet", "model_year"}]].copy()
            base = base.merge(tr, on="trailer_id", how="left", suffixes=("", "_trailer"))

        if not events.empty:
            delivery = events[events["event_type"].astype(str).str.lower().eq("delivery")].copy()
            keep = [c for c in [
                "load_id", "trip_id", "scheduled_datetime", "actual_datetime", "detention_minutes",
                "on_time_flag", "facility_id", "location_city", "location_state"
            ] if c in delivery.columns]
            delivery = delivery[keep].drop_duplicates(subset=[c for c in ["load_id", "trip_id"] if c in keep])
            keys = [c for c in ["load_id", "trip_id"] if c in base.columns and c in delivery.columns]
            base = base.merge(delivery, on=keys or ["load_id"], how="left", suffixes=("", "_delivery"))

        if not fuel.empty and "trip_id" in fuel.columns and "trip_id" in base.columns:
            fagg = fuel.groupby("trip_id", as_index=False).agg(
                fuel_cost_usd=("total_cost", "sum"), fuel_gallons_purchased=("gallons", "sum")
            )
            base = base.merge(fagg, on="trip_id", how="left")
        if not maintenance.empty and "truck_id" in maintenance.columns and "truck_id" in base.columns:
            magg = maintenance.groupby("truck_id", as_index=False).agg(
                maintenance_cost_usd=("total_cost", "sum")
            )
            base = base.merge(magg, on="truck_id", how="left")

        out = pd.DataFrame(index=base.index)
        out["timestamp"] = _first(base, "load_date", "dispatch_date")
        out["shipment_id"] = _first(base, "load_id")
        out["route_id"] = _first(base, "route_id")
        out["origin"] = _first(base, "origin_city")
        out["destination"] = _first(base, "destination_city")
        out["distance_km"] = pd.to_numeric(_first(base, "typical_distance_miles"), errors="coerce") * 1.609344
        out["driver_id"] = _first(base, "driver_id")
        out["truck_id"] = _first(base, "truck_id")
        out["revenue_usd"] = _first(base, "revenue")
        out["fuel_cost_usd"] = _first(base, "fuel_cost_usd")
        out["maintenance_cost_usd"] = _first(base, "maintenance_cost_usd")
        out["detention_minutes"] = _first(base, "detention_minutes")
        out["on_time"] = _first(base, "on_time_flag")
        out["delayed"] = ~_boolish(out["on_time"])
        out["vehicle_type"] = _first(base, "trailer_type")
        out["planned_eta"] = _first(base, "scheduled_datetime")
        out["actual_eta"] = _first(base, "actual_datetime")
        planned = pd.to_datetime(out["planned_eta"], errors="coerce")
        actual = pd.to_datetime(out["actual_eta"], errors="coerce")
        out["service_deviation_hours"] = (actual - planned).dt.total_seconds() / 3600.0
        out["delay_hours"] = out["service_deviation_hours"]
        out["free_text"] = base.astype(str).apply(
            lambda r: " | ".join(f"{c}={r[c]}" for c in base.columns if r[c] not in ("nan", "None")), axis=1
        )
        return _finalize(out, self.source_name)


@dataclass
class USLogisticsPerformanceAdapter:
    source_name: str = "shahriarkabir/us-logistics-performance-dataset"

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [_snake(c) for c in df.columns]
        out = pd.DataFrame(index=df.index)

        out["timestamp"] = _first(df, "shipment_date")
        out["shipment_id"] = _first(df, "shipment_id")
        out["origin"] = _first(df, "origin_warehouse")
        out["destination"] = _first(df, "destination")
        out["carrier"] = _first(df, "carrier")
        out["shipment_status"] = _first(df, "status")
        out["weight_kg"] = _first(df, "weight_kg")
        out["shipping_cost_usd"] = _first(df, "cost")
        out["transit_days"] = _first(df, "transit_days")
        out["distance_km"] = pd.to_numeric(_first(df, "distance_miles"), errors="coerce") * 1.609344
        out["transport_mode"] = "parcel/shipment"
        out["actual_eta"] = _first(df, "delivery_date")
        out["route_id"] = (
            out["origin"].astype("string").fillna("")
            + "->"
            + out["destination"].astype("string").fillna("")
        )

        status = out["shipment_status"].astype("string").str.strip().str.lower()
        out["delayed"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out["on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out.loc[status.eq("delayed"), "delayed"] = True
        out.loc[status.eq("delayed"), "on_time"] = False
        out.loc[status.eq("delivered"), "delayed"] = False
        out.loc[status.eq("delivered"), "on_time"] = True

        shipped = pd.to_datetime(_first(df, "shipment_date"), errors="coerce")
        delivered = pd.to_datetime(_first(df, "delivery_date"), errors="coerce")
        calculated = (delivered - shipped).dt.total_seconds() / 86400.0
        recorded = pd.to_numeric(out["transit_days"], errors="coerce")
        discrepancy = calculated - recorded
        out["date_quality_flag"] = "ok"
        out.loc[delivered.isna(), "date_quality_flag"] = "missing_delivery_date"
        out.loc[calculated.lt(0), "date_quality_flag"] = "impossible_negative_transit"
        out.loc[calculated.ge(0) & discrepancy.abs().gt(0.01), "date_quality_flag"] = "transit_date_mismatch"

        out["free_text"] = df.astype(str).apply(
            lambda r: " | ".join(f"{c}={r[c]}" for c in df.columns if r[c] not in ("nan", "None")), axis=1
        )
        return _finalize(out, self.source_name)


def discover_and_load(raw_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for p in raw_root.rglob("dynamic_supply_chain_logistics_dataset.csv"):
        frames.append(DynamicSupplyChainAdapter().load(p))
        break

    tracking_candidates = list(raw_root.rglob("*.xlsx")) + list(raw_root.rglob("*.xls"))
    for p in tracking_candidates:
        try:
            d = TransportationTrackingAdapter().load(p)
            score = sum(d[c].notna().any() for c in ("shipment_id", "origin", "destination", "planned_eta", "distance_km"))
            if score >= 3:
                frames.append(d)
                break
        except Exception:
            continue

    for p in raw_root.rglob("loads.csv"):
        try:
            frames.append(LogisticsOperationsAdapter().load(p.parent))
            break
        except Exception:
            continue

    for p in raw_root.rglob("logistics_shipments_dataset.csv"):
        try:
            frames.append(USLogisticsPerformanceAdapter().load(p))
            break
        except Exception:
            continue

    if not frames:
        raise FileNotFoundError(
            f"No supported Kaggle dataset files were discovered under {raw_root}. "
            "Run scripts/download_kaggle.py or place extracted files there."
        )
    return pd.concat(frames, ignore_index=True, sort=False)
