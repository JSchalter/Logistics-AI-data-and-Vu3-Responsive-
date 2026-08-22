from __future__ import annotations

UNIFIED_COLUMNS = [
    "record_id", "source_dataset", "timestamp", "shipment_id", "route_id",
    "origin", "destination", "origin_lat", "origin_lon", "destination_lat",
    "destination_lon", "current_lat", "current_lon", "distance_km",
    "vehicle_type", "transport_mode", "carrier", "shipment_status", "weight_kg",
    "transit_days", "date_quality_flag", "planned_eta", "actual_eta", "delay_hours",
    "delayed", "on_time", "traffic_level", "weather_severity", "weather_condition",
    "shipping_cost_usd", "route_risk", "disruption_likelihood", "delay_probability",
    "risk_class", "supplier_reliability", "driver_behavior_score", "fatigue_score",
    "fuel_consumption_rate", "customer_rating", "route_rating", "loading_time_hours",
    "port_congestion", "inventory_level", "cargo_condition", "customer_id", "driver_id",
    "truck_id", "revenue_usd", "fuel_cost_usd", "maintenance_cost_usd", "detention_minutes",
    # V2 source-aware fields retained for retrieval/analytics. These are not fed to every model.
    "eta_variation_hours", "delivery_time_deviation", "order_fulfillment_score",
    "handling_equipment_availability", "lead_time_days", "historical_demand",
    "iot_temperature", "customs_clearance_time", "minimum_km_per_day",
    "planned_transit_hours", "service_deviation_hours", "free_text"
]

NUMERIC_COLUMNS = [
    "origin_lat", "origin_lon", "destination_lat", "destination_lon", "current_lat", "current_lon",
    "distance_km", "weight_kg", "transit_days", "delay_hours", "traffic_level", "weather_severity",
    "shipping_cost_usd", "route_risk", "disruption_likelihood", "delay_probability",
    "supplier_reliability", "driver_behavior_score", "fatigue_score", "fuel_consumption_rate",
    "customer_rating", "route_rating", "loading_time_hours", "port_congestion", "inventory_level",
    "cargo_condition", "revenue_usd", "fuel_cost_usd", "maintenance_cost_usd", "detention_minutes",
    "eta_variation_hours", "delivery_time_deviation", "order_fulfillment_score",
    "handling_equipment_availability", "lead_time_days", "historical_demand",
    "iot_temperature", "customs_clearance_time", "minimum_km_per_day",
    "planned_transit_hours", "service_deviation_hours"
]

CATEGORICAL_COLUMNS = [
    "source_dataset", "origin", "destination", "vehicle_type", "transport_mode", "carrier",
    "shipment_status", "date_quality_flag", "weather_condition", "risk_class", "route_id"
]
