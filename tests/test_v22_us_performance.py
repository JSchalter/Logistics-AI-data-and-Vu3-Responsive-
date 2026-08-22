from pathlib import Path

import pandas as pd

from backend.app.data.adapters import USLogisticsPerformanceAdapter
from backend.app.ml.us_performance import build_us_feature_row, prepare_us_performance


def _fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Shipment_ID": "SH1",
                "Origin_Warehouse": "Warehouse_MIA",
                "Destination": "Detroit",
                "Carrier": "UPS",
                "Shipment_Date": "2023-08-10",
                "Delivery_Date": "2023-08-15",
                "Weight_kg": 30.0,
                "Cost": 220.0,
                "Status": "Delivered",
                "Distance_miles": 1200,
                "Transit_Days": 5,
            },
            {
                "Shipment_ID": "SH2",
                "Origin_Warehouse": "Warehouse_MIA",
                "Destination": "Detroit",
                "Carrier": "USPS",
                "Shipment_Date": "2023-08-10",
                "Delivery_Date": "2023-07-11",
                "Weight_kg": 25.0,
                "Cost": None,
                "Status": "Delayed",
                "Distance_miles": 1200,
                "Transit_Days": 5,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_us_adapter_and_quality_flag(tmp_path):
    path = _fixture(tmp_path / "us_performance" / "logistics_shipments_dataset.csv")
    unified = USLogisticsPerformanceAdapter().load(path)
    assert len(unified) == 2
    assert set(unified["carrier"].dropna()) == {"UPS", "USPS"}
    assert "impossible_negative_transit" in set(unified["date_quality_flag"].dropna())
    assert unified["shipping_cost_usd"].notna().sum() == 1


def test_us_preparation_and_inference_features(tmp_path):
    _fixture(tmp_path / "us_performance" / "logistics_shipments_dataset.csv")
    frame, diag = prepare_us_performance(tmp_path)
    assert diag["rows"] == 2
    assert diag["impossible_negative_transit"] == 1
    row = build_us_feature_row(
        {
            "origin_warehouse": "Warehouse_MIA",
            "destination": "Detroit",
            "carrier": "UPS",
            "shipment_date": "2026-08-21",
            "weight_kg": 30,
            "distance_miles": 1200,
        }
    )
    assert row.loc[0, "shipment_month"] == 8
    assert row.loc[0, "Carrier"] == "UPS"
    assert frame["is_delayed"].notna().all()
