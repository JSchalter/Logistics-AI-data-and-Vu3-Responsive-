from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo_dataset(path: Path, n: int = 800, seed: int = 42) -> Path:
    """Generate a synthetic validation fixture. This is never labeled as Kaggle/real data."""
    rng = np.random.default_rng(seed)
    routes = np.array(["Detroit -> Chicago", "Detroit -> Cleveland", "Chicago -> Indianapolis", "Cleveland -> Pittsburgh"])
    route = rng.choice(routes, n)
    traffic = rng.uniform(0, 10, n)
    weather = rng.uniform(0, 1, n)
    risk = np.clip(0.45 * traffic + 4 * weather + rng.normal(0, 1, n), 0, 10)
    distance = np.array([455 if r.startswith("Detroit -> Chicago") else 275 if "Cleveland" in r and r.startswith("Detroit") else 295 if r.startswith("Chicago") else 215 for r in route]) + rng.normal(0, 15, n)
    shipping = distance * rng.uniform(1.3, 2.2, n) + traffic * 18
    delay_prob = 1 / (1 + np.exp(-(-3.2 + 0.32 * traffic + 1.8 * weather + 0.18 * risk)))
    delayed = rng.binomial(1, delay_prob)
    delay_hours = np.maximum(0, delayed * (0.3 + 0.25 * traffic + 1.6 * weather + rng.normal(0, .7, n)))
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "Route": route,
        "Traffic Congestion Level": traffic,
        "Weather Condition Severity": weather,
        "Route Risk Level": risk,
        "Shipping Costs": shipping,
        "Delivery Time Deviation": delay_hours,
        "Delay Probability": delay_prob,
        "Risk Classification": pd.cut(risk, [-1,3.3,6.6,11], labels=["Low Risk","Moderate Risk","High Risk"]),
        "Supplier Reliability Score": rng.uniform(.65, .99, n),
        "Driver Behavior Score": rng.uniform(.55, .99, n),
        "Fatigue Monitoring Score": rng.uniform(.02, .8, n),
        "Order Fulfillment Status": 1-delayed,
        "Fuel Consumption Rate": rng.uniform(10, 32, n),
        "Warehouse Inventory Level": rng.integers(100, 10000, n),
        "Loading/Unloading Time": rng.uniform(.2, 4.5, n),
        "Port Congestion Level": rng.uniform(0, 10, n),
        "Cargo Condition Status": rng.binomial(1, .97, n),
        "Vehicle GPS Latitude": rng.uniform(40.0, 43.5, n),
        "Vehicle GPS Longitude": rng.uniform(-87.8, -81.5, n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
