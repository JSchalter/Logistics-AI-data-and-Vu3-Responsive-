from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

# Figure generation also runs from the API-triggered training path; keep it
# headless and thread-safe rather than trying to open a Tk desktop window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.config import settings

US_CATEGORICAL_FEATURES = ["Origin_Warehouse", "Destination", "Carrier"]
US_NUMERIC_FEATURES = [
    "Weight_kg",
    "Distance_miles",
    "shipment_month",
    "shipment_weekday",
    "shipment_quarter",
    "month_sin",
    "month_cos",
]
US_FEATURES = US_CATEGORICAL_FEATURES + US_NUMERIC_FEATURES


def _us_file(raw_root: Path) -> Path:
    direct = raw_root / "us_performance" / "logistics_shipments_dataset.csv"
    if direct.exists():
        return direct
    hits = list(raw_root.rglob("logistics_shipments_dataset.csv"))
    if not hits:
        raise FileNotFoundError(
            "US Logistics Performance CSV not found. Expected data/raw/us_performance/logistics_shipments_dataset.csv"
        )
    return hits[0]


def prepare_us_performance(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(_us_file(raw_root)).copy()

    for c in US_CATEGORICAL_FEATURES + ["Status", "Shipment_ID"]:
        if c in df:
            df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)

    df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"], errors="coerce")
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
    df["shipment_month"] = df["Shipment_Date"].dt.month
    df["shipment_weekday"] = df["Shipment_Date"].dt.dayofweek
    df["shipment_quarter"] = df["Shipment_Date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["shipment_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["shipment_month"] / 12.0)

    df["calculated_transit_days"] = (
        df["Delivery_Date"] - df["Shipment_Date"]
    ).dt.total_seconds() / 86400.0
    df["transit_discrepancy"] = df["calculated_transit_days"] - pd.to_numeric(
        df["Transit_Days"], errors="coerce"
    )
    df["date_quality_flag"] = "ok"
    df.loc[df["Delivery_Date"].isna(), "date_quality_flag"] = "missing_delivery_date"
    df.loc[df["calculated_transit_days"].lt(0), "date_quality_flag"] = "impossible_negative_transit"
    mismatch = (
        df["calculated_transit_days"].ge(0)
        & df["transit_discrepancy"].abs().gt(0.01)
    )
    df.loc[mismatch, "date_quality_flag"] = "transit_date_mismatch"

    df["lane"] = df["Origin_Warehouse"].astype("string") + " -> " + df["Destination"].astype("string")
    df["cost_per_mile"] = pd.to_numeric(df["Cost"], errors="coerce") / pd.to_numeric(
        df["Distance_miles"], errors="coerce"
    ).replace(0, np.nan)
    df["cost_per_kg"] = pd.to_numeric(df["Cost"], errors="coerce") / pd.to_numeric(
        df["Weight_kg"], errors="coerce"
    ).replace(0, np.nan)

    delay_mask = df["Status"].isin(["Delivered", "Delayed"])
    df["is_delayed"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[delay_mask, "is_delayed"] = df.loc[delay_mask, "Status"].eq("Delayed").astype(int)

    final_mask = ~df["Status"].eq("In Transit")
    df["is_exception"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[final_mask, "is_exception"] = df.loc[final_mask, "Status"].isin(
        ["Delayed", "Lost", "Returned"]
    ).astype(int)

    valid_dates = df["calculated_transit_days"].notna()
    exact = np.isclose(
        pd.to_numeric(df.loc[valid_dates, "Transit_Days"], errors="coerce").to_numpy(),
        df.loc[valid_dates, "calculated_transit_days"].to_numpy(),
    )
    diag = {
        "rows": int(len(df)),
        "status_distribution": {str(k): int(v) for k, v in df["Status"].value_counts().items()},
        "carrier_distribution": {str(k): int(v) for k, v in df["Carrier"].value_counts().items()},
        "missing_cost": int(df["Cost"].isna().sum()),
        "missing_delivery_date": int(df["Delivery_Date"].isna().sum()),
        "date_transit_exact_agreement": int(exact.sum()),
        "date_transit_disagreement": int((~exact).sum()),
        "impossible_negative_transit": int(df["calculated_transit_days"].lt(0).sum()),
        "lane_count": int(df["lane"].nunique()),
        "carrier_lane_combinations": int(df.groupby(["lane", "Carrier"]).ngroups),
    }
    return df, diag


def build_us_feature_row(payload: dict) -> pd.DataFrame:
    date = pd.to_datetime(payload.get("shipment_date"), errors="coerce")
    if pd.isna(date):
        date = pd.Timestamp.now().normalize()
    row = {
        "Origin_Warehouse": str(payload.get("origin_warehouse", "")).strip(),
        "Destination": str(payload.get("destination", "")).strip(),
        "Carrier": str(payload.get("carrier", "")).strip(),
        "Weight_kg": payload.get("weight_kg"),
        "Distance_miles": payload.get("distance_miles"),
        "shipment_month": int(date.month),
        "shipment_weekday": int(date.dayofweek),
        "shipment_quarter": int(date.quarter),
        "month_sin": float(np.sin(2 * np.pi * date.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * date.month / 12.0)),
    }
    return pd.DataFrame([row], columns=US_FEATURES)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                US_CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                US_NUMERIC_FEATURES,
            ),
        ]
    )


def _regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "median_absolute_error": float(median_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def _classification_metrics(y_true: pd.Series, pred: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def _train_regression(df: pd.DataFrame, target: str, artifact_name: str) -> dict:
    work = df.dropna(subset=[target]).copy()
    X = work[US_FEATURES]
    y = pd.to_numeric(work[target], errors="coerce")
    good = y.notna()
    X, y = X.loc[good], y.loc[good]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=settings.random_state
    )
    models = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Pipeline([("prep", _preprocessor()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        random_state=settings.random_state,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=350,
                        random_state=settings.random_state,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }
    results: dict[str, dict] = {}
    best_name = None
    best_model = None
    best_mae = np.inf
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rec = _regression_metrics(yte, pred)
        results[name] = rec
        if not name.startswith("dummy") and rec["mae"] < best_mae:
            best_mae = rec["mae"]
            best_name = name
            best_model = model

    baseline = results["dummy_median"]["mae"]
    improvement = baseline - best_mae
    relative = improvement / baseline if baseline else 0.0
    best_rec = results[best_name] if best_name else {}
    deploy = bool(
        best_name
        and best_mae < baseline
        and best_rec.get("r2", -np.inf) >= 0.10
        and relative >= 0.05
    )
    if deploy and best_model is not None:
        joblib.dump(best_model, settings.models_dir / f"{artifact_name}.joblib")
    return {
        "rows": int(len(X)),
        "features": US_FEATURES,
        "selected_model": best_name,
        "models": results,
        "beats_dummy_mae": bool(best_mae < baseline),
        "mae_improvement_vs_dummy": float(improvement),
        "relative_mae_improvement_vs_dummy": float(relative),
        "deployment_recommended": deploy,
        "artifact": f"{artifact_name}.joblib" if deploy else None,
        "deployment_gate": "MAE beats median dummy by >=5% AND R2 >=0.10",
    }


def _train_classification(df: pd.DataFrame, target: str) -> dict:
    work = df.dropna(subset=[target]).copy()
    X = work[US_FEATURES]
    y = pd.to_numeric(work[target], errors="coerce").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=settings.random_state,
        stratify=y,
    )
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_balanced": Pipeline(
            [
                ("prep", _preprocessor()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2500)),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        random_state=settings.random_state,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=350,
                        random_state=settings.random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }
    results: dict[str, dict] = {}
    best_name = None
    best_score = -np.inf
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        rec = _classification_metrics(yte, pred, prob)
        results[name] = rec
        if not name.startswith("dummy") and rec["macro_f1"] > best_score:
            best_score = rec["macro_f1"]
            best_name = name
    best = results[best_name] if best_name else {}
    deploy = bool(
        best.get("balanced_accuracy", 0) >= 0.60
        and best.get("macro_f1", 0) >= 0.60
        and best.get("roc_auc", 0) >= 0.60
    )
    return {
        "rows": int(len(work)),
        "features": US_FEATURES,
        "selected_model": best_name,
        "models": results,
        "deployment_recommended": deploy,
        "artifact": None,
        "deployment_gate": "balanced accuracy >=0.60 AND macro-F1 >=0.60 AND ROC-AUC >=0.60",
        "note": "Evaluated for research/governance only; failed classifiers are intentionally not persisted for production inference.",
    }


def _plot_us_targets(df: pd.DataFrame) -> None:
    settings.figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    df["Status"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("US Logistics Performance: Shipment Status")
    ax.set_ylabel("Shipments")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "us_performance_status_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    df["Cost"].dropna().clip(upper=df["Cost"].quantile(0.99)).hist(bins=35, ax=ax)
    ax.set_title("US Logistics Performance: Shipment Cost (clipped at 99th percentile)")
    ax.set_xlabel("Cost (USD)")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "us_performance_cost_distribution.png", dpi=160)
    plt.close(fig)

    terminal = df[df["Status"] != "In Transit"].copy()
    terminal["is_exception_hist"] = terminal["Status"].isin(["Delayed", "Lost", "Returned"]).astype(int)
    rates = terminal.groupby("Carrier")["is_exception_hist"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    rates.mul(100).plot(kind="bar", ax=ax)
    ax.set_title("Historical Carrier Exception Rate")
    ax.set_ylabel("Exception rate (%)")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "us_performance_carrier_exception_rate.png", dpi=160)
    plt.close(fig)

    sample = df[["Distance_miles", "Transit_Days"]].dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=settings.random_state)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sample["Distance_miles"], sample["Transit_Days"], alpha=0.35, s=14)
    ax.set_title("Distance vs Transit Days")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Transit days")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "us_performance_distance_vs_transit.png", dpi=160)
    plt.close(fig)


def train_us_performance_models(raw_root: Path) -> tuple[dict, dict]:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)

    df, diag = prepare_us_performance(raw_root)
    df.to_pickle(settings.processed_dir / "us_performance_model_frame.pkl")

    cost = _train_regression(df, "Cost", "us_cost_regressor")
    transit = _train_regression(df, "Transit_Days", "us_transit_days_regressor")
    delay = _train_classification(df[df["is_delayed"].notna()].copy(), "is_delayed")
    exception = _train_classification(df[df["is_exception"].notna()].copy(), "is_exception")
    _plot_us_targets(df)

    metrics = {
        "us_cost_regression": cost,
        "us_transit_days_regression": transit,
        "us_delay_classification_research_only": delay,
        "us_exception_classification_research_only": exception,
    }
    (settings.metrics_dir / "us_performance_metrics.json").write_text(
        json.dumps({"diagnostics": diag, "models": metrics}, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics, diag
