from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd

# Training can be invoked by a FastAPI worker thread.  Use a file-only backend
# so figure generation never initializes Tk or another desktop GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    average_precision_score,
    r2_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.config import settings
from backend.app.data.adapters import discover_and_load
from backend.app.ml.training import build_retrieval_index
from backend.app.ml.us_performance import train_us_performance_models


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _tracking_file(raw_root: Path) -> Path:
    files = list((raw_root / "tracking").glob("*.xlsx")) + list((raw_root / "tracking").glob("*.xls"))
    if not files:
        files = list(raw_root.rglob("*Tracking Dataset*.xlsx"))
    if not files:
        raise FileNotFoundError("Tracking XLSX was not found under data/raw/tracking")
    return files[0]


def _supply_file(raw_root: Path) -> Path:
    p = raw_root / "supply_chain" / "dynamic_supply_chain_logistics_dataset.csv"
    if p.exists():
        return p
    hits = list(raw_root.rglob("dynamic_supply_chain_logistics_dataset.csv"))
    if not hits:
        raise FileNotFoundError("Supply-chain CSV was not found")
    return hits[0]


def _operations_dir(raw_root: Path) -> Path:
    p = raw_root / "operations"
    if (p / "loads.csv").exists():
        return p
    hits = list(raw_root.rglob("loads.csv"))
    if not hits:
        raise FileNotFoundError("Operations database was not found")
    return hits[0].parent


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out


def _available_numeric(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    out = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                df[c] = s
                out.append(c)
    return out


def _available_cat(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    """Return usable categorical columns after enforcing a uniform string dtype.

    Kaggle identifier/category columns can contain mixed Python values (for example,
    integer customer IDs alongside string IDs). scikit-learn's OneHotEncoder requires
    each input feature to contain a uniform primitive type, so normalize every
    non-missing categorical value to a stripped string and preserve missing values as
    np.nan for SimpleImputer.
    """
    out: list[str] = []
    for c in cols:
        if c not in df.columns or not df[c].notna().any():
            continue
        df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        out.append(c)
    return out


def _preprocessor(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[ColumnTransformer, list[str]]:
    n = _available_numeric(df, numeric)
    c = _available_cat(df, categorical)
    if not n and not c:
        raise ValueError("No usable features were available for this model")
    transformers = []
    if n:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            n,
        ))
    if c:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, max_categories=80)),
            ]),
            c,
        ))
    return ColumnTransformer(transformers), n + c


def _classification_metrics(y_true, pred, prob=None) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    rec = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }
    if prob is not None and pd.Series(y_true).nunique() == 2:
        rec["roc_auc"] = float(roc_auc_score(y_true, prob))
        rec["pr_auc"] = float(average_precision_score(y_true, prob))
    return rec


def _regression_metrics(y_true, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "median_absolute_error": float(median_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def _fit_classifiers(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    model_prefix: str,
    figure_name: str,
) -> dict:
    work = frame[frame[target].notna()].copy()
    y = work[target]
    if pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce").astype(int)
    else:
        y = y.astype(str)
    if y.nunique() < 2:
        raise ValueError(f"{target} has fewer than two classes")
    pre, features = _preprocessor(work, numeric, categorical)
    X = work[features]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=settings.random_state, stratify=y
    )
    candidates = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": Pipeline([
            ("prep", clone(pre)),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced")),
        ]),
        "random_forest_balanced": Pipeline([
            ("prep", clone(pre)),
            ("model", RandomForestClassifier(
                n_estimators=220,
                random_state=settings.random_state,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
        "extra_trees_balanced": Pipeline([
            ("prep", clone(pre)),
            ("model", ExtraTreesClassifier(
                n_estimators=220,
                random_state=settings.random_state,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
    }
    results = {}
    best_name = None
    best_score = -np.inf
    best_model = None
    for name, model in candidates.items():
        if name.startswith("dummy"):
            model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = None
        if hasattr(model, "predict_proba") and pd.Series(yte).nunique() == 2:
            probs = model.predict_proba(Xte)
            # Positive class is the second sorted sklearn class.
            prob = probs[:, 1]
        rec = _classification_metrics(yte, pred, prob)
        results[name] = rec
        if not name.startswith("dummy") and rec["macro_f1"] > best_score:
            best_score = rec["macro_f1"]
            best_name = name
            best_model = model
    dummy_f1 = results["dummy_most_frequent"]["macro_f1"]
    beats_dummy = bool(best_score > dummy_f1)
    best_rec = results.get(best_name, {})
    if "roc_auc" in best_rec:
        deploy = bool(
            best_rec.get("balanced_accuracy", 0) >= 0.60
            and best_rec.get("macro_f1", 0) >= 0.60
            and best_rec.get("roc_auc", 0) >= 0.60
        )
        gate = "balanced accuracy >=0.60 AND macro-F1 >=0.60 AND ROC-AUC >=0.60"
    else:
        deploy = bool(
            best_rec.get("balanced_accuracy", 0) >= 0.60
            and best_rec.get("macro_f1", 0) >= 0.60
        )
        gate = "balanced accuracy >=0.60 AND macro-F1 >=0.60"
    if best_model is not None:
        # Persist the research artifact for reproducibility, but production APIs must check deployment_recommended.
        joblib.dump(best_model, settings.models_dir / f"{model_prefix}.joblib")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(best_model, Xte, yte, ax=ax, xticks_rotation=20)
        ax.set_title(f"{model_prefix.replace('_', ' ').title()} Confusion Matrix")
        fig.tight_layout()
        fig.savefig(settings.figures_dir / figure_name, dpi=160)
        plt.close(fig)
    return {
        "rows": int(len(work)),
        "features": features,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts(dropna=False).items()},
        "selected_model": best_name,
        "models": results,
        "beats_dummy_macro_f1": beats_dummy,
        "macro_f1_improvement_vs_dummy": float(best_score - dummy_f1),
        "deployment_recommended": deploy,
        "deployment_gate": gate,
    }


def _fit_regressors(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    model_prefix: str,
) -> dict:
    work = frame[frame[target].notna()].copy()
    pre, features = _preprocessor(work, numeric, categorical)
    X = work[features]
    y = pd.to_numeric(work[target], errors="coerce")
    good = y.notna()
    X, y = X.loc[good], y.loc[good]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=settings.random_state
    )
    candidates = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Pipeline([
            ("prep", clone(pre)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("prep", clone(pre)),
            ("model", RandomForestRegressor(
                n_estimators=180,
                random_state=settings.random_state,
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("prep", clone(pre)),
            ("model", ExtraTreesRegressor(
                n_estimators=180,
                random_state=settings.random_state,
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
    }
    results = {}
    best_name = None
    best_mae = np.inf
    best_model = None
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rec = _regression_metrics(yte, pred)
        results[name] = rec
        if not name.startswith("dummy") and rec["mae"] < best_mae:
            best_mae = rec["mae"]
            best_name = name
            best_model = model
    if best_model is not None:
        joblib.dump(best_model, settings.models_dir / f"{model_prefix}.joblib")
    baseline_mae = results["dummy_median"]["mae"]
    improvement = baseline_mae - best_mae
    relative = improvement / baseline_mae if baseline_mae else 0.0
    best_rec = results.get(best_name, {})
    deploy = bool(
        best_mae < baseline_mae
        and best_rec.get("r2", -np.inf) >= 0.10
        and relative >= 0.05
    )
    return {
        "rows": int(len(X)),
        "features": features,
        "selected_model": best_name,
        "models": results,
        "beats_dummy_mae": bool(best_mae < baseline_mae),
        "mae_improvement_vs_dummy": float(improvement),
        "relative_mae_improvement_vs_dummy": float(relative),
        "deployment_recommended": deploy,
        "deployment_gate": "MAE beats median dummy by >=5% AND R2 >=0.10",
    }


def prepare_tracking(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(_tracking_file(raw_root))
    df = _clean_cols(raw)
    # Column normalization for the Kaggle release.
    rename = {
        "market_regular_": "market_regular",
        "transportation_distance_in_km": "distance_km",
        "minimum_kms_to_be_covered_in_a_day": "minimum_km_per_day",
        "vehicletype": "vehicle_type",
        "originlocation_code": "origin_code",
        "destinationlocation_code": "destination_code",
        "material_shipped": "material_shipped",
        "bookingid_date": "booking_date",
        "bookingid": "booking_id",
        "gpsprovider": "gps_provider",
        "customerid": "customer_id",
        "supplierid": "supplier_id",
    }
    df = df.rename(columns=rename)

    delay = df["delay"].astype("string").str.strip().str.upper()
    ontime = df["ontime"].astype("string").str.strip().str.upper()
    delay_r = delay.eq("R").fillna(False)
    ontime_g = ontime.eq("G").fillna(False)
    df["is_delayed"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[delay_r & ~ontime_g, "is_delayed"] = 1
    df.loc[ontime_g & ~delay_r, "is_delayed"] = 0

    df["planned_eta"] = pd.to_datetime(df["planned_eta"], errors="coerce")
    df["actual_eta"] = pd.to_datetime(df["actual_eta"], errors="coerce")
    df["trip_start_date"] = pd.to_datetime(df["trip_start_date"], errors="coerce")
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["delay_hours_target"] = (df["actual_eta"] - df["planned_eta"]).dt.total_seconds() / 3600
    df["planned_transit_hours"] = (df["planned_eta"] - df["trip_start_date"]).dt.total_seconds() / 3600
    df["booking_to_start_hours"] = (df["trip_start_date"] - df["booking_date"]).dt.total_seconds() / 3600
    df["start_month"] = df["trip_start_date"].dt.month
    df["start_dayofweek"] = df["trip_start_date"].dt.dayofweek
    df["start_hour"] = df["trip_start_date"].dt.hour
    df["route_id"] = (
        df.get("origin_code", pd.Series(index=df.index, dtype="string")).astype("string").fillna("")
        + "->" +
        df.get("destination_code", pd.Series(index=df.index, dtype="string")).astype("string").fillna("")
    )
    diag = {
        "rows": int(len(df)),
        "clean_labeled_rows": int(df["is_delayed"].notna().sum()),
        "contradictory_r_and_g": int((delay_r & ontime_g).sum()),
        "unlabeled_neither": int((~delay_r & ~ontime_g).sum()),
    }
    return df, diag


def prepare_operations(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    root = _operations_dir(raw_root)
    events = _read_csv(root / "delivery_events.csv")
    deliveries = events[events["event_type"].astype(str).str.lower().eq("delivery")].copy()
    loads = _read_csv(root / "loads.csv")
    trips = _read_csv(root / "trips.csv")
    routes = _read_csv(root / "routes.csv")
    drivers = _read_csv(root / "drivers.csv")
    trucks = _read_csv(root / "trucks.csv")
    trailers = _read_csv(root / "trailers.csv")
    customers = _read_csv(root / "customers.csv")

    drivers = drivers.rename(columns={c: f"driver_{c}" for c in drivers.columns if c != "driver_id"})
    trucks = trucks.rename(columns={c: f"truck_{c}" for c in trucks.columns if c != "truck_id"})
    trailers = trailers.rename(columns={c: f"trailer_{c}" for c in trailers.columns if c != "trailer_id"})
    customers = customers.rename(columns={c: f"customer_{c}" for c in customers.columns if c != "customer_id"})

    df = deliveries.merge(loads, on="load_id", how="left", validate="many_to_one")
    df = df.merge(trips, on=["trip_id", "load_id"], how="left", validate="many_to_one")
    df = df.merge(routes, on="route_id", how="left", validate="many_to_one")
    df = df.merge(drivers, on="driver_id", how="left", validate="many_to_one")
    df = df.merge(trucks, on="truck_id", how="left", validate="many_to_one")
    df = df.merge(trailers, on="trailer_id", how="left", validate="many_to_one")
    df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")

    df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"], errors="coerce")
    df["actual_datetime"] = pd.to_datetime(df["actual_datetime"], errors="coerce")
    df["service_deviation_hours"] = (
        df["actual_datetime"] - df["scheduled_datetime"]
    ).dt.total_seconds() / 3600
    df["abs_service_deviation_hours"] = df["service_deviation_hours"].abs()
    df["on_time_target"] = df["on_time_flag"].astype("boolean").astype("Int64")
    df["scheduled_month"] = df["scheduled_datetime"].dt.month
    df["scheduled_dayofweek"] = df["scheduled_datetime"].dt.dayofweek
    df["scheduled_hour"] = df["scheduled_datetime"].dt.hour

    computed_window = df["abs_service_deviation_hours"] <= 2.0
    stored = df["on_time_flag"].astype("boolean")
    agreement = float((computed_window.astype("boolean") == stored).mean())
    diag = {
        "delivery_rows": int(len(df)),
        "on_time_distribution": {str(k): int(v) for k, v in stored.value_counts().items()},
        "plus_minus_2h_window_agreement": agreement,
        "service_deviation_summary": {
            k: float(v) for k, v in df["service_deviation_hours"].describe().to_dict().items()
        },
    }
    return df, diag


def prepare_supply(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    df = _read_csv(_supply_file(raw_root)).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp_month"] = df["timestamp"].dt.month
    df["timestamp_dayofweek"] = df["timestamp"].dt.dayofweek
    df["timestamp_hour"] = df["timestamp"].dt.hour

    def risk_from_disruption(s: pd.Series) -> pd.Series:
        return pd.cut(
            pd.to_numeric(s, errors="coerce"),
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=["Low Risk", "Moderate Risk", "High Risk"],
            right=False,
        ).astype("string")

    derived = risk_from_disruption(df["disruption_likelihood_score"])
    actual = df["risk_classification"].astype("string")
    agreement = float((derived == actual).mean())
    diag = {
        "rows": int(len(df)),
        "risk_distribution": {str(k): int(v) for k, v in actual.value_counts().items()},
        "risk_class_vs_disruption_threshold_agreement": agreement,
        "risk_threshold_rule": "Low < 0.3; Moderate 0.3 to <0.7; High >= 0.7",
    }
    return df, diag


def _plot_source_targets(tracking: pd.DataFrame, operations: pd.DataFrame, supply: pd.DataFrame) -> None:
    settings.figures_dir.mkdir(parents=True, exist_ok=True)

    clean = tracking["is_delayed"].dropna().astype(int).map({0: "On Time", 1: "Delayed"})
    fig, ax = plt.subplots(figsize=(6, 4))
    clean.value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Tracking Dataset: Clean Delay Labels")
    ax.set_ylabel("Shipments")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "tracking_delay_target_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    operations["on_time_flag"].map({True: "On Time", False: "Outside Service Window"}).value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Operations Dataset: Delivery Service Window")
    ax.set_ylabel("Deliveries")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "operations_on_time_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    operations["service_deviation_hours"].dropna().hist(bins=40, ax=ax)
    ax.axvline(-2, linestyle="--")
    ax.axvline(2, linestyle="--")
    ax.set_title("Delivery Service Deviation (±2h Window)")
    ax.set_xlabel("Actual minus scheduled hours")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "operations_service_deviation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    supply["risk_classification"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Supply-Chain Risk Class Distribution")
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "supply_risk_distribution.png", dpi=160)
    plt.close(fig)

    order = ["Low Risk", "Moderate Risk", "High Risk"]
    data = [
        supply.loc[supply["risk_classification"].eq(c), "disruption_likelihood_score"].dropna().values
        for c in order
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=order)
    ax.axhline(0.3, linestyle="--")
    ax.axhline(0.7, linestyle="--")
    ax.set_title("Risk Class Is Determined by Disruption Likelihood Bands")
    ax.set_ylabel("Disruption likelihood score")
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "risk_class_disruption_bands.png", dpi=160)
    plt.close(fig)


def train_source_models(raw_root: Path) -> dict:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    tracking, tracking_diag = prepare_tracking(raw_root)
    operations, operations_diag = prepare_operations(raw_root)
    supply, supply_diag = prepare_supply(raw_root)
    us_models, us_diag = train_us_performance_models(raw_root)

    # Persist source-specific processed frames for reproducibility and inspection.
    tracking.to_pickle(settings.processed_dir / "tracking_model_frame.pkl")
    operations.to_pickle(settings.processed_dir / "operations_delivery_model_frame.pkl")
    supply.to_pickle(settings.processed_dir / "supply_chain_model_frame.pkl")

    tracking_num = [
        "distance_km", "minimum_km_per_day", "planned_transit_hours", "booking_to_start_hours",
        "start_month", "start_dayofweek", "start_hour",
    ]
    tracking_cat = [
        "market_regular", "vehicle_type", "origin_code", "destination_code", "route_id",
        "customer_id", "supplier_id", "material_shipped", "gps_provider",
    ]
    tracking_class = _fit_classifiers(
        tracking, "is_delayed", tracking_num, tracking_cat,
        "tracking_delay_classifier", "tracking_delay_confusion_matrix.png"
    )
    tracking_reg = _fit_regressors(
        tracking, "delay_hours_target", tracking_num, tracking_cat,
        "tracking_delay_hours_regressor"
    )

    operations_num = [
        "weight_lbs", "pieces", "revenue", "fuel_surcharge", "accessorial_charges",
        "typical_distance_miles", "base_rate_per_mile", "fuel_surcharge_rate", "typical_transit_days",
        "driver_years_experience", "truck_model_year", "truck_acquisition_mileage",
        "truck_tank_capacity_gallons", "trailer_length_feet", "customer_credit_terms_days",
        "customer_annual_revenue_potential", "scheduled_month", "scheduled_dayofweek", "scheduled_hour",
    ]
    operations_cat = [
        "load_type", "booking_type", "route_id", "origin_state", "destination_state",
        "driver_license_state", "driver_home_terminal", "driver_cdl_class",
        "truck_make", "truck_fuel_type", "truck_home_terminal", "trailer_trailer_type",
        "customer_customer_type", "customer_primary_freight_type",
    ]
    operations_class = _fit_classifiers(
        operations, "on_time_target", operations_num, operations_cat,
        "operations_on_time_classifier", "operations_on_time_confusion_matrix.png"
    )
    operations_reg = _fit_regressors(
        operations, "service_deviation_hours", operations_num, operations_cat,
        "operations_service_deviation_regressor"
    )

    # Leakage-safe supply features.  Direct outcome/score fields are deliberately excluded.
    supply_safe_num = [
        "fuel_consumption_rate", "traffic_congestion_level", "warehouse_inventory_level",
        "loading_unloading_time", "handling_equipment_availability", "order_fulfillment_status",
        "weather_condition_severity", "port_congestion_level", "shipping_costs",
        "supplier_reliability_score", "lead_time_days", "historical_demand", "iot_temperature",
        "cargo_condition_status", "customs_clearance_time", "driver_behavior_score",
        "fatigue_monitoring_score", "timestamp_month", "timestamp_dayofweek", "timestamp_hour",
    ]
    supply_risk = _fit_classifiers(
        supply, "risk_classification", supply_safe_num, [],
        "supply_risk_classifier_leakage_safe", "supply_risk_confusion_matrix.png"
    )

    # Predict disruption score from operational conditions; convert to the dataset's documented risk bands downstream.
    supply_disruption = _fit_regressors(
        supply, "disruption_likelihood_score", supply_safe_num, [],
        "supply_disruption_regressor"
    )

    # Route-decision context may legitimately use route_risk and disruption as currently observed conditions.
    supply_delay_features = supply_safe_num + ["route_risk_level", "disruption_likelihood_score"]
    supply_delay_probability = _fit_regressors(
        supply, "delay_probability", supply_delay_features, [],
        "supply_delay_probability_regressor"
    )
    supply_delivery_deviation = _fit_regressors(
        supply, "delivery_time_deviation", supply_delay_features, [],
        "supply_delivery_time_deviation_regressor"
    )

    _plot_source_targets(tracking, operations, supply)

    # Unified knowledge/retrieval layer is still valuable, but no universal supervised target is trained on it.
    unified = discover_and_load(raw_root)
    retrieval_meta = build_retrieval_index(unified)

    metrics = {
        "version": "2.2-four-source-decision-intelligence",
        "design": {
            "unified_retrieval": True,
            "universal_supervised_model": False,
            "reason": "The four datasets have different grains and target semantics; supervised models are trained source-specifically and only models passing explicit deployment gates are exposed for prediction.",
        },
        "unified_rows": int(len(unified)),
        "sources": {str(k): int(v) for k, v in unified["source_dataset"].value_counts().items()},
        "diagnostics": {
            "tracking": tracking_diag,
            "operations": operations_diag,
            "supply_chain": supply_diag,
            "us_performance": us_diag,
        },
        "models": {
            "tracking_delay_classification": tracking_class,
            "tracking_delay_hours_regression": tracking_reg,
            "operations_on_time_classification": operations_class,
            "operations_service_deviation_regression": operations_reg,
            "supply_risk_classification_leakage_safe": supply_risk,
            "supply_disruption_regression": supply_disruption,
            "supply_delay_probability_regression": supply_delay_probability,
            "supply_delivery_time_deviation_regression": supply_delivery_deviation,
            **us_models,
        },
        "retrieval": retrieval_meta,
    }

    text = json.dumps(metrics, indent=2, default=str)
    (settings.metrics_dir / "training_metrics_v2.json").write_text(text, encoding="utf-8")
    # Keep API compatibility with /models/metadata.
    (settings.metrics_dir / "training_metrics.json").write_text(text, encoding="utf-8")
    return metrics


def train_v2(raw_root: Path | None = None) -> dict:
    return train_source_models(raw_root or settings.raw_dir)
