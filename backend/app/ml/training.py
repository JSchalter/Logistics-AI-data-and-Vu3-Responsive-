from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import settings


def build_documents(df: pd.DataFrame) -> pd.Series:
    fields = [
        "shipment_id", "route_id", "origin", "destination", "vehicle_type", "transport_mode",
        "carrier", "shipment_status", "weight_kg", "transit_days", "date_quality_flag",
        "planned_eta", "actual_eta", "delay_hours", "on_time", "traffic_level", "weather_severity",
        "weather_condition", "shipping_cost_usd", "route_risk", "disruption_likelihood",
        "delay_probability", "risk_class", "supplier_reliability", "driver_behavior_score",
        "fatigue_score", "detention_minutes", "eta_variation_hours", "delivery_time_deviation",
        "order_fulfillment_score", "lead_time_days", "source_dataset",
    ]

    def one(r):
        parts = []
        for f in fields:
            if f not in r.index:
                continue
            v = r.get(f)
            if pd.notna(v):
                parts.append(f"{f.replace('_', ' ')}: {v}")
        if "free_text" in r.index and pd.notna(r.get("free_text")):
            parts.append(str(r.get("free_text"))[:1200])
        return ". ".join(parts)

    return df.apply(one, axis=1)


def build_retrieval_index(df: pd.DataFrame) -> dict:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    docs = build_documents(df).fillna("")
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=60000, sublinear_tf=True
    )
    mat = vec.fit_transform(docs)
    joblib.dump(vec, settings.index_dir / "retrieval_vectorizer.joblib")
    joblib.dump(mat, settings.index_dir / "retrieval_matrix.joblib")
    df.to_pickle(settings.processed_dir / "unified_logistics.pkl")
    return {
        "documents": int(len(df)),
        "vocabulary_size": int(len(vec.vocabulary_)),
        "matrix_shape": [int(mat.shape[0]), int(mat.shape[1])],
    }


def train_all(df: pd.DataFrame) -> dict:
    """Compatibility wrapper.

    V2 intentionally does not train one universal supervised model over the concatenated datasets.
    This wrapper only builds the unified retrieval index.  Use scripts/train.py or train_v2()
    for source-specific supervised training.
    """
    meta = build_retrieval_index(df)
    out = {
        "version": "2.0-retrieval-only-wrapper",
        "rows": int(len(df)),
        "sources": df.source_dataset.value_counts().to_dict(),
        "retrieval": meta,
        "note": "Supervised models are source-specific in V2; run backend.app.ml.source_training.train_v2.",
    }
    (settings.metrics_dir / "retrieval_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def retrieve(query: str, k: int = 8):
    df = pd.read_pickle(settings.processed_dir / "unified_logistics.pkl")
    vec = joblib.load(settings.index_dir / "retrieval_vectorizer.joblib")
    mat = joblib.load(settings.index_dir / "retrieval_matrix.joblib")
    q = vec.transform([query])
    scores = cosine_similarity(q, mat).ravel()
    idx = np.argsort(-scores)[:k]
    cols = [
        "record_id", "source_dataset", "shipment_id", "route_id", "origin", "destination",
        "carrier", "shipment_status", "weight_kg", "transit_days", "date_quality_flag",
        "delay_hours", "traffic_level", "weather_severity", "shipping_cost_usd", "route_risk",
        "disruption_likelihood", "delay_probability", "risk_class", "detention_minutes",
    ]
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        item = {}
        for c in cols:
            if c in row.index:
                v = row[c]
                item[c] = None if pd.isna(v) else v
        item["score"] = float(scores[i])
        out.append(item)
    return out
