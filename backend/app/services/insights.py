from __future__ import annotations

import json

import pandas as pd
import numpy as np
from backend.app.core.config import settings
from backend.app.ml.training import retrieve
from backend.app.services.ollama import OllamaUnavailable, embed, grounded_answer


def load_df():
    return pd.read_pickle(settings.processed_dir / "unified_logistics.pkl")


def summary():
    df = load_df()
    out = {"rows": len(df), "sources": df.source_dataset.value_counts().to_dict()}
    for c in [
        "delay_hours", "shipping_cost_usd", "transit_days", "weight_kg", "traffic_level", "weather_severity", "route_risk",
        "disruption_likelihood", "delay_probability", "detention_minutes",
    ]:
        if c in df and df[c].notna().any():
            out[c] = {
                "mean": float(pd.to_numeric(df[c], errors="coerce").mean()),
                "median": float(pd.to_numeric(df[c], errors="coerce").median()),
                "max": float(pd.to_numeric(df[c], errors="coerce").max()),
            }
    if "on_time" in df and df.on_time.notna().any():
        out["on_time_rate"] = float(df.on_time.astype(float).mean())
    if "carrier" in df and df.carrier.notna().any():
        out["carriers"] = int(df.carrier.nunique())
    if "shipment_status" in df and df.shipment_status.notna().any():
        out["shipment_status"] = {str(k): int(v) for k, v in df.shipment_status.value_counts().items()}
    return out


def _evidence_from_rows(rows: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    cols = [
        "record_id", "source_dataset", "shipment_id", "route_id", "origin", "destination",
        "carrier", "shipment_status", "weight_kg", "transit_days", "date_quality_flag",
        "delay_hours", "traffic_level", "weather_severity", "shipping_cost_usd", "route_risk",
        "disruption_likelihood", "delay_probability", "risk_class", "detention_minutes",
    ]
    evidence = []
    for (_, row), score in zip(rows.iterrows(), scores):
        item = {column: (None if pd.isna(row[column]) else row[column]) for column in cols if column in row.index}
        item["score"] = float(score)
        evidence.append(item)
    return evidence


def _semantic_rerank(question: str, candidates: list[dict], k: int) -> list[dict] | None:
    """Use qwen3 embeddings to semantically rerank lexical retrieval candidates.

    This two-stage design avoids storing a multi-gigabyte embedding matrix for
    the full historical corpus while still applying local semantic retrieval.
    """
    if not candidates:
        return []
    try:
        candidate_texts = [json.dumps(candidate, ensure_ascii=False, default=str) for candidate in candidates]
        vectors = embed([question, *candidate_texts])
    except OllamaUnavailable:
        return None
    if vectors.ndim != 2 or len(vectors) != len(candidates) + 1:
        return None
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    scores = vectors[1:] @ vectors[0]
    top = np.argsort(-scores)[:k]
    evidence = []
    for position in top:
        item = {**candidates[int(position)], "score": float(scores[position])}
        evidence.append(item)
    return evidence


def answer(question: str, k: int = 8):
    df = load_df()
    candidates = retrieve(question, max(k, settings.ollama_rerank_candidates))
    evidence = _semantic_rerank(question, candidates, k)
    retrieval_engine = "qwen3-embedding:4b hybrid semantic reranking" if evidence is not None else "tf-idf-fallback"
    if evidence is None:
        evidence = candidates[:k]
    ids = {e["record_id"] for e in evidence}
    sub = df[df.record_id.isin(ids)].copy()
    q = question.lower()
    lines = []

    if any(w in q for w in ["delay", "late", "eta"]):
        vals = pd.to_numeric(sub.delay_hours, errors="coerce").dropna()
        if len(vals):
            lines.append(
                f"Among the {len(sub)} most relevant records, median schedule/deviation value is "
                f"{vals.median():.2f} hours and mean is {vals.mean():.2f} hours."
            )
    if any(w in q for w in ["risk", "risky", "disruption"]):
        if "disruption_likelihood" in sub:
            vals = pd.to_numeric(sub.disruption_likelihood, errors="coerce").dropna()
            if len(vals):
                lines.append(f"Relevant records have mean disruption likelihood {vals.mean():.3f}.")
        vals = pd.to_numeric(sub.route_risk, errors="coerce").dropna()
        if len(vals):
            lines.append(f"Relevant records have mean route-risk score {vals.mean():.2f} on the source scale.")
        if sub.risk_class.notna().any():
            lines.append("Risk classes: " + ", ".join(f"{a}={b}" for a, b in sub.risk_class.value_counts().items()))
    if any(w in q for w in ["detention", "dwell"]):
        vals = pd.to_numeric(sub.detention_minutes, errors="coerce").dropna()
        if len(vals):
            lines.append(f"Relevant delivery records average {vals.mean():.1f} detention minutes.")
    if any(w in q for w in ["cost", "price", "expensive"]):
        vals = pd.to_numeric(sub.shipping_cost_usd, errors="coerce").dropna()
        if len(vals):
            lines.append(f"Relevant shipping cost averages ${vals.mean():,.2f} (median ${vals.median():,.2f}).")
    if any(w in q for w in ["carrier", "ups", "fedex", "dhl", "usps", "ontrac", "lasership", "amazon"]):
        if "carrier" in sub and sub.carrier.notna().any():
            carriers = sub.carrier.dropna().astype(str).value_counts().head(5)
            lines.append("Most represented carriers in the retrieved evidence: " + ", ".join(f"{c} ({n})" for c, n in carriers.items()) + ".")
    if any(w in q for w in ["transit", "days", "fastest", "delivery time"]):
        if "transit_days" in sub:
            vals = pd.to_numeric(sub.transit_days, errors="coerce").dropna()
            if len(vals):
                lines.append(f"Relevant US shipment records average {vals.mean():.2f} transit days (median {vals.median():.2f}).")
    if not lines:
        routes = sub.route_id.dropna().astype(str).value_counts().head(3)
        if len(routes):
            lines.append("Most represented relevant lanes: " + ", ".join(f"{r} ({n})" for r, n in routes.items()))
        lines.append("The response is grounded in the retrieved logistics records shown below; no live condition data is assumed.")
    deterministic_answer = " ".join(lines)
    try:
        response = grounded_answer(question, evidence)
        answer_model = settings.ollama_chat_model
    except OllamaUnavailable:
        response = deterministic_answer
        answer_model = None
    return {
        "answer": response,
        "evidence": evidence,
        "grounding": "dataset evidence; Ollama response constrained to retrieved records" if answer_model else "dataset (deterministic fallback)",
        "retrieval_engine": retrieval_engine,
        "answer_model": answer_model,
        "live_conditions_used": False,
    }


def route_recommend(candidates: list[dict]):
    if not candidates:
        return []
    df = pd.DataFrame(candidates).copy()
    # V2 deliberately avoids calling a source-specific model on a mismatched candidate schema.
    # If delay_probability is supplied or produced by a dedicated integration, it is used directly.
    factors = {
        "delay_probability": .34,
        "disruption_likelihood": .18,
        "route_risk": .18,
        "traffic_level": .10,
        "weather_severity": .08,
        "shipping_cost_usd": .07,
        "distance_km": .05,
    }

    def norm(s):
        s = pd.to_numeric(s, errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            return (s - lo) / (hi - lo)
        return pd.Series([0.5] * len(s), index=s.index)

    score = pd.Series(0.0, index=df.index)
    used = []
    for c, w in factors.items():
        if c in df and df[c].notna().any():
            score += w * norm(df[c])
            used.append(c)
    if "supplier_reliability" in df and df.supplier_reliability.notna().any():
        score += .08 * (1 - norm(df.supplier_reliability))
        used.append("supplier_reliability")

    df["smart_route_score"] = score
    df["rank"] = df.smart_route_score.rank(method="dense").astype(int)
    return [
        {
            **r,
            "score_factors_used": used,
            "interpretation": "lower score is preferred",
            "live_conditions_used": False,
        }
        for r in df.sort_values("smart_route_score").to_dict("records")
    ]
