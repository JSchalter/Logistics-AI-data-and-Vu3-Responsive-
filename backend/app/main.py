from __future__ import annotations

import json
from datetime import date, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.ml.source_training import train_v2
from backend.app.services.insights import summary, answer, route_recommend
from backend.app.services.ollama import is_available
from backend.app.services.decision_engine import (
    carrier_analytics,
    recommend_carriers,
    tracking_predict,
    us_anomalies,
    us_predict,
)

app = FastAPI(title="Logistics AI Intelligence Platform", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskReq(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=8, ge=1, le=25)


class RouteReq(BaseModel):
    candidates: list[dict]


class TrackingPredictReq(BaseModel):
    distance_km: float | None = Field(default=None, ge=0)
    minimum_km_per_day: float | None = Field(default=None, ge=0)
    planned_transit_hours: float | None = None
    booking_to_start_hours: float | None = None
    trip_start_date: datetime | None = None
    planned_eta: datetime | None = None
    booking_date: datetime | None = None
    start_month: int | None = Field(default=None, ge=1, le=12)
    start_dayofweek: int | None = Field(default=None, ge=0, le=6)
    start_hour: int | None = Field(default=None, ge=0, le=23)
    market_regular: str | None = None
    vehicle_type: str | None = None
    origin_code: str | None = None
    destination_code: str | None = None
    route_id: str | None = None
    customer_id: str | int | None = None
    supplier_id: str | int | None = None
    material_shipped: str | None = None
    gps_provider: str | None = None


class USPredictReq(BaseModel):
    origin_warehouse: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    carrier: str = Field(min_length=1)
    shipment_date: date
    weight_kg: float = Field(gt=0)
    distance_miles: float = Field(gt=0)


class CarrierRecommendReq(BaseModel):
    origin_warehouse: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    shipment_date: date
    weight_kg: float = Field(gt=0)
    distance_miles: float = Field(gt=0)
    objective: Literal["cheapest", "fastest", "most_reliable", "balanced"] = "balanced"


@app.get("/health")
def health():
    metrics = settings.metrics_dir / "training_metrics_v2.json"
    return {
        "status": "ok",
        "data_ready": (settings.processed_dir / "unified_logistics.pkl").exists(),
        "mode": settings.data_mode,
        "training_architecture": "source-aware-v2.2-four-source",
        "metrics_ready": metrics.exists(),
        "us_performance_ready": (settings.processed_dir / "us_performance_model_frame.pkl").exists(),
        "ollama": {
            "enabled": settings.ollama_enabled,
            "reachable_with_configured_models": is_available(),
            "chat_model": settings.ollama_chat_model,
            "embedding_model": settings.ollama_embedding_model,
            "semantic_index_ready": settings.ollama_semantic_manifest_path.exists(),
            "semantic_retrieval_mode": "qwen3 hybrid semantic reranking (no corpus-wide vector index)",
        },
    }


@app.post("/train")
def train():
    try:
        if settings.data_mode != "real":
            raise ValueError("V2.2 /train requires DATA_MODE=real and all four Kaggle datasets under data/raw.")
        return train_v2(settings.raw_dir)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/analytics/summary")
def analytics_summary():
    try:
        return summary()
    except Exception as e:
        raise HTTPException(503, str(e))


@app.post("/ask")
def ask(req: AskReq):
    try:
        return answer(req.question, req.top_k)
    except Exception as e:
        raise HTTPException(503, str(e))


@app.post("/tracking/predict")
def tracking_prediction(req: TrackingPredictReq):
    try:
        return tracking_predict(req.model_dump(mode="json"))
    except Exception as e:
        raise HTTPException(503, str(e))


@app.post("/us-performance/predict")
def us_performance_prediction(req: USPredictReq):
    try:
        return us_predict(req.model_dump(mode="json"))
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/carriers/analytics")
def carriers_analytics():
    try:
        return carrier_analytics()
    except Exception as e:
        raise HTTPException(503, str(e))


@app.post("/carriers/recommend")
def carriers_recommend(req: CarrierRecommendReq):
    try:
        return recommend_carriers(req.model_dump(mode="json"))
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/us-performance/anomalies")
def performance_anomalies(limit: int = Query(default=25, ge=1, le=200)):
    try:
        return us_anomalies(limit)
    except Exception as e:
        raise HTTPException(503, str(e))


# Legacy generic route-ranking endpoint retained for backward compatibility.
@app.post("/routes/recommend")
def routes(req: RouteReq):
    return {"recommendations": route_recommend(req.candidates), "live_conditions_used": False}


@app.get("/models/metadata")
def model_metadata():
    mp = settings.metrics_dir / "training_metrics_v2.json"
    if not mp.exists():
        mp = settings.metrics_dir / "training_metrics.json"
    return json.loads(mp.read_text()) if mp.exists() else {"trained": False, "models": {}}


@app.get("/figures/list")
def figures():
    return {"figures": [p.name for p in sorted(settings.figures_dir.glob("*.png"))]}


app.mount("/figures", StaticFiles(directory=settings.figures_dir), name="figures")
