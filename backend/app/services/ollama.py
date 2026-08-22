"""Local Ollama integration for grounded retrieval and explanation.

This module never calls the platform's sklearn prediction models.  Those models
remain the authoritative source for prediction endpoints; Ollama only retrieves
dataset evidence and writes an explanation constrained to that evidence.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np

from backend.app.core.config import settings


class OllamaUnavailable(RuntimeError):
    """Raised when local Ollama cannot provide the requested capability."""


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)


def _request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.ollama_enabled:
        raise OllamaUnavailable("Ollama is disabled (set OLLAMA_ENABLED=true to enable it).")
    try:
        with _client() as client:
            response = client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OllamaUnavailable(f"Local Ollama request failed: {exc}") from exc


def embed(texts: list[str]) -> np.ndarray:
    """Embed text using Ollama's current batch embedding endpoint."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    data = _request("/api/embed", {"model": settings.ollama_embedding_model, "input": texts})
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise OllamaUnavailable("Ollama returned an invalid embedding response.")
    return np.asarray(vectors, dtype=np.float32)


def is_available() -> bool:
    if not settings.ollama_enabled:
        return False
    try:
        # Health must remain responsive even when a local Ollama process is hung.
        with httpx.Client(base_url=settings.ollama_base_url, timeout=min(settings.ollama_timeout_seconds, 5.0)) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            names = {model.get("name") for model in response.json().get("models", [])}
        return settings.ollama_chat_model in names and settings.ollama_embedding_model in names
    except (httpx.HTTPError, ValueError):
        return False


def grounded_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    """Ask Gemma for a concise answer using only supplied retrieval evidence."""
    context = json.dumps(evidence, ensure_ascii=False, default=str)
    prompt = f"""You are Ask Logistics, a decision-support assistant. Answer the user's question only from the supplied retrieved dataset records. Do not invent facts, live conditions, policies, or predictions. If the records are insufficient, say so clearly. Existing sklearn model predictions are authoritative and must not be recalculated or contradicted. Cite record_id values inline where useful. Keep the answer concise.\n\nQuestion: {question}\n\nRetrieved evidence (JSON):\n{context}"""
    data = _request(
        "/api/generate",
        {"model": settings.ollama_chat_model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
    )
    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaUnavailable("Ollama returned an empty chat response.")
    return answer.strip()
