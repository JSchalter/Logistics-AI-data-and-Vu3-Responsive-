"""Validate local semantic RAG without embedding the entire historical corpus.

Ask Logistics uses TF-IDF for candidate recall and qwen3 embeddings to rerank
only those candidates. A full 2,560-dimension corpus index would be large and
slow, so it is deliberately not built.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.services.ollama import OllamaUnavailable, embed  # noqa: E402


def main() -> None:
    vector = embed(["Logistics semantic retrieval readiness check."])
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "hybrid-tfidf-candidate-retrieval-plus-qwen3-semantic-rerank",
        "embedding_model": settings.ollama_embedding_model,
        "embedding_dimensions": int(vector.shape[1]),
        "rerank_candidates": settings.ollama_rerank_candidates,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    settings.ollama_semantic_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Hybrid qwen3 semantic RAG is ready; no corpus-wide embedding index was created.")
    print(f"Wrote readiness manifest to {settings.ollama_semantic_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except OllamaUnavailable as exc:
        raise SystemExit(str(exc))
