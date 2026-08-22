from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


# Load project-local configuration for scripts, uvicorn, and tests without
# overriding values explicitly supplied by the operating system.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    data_root: Path = project_root / "data"
    raw_dir: Path = data_root / "raw"
    processed_dir: Path = data_root / "processed"
    demo_dir: Path = data_root / "demo"
    artifacts_dir: Path = project_root / "artifacts"
    models_dir: Path = artifacts_dir / "models"
    figures_dir: Path = artifacts_dir / "figures"
    metrics_dir: Path = artifacts_dir / "metrics"
    index_dir: Path = artifacts_dir / "index"
    ollama_semantic_manifest_path: Path = index_dir / "ollama_qwen3_semantic_rerank.json"
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    data_mode: str = os.getenv("DATA_MODE", "real").lower()
    cors_origins: tuple[str, ...] = tuple(
        x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip()
    )
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:12b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    ollama_rerank_candidates: int = int(os.getenv("OLLAMA_RERANK_CANDIDATES", "16"))


settings = Settings()
for _path in (
    settings.raw_dir,
    settings.processed_dir,
    settings.demo_dir,
    settings.models_dir,
    settings.figures_dir,
    settings.metrics_dir,
    settings.index_dir,
):
    _path.mkdir(parents=True, exist_ok=True)
