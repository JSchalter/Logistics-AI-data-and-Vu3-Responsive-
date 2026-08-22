import numpy as np

from backend.app.services import ollama
from backend.app.services import insights


def test_embed_uses_configured_ollama_embedding_model(monkeypatch):
    captured = {}

    def fake_request(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}

    monkeypatch.setattr(ollama, "_request", fake_request)
    vectors = ollama.embed(["first", "second"])

    assert captured["path"] == "/api/embed"
    assert captured["payload"]["model"] == ollama.settings.ollama_embedding_model
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)


def test_grounded_answer_sends_evidence_and_disables_creativity(monkeypatch):
    captured = {}

    def fake_request(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"response": "Evidence-grounded response."}

    monkeypatch.setattr(ollama, "_request", fake_request)
    response = ollama.grounded_answer("What delayed shipment A?", [{"record_id": "A", "delay_hours": 4}])

    assert response == "Evidence-grounded response."
    assert captured["path"] == "/api/generate"
    assert captured["payload"]["model"] == ollama.settings.ollama_chat_model
    assert captured["payload"]["options"]["temperature"] == 0
    assert '"record_id": "A"' in captured["payload"]["prompt"]


def test_semantic_rerank_returns_qwen_ranked_candidates(monkeypatch):
    def fake_embed(_inputs):
        # Query aligns with the second candidate.
        return np.asarray([[1.0, 0.0], [0.1, 1.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(insights, "embed", fake_embed)
    candidates = [{"record_id": "first", "score": 0.9}, {"record_id": "second", "score": 0.8}]

    ranked = insights._semantic_rerank("question", candidates, 1)

    assert ranked[0]["record_id"] == "second"
    assert ranked[0]["score"] == 1.0
