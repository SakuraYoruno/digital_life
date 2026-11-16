from __future__ import annotations

import asyncio
import math

import pytest

from digital_life.embedding import OllamaEmbedder
from digital_life.memory import MemoryStore


class FakeResponse:
    def __init__(self, data, status: int = 200) -> None:
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeClient:
    def __init__(self) -> None:
        self.posts: list[str] = []
        self.payloads: list[dict] = []

    def get(self, url: str):
        return FakeResponse(
            {"models": [{"name": "bge-m3:latest"}]}
        )

    def post(self, url: str, json=None):
        self.posts.append(url)
        self.payloads.append(dict(json or {}))
        if url.endswith("/api/embed"):
            return FakeResponse({"embeddings": [[0.1] * 1024]})
        return FakeResponse({"embedding": [0.2] * 1024})

    def close(self) -> None:
        pass


def test_ollama_embedder_available_and_normalized():
    client = FakeClient()
    embedder = OllamaEmbedder("bge-m3", client=client)
    assert embedder.available()
    vec = embedder("今天想你了")
    assert len(vec) == 1024
    assert math.sqrt(sum(x * x for x in vec)) == pytest.approx(1.0)
    assert client.posts[-1].endswith("/api/embed")
    assert client.payloads[-1]["keep_alive"] == "5m"


def test_default_embed_model_is_bge_m3_for_vram():
    from digital_life.embedding import DEFAULT_EMBED_MODEL

    assert DEFAULT_EMBED_MODEL == "bge-m3"
    embedder = OllamaEmbedder(client=FakeClient())
    assert embedder.dim == 1024
    assert embedder.keep_alive == "5m"


def test_ollama_embedder_legacy_fallback():
    class LegacyClient(FakeClient):
        def post(self, url, json=None):
            self.posts.append(url)
            if url.endswith("/api/embed"):
                return FakeResponse({}, status=404)
            return FakeResponse({"embedding": [0.3] * 1024})

    embedder = OllamaEmbedder("bge-m3", client=LegacyClient())
    assert len(embedder("你好")) == 1024


class TinyEmbedder:
    name = "tiny"
    dim = 8

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * 8
        for i, ch in enumerate(text[:8]):
            vec[i % 8] += ord(ch) % 7 - 3
        return vec or [1.0] + [0.0] * 7


def test_custom_embedding_dim_uses_vec0_and_rebuilds(tmp_path):
    async def run():
        path = tmp_path / "m.db"
        mem = MemoryStore(path)
        await mem.write("u1", "episodic", "用户喜欢玩龙卡")
        assert "192d" in mem.vector_backend
        mem.close()

        mem2 = MemoryStore(path, embedder=TinyEmbedder())
        assert "tiny 8d" in mem2.vector_backend
        assert mem2._conn.execute(
            "SELECT COUNT(*) FROM vec_memories"
        ).fetchone()[0] == 1
        hits = await mem2.search("u1", "龙卡")
        assert hits and hits[0].text == "用户喜欢玩龙卡"
        mem2.close()

    asyncio.run(run())
