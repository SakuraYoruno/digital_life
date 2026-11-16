"""Ollama embedding 适配器。

sqlite-vec 只负责索引/KNN，不产生向量；这个模块把 Ollama 的 embedding
接口接到 MemoryStore.embedder 上。

默认 bge-m3（1024 维，中文强，实测显存约 689MB）。
qwen3-embedding:0.6b 文件更小，但 f16 权重常驻显存约 1.2GB，
如果主要看显存，bge-m3 更划算；nomic-embed-text（335MB）/all-minilm（27MB）
更省显存，但中文语义明显更弱。
"""

from __future__ import annotations

import math
import os
from typing import Any

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
KNOWN_DIMS = {
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding": 1024,
    "bge-m3": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "paraphrase-multilingual": 768,
}


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return list(vec)
    return [v / norm for v in vec]


class OllamaEmbedder:
    """同步 embedding 客户端。

    MemoryStore 的写/检索路径是同步 SQLite 调用，这里用同步 httpx 保持接口简单；
    Ollama 在本机时单次 embedding 通常几十毫秒，不会明显拖慢主循环。
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        base_url: str | None = None,
        dim: int | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("DIGITAL_LIFE_OLLAMA_URL")
            or DEFAULT_OLLAMA_URL
        ).rstrip("/")
        self.dim = int(dim or KNOWN_DIMS.get(model, 1024))
        self.timeout = timeout
        self.keep_alive = (
            keep_alive
            or os.getenv("DIGITAL_LIFE_EMBED_KEEP_ALIVE")
            or "5m"
        )
        self.name = f"ollama:{model}"
        if client is not None:
            self._client = client
        else:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "需要 httpx 才能连接 Ollama：python -m pip install httpx"
                ) from exc
            self._client = httpx.Client(timeout=timeout, trust_env=False)

    # -- 探测 ---------------------------------------------------------
    def available(self) -> bool:
        """Ollama 在跑且模型存在。失败一律返回 False，便于 auto 回退。"""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            names = {
                str(item.get("name") or item.get("model") or "").split(":")[0]
                for item in models
                if isinstance(item, dict)
            }
            return self.model.split(":")[0] in names
        except Exception:
            return False

    # -- 接口 ---------------------------------------------------------
    def _request(self, text: str) -> list[float]:
        # 新版 /api/embed
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                    "keep_alive": self.keep_alive,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings")
                if embeddings:
                    return [float(x) for x in embeddings[0]]
        except Exception:
            pass
        # 兼容旧版 /api/embeddings
        resp = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text, "keep_alive": self.keep_alive},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama 没有返回 embedding：{data}")
        return [float(x) for x in embedding]

    def __call__(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * self.dim
        vec = _normalize(self._request(text))
        if len(vec) != self.dim:
            raise ValueError(
                f"embedding 维度 {len(vec)} 与配置的 {self.dim} 不一致；"
                f"请设置 --embed-dim 或 DIGITAL_LIFE_EMBED_DIM"
            )
        return vec

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "dim": self.dim,
            "keep_alive": self.keep_alive,
            "available": self.available(),
        }


def make_embedder(
    backend: str = "auto",
    *,
    model: str | None = None,
    dim: int | None = None,
    base_url: str | None = None,
) -> OllamaEmbedder | None:
    """backend: auto | hash | ollama。

    auto 只在 Ollama 正在运行且模型已存在时启用，否则返回 None（内置哈希）。
    """
    backend = (backend or "auto").lower()
    if backend == "hash":
        return None
    model = model or os.getenv("DIGITAL_LIFE_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    dim = dim or (
        int(os.environ["DIGITAL_LIFE_EMBED_DIM"])
        if os.getenv("DIGITAL_LIFE_EMBED_DIM")
        else None
    )
    embedder = OllamaEmbedder(model, base_url=base_url, dim=dim)
    if backend == "ollama":
        return embedder
    if embedder.available():
        return embedder
    embedder.close()
    return None


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_URL",
    "KNOWN_DIMS",
    "OllamaEmbedder",
    "make_embedder",
]
