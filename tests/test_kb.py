"""知识库导入与命中阈值。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from digital_life.kb import chunk_markdown, import_markdown
from digital_life.memory import MemoryStore, _hit_threshold

DOC = """# 龙族：卡塞尔之门

游戏简介。

## 绘梨衣阵容

绘梨衣是水系输出，前排配源稚生。

## 新手建议

攒够三十抽再决定主 C。
"""


def test_chunk_markdown_splits_by_heading():
    chunks = chunk_markdown(DOC)
    titles = [title for title, _ in chunks]
    assert "绘梨衣阵容" in titles
    assert "新手建议" in titles
    assert all(len(body) <= 700 for _, body in chunks)


def test_chunk_markdown_splits_long_section():
    long_text = "## 长文\n\n" + ("这是一段很长的说明。" * 120)
    chunks = chunk_markdown(long_text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(body) <= 200 for _, body in chunks)


def test_import_is_idempotent(tmp_path):
    doc = tmp_path / "game.md"
    doc.write_text(DOC, encoding="utf-8")

    async def run():
        mem = MemoryStore(
            tmp_path / "m.db", embedder=lambda text: [0.1] * 8, embedding_dim=8
        )
        first = await import_markdown(mem, doc)
        second = await import_markdown(mem, doc)
        mem.close()
        return first, second

    first, second = asyncio.run(run())
    assert first["written"] == first["chunks"] > 0
    assert second["written"] == 0
    assert second["skipped"] == second["chunks"]


def test_hit_threshold_follows_embedder():
    hash_like = SimpleNamespace(embedder_name="hash")
    bge_like = SimpleNamespace(embedder_name="ollama:bge-m3")

    assert _hit_threshold(hash_like, 0.4) == 0.4
    assert _hit_threshold(bge_like, 0.4) == 0.82
    assert _hit_threshold(bge_like, 0.55) == 0.82
