from __future__ import annotations

import asyncio

import pytest

from digital_life.constants import estimate_tokens
from digital_life.llm import MockLLM
from digital_life.memory import (
    MemoryStore,
    RawPercept,
    WorkingMemory,
    format_recalled,
    maybe_write_episodic,
    perceive,
    recall_full,
    roll_summary,
)
from digital_life.personas import Relation


class Clock:
    def __init__(self, value: float = 1_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_fts_write_and_search(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        await mem.write("u1", "episodic", "用户喜欢玩龙卡", importance=0.8)
        await mem.write("u1", "episodic", "今天一起看了电影", importance=0.5)
        hits = await mem.search("u1", "龙卡")
        assert hits
        assert any("龙卡" in h.text for h in hits)
        assert hits[0].id > 0
        mem.close()

    asyncio.run(run())


def test_uid_isolation(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        await mem.write("u1", "episodic", "阿暖的秘密是喜欢龙卡")
        await mem.write("u2", "episodic", "小冷的秘密是讨厌下雨")
        assert all("龙卡" not in h.text for h in await mem.search("u2", "龙卡", k=10))
        assert all("小冷" not in h.text for h in await mem.search("u1", "小冷", k=10))
        mem.close()

    asyncio.run(run())


def test_ttl_and_delete_expired(tmp_path):
    async def run():
        clock = Clock()
        mem = MemoryStore(tmp_path / "m.db", clock=clock)
        await mem.write("u1", "knowledge", "限时知识", ttl=10, now=clock())
        assert await mem.search("u1", "限时知识", now=clock())
        clock.advance(11)
        assert not await mem.search("u1", "限时知识", now=clock())
        deleted = await mem.delete_expired("u1", now=clock())
        assert deleted == 1
        mem.close()

    asyncio.run(run())


def test_update_reindexes(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        mem_id = await mem.write("u1", "semantic", "用户喜欢猫")
        assert await mem.search("u1", "猫")
        await mem.update(mem_id, text="用户喜欢狗")
        assert await mem.search("u1", "狗")
        assert not await mem.search("u1", "猫")
        mem.close()

    asyncio.run(run())


def test_find_similar(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        await mem.write("u1", "episodic", "和用户一起玩龙卡，很开心")
        await mem.write("u1", "episodic", "和用户一起打龙卡，超开心")
        await mem.write("u1", "episodic", "今天下雨没出门")
        base = (await mem.search("u1", "龙卡"))[0]
        similar = await mem.find_similar(base, threshold=0.5)
        assert similar
        assert any("龙卡" in h.text for h in similar)
        mem.close()

    asyncio.run(run())


def test_recall_full_merges_layers_and_dedups(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        await mem.write("u1", "episodic", "用户说想玩龙卡", importance=0.9)
        await mem.write("u1", "semantic", "用户喜欢卡牌游戏", importance=0.8)
        await mem.write("__knowledge__", "knowledge", "龙卡：一款卡牌游戏", importance=0.9)
        hits = await recall_full(mem, "u1", "龙卡", mood="开心", k=8)
        assert hits
        assert len({h.id for h in hits}) == len(hits)
        assert any(h.kind == "episodic" for h in hits)
        assert any(h.kind == "knowledge" for h in hits)
        mem.close()

    asyncio.run(run())


def test_working_memory_rolls_and_compresses():
    batches: list[list[str]] = []

    async def compress(turns):
        batches.append([t.text for t in turns])

    async def run():
        wm = WorkingMemory(
            max_turns=5,
            max_tokens=400,
            compress_batch=2,
            on_compress=compress,
        )
        for i in range(13):
            wm.add("user", f"消息{i}", "平静")
        assert len(wm.turns) <= 5
        await wm.flush()
        assert batches
        assert sum(len(b) for b in batches) >= 8
        assert not wm.pending

    asyncio.run(run())


def test_working_memory_to_text_budget_and_ends():
    wm = WorkingMemory(max_turns=50, max_tokens=10_000)
    for i in range(20):
        wm.add("user", f"这是第{i}条比较长的消息" * 5, "平静")
    text = wm.to_text(max_tokens=80)
    assert estimate_tokens(text) <= 80
    assert "省略" in text
    # 最近一条一定保留
    assert "第19条" in text


def test_finalize_clears_turns():
    async def run():
        batches = []

        async def compress(turns):
            batches.append(len(turns))

        wm = WorkingMemory(max_turns=5, on_compress=compress)
        for i in range(4):
            wm.add("user", f"m{i}")
        wm.finalize()
        assert not wm.turns
        await wm.flush()
        assert batches == [4]

    asyncio.run(run())


def test_roll_summary_updates_relation():
    async def run():
        rel = Relation()
        wm = WorkingMemory()
        wm.add("user", "我今天很难过")
        wm.add("assistant", "怎么了")
        await roll_summary(MockLLM(1), rel, list(wm.turns))
        assert rel.session_summary
        assert len(rel.session_summary) <= 100

    asyncio.run(run())


def test_perceive_extracts_entities_and_cleans():
    async def run():
        percept = await perceive(MockLLM(1), "@小明 我最近在玩龙卡", "u1")
        assert "龙卡" in percept.entities
        assert "@小明" not in percept.clean_text

    asyncio.run(run())


def test_maybe_write_episodic_threshold(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        low = RawPercept("u1", "嗯", "嗯", [], 0.0)
        assert await maybe_write_episodic(MockLLM(1), mem, "u1", low, "哦", 0.1) is None
        high = RawPercept("u1", "我明天帮你带饭", "我明天帮你带饭", [], 0.0)
        mem_id = await maybe_write_episodic(
            MockLLM(1), mem, "u1", high, "好", 0.2
        )
        assert mem_id is not None
        hits = await mem.recent("u1", kind="episodic")
        assert hits
        mem.close()

    asyncio.run(run())


def test_format_recalled_budget():
    from digital_life.memory import MemoryHit

    hits = [
        MemoryHit(i, "u", "episodic", "很长的记忆" * 20, 0.9, 0.8, 0.0)
        for i in range(10)
    ]
    text = format_recalled(hits, max_tokens=30)
    assert estimate_tokens(text) <= 30


def test_sqlite_vec_backend_indexes_update_and_delete(tmp_path):
    pytest.importorskip("sqlite_vec")

    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        assert mem.vector_backend.startswith("sqlite-vec")
        mem_id = await mem.write("u1", "episodic", "用户喜欢玩龙卡")
        await mem.write("u2", "episodic", "小冷喜欢下雨")
        assert mem._conn.execute(
            "SELECT COUNT(*) FROM vec_memories"
        ).fetchone()[0] == 2
        hits = await mem.search("u1", "龙卡")
        assert hits and hits[0].text == "用户喜欢玩龙卡"

        await mem.update(mem_id, text="用户喜欢玩原神")
        assert mem._conn.execute(
            "SELECT COUNT(*) FROM vec_memories"
        ).fetchone()[0] == 2
        assert (await mem.search("u1", "原神"))[0].text == "用户喜欢玩原神"

        await mem.delete(mem_id)
        assert mem._conn.execute(
            "SELECT COUNT(*) FROM vec_memories"
        ).fetchone()[0] == 1
        mem.close()

    asyncio.run(run())


def test_hash_vector_fallback_still_searches(tmp_path):
    async def run():
        mem = MemoryStore(tmp_path / "m.db")
        await mem.write("u1", "episodic", "用户喜欢玩龙卡")
        mem._vec_ready = False
        mem.vector_backend = "hash-cosine"
        hits = await mem.search("u1", "龙卡")
        assert hits and hits[0].text == "用户喜欢玩龙卡"
        mem.close()

    asyncio.run(run())


def test_same_dim_embedder_switch_rebuilds_vectors(tmp_path):
    class _Tiny:
        name = "tiny"
        dim = 8

        def __call__(self, text: str) -> list[float]:
            vec = [0.0] * 8
            for i, ch in enumerate(text[:8]):
                vec[i % 8] += ord(ch) % 7 - 3
            return vec

    class OtherTiny(_Tiny):
        name = "tiny-b"

    async def run():
        path = tmp_path / "m.db"
        mem = MemoryStore(path, embedder=_Tiny())
        await mem.write("u1", "episodic", "用户喜欢玩龙卡")
        assert mem._conn.execute(
            "SELECT value FROM kv WHERE key='vec_embedder'"
        ).fetchone()[0] == "tiny"
        mem.close()

        mem2 = MemoryStore(path, embedder=OtherTiny())
        assert mem2._conn.execute(
            "SELECT value FROM kv WHERE key='vec_embedder'"
        ).fetchone()[0] == "tiny-b"
        assert mem2._conn.execute(
            "SELECT COUNT(*) FROM vec_memories"
        ).fetchone()[0] == 1
        assert (await mem2.search("u1", "龙卡"))[0].text == "用户喜欢玩龙卡"
        mem2.close()

    asyncio.run(run())


def test_vec_table_uses_small_chunks(tmp_path):
    """vec0 默认 chunk=1024 条向量（1024 维下 4MB/人），必须显式调小。"""
    from digital_life.memory import MemoryStore

    store = MemoryStore(tmp_path / "m.db", embedder=lambda text: [0.1] * 8, embedding_dim=8)
    try:
        if not store._vec_ready:
            return  # 没装 sqlite-vec 就跳过
        sql = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='vec_memories'"
        ).fetchone()[0]
        assert "chunk_size" in sql
        assert store.vec_chunk_size <= 128
    finally:
        store.close()
