from __future__ import annotations

import asyncio
import random

from digital_life.engine import DigitalLife, EngineConfig
from digital_life.io import CollectingIO
from digital_life.llm import MockLLM
from digital_life.memory import MemoryStore
from digital_life.persistence import StateStore
from digital_life.personas import LifeState, Relation, xiami_state


class Clock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class BadChatLLM(MockLLM):
    def _chat(self, prompt: str) -> str:
        return "在吗"


def _engine(tmp_path, *, big=None, clock=None, mem_clock=None):
    clock = clock or Clock()
    mem = MemoryStore(":memory:", clock=mem_clock or clock)
    store = StateStore(":memory:", clock=clock)
    io = CollectingIO(clock=clock)
    engine = DigitalLife(
        xiami_state(),
        MockLLM(1),
        big or MockLLM(2),
        mem,
        io,
        state_store=store,
        config=EngineConfig(
            clock=clock,
            rng=random.Random(7),
            session_timeout=7200,
            thought_probability=0.0,
        ),
    )
    return engine, io, store, clock, mem


def test_on_message_updates_state_and_io(tmp_path):
    async def run():
        engine, io, store, clock, mem = _engine(tmp_path)
        result = await engine.on_message("u1", "你今天真可爱，我最喜欢你了", now=clock())
        assert result.reply
        assert not result.blocked
        assert io.texts("u1")
        assert engine.st.drive["approval"] > 0.5
        rel = engine.st.rel("u1")
        assert rel.session.n_turns == 1
        assert rel.session_active
        assert engine.stats.ctx_tokens[-1] <= 8000
        assert engine.stats.p_samples
        await engine.close()
        mem.close()

    asyncio.run(run())


def test_safety_gate_blocks_and_does_not_send(tmp_path):
    async def run():
        engine, io, store, clock, mem = _engine(tmp_path, big=BadChatLLM(2))
        result = await engine.on_message("u1", "你好呀", now=clock())
        assert result.blocked
        assert result.reply == ""
        assert io.texts("u1") == []
        assert engine.stats.safety_gate_hits == 1
        await engine.close()
        mem.close()

    asyncio.run(run())


def test_session_end_updates_affinity_and_summary(tmp_path):
    async def run():
        engine, io, store, clock, mem = _engine(tmp_path)
        await engine.on_message("u1", "我今天好开心，谢谢你陪我", now=clock())
        clock.advance(600)
        await engine.on_message("u1", "明天也一起玩吧", now=clock())
        before = engine.st.rel("u1").affinity
        clock.advance(3 * 3600)
        tick = await engine.life_tick(dt=3 * 3600, now=clock())
        assert tick.dt == 3 * 3600
        assert engine.stats.session_endings >= 1
        rel = engine.st.rel("u1")
        assert not rel.session_active
        assert rel.affinity > before
        assert rel.session_summary
        assert engine.stats.affinity_history
        await engine.close()
        mem.close()

    asyncio.run(run())


def test_life_tick_sends_proactive_when_idle(tmp_path):
    async def run():
        clock = Clock()
        engine, io, store, clock, mem = _engine(tmp_path, clock=clock)
        rel = engine.st.rel("u1")
        rel.last_interact = clock() - 6 * 3600
        rel.affinity = 0.5
        engine.st.drive["bond"] = 0.2
        tick = await engine.life_tick(dt=60, now=clock())
        assert tick.sent
        assert engine.stats.proactive_count == 1
        assert io.texts("u1")
        assert engine.st.rel("u1").proactive_awaiting_reply
        await engine.close()
        mem.close()

    asyncio.run(run())


def test_state_roundtrip_and_store(tmp_path):
    st = xiami_state()
    st.P = 0.42
    st.drive["bond"] = 0.11
    rel = st.rel("u1")
    rel.persona.nickname = "阿暖"
    rel.persona.boundaries = ["别提家人"]
    rel.affinity = 0.77
    restored = LifeState.from_dict(st.to_dict())
    assert restored.P == 0.42
    assert restored.drive["bond"] == 0.11
    assert restored.relations["u1"].persona.nickname == "阿暖"
    assert restored.relations["u1"].persona.boundaries == ["别提家人"]
    assert restored.core.name == "夏弥"

    store = StateStore(":memory:")
    store.save(restored)
    loaded = store.load()
    assert loaded is not None
    assert loaded.relations["u1"].affinity == 0.77
    store.close()
