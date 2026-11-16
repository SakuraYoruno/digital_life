"""群聊共享时间线 + 会话结束更新画像。"""

from __future__ import annotations

import asyncio
import time

from digital_life.engine import DigitalLife, EngineConfig
from digital_life.io import CollectingIO
from digital_life.llm import MockLLM
from digital_life.memory import MemoryStore
from digital_life.personas import xiami_state


class _Spy:
    def __init__(self, inner):
        self.inner = inner
        self.prompts: list[str] = []

    async def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return await self.inner.complete(prompt, **kwargs)


def _engine(spy: _Spy | None = None, **config):
    st = xiami_state()
    mem = MemoryStore(":memory:", clock=time.time)
    io = CollectingIO(clock=time.time)
    big = spy or _Spy(MockLLM(1))
    engine = DigitalLife(
        st,
        MockLLM(2),
        big,
        mem,
        io,
        config=EngineConfig(clock=time.time, **config),
    )
    return engine, st, mem


def test_group_timeline_reaches_prompt():
    async def run():
        spy = _Spy(MockLLM(1))
        engine, _st, mem = _engine(spy)
        now = time.time()
        await engine.on_message("111", "今天好热啊", "9001", now=now, speaker="小虾米")
        await engine.on_message("222", "空调坏了？", "9001", now=now + 10, speaker="赦")

        last = spy.prompts[-1]
        assert "【群里刚才】" in last
        assert "小虾米：今天好热啊" in last
        assert "赦：" not in last.split("【群里刚才】")[1]  # 当前这条不重复列

        await engine.drain()
        mem.close()

    asyncio.run(run())


def test_private_chat_has_no_group_block():
    async def run():
        spy = _Spy(MockLLM(1))
        engine, _st, mem = _engine(spy)
        await engine.on_message("111", "在吗", now=time.time(), speaker="小虾米")
        assert "【群里刚才】" not in spy.prompts[-1]
        await engine.drain()
        mem.close()

    asyncio.run(run())


def test_group_timeline_is_per_group():
    async def run():
        spy = _Spy(MockLLM(1))
        engine, _st, mem = _engine(spy)
        now = time.time()
        await engine.on_message("111", "A群的悄悄话", "1001", now=now, speaker="甲")
        await engine.on_message("222", "B群的话题", "1002", now=now + 10, speaker="乙")
        last = spy.prompts[-1]
        assert "A群的悄悄话" not in last
        await engine.drain()
        mem.close()

    asyncio.run(run())


def test_session_end_refreshes_profile_when_enough_turns(monkeypatch):
    calls: list[str] = []

    async def fake_update(small, mem, uid, rel, *, now=None):
        calls.append(uid)
        return True

    monkeypatch.setattr("digital_life.engine.update_user_profile", fake_update)

    async def run():
        engine, st, mem = _engine()
        rel = st.rel("111")
        rel.session_active = True
        rel.session.n_turns = 5
        await engine._end_session("111", now=time.time())
        await engine.drain()
        mem.close()

    asyncio.run(run())
    assert calls == ["111"]


def test_session_end_skips_profile_for_short_chat(monkeypatch):
    calls: list[str] = []

    async def fake_update(small, mem, uid, rel, *, now=None):
        calls.append(uid)
        return True

    monkeypatch.setattr("digital_life.engine.update_user_profile", fake_update)

    async def run():
        engine, st, mem = _engine()
        rel = st.rel("111")
        rel.session_active = True
        rel.session.n_turns = 1
        await engine._end_session("111", now=time.time())
        await engine.drain()
        mem.close()

    asyncio.run(run())
    assert calls == []
