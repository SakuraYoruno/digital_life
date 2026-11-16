from __future__ import annotations

import asyncio

from digital_life.llm import MockLLM
from digital_life.memory import MemoryStore
from digital_life.persistence import ObsidianReflector
from digital_life.persona_evolution import (
    evolve_persona,
    extract_semantic_facts,
    memory_gc,
    nightly_reflect,
    update_user_profile,
)
from digital_life.personas import Relation, xiami_state


class ProfileLLM(MockLLM):
    async def complete(self, prompt, **kwargs):
        if "结构化用户模型" in prompt:
            return (
                '{"preferred_name":"阿暖","sleep_start":1,"sleep_end":9,'
                '"notes":{"雷区":"别提家人","在意的事":"龙卡"}}'
            )
        return await super().complete(prompt, **kwargs)


def test_evolve_persona_constraints():
    st = xiami_state()
    rel = Relation()
    before = dict(st.plastic.style_traits)
    evolve_persona(
        st,
        rel,
        {
            "style_drift": {"exclaim_rate": 0.5, "emoji_rate": -0.9, "unknown": 1},
            "boundary": "别提家人",
            "insight": "先接住话头，再讲道理",
        },
    )
    assert st.plastic.style_traits["exclaim_rate"] == min(1.0, before["exclaim_rate"] + 0.05)
    assert st.plastic.style_traits["emoji_rate"] == max(0.0, before["emoji_rate"] - 0.05)
    assert "unknown" not in st.plastic.style_traits
    assert rel.persona.boundaries == ["别提家人"]
    assert rel.persona.strategy == "先接住话头，再讲道理"
    assert st.plastic.version == 1
    assert st.core.name == "夏弥"
    assert "不提家人" not in st.core.values
    # 边界只增不减
    evolve_persona(st, rel, {"boundary": "别提家人", "insight": "换一句"})
    assert rel.persona.boundaries == ["别提家人"]


def test_nightly_reflect_updates_strategy_and_writes_markdown(tmp_path):
    async def run():
        now = 1_800_000_000.0
        st = xiami_state()
        rel = st.rel("u1")
        rel.persona.nickname = "阿暖"
        mem = MemoryStore(tmp_path / "m.db", clock=lambda: now)
        for i in range(6):
            await mem.write(
                "u1", "episodic", f"和用户聊了游戏龙卡的第{i}次", importance=0.8, now=now
            )
        reflector = ObsidianReflector(tmp_path / "vault")
        result = await nightly_reflect(
            MockLLM(1), st, mem, now=now, reflector=reflector, run_gc=False
        )
        assert result["uids"] == ["u1"]
        assert result["evolved"] == 1
        assert rel.persona.strategy
        assert st.self_narrative
        files = list((tmp_path / "vault" / "数字生命" / "反思").glob("*.md"))
        assert files
        assert "感悟" in files[0].read_text(encoding="utf-8")
        assert (tmp_path / "vault" / "数字生命" / "关系图谱.md").exists()
        mem.close()

    asyncio.run(run())


def test_extract_semantic_facts_does_not_crash_on_empty():
    async def run():
        mem = MemoryStore(":memory:")
        count = await extract_semantic_facts(MockLLM(1), mem, "u1")
        assert count == 0
        mem.close()

    asyncio.run(run())


def test_update_user_profile_from_semantic_memory():
    async def run():
        mem = MemoryStore(":memory:")
        await mem.write("u1", "semantic", "用户喜欢玩龙卡", importance=0.9)
        rel = Relation()
        changed = await update_user_profile(ProfileLLM(1), mem, "u1", rel)
        assert changed
        assert rel.profile.preferred_name == "阿暖"
        assert rel.profile.sleep_start == 1
        assert rel.profile.sleep_end == 9
        assert rel.profile.notes["雷区"] == "别提家人"
        mem.close()

    asyncio.run(run())


def test_memory_gc_forgets_merges_and_expires(tmp_path):
    async def run():
        now = 1_800_000_000.0
        mem = MemoryStore(tmp_path / "m.db", clock=lambda: now)
        old = now - 100 * 86400
        # 该遗忘：低重要性 + 未命中 + 超 30 天
        await mem.write("u1", "episodic", "很久以前的一件小事", importance=0.1, now=old)
        # 相似簇：应合并成一条语义记忆
        for i in range(4):
            await mem.write(
                "u1",
                "episodic",
                f"和用户一起玩龙卡很开心，第{i}次",
                importance=0.6,
                now=now - (i + 1) * 86400,
            )
        # 超 90 天：重要性减半
        await mem.write("u1", "episodic", "更早以前的一件重要的事", importance=0.8, now=old)
        # TTL 过期
        await mem.write("u1", "knowledge", "过期知识", ttl=10, now=now - 100)

        stats = await memory_gc(MockLLM(1), mem, "u1", now=now)
        assert stats["forgotten"] >= 1
        assert stats["merged"] >= 1
        assert stats["downgraded"] >= 1
        assert stats["expired"] >= 1
        assert await mem.count("u1", "semantic") >= 1
        assert await mem.count("u1", "knowledge") == 0
        mem.close()

    asyncio.run(run())
