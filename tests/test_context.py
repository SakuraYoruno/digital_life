from __future__ import annotations

import asyncio

from digital_life.constants import estimate_tokens
from digital_life.context import (
    BUDGET,
    ContextAssembler,
    assemble_in_natural_order,
)
from digital_life.memory import MemoryStore, WorkingMemory
from digital_life.personas import Relation, build_system_prompt, xiami_state


def test_assemble_order():
    text = assemble_in_natural_order(
        [("user", "你好"), ("summary", "聊过"), ("working", "上一句")]
    )
    assert text.index("【近期对话】") < text.index("【会话摘要】")
    assert text.index("【会话摘要】") < text.index("【对方刚说】")
    assert text.endswith("你好")


def test_context_budget_and_layers(tmp_path):
    async def run():
        st = xiami_state()
        rel = Relation()
        rel.session_summary = "很长的摘要" * 200
        mem = MemoryStore(tmp_path / "m.db")
        for i in range(20):
            await mem.write(
                "u1", "episodic", f"用户和龙卡有关的长记忆{i}" * 10, importance=0.7
            )
        wm = WorkingMemory(max_turns=20, max_tokens=3000)
        for i in range(60):
            wm.add("user", f"第{i}条比较长的工作记忆内容" * 8, "平静")
        sys_prompt, _ = build_system_prompt(st, rel)
        ctx = await ContextAssembler().build(
            uid="u1",
            user_msg="你还记得龙卡吗",
            wm=wm,
            mem=mem,
            sys_prompt=sys_prompt,
            mood="开心",
            summary=rel.session_summary,
            knobs={},
        )
        assert ctx.tokens <= 8000
        assert ctx.tokens == estimate_tokens(sys_prompt) + estimate_tokens(ctx.text)
        assert "【近期对话】" in ctx.text
        assert "【相关记忆】" in ctx.text
        assert "【对方刚说】" in ctx.text
        assert "你还记得龙卡吗" in ctx.text
        assert ctx.recalled
        assert all(h.uid == "u1" for h in ctx.recalled if h.kind in ("episodic", "semantic"))
        mem.close()

    asyncio.run(run())


def test_budget_constants_leave_headroom():
    assert sum(BUDGET.values()) < 8000
