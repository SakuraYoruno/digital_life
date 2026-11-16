"""上下文压缩与组装。

对应开发文档 §10。token 预算按固定比例分给各层，最终硬上限 8000。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .constants import estimate_tokens, truncate_to_tokens
from .memory import (
    MemoryHit,
    MemoryStore,
    WorkingMemory,
    format_recalled,
    maybe_fetch_knowledge,
    recall_full,
)

BUDGET = {
    "system": 500,
    "working": 3000,
    "summary": 300,
    "memory": 800,
    "group": 220,
    "knowledge": 500,
    "user": 500,
}

MAX_CONTEXT_TOKENS = 8000

_HEADERS = {
    "working": "【近期对话】",
    "summary": "【会话摘要】",
    "memory": "【相关记忆】",
    "group": "【群里刚才】",
    "knowledge": "【资料】",
    "user": "【对方刚说】",
    "system": "【系统】",
}


@dataclass
class ContextBundle:
    system_prompt: str
    text: str
    tokens: int
    parts: dict[str, str] = field(default_factory=dict)
    recalled: list[MemoryHit] = field(default_factory=list)
    knobs: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


_ORDER = {
    "system": 0,
    "working": 1,
    "summary": 2,
    "memory": 3,
    "group": 4,
    "knowledge": 5,
    "user": 6,
}


def assemble_in_natural_order(parts: list[tuple[str, str]]) -> str:
    ordered = sorted(
        enumerate(parts),
        key=lambda pair: (_ORDER.get(pair[1][0], len(_ORDER)), pair[0]),
    )
    blocks: list[str] = []
    for _, (name, content) in ordered:
        if not content:
            continue
        header = _HEADERS.get(name, f"【{name}】")
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


class ContextAssembler:
    def __init__(
        self,
        *,
        search_fn: Callable[[str], Any] | None = None,
        kb: Any | None = None,
        budget: dict[str, int] | None = None,
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self.search_fn = search_fn
        # 知识库直接读 Markdown 源文件；命中就不必再问向量库/联网
        self.kb = kb
        self.budget = dict(BUDGET if budget is None else budget)
        self.max_tokens = max_tokens

    async def build(
        self,
        *,
        uid: str,
        user_msg: str,
        wm: WorkingMemory,
        mem: MemoryStore,
        sys_prompt: str,
        mood: str = "",
        summary: str = "",
        group_id: str | None = None,
        group_context: str = "",
        knobs: dict[str, Any] | None = None,
        small_llm: Any = None,
        now: float | None = None,
    ) -> ContextBundle:
        working = wm.to_text(max_tokens=self.budget["working"])
        summary_text = truncate_to_tokens(
            summary or "", self.budget["summary"]
        )

        recalled = await recall_full(
            mem,
            uid,
            user_msg,
            mood=mood,
            group_id=group_id,
            k=8,
            now=now,
        )
        memory_text = format_recalled(recalled, self.budget["memory"])

        knowledge = ""
        if self.kb is not None:
            try:
                knowledge = self.kb.context(user_msg)
            except Exception:
                knowledge = ""
        if not knowledge and small_llm is not None:
            try:
                knowledge = await maybe_fetch_knowledge(
                    small_llm,
                    mem,
                    user_msg,
                    group_id=group_id,
                    search_fn=self.search_fn,
                    now=now,
                )
            except Exception:
                knowledge = ""
        knowledge = truncate_to_tokens(knowledge, self.budget["knowledge"])
        group_text = truncate_to_tokens(
            group_context or "", self.budget.get("group", 220)
        )

        parts = [
            ("working", working),
            ("summary", summary_text),
            ("memory", memory_text),
            ("group", group_text),
            ("knowledge", knowledge),
            ("user", user_msg),
        ]
        text = assemble_in_natural_order(parts)
        total = estimate_tokens(sys_prompt) + estimate_tokens(text)

        # 硬上限保护：先丢资料，再丢群聊回声，再丢召回，再削工作记忆；
        # system 与 user 永不动。
        if total > self.max_tokens:
            parts = [(name, content) for name, content in parts if name != "knowledge"]
            text = assemble_in_natural_order(parts)
            total = estimate_tokens(sys_prompt) + estimate_tokens(text)
        if total > self.max_tokens:
            parts = [
                (name, content)
                for name, content in parts
                if name not in ("knowledge", "group")
            ]
            text = assemble_in_natural_order(parts)
            total = estimate_tokens(sys_prompt) + estimate_tokens(text)
        if total > self.max_tokens:
            parts = [
                (name, content)
                for name, content in parts
                if name not in ("knowledge", "group", "memory")
            ]
            text = assemble_in_natural_order(parts)
            total = estimate_tokens(sys_prompt) + estimate_tokens(text)
        if total > self.max_tokens:
            room = max(200, self.max_tokens - estimate_tokens(sys_prompt) - 200)
            for i, (name, content) in enumerate(parts):
                if name == "working":
                    parts[i] = (name, truncate_to_tokens(content, room))
                    break
            text = assemble_in_natural_order(parts)
            total = estimate_tokens(sys_prompt) + estimate_tokens(text)
        if total > self.max_tokens:
            # 最后一层保护：当前消息很长时截断它，仍然保留开头。
            fixed = estimate_tokens(sys_prompt) + sum(
                estimate_tokens(content)
                for name, content in parts
                if name != "user"
            )
            room = max(80, self.max_tokens - fixed - 40)
            for i, (name, content) in enumerate(parts):
                if name == "user":
                    parts[i] = (name, truncate_to_tokens(content, room))
                    break
            text = assemble_in_natural_order(parts)
            total = estimate_tokens(sys_prompt) + estimate_tokens(text)

        return ContextBundle(
            system_prompt=sys_prompt,
            text=text,
            tokens=total,
            parts={name: content for name, content in parts},
            recalled=recalled,
            knobs=dict(knobs or {}),
        )
