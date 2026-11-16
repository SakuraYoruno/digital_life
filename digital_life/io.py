"""出站 IO：把回复按 A 维旋钮切成气泡发送。

对应开发文档 §5.1。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class OutboundMessage:
    uid: str
    text: str
    ts: float
    knobs: dict[str, Any] = field(default_factory=dict)


class OutboundIO(Protocol):
    async def send(self, uid: str, text: str, knobs: dict[str, Any]) -> None: ...


_PUNCT = "。！？!?…～~"


def split_bubbles(text: str, knobs: dict[str, Any] | None = None) -> list[str]:
    if not text or not text.strip():
        return []

    # 模型自己用换行断的，就是它想分的气泡——这是它的意图，不该被旋钮吞掉。
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        if len(lines) > 4:
            lines = lines[:3] + ["".join(lines[3:])]
        return lines

    line = lines[0] if lines else text.strip()
    if not knobs or not knobs.get("split_bubbles"):
        return [line]

    # 旋钮要求拆气泡时，再按句末标点切；不拆英文缩写（这里只处理中文句末标点，足够）。
    raw_parts = re.findall(r"[^。！？!?…\n]+[。！？!?…]*", line)
    parts = [p.strip() for p in raw_parts if p.strip()]
    if len(parts) <= 1:
        return [line]
    # 太碎的片段并到前一条，避免“哈。”“？”这种单字气泡。
    merged: list[str] = []
    for part in parts:
        if merged and len(part) <= 2:
            merged[-1] += part
        else:
            merged.append(part)
    # 最多 4 条，剩下的并到最后一口气说完。
    if len(merged) > 4:
        merged = merged[:3] + ["".join(merged[3:])]
    return merged


class NullIO:
    async def send(self, uid: str, text: str, knobs: dict[str, Any]) -> None:
        return None


class CollectingIO:
    """测试/仿真用：记录所有出站气泡。"""

    def __init__(self, *, clock=time.time) -> None:
        self.messages: list[OutboundMessage] = []
        self.clock = clock

    async def send(self, uid: str, text: str, knobs: dict[str, Any]) -> None:
        for bubble in split_bubbles(text, knobs):
            self.messages.append(
                OutboundMessage(str(uid), bubble, self.clock(), dict(knobs or {}))
            )

    def texts(self, uid: str | None = None) -> list[str]:
        if uid is None:
            return [m.text for m in self.messages]
        return [m.text for m in self.messages if m.uid == str(uid)]

    def last(self, uid: str | None = None) -> str | None:
        texts = self.texts(uid)
        return texts[-1] if texts else None

    def clear(self) -> None:
        self.messages.clear()


class ConsoleIO:
    """控制台聊天：直接打印，按 uid 加前缀。"""

    def __init__(self, *, prefix: str = "她", clock=time.time) -> None:
        self.prefix = prefix
        self.clock = clock

    async def send(self, uid: str, text: str, knobs: dict[str, Any]) -> None:
        bubbles = split_bubbles(text, knobs)
        for i, bubble in enumerate(bubbles):
            label = self.prefix if i == 0 else "　"
            print(f"{label}：{bubble}")


__all__ = [
    "OutboundMessage",
    "OutboundIO",
    "split_bubbles",
    "NullIO",
    "CollectingIO",
    "ConsoleIO",
]
