"""念头流：没人说话时“她在想什么”。

对应开发文档 §6、§7。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .constants import clamp
from .llm import call_llm
from .personas import LifeState

THOUGHT_TYPES: dict[str, Callable[[LifeState], float]] = {
    "想念对方": lambda st: 1.2 - st.drive["bond"],
    "回味刚才": lambda st: st.lingering,
    "好奇发散": lambda st: st.drive["curiosity"] * 0.8,
    "自我怀疑": lambda st: (0.6 - st.P) if st.P < 0.6 else 0,
    "期待分享": lambda st: st.drive["curiosity"] * (0.5 + st.P),
    "放空": lambda st: 0.3,
}

# 念头对 PAD 的反扰幅度；靠累积，单次很小。
THOUGHT_NUDGES: dict[str, tuple[float, float, float]] = {
    "想念对方": (-0.02, 0.03, 0.0),
    "自我怀疑": (-0.05, 0.04, -0.03),
    "回味刚才": (0.03, 0.0, 0.0),
    "期待分享": (0.04, 0.03, 0.02),
}


@dataclass
class ThoughtResult:
    type: str
    text: str
    nudge: tuple[float, float, float]


def _pick_type(st: LifeState, rng: random.Random) -> str:
    weights = {t: max(0.0, f(st)) for t, f in THOUGHT_TYPES.items()}
    total = sum(weights.values()) or 1.0
    r = rng.random() * total
    acc = 0.0
    pick = "放空"
    for thought_type, weight in weights.items():
        acc += weight
        if r <= acc:
            pick = thought_type
            break
    return pick


async def thought_tick(
    small_llm: Any,
    st: LifeState,
    history: deque[str],
    *,
    rng: random.Random | None = None,
) -> ThoughtResult:
    """采样念头类型 → 小模型生成独白 → 按 lingering 衰减后微扰 PAD。

    history 只存念头类型，用于闸1；念头正文由调用方另行保存，
    作为主动发言的素材（§7 出口2）。
    """
    rng = rng or random
    pick = _pick_type(st, rng)

    # 闸1 多样性：最近三条同类型，强制换成放空。
    if list(history)[-3:].count(pick) >= 3:
        pick = "放空"

    prompt = (
        f"用一句内心独白表达'{pick}'，口吻是{st.core.style_skeleton}，"
        f"当前{st.mood}。只输出这一句，不要引号。"
    )
    try:
        text = (await call_llm(small_llm, prompt, temperature=0.9)).strip()
    except Exception:
        text = "……"
    if not text:
        text = "……"
    history.append(pick)

    # 闸2 衰减：余味越低，反扰越弱，避免无限自激。
    scale = 0.3 + 0.7 * st.lingering
    nudge = THOUGHT_NUDGES.get(pick, (0.0, 0.0, 0.0))
    st.P = clamp(st.P + nudge[0] * scale, -1.0, 1.0)
    st.A = clamp(st.A + nudge[1] * scale, 0.0, 1.0)
    st.D = clamp(st.D + nudge[2] * scale, -1.0, 1.0)
    return ThoughtResult(type=pick, text=text, nudge=nudge)


__all__ = ["THOUGHT_TYPES", "THOUGHT_NUDGES", "ThoughtResult", "thought_tick"]
