"""难点一：appraisal。让 LLM 只做定性评估，数值交给确定性公式。

对应开发文档 §2。
"""

from __future__ import annotations

from typing import Any

from .constants import clamp, safe_json
from .llm import call_llm

APPRAISE_SCHEMA = {"drive", "valence", "intensity", "reason"}
VALID_DRIVES = {"bond", "approval", "curiosity", "coherence"}

APPRAISE_FALLBACK: dict[str, Any] = {
    "drive": "bond",
    "valence": 1,
    "intensity": 0.1,
    "reason": "appraise_fallback",
}


def parse_appraisal(raw: Any) -> dict[str, Any]:
    """校验并归一化模型输出；失败走兜底，绝不让主流程崩。"""
    data = safe_json(raw, None)
    if not isinstance(data, dict):
        return dict(APPRAISE_FALLBACK)
    if not APPRAISE_SCHEMA.issubset(set(data)):
        return dict(APPRAISE_FALLBACK)
    drive = data.get("drive")
    if drive not in VALID_DRIVES:
        return dict(APPRAISE_FALLBACK)
    try:
        valence = 1 if float(data["valence"]) >= 0 else -1
        intensity = clamp(float(data["intensity"]), 0.0, 1.0)
    except (TypeError, ValueError):
        return dict(APPRAISE_FALLBACK)
    return {
        "drive": drive,
        "valence": valence,
        "intensity": intensity,
        "reason": str(data.get("reason", ""))[:200],
    }


async def llm_appraise(small_llm: Any, event_text: str, ctx: str = "") -> dict[str, Any]:
    """让 small_llm 判断“这件事触动了哪条驱力、正向还是负向、多强”。"""
    prompt = f"""你在评估一条消息对“她”的内在触动。只输出 JSON，不要解释。
消息：{event_text}
上下文：{ctx or '（无）'}
字段：
  drive: 被最强触动的驱力，取值 bond|approval|curiosity|coherence
  valence: +1 表示满足/正向，-1 表示威胁/负向
  intensity: 触动强度 0~1
  reason: 一句话理由
"""
    raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.2)
    return parse_appraisal(raw)
