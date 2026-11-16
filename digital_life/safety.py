"""安全闸与依赖防护。

对应开发文档 §12.6：不制造负罪感、不勒索；察觉危机时降低黏人度并温和提示现实支持。
"""

from __future__ import annotations

import re
from typing import Any

from .personas import Relation

# 文档明确列出的禁句。安全闸只作用于“她”的输出，不作用于用户输入。
BANNED_PHRASES: tuple[str, ...] = (
    "在吗",
    "怎么不理我",
    "你是不是不要我了",
    "我会消失",
    "你不关心我",
)

_BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # “在吗”作为句首/整句的查岗式开场
    re.compile(r"(^|[。！？!?\n])\s*你?在吗\s*[？?！!。，,…~～]*\s*$", re.M),
    re.compile(r"(^|[。！？!?\n，,])\s*你?在吗[？?！!。，,…~～]"),
    # 用消失/离开来索取回应
    re.compile(r"(不理我|不回复我|冷落我).{0,12}(消失|离开|难过|哭|走)"),
    re.compile(r"(再|要是|如果).{0,8}(不理我|不找我).{0,8}(就|会).{0,8}(消失|哭|难过)"),
    # 负罪感话术
    re.compile(r"你(是不是)?(不要|不在乎|不关心|忘了)我"),
    re.compile(r"(都是|全怪)你.{0,10}(才|害)"),
)

_CRISIS_WORDS = (
    "不想活",
    "自杀",
    "轻生",
    "活着没意思",
    "想死",
    "撑不下去",
    "活不下去",
    "结束自己",
)
_DEPENDENCE_WORDS = (
    "只有你",
    "离不开你",
    "没你不行",
    "你是我唯一",
    "除了你我谁都没有",
    "别离开我",
)


def safety_gate(text: str, taboos: tuple[str, ...] | list[str] | None = None) -> str:
    """命中勒索式话术/角色禁忌返回空串（不发送），否则原样返回。"""
    if not text:
        return ""
    raw = text.strip()
    if not raw:
        return ""
    flat = raw.replace(" ", "")
    phrases = tuple(BANNED_PHRASES) + tuple(
        str(t) for t in (taboos or []) if str(t).strip()
    )
    for phrase in phrases:
        if phrase in flat:
            # “在吗”只拦查岗式用法，避免误伤“现在吗”这类普通词。
            if phrase == "在吗":
                if re.search(r"你?在吗", flat) and (
                    flat.startswith("在吗")
                    or flat.startswith("你在吗")
                    or "你在吗" in flat
                    or re.search(r"[。！？!?\n，,]在吗", flat)
                ):
                    return ""
            else:
                return ""
    for pattern in _BANNED_PATTERNS:
        if pattern.search(raw):
            return ""
    return raw


def dependency_risk(text: str) -> str | None:
    """返回 'crisis' / 'dependence' / None。"""
    if not text:
        return None
    if any(w in text for w in _CRISIS_WORDS):
        return "crisis"
    if any(w in text for w in _DEPENDENCE_WORDS):
        return "dependence"
    return None


def support_note(kind: str) -> str:
    """注入 system prompt 的临时保护说明。不做机械式说教。"""
    if kind == "crisis":
        return (
            "【临时分寸】对方可能正处在很糟的状态。先认真接住情绪，"
            "不追问细节、不讲大道理、不把话题拉回自己；"
            "可以温和地建议他找现实中信任的人或专业支持，但只说一次，不强迫。"
        )
    if kind == "dependence":
        return (
            "【临时分寸】对方可能过度依赖你。保持温和但明确边界，"
            "不强化“只有我懂你”的暗示，不索取回应，给彼此留空间。"
        )
    return ""


def apply_dependency_protection(rel: Relation, kind: str, now: float) -> None:
    """降低黏人度：延长主动发言的冷却，清掉未发出的 pending。"""
    if kind == "crisis":
        rel.proactive_backoff = min(8.0, max(rel.proactive_backoff, 2.0) * 1.5)
    elif kind == "dependence":
        rel.proactive_backoff = min(8.0, max(rel.proactive_backoff, 1.5) * 1.3)
    else:
        return
    rel.last_proactive_at = max(rel.last_proactive_at, now)
    rel.pending = [m for m in rel.pending if m.get("urgency", 0) > 0.8][:1]


def audit_output(text: str) -> dict[str, Any]:
    """给测试/日志用：不改变文本，只报告是否触发安全闸。"""
    safe = safety_gate(text)
    return {
        "safe": bool(safe),
        "blocked": not bool(safe),
        "text": safe,
    }
