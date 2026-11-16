"""全局常量与纯工具函数。

这里不保存任何状态，所有函数都是无副作用的纯函数，方便测试与并发调用。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

# 文档 §0 的时间基准：所有 rate 都是“每 BASE_DT 秒”的比例。
BASE_DT: float = 60.0

# 文档 §8.1 给出的关键阈值。不要随手调成 0.5，否则几乎永远不会主动说话。
SEND_THRESHOLD: float = 0.25

# 情绪锚点数量固定为 8；mood 是离散词，不参与数值运算。
MOODS: tuple[str, ...] = (
    "开心",
    "兴奋",
    "平静",
    "无聊",
    "低落",
    "焦虑",
    "委屈",
    "烦躁",
)


def clamp(x: float, lo: float, hi: float) -> float:
    """把 x 限制在 [lo, hi]。lo > hi 时返回 lo。"""
    if lo > hi:
        return lo
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid，避免 exp 溢出。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def scaled(rate: float, dt: float) -> float:
    """按真实 dt 缩放“每个 BASE_DT 的比例”。

    scaled(rate, BASE_DT) == rate；
    scaled(rate, 2*BASE_DT) == 1-(1-rate)^2。

    这样 60 秒 tick 与 240 秒 tick 在连续时间上一致。
    """
    rate = clamp(float(rate), 0.0, 1.0)
    dt = max(0.0, float(dt))
    if dt == 0.0:
        return 0.0
    return 1.0 - (1.0 - rate) ** (dt / BASE_DT)


_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def estimate_tokens(text: str | None) -> int:
    """一个足够稳定的本地 token 估算。

    中文/标点/emoji 按 1 token，ASCII 按 4 字符 1 token。
    不依赖 tiktoken，跨模型差异由预算余量吸收。
    """
    if not text:
        return 0
    cjk = len(_NON_ASCII.findall(text))
    ascii_len = len(text) - cjk
    return max(1, cjk + math.ceil(ascii_len / 4))


def truncate_to_tokens(text: str, limit: int, suffix: str = "…") -> str:
    """按 estimate_tokens 截断，给后缀留出预算。"""
    if not text or limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    suffix_tokens = estimate_tokens(suffix)
    room = max(0, limit - suffix_tokens)
    out: list[str] = []
    used = 0.0
    for ch in text:
        cost = 1.0 if ord(ch) > 127 else 0.25
        if used + cost > room:
            break
        out.append(ch)
        used += cost
    return "".join(out).rstrip() + suffix


def strip_code_fence(raw: str) -> str:
    """去掉模型常用的 ```json ... ``` 包裹。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_slice(text: str, start: int) -> str:
    """从 start 处的 { 或 [ 开始，找到配对闭合位置（忽略字符串里的括号）。"""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def safe_json(raw: Any, default: Any = None) -> Any:
    """尽量把模型输出解析成 JSON；失败返回 default，不抛异常。"""
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    text = strip_code_fence(str(raw))
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    positions = [p for p in (text.find("{"), text.find("[")) if p >= 0]
    if positions:
        candidate = _balanced_slice(text, min(positions))
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            return default
    return default


def json_text(value: Any) -> str:
    """稳定、紧凑的 JSON 序列化。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def unique_str_list(value: Iterable[Any], limit: int = 12) -> list[str]:
    out: list[str] = []
    for item in value or []:
        if not isinstance(item, str):
            item = str(item)
        item = item.strip()
        if item and item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def has_time_ref(text: str) -> bool:
    """判断 query 里是否含时间指代，供多维召回使用。"""
    return bool(
        re.search(
            r"昨天|前天|今天|昨晚|上次|上周|上个月|去年|前年|"
            r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]|"
            r"\d{1,2}\s*[点时]|"
            r"星期[一二三四五六日天]|周[一二三四五六日天]",
            text or "",
        )
    )


def parse_time_ref(text: str, now: float) -> tuple[float, float] | None:
    """把中文相对时间解析成 [start, end) 时间戳。

    只覆盖日常对话里常见的粒度；解析不到时返回 None，召回层会跳过时间维。
    """
    import datetime as _dt

    if not text:
        return None
    today = _dt.datetime.fromtimestamp(now)
    start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)

    def day_range(days_ago: int) -> tuple[float, float]:
        start = start_of_today - _dt.timedelta(days=days_ago)
        return start.timestamp(), (start + _dt.timedelta(days=1)).timestamp()

    if "前天" in text:
        return day_range(2)
    if "昨天" in text or "昨晚" in text:
        return day_range(1)
    if "今天" in text:
        return day_range(0)
    if "上周" in text:
        start = start_of_today - _dt.timedelta(days=7)
        return start.timestamp(), (start + _dt.timedelta(days=7)).timestamp()
    if "上个月" in text:
        first = start_of_today.replace(day=1)
        prev_end = first
        prev_start = (first - _dt.timedelta(days=1)).replace(day=1)
        return prev_start.timestamp(), prev_end.timestamp()
    if "去年" in text:
        start = start_of_today.replace(month=1, day=1) - _dt.timedelta(days=365)
        return start.timestamp(), (start + _dt.timedelta(days=365)).timestamp()

    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            start = start_of_today.replace(month=month, day=day)
            if start > start_of_today:
                start = start.replace(year=start.year - 1)
            return start.timestamp(), (start + _dt.timedelta(days=1)).timestamp()
        except ValueError:
            return None

    m = re.search(r"(\d{1,2})\s*[点时]", text)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            start = start_of_today.replace(hour=hour)
            return start.timestamp(), (start + _dt.timedelta(hours=1)).timestamp()
    return None


async def maybe_await(value: Any) -> Any:
    """接受同步/异步返回值。"""
    if hasattr(value, "__await__"):
        return await value
    return value
