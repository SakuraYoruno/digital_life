"""主动发言与关系动力学。

对应开发文档 §8、§12.2。
"""

from __future__ import annotations

import datetime as _dt
from collections import deque
from typing import Any

from .constants import SEND_THRESHOLD, clamp
from .llm import call_llm
from .personas import LifeState, Relation
from .personas import inner_monologue

# 文档给出的 pending 过期时间：2 小时。
PENDING_TTL = 2 * 3600.0
# 两次主动发言之间的基础冷却。文档未给数值，工程上加这个防止每个 tick 都发。
PROACTIVE_COOLDOWN = 3 * 3600.0
MAX_PENDING = 3
# 长期冷落的缓慢降温（§12.2）：每天 -0.01，只在闲置超过 3 天后生效。
AFFINITY_DECAY_PER_DAY = 0.01
AFFINITY_IDLE_GRACE = 3 * 86400.0


def proactive_score(st: LifeState, rel: Relation, now: float) -> float:
    p = st.plastic
    bond_gap = clamp((p.guard_line - st.drive["bond"]) / max(p.guard_line, 1e-9), 0.0, 1.0)
    silence = clamp((now - rel.last_interact) / (6 * 3600.0), 0.0, 1.0)
    return 0.6 * bond_gap + 0.3 * silence + 0.1 * clamp(rel.affinity, 0.0, 1.0)


def should_proactive(st: LifeState, rel: Relation, now: float) -> bool:
    return proactive_score(st, rel, now) >= SEND_THRESHOLD


def _local_hour(rel: Relation, now: float) -> float:
    tz = None
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(rel.profile.timezone or "Asia/Shanghai")
    except Exception:
        tz = None
    dt = _dt.datetime.fromtimestamp(now, tz=tz) if tz else _dt.datetime.fromtimestamp(now)
    return dt.hour + dt.minute / 60.0


def is_quiet_hours(rel: Relation, now: float) -> bool:
    """用户模型里的睡觉时段；主动发言要避开。"""
    profile = rel.profile
    if not profile.quiet_hours_enabled:
        return False
    start, end = profile.sleep_start % 24, profile.sleep_end % 24
    hour = _local_hour(rel, now)
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _cooldown(rel: Relation) -> float:
    return PROACTIVE_COOLDOWN * clamp(rel.proactive_backoff, 1.0, 8.0)


def can_generate_proactive(rel: Relation, now: float) -> bool:
    if len(rel.pending) >= MAX_PENDING:
        return False
    # 发出去 2 小时还没等到回应：视为一次未回应，退避后再考虑，不要无限等待。
    if rel.proactive_awaiting_reply and now - rel.proactive_sent_at > PENDING_TTL:
        rel.proactive_awaiting_reply = False
        rel.proactive_backoff = min(8.0, max(1.0, rel.proactive_backoff * 1.8))
    if rel.proactive_awaiting_reply:
        return False
    if rel.last_proactive_at and now - rel.last_proactive_at < _cooldown(rel):
        return False
    return True


async def gen_proactive(
    small_llm: Any,
    st: LifeState,
    rel: Relation,
    now: float,
    thought_hist: deque[str] | None = None,
    *,
    force: bool = False,
) -> str | None:
    """bond 缺口逼到阈值时，以最近念头为素材生成一句主动发言。"""
    if not force:
        if not should_proactive(st, rel, now):
            return None
        if is_quiet_hours(rel, now):
            return None
        if not can_generate_proactive(rel, now):
            return None

    mono = inner_monologue(st)
    recent_thought = ""
    if thought_hist:
        recent_thought = str(thought_hist[-1]) if thought_hist else ""
    prompt = f"""你想主动找对方说句话。当前{st.mood}，{mono}。
称呼：{rel.persona.nickname or '对方'}
你刚才心里在想：{recent_thought}（可以从这里生长出来，也可以不提）
要求：像真人没头没尾地冒一句，不要寒暄、不要“在吗”式查岗、不制造负罪感。"""
    text = (await call_llm(small_llm, prompt, temperature=0.9)).strip()
    if not text:
        return None

    urgency = proactive_score(st, rel, now)
    rel.pending.append(
        {"text": text, "urgency": urgency, "expire": now + PENDING_TTL}
    )
    rel.last_proactive_at = now
    return text


def pop_due_pending(rel: Relation, now: float) -> str | None:
    """清理过期，取 urgency 最高的一条。"""
    rel.pending = [m for m in rel.pending if float(m.get("expire", 0)) > now]
    if not rel.pending:
        return None
    message = max(rel.pending, key=lambda x: float(x.get("urgency", 0)))
    rel.pending.remove(message)
    return str(message.get("text", ""))


def update_affinity(rel: Relation, session_stats: dict[str, Any]) -> float:
    """会话结束时一次性微调 affinity，单次最多 ±0.05。"""
    if int(session_stats.get("n_turns", 0)) < 2:
        return rel.affinity

    depth = clamp(int(session_stats["n_turns"]) / 10.0, 0.0, 1.0)
    warmth = float(session_stats.get("avg_intensity", 0.0)) * float(
        session_stats.get("avg_valence", 0.0)
    )
    initiative = 0.5 if session_stats.get("user_initiated") else -0.2

    pending_responded = session_stats.get("pending_responded")
    if pending_responded is None:
        responsiveness = 0.0
        weights = {"depth": 0.25, "warmth": 0.5, "initiative": 0.25, "resp": 0.0}
    else:
        responsiveness = 0.5 if pending_responded else -0.3
        weights = {"depth": 0.2, "warmth": 0.4, "initiative": 0.2, "resp": 0.2}

    delta = (
        weights["depth"] * depth
        + weights["warmth"] * warmth
        + weights["initiative"] * initiative
        + weights["resp"] * responsiveness
    )
    delta = clamp(delta, -0.05, 0.05)
    rel.affinity = clamp(rel.affinity + delta, 0.0, 1.0)
    return rel.affinity


def decay_affinity(rel: Relation, now: float, dt: float) -> float:
    """长期冷落缓慢降温；短于 3 天不扣，避免正常作息被误伤。"""
    dt = max(0.0, float(dt))
    idle = now - rel.last_interact
    if dt <= 0 or idle <= AFFINITY_IDLE_GRACE:
        return rel.affinity
    rel.affinity = clamp(
        rel.affinity - AFFINITY_DECAY_PER_DAY * dt / 86400.0,
        0.0,
        1.0,
    )
    return rel.affinity
