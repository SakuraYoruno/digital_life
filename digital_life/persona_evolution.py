"""人格养成、夜间反思、自我叙事与记忆治理。

对应开发文档 §11、§12.1、§12.4、§12.5。
"""

from __future__ import annotations

import time
from typing import Any

from .constants import clamp, safe_json, truncate_to_tokens
from .llm import call_llm
from .memory import MemoryStore
from .personas import LifeState, Relation


def evolve_persona(st: LifeState, rel: Relation, reflection: dict[str, Any]) -> None:
    """三约束：Core 锁死；Plastic 单次 ±0.05；边界只进 Relational。"""
    p = st.plastic

    drift = reflection.get("style_drift") or {}
    if isinstance(drift, dict):
        for key, delta in drift.items():
            if key in p.style_traits:
                try:
                    step = clamp(float(delta), -0.05, 0.05)
                except (TypeError, ValueError):
                    continue
                p.style_traits[key] = clamp(p.style_traits[key] + step, 0.0, 1.0)

    boundary = str(reflection.get("boundary") or "").strip()
    if boundary and boundary not in rel.persona.boundaries:
        rel.persona.boundaries.append(boundary)

    insight = str(reflection.get("insight") or "").strip()
    if insight:
        rel.persona.strategy = insight

    p.version += 1


async def update_user_profile(
    small_llm: Any,
    mem: MemoryStore,
    uid: str,
    rel: Relation,
    *,
    now: float | None = None,
) -> bool:
    """从语义记忆维护结构化用户模型（§12.3）。"""
    facts = await mem.recent(uid, kind="semantic", k=20, now=now)
    if not facts:
        return False
    prompt = f"""从以下关于用户的稳定事实中，更新结构化用户模型，输出JSON：
{{"preferred_name": "称呼（不确定则空串）",
  "sleep_start": 23, "sleep_end": 8,
  "notes": {{"雷区": "...", "在意的事": "..."}}}}
只写有依据的字段，无法判断的不要编。事实：{[f.text for f in facts]}"""
    try:
        raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.2)
        data = safe_json(raw, {})
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    changed = False
    preferred = str(data.get("preferred_name") or "").strip()
    if preferred:
        rel.profile.preferred_name = preferred
        changed = True
    for field_name in ("sleep_start", "sleep_end"):
        try:
            hour = int(data.get(field_name, rel.profile.sleep_start))
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23 and hour != getattr(rel.profile, field_name):
            setattr(rel.profile, field_name, hour)
            changed = True
    notes = data.get("notes")
    if isinstance(notes, dict):
        for key, value in list(notes.items())[:8]:
            if value in (None, "", [], {}):
                continue
            rel.profile.notes[str(key)] = value
            changed = True
    return changed


async def extract_semantic_facts(
    small_llm: Any,
    mem: MemoryStore,
    uid: str,
    *,
    now: float | None = None,
) -> int:
    """夜间从近期情景记忆提炼稳定事实，不手动写。"""
    episodes = await mem.recent(uid, kind="episodic", k=30, now=now)
    if not episodes:
        return 0
    prompt = f"""从以下情景记忆提炼关于该用户的稳定事实，输出JSON数组：
[{{"fact": "用户喜欢玩龙卡", "confidence": 0.9, "category": "兴趣"}}]
类别：兴趣/作息/性格/雷区/习惯
记忆：{[e.text for e in episodes]}"""
    try:
        raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.2)
        facts = safe_json(raw, [])
    except Exception:
        return 0
    if not isinstance(facts, list):
        return 0

    written = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("fact") or "").strip()
        if not text:
            continue
        try:
            confidence = float(fact.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < 0.6:
            continue
        existing = await mem.search(uid, text, k=1, kind="semantic", now=now)
        if existing and existing[0].score > 0.9:
            await mem.update(existing[0].id, text=text)
        else:
            await mem.write(
                uid,
                "semantic",
                text,
                importance=confidence,
                category=str(fact.get("category") or ""),
                now=now,
            )
        written += 1
    return written


async def update_self_narrative(
    small_llm: Any,
    st: LifeState,
    insights: list[str],
    *,
    now: float | None = None,
) -> str:
    """增量维护“我是谁”的第一人称自述（§12.1）。"""
    if not insights and st.self_narrative:
        return st.self_narrative
    prompt = f"""更新你对自己的自述，第一人称，150字以内。
保留“我是谁”的连续性：来历、喜好、与这个人的故事线都可以写一点，
但不要编造没发生过的具体事件。只输出自述正文。
名字：{st.core.name}
风格：{st.core.style_skeleton}
价值观：{st.core.values}
已有自述：{st.self_narrative or '（无）'}
最近的感悟：{insights}"""
    try:
        text = (await call_llm(small_llm, prompt, temperature=0.6)).strip()
    except Exception:
        return st.self_narrative
    if text:
        st.self_narrative = truncate_to_tokens(text, 500)
    return st.self_narrative


async def nightly_reflect(
    small_llm: Any,
    st: LifeState,
    mem: MemoryStore,
    *,
    now: float | None = None,
    reflector: Any | None = None,
    run_gc: bool | None = None,
) -> dict[str, Any]:
    """每天一次：关系反思 + 人格演化 + 语义提炼 + 自我叙事 + 记忆治理。"""
    now = time.time() if now is None else now
    result: dict[str, Any] = {
        "uids": [],
        "evolved": 0,
        "facts": 0,
        "gc": {},
        "insights": [],
    }

    for uid, rel in list(st.relations.items()):
        recent = await mem.recent(uid, k=20, now=now)
        if not recent:
            continue
        result["uids"].append(uid)
        prompt = f"""回顾与该用户最近互动，输出 JSON：
  insight: 一条关于如何跟TA相处的感悟
  boundary: 新发现需遵守的边界（没有则 ""）
  style_drift: {{exclaim_rate,emoji_rate,avg_len}} 的微调建议(每项±0.05内)
素材：{[h.text for h in recent]}"""
        try:
            raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.4)
            reflection = safe_json(raw, {})
        except Exception:
            reflection = {}
        if not isinstance(reflection, dict):
            reflection = {}
        evolve_persona(st, rel, reflection)
        result["evolved"] += 1
        insight = str(reflection.get("insight") or "")
        if insight:
            result["insights"].append(insight)
            await mem.kv_set(f"reflection:{uid}", insight, now=now)
            await mem.write(
                uid,
                "reflection",
                insight,
                importance=0.8,
                category="相处感悟",
                now=now,
            )
        if reflector is not None:
            try:
                reflector.write_reflection(uid, reflection, rel=rel, now=now)
            except Exception:
                pass
        facts_written = await extract_semantic_facts(
            small_llm, mem, uid, now=now
        )
        result["facts"] += facts_written
        if facts_written:
            await update_user_profile(small_llm, mem, uid, rel, now=now)

    if result["insights"]:
        await update_self_narrative(small_llm, st, result["insights"], now=now)
        await mem.kv_set("self_narrative", st.self_narrative, now=now)

    if run_gc is None:
        last_raw = await mem.kv_get("last_gc")
        try:
            last_gc = float(last_raw) if last_raw else 0.0
        except (TypeError, ValueError):
            last_gc = 0.0
        run_gc = (now - last_gc) >= 7 * 86400 or last_gc == 0.0

    if run_gc:
        for uid in list(st.relations):
            result["gc"][uid] = await memory_gc(small_llm, mem, uid, now=now)
        await mem.kv_set("last_gc", str(now), now=now)

    if reflector is not None:
        try:
            reflector.write_self_narrative(st)
            reflector.write_relation_graph(st)
        except Exception:
            pass
    return result


async def memory_gc(
    small_llm: Any,
    mem: MemoryStore,
    uid: str,
    *,
    now: float | None = None,
) -> dict[str, int]:
    """每周 GC：遗忘 / 合并 / 降级 / TTL 清理。"""
    now = time.time() if now is None else now
    stats = {"forgotten": 0, "merged": 0, "downgraded": 0, "expired": 0}
    items = await mem.list_all(uid, kind="episodic")
    deleted: set[int] = set()

    for item in items:
        if item.id in deleted:
            continue
        age_days = max(0.0, (now - item.ts) / 86400.0)

        if item.importance < 0.3 and item.hit_count == 0 and age_days > 30:
            await mem.delete(item.id)
            deleted.add(item.id)
            stats["forgotten"] += 1
            continue

        similar = await mem.find_similar(
            item, threshold=0.82, uid=uid, kind="episodic"
        )
        similar = [s for s in similar if s.id not in deleted]
        if len(similar) >= 3:
            source = [item.text] + [s.text for s in similar]
            try:
                merged = await call_llm(
                    small_llm,
                    f"把这几条相似记忆合并成一条语义事实：{source}",
                    temperature=0.3,
                )
            except Exception:
                merged = source[0]
            merged = (merged or source[0]).strip()
            await mem.write(
                uid, "semantic", merged, importance=0.8, category="合并", now=now
            )
            for s in [item] + similar:
                await mem.delete(s.id)
                deleted.add(s.id)
            stats["merged"] += 1
            continue

        if age_days > 90:
            await mem.update_importance(item.id, item.importance * 0.5)
            stats["downgraded"] += 1

    stats["expired"] = await mem.delete_expired(uid, now=now)
    return stats
