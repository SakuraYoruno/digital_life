"""主循环整合：快循环 on_message + 慢循环 life_tick，共用一把状态锁。

对应开发文档 §13、§16。
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import random
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .appraisal import llm_appraise
from .constants import clamp
from .context import ContextAssembler, ContextBundle
from .dynamics import decay_tick, react_to_event
from .llm import call_llm
from .memory import (
    MemoryStore,
    WorkingMemory,
    detect_group_slang,
    enrich_knowledge,
    maybe_write_episodic,
    perceive,
    roll_summary,
)
from .personas import (
    LifeState,
    Relation,
    SessionStats,
    arousal_knobs,
    build_system_prompt,
    default_state,
    map_mood,
    user_model_text,
)
from .persistence import ObsidianReflector, StateStore
from .persona_evolution import nightly_reflect, update_user_profile
from .proactive import (
    decay_affinity,
    gen_proactive,
    is_quiet_hours,
    pop_due_pending,
    update_affinity,
)
from .safety import (
    apply_dependency_protection,
    dependency_risk,
    safety_gate,
    support_note,
)
from .thoughts import ThoughtResult, thought_tick


@dataclass
class EngineConfig:
    tick_seconds: float = 60.0
    thought_probability: float = 0.5
    session_timeout: float = 2 * 3600.0
    night_hour: int = 2
    autosave_seconds: float = 600.0
    search_fn: Callable[[str], Any] | None = None
    preprocess: Callable[[str], Any] | None = None
    clock: Callable[[], float] | None = None
    rng: random.Random | None = None
    vault_path: str | None = None
    log: Callable[[str], None] | None = None
    group_slang_every: int = 50
    # 直接读 Markdown 的知识库（vault.VaultKB）；给了就优先于向量知识层
    kb: Any = None
    # 会话结束更新画像的门槛：至少说了几条才值得花一次小模型调用
    profile_min_turns: int = 3


@dataclass
class EngineStats:
    messages: int = 0
    proactive_count: int = 0
    safety_gate_hits: int = 0
    thoughts: int = 0
    session_endings: int = 0
    ctx_tokens: list[int] = field(default_factory=list)
    p_samples: list[float] = field(default_factory=list)
    a_samples: list[float] = field(default_factory=list)
    d_samples: list[float] = field(default_factory=list)
    mood_counts: Counter[str] = field(default_factory=Counter)
    recalled_by_uid: dict[str, list[str]] = field(default_factory=dict)
    recalled_ids_by_uid: dict[str, list[int]] = field(default_factory=dict)
    proactive_texts: list[tuple[str, str]] = field(default_factory=list)
    affinity_history: list[tuple[str, float, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_state(self, st: LifeState) -> None:
        self.p_samples.append(st.P)
        self.a_samples.append(st.A)
        self.d_samples.append(st.D)
        self.mood_counts[st.mood] += 1

    @property
    def p_std(self) -> float:
        if len(self.p_samples) < 2:
            return 0.0
        return float(statistics.pstdev(self.p_samples))

    @property
    def max_ctx_tokens(self) -> int:
        return max(self.ctx_tokens, default=0)

    def summary(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "proactive_count": self.proactive_count,
            "safety_gate_hits": self.safety_gate_hits,
            "thoughts": self.thoughts,
            "session_endings": self.session_endings,
            "p_std": round(self.p_std, 4),
            "max_ctx_tokens": self.max_ctx_tokens,
            "moods": dict(self.mood_counts),
            "affinity": {
                uid: round(value, 4)
                for uid, value, _ in self.affinity_history[-10:]
            },
            "errors": self.errors[-10:],
        }


@dataclass
class MessageResult:
    reply: str
    raw_reply: str
    mood: str
    appraisal: dict[str, Any]
    knobs: dict[str, Any]
    ctx_tokens: int
    recalled: list[Any] = field(default_factory=list)
    blocked: bool = False


@dataclass
class TickResult:
    dt: float
    thought: ThoughtResult | None = None
    sent: list[tuple[str, str]] = field(default_factory=list)


class DigitalLife:
    def __init__(
        self,
        st: LifeState,
        small_llm: Any,
        big_llm: Any,
        mem: MemoryStore,
        io: Any,
        *,
        state_store: StateStore | None = None,
        config: EngineConfig | None = None,
        reflector: ObsidianReflector | None = None,
    ) -> None:
        self.st = st or default_state()
        self.small = small_llm
        self.big = big_llm
        self.mem = mem
        self.io = io
        self.config = config or EngineConfig()
        self.state_store = state_store
        self.clock = self.config.clock or time.time
        self.rng = self.config.rng or random.Random()
        self.lock = asyncio.Lock()
        self.thought_hist: deque[str] = deque(maxlen=20)
        self.thought_texts: deque[str] = deque(maxlen=20)
        self._wm: dict[str, WorkingMemory] = {}
        self._uid_locks: dict[str, asyncio.Lock] = {}
        self._group_msgs: dict[str, deque[str]] = {}
        # 群聊共享时间线：最近若干条（谁说的、说了什么、什么时候）。
        # 工作记忆是按 uid 分的，缺了这块，她接不住"别人刚才说的话"。
        self._group_turns: dict[str, deque[tuple[str, str, float]]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._support_until: dict[str, tuple[float, str]] = {}
        self._last_tick = self.clock()
        self._last_save = self.clock()
        self._last_reflect_date: _dt.date | None = None
        self._running = False
        self._closed = False

        self.stats = EngineStats()
        self.assembler = ContextAssembler(
            search_fn=self.config.search_fn, kb=self.config.kb
        )

        if reflector is not None:
            self.reflector = reflector
        elif self.config.vault_path:
            self.reflector = ObsidianReflector(self.config.vault_path)
        else:
            self.reflector = None

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _now(self, now: float | None = None) -> float:
        return float(self.clock() if now is None else now)

    def _uid_lock(self, uid: str) -> asyncio.Lock:
        if uid not in self._uid_locks:
            self._uid_locks[uid] = asyncio.Lock()
        return self._uid_locks[uid]

    def _get_wm(self, uid: str, rel: Relation) -> WorkingMemory:
        if uid not in self._wm:
            self._wm[uid] = WorkingMemory(
                max_turns=20,
                max_tokens=3000,
                keep_head=2,
                keep_tail=6,
                compress_batch=5,
                on_compress=lambda turns: self._compress_wm(uid, turns),
                clock=self.clock,
            )
        return self._wm[uid]

    async def _compress_wm(self, uid: str, turns: list[Any]) -> None:
        async with self.lock:
            rel = self.st.rel(uid)
            await roll_summary(self.small, rel, turns)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            self._tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.stats.errors.append(f"{type(exc).__name__}: {exc}")

        task.add_done_callback(_done)
        return task

    async def drain(self) -> None:
        """等待后台知识补全 / 摘要压缩结束。"""
        for _ in range(3):
            if not self._tasks:
                break
            tasks = list(self._tasks)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self.stats.errors.append(f"{type(result).__name__}: {result}")
            self._tasks.difference_update(tasks)
        for wm in list(self._wm.values()):
            await wm.flush()

    def _group_context(
        self, group_id: str, *, now: float, window: float = 1800.0, limit: int = 8
    ) -> str:
        """群里刚才在聊什么——不含当前这一条（它已经作为【对方刚说】出现了）。"""
        timeline = self._group_turns.get(group_id)
        if not timeline:
            return ""
        rows = [(who, text) for who, text, ts in timeline if now - ts <= window]
        if len(rows) > 1:
            rows = rows[:-1]
        if not rows:
            return ""
        return "\n".join(f"{who}：{text}" for who, text in rows[-limit:])

    def _active_support(self, uid: str, now: float) -> str:
        item = self._support_until.get(uid)
        if not item:
            return ""
        if item[0] > now:
            return item[1]
        self._support_until.pop(uid, None)
        return ""

    def _record_recalled(self, uid: str, recalled: list[Any]) -> None:
        bucket = self.stats.recalled_by_uid.setdefault(uid, [])
        ids = self.stats.recalled_ids_by_uid.setdefault(uid, [])
        for hit in recalled:
            if getattr(hit, "kind", "") not in ("episodic", "semantic", "reflection"):
                continue
            text = getattr(hit, "text", "")
            mem_id = int(getattr(hit, "id", 0))
            if text and text not in bucket:
                bucket.append(text)
            if mem_id and mem_id not in ids:
                ids.append(mem_id)
        if len(bucket) > 500:
            del bucket[:-500]
        if len(ids) > 500:
            del ids[:-500]

    # ------------------------------------------------------------------
    # 快循环
    # ------------------------------------------------------------------
    async def on_message(
        self,
        uid: str,
        raw_text: str,
        group_id: str | None = None,
        *,
        now: float | None = None,
        speak: bool = True,
        speaker: str = "",
    ) -> MessageResult:
        """处理一条消息。

        `speak=False` 是"只听不说"：感知、appraisal、情绪状态、工作记忆照常更新，
        但不调用主模型、不发送。群聊里"她听见了但没搭话"走这条路——代价只有
        本地小模型，不需要为每条没接的话付一次大模型的钱。
        """
        uid = str(uid)
        async with self._uid_lock(uid):
            now = self._now(now)
            rel = self.st.rel(uid)
            if (
                rel.session_active
                and now - rel.last_interact > self.config.session_timeout
            ):
                await self._end_session(uid, now)

            # 层0：感知缓冲（实体抽取不碰共享状态，放在锁外）
            percept = await perceive(
                self.small, raw_text, uid, preprocess=self.config.preprocess
            )
            if group_id:
                bucket = self._group_msgs.setdefault(
                    group_id, deque(maxlen=max(10, self.config.group_slang_every))
                )
                bucket.append(percept.clean_text)
                # 共享时间线：不管她这次说没说话，都先记下来。
                timeline = self._group_turns.setdefault(group_id, deque(maxlen=60))
                timeline.append(
                    ((speaker or uid), percept.clean_text[:80], now)
                )
                if len(bucket) >= self.config.group_slang_every:
                    batch = list(bucket)
                    bucket.clear()
                    self._spawn(
                        detect_group_slang(
                            self.small,
                            self.mem,
                            group_id,
                            batch,
                            now=now,
                        )
                    )
            rel = self.st.rel(uid)
            appraisal_ctx = rel.session_summary
            profile_text = user_model_text(rel)
            if profile_text:
                appraisal_ctx = f"{appraisal_ctx}\n用户模型：{profile_text}"
            appraisal = await llm_appraise(
                self.small, percept.clean_text, appraisal_ctx
            )
            risk = dependency_risk(percept.clean_text)

            # 状态更新 + 组装 system prompt，全部在锁内完成
            async with self.lock:
                rel = self.st.rel(uid)
                wm = self._get_wm(uid, rel)
                if not rel.session_active:
                    rel.session = SessionStats()
                    rel.session_active = True
                if rel.session.n_turns == 0:
                    rel.session.user_initiated = not rel.proactive_awaiting_reply
                if rel.proactive_awaiting_reply:
                    rel.session.pending_responded = True
                    rel.proactive_awaiting_reply = False
                    rel.proactive_backoff = 1.0

                react_to_event(self.st, appraisal)
                map_mood(self.st)
                rel.last_interact = now
                rel.session.observe(appraisal["valence"], appraisal["intensity"])
                self.stats.record_state(self.st)

                if not speak:
                    # 只听不说：把话记进工作记忆，情绪该起反应也起过了，到此为止。
                    wm.add("user", percept.clean_text, self.st.mood)
                    self._maybe_save(now)
                    return MessageResult(
                        reply="",
                        raw_reply="",
                        mood=self.st.mood,
                        appraisal=appraisal,
                        knobs={},
                        ctx_tokens=0,
                        blocked=False,
                    )

                self.stats.messages += 1

                if risk:
                    self._support_until[uid] = (now + 3600.0, risk)
                    apply_dependency_protection(rel, risk, now)
                support_kind = self._active_support(uid, now)

                # 群聊里第一次见到这个人，就把显示名记成"她对你的称呼"，
                # 免得下次还要靠上下文猜"刚才那个小虾米是谁"。
                if speaker and not rel.persona.nickname:
                    rel.persona.nickname = str(speaker)[:16]

                knobs = arousal_knobs(self.st, self.rng)
                sys_prompt, knobs = build_system_prompt(
                    self.st,
                    rel,
                    knobs=knobs,
                    self_narrative=self.st.self_narrative,
                    support_note=support_note(support_kind) if support_kind else "",
                    speaker=speaker,
                )
                mood = self.st.mood
                summary = rel.session_summary

            ctx: ContextBundle = await self.assembler.build(
                uid=uid,
                user_msg=percept.clean_text,
                wm=wm,
                mem=self.mem,
                sys_prompt=sys_prompt,
                mood=mood,
                summary=summary,
                group_id=group_id,
                group_context=(
                    self._group_context(group_id, now=now) if group_id else ""
                ),
                knobs=knobs,
                small_llm=self.small,
                now=now,
            )
            self.stats.ctx_tokens.append(ctx.tokens)
            self._record_recalled(uid, ctx.recalled)

            raw_reply = await call_llm(
                self.big,
                ctx.text,
                system=ctx.system_prompt,
                temperature=knobs["temperature"],
            )
            safe = safety_gate(raw_reply, self.st.core.taboos)
            blocked = not bool(safe)
            if blocked:
                self.stats.safety_gate_hits += 1
                retry = await self._regenerate_safe(ctx, knobs)
                safe = retry
                blocked = not bool(safe)

            # 写入工作记忆（用户轮永远保留；被拦下的回复不写）
            wm.add("user", percept.clean_text, mood)
            if safe:
                wm.add("assistant", safe, mood)

            if safe:
                await maybe_write_episodic(
                    self.small,
                    self.mem,
                    uid,
                    percept,
                    safe,
                    float(appraisal["intensity"]),
                    now=now,
                )

            if percept.entities:
                self._spawn(
                    enrich_knowledge(
                        self.small,
                        self.mem,
                        percept.entities,
                        group_id=group_id,
                        search_fn=self.config.search_fn,
                        now=now,
                    )
                )

            if safe:
                await self.io.send(uid, safe, knobs)

            self._maybe_save(now)
            return MessageResult(
                reply=safe,
                raw_reply=raw_reply,
                mood=mood,
                appraisal=appraisal,
                knobs=knobs,
                ctx_tokens=ctx.tokens,
                recalled=ctx.recalled,
                blocked=blocked,
            )

    async def _regenerate_safe(self, ctx: ContextBundle, knobs: dict[str, Any]) -> str:
        prompt = (
            ctx.text
            + "\n\n【修正】刚才那句碰到了安全底线（查岗/勒索/负罪感）。"
            "换一种不索取、不施压的说法，只输出一句话。"
        )
        try:
            raw = await call_llm(
                self.big,
                prompt,
                system=ctx.system_prompt,
                temperature=min(1.0, float(knobs.get("temperature", 0.8))),
            )
        except Exception:
            return ""
        return safety_gate(raw, self.st.core.taboos)

    # ------------------------------------------------------------------
    # 慢循环
    # ------------------------------------------------------------------
    async def life_tick(
        self, dt: float | None = None, *, now: float | None = None
    ) -> TickResult:
        now = self._now(now)
        elapsed = (now - self._last_tick) if dt is None else float(dt)
        elapsed = clamp(elapsed, 0.0, 12 * 3600.0)
        self._last_tick = now

        async with self.lock:
            decay_tick(self.st, elapsed)
            for rel in self.st.relations.values():
                decay_affinity(rel, now, elapsed)
            map_mood(self.st)
            self.stats.record_state(self.st)
            thought: ThoughtResult | None = None
            if self.rng.random() < self.config.thought_probability:
                thought = await thought_tick(
                    self.small, self.st, self.thought_hist, rng=self.rng
                )
                self.thought_texts.append(thought.text)
                self.stats.thoughts += 1
            expired = [
                uid
                for uid, rel in self.st.relations.items()
                if rel.session_active
                and now - rel.last_interact > self.config.session_timeout
            ]

        # 会话结束要先于主动发言：结算完旧会话，再决定要不要找她。
        for uid in expired:
            await self._end_session(uid, now)

        due: list[tuple[str, str]] = []
        async with self.lock:
            for uid, rel in list(self.st.relations.items()):
                if (
                    rel.session_active
                    and now - rel.last_interact < self.config.session_timeout
                ):
                    continue
                if is_quiet_hours(rel, now):
                    continue
                await gen_proactive(
                    self.small,
                    self.st,
                    rel,
                    now,
                    self.thought_texts,
                )
                message = pop_due_pending(rel, now)
                if message:
                    due.append((uid, message))

        sent: list[tuple[str, str]] = []
        for uid, message in due:
            safe = safety_gate(message, self.st.core.taboos)
            if not safe:
                self.stats.safety_gate_hits += 1
                continue
            await self.io.send(uid, safe, arousal_knobs(self.st, self.rng))
            self.stats.proactive_count += 1
            self.stats.proactive_texts.append((uid, safe))
            sent.append((uid, safe))
            async with self.lock:
                rel = self.st.rel(uid)
                rel.proactive_awaiting_reply = True
                rel.proactive_sent_at = now

        self._maybe_save(now)
        return TickResult(dt=elapsed, thought=thought, sent=sent)

    async def _end_session(self, uid: str, now: float | None = None) -> dict[str, Any] | None:
        uid = str(uid)
        now = self._now(now)
        async with self.lock:
            rel = self.st.rel(uid)
            if not rel.session_active:
                return None
            snapshot = rel.session.as_dict()
            if rel.proactive_awaiting_reply:
                snapshot["pending_responded"] = False
            rel.session_active = False
            wm = self._wm.get(uid)

        if wm is not None:
            wm.finalize()
            await wm.flush()

        async with self.lock:
            rel = self.st.rel(uid)
            update_affinity(rel, snapshot)
            rel.session = SessionStats()
            if rel.proactive_awaiting_reply:
                rel.proactive_backoff = min(
                    8.0, max(1.0, rel.proactive_backoff * 1.8)
                )
                rel.proactive_awaiting_reply = False
            self.stats.session_endings += 1
            self.stats.affinity_history.append((uid, rel.affinity, now))
            enough_turns = int(snapshot.get("n_turns", 0)) >= self.config.profile_min_turns

        self.save_state(now)

        # 画像在这里更新，而不是只在夜里：夜里一次要跑所有 uid，白天聊完随手就更新，
        # 下次开口就能用上「他讨厌被叫全名」这类结论。只对聊够几条的人做，省调用。
        if enough_turns:
            self._spawn(self._refresh_profile(uid, now))
        return snapshot

    async def _refresh_profile(self, uid: str, now: float) -> None:
        try:
            await update_user_profile(self.small, self.mem, uid, self.st.rel(uid), now=now)
        except Exception as exc:  # noqa: BLE001 - 画像失败不能影响主流程
            self.stats.errors.append(f"user_profile: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 夜间反思 / 持久化 / 运行
    # ------------------------------------------------------------------
    async def _maybe_night_reflect(self, now: float | None = None) -> None:
        now = self._now(now)
        stamp = _dt.datetime.fromtimestamp(now)
        if stamp.hour != int(self.config.night_hour):
            return
        if self._last_reflect_date == stamp.date():
            return
        self._last_reflect_date = stamp.date()
        self._spawn(
            nightly_reflect(
                self.small,
                self.st,
                self.mem,
                now=now,
                reflector=self.reflector,
            )
        )

    def save_state(self, now: float | None = None) -> None:
        if self.state_store is None:
            return
        now = self._now(now)
        try:
            self.state_store.save(self.st, now=now)
            self._last_save = now
        except Exception as exc:
            self.stats.errors.append(f"save_state: {type(exc).__name__}: {exc}")

    def _maybe_save(self, now: float) -> None:
        if (
            self.state_store is not None
            and now - self._last_save >= self.config.autosave_seconds
        ):
            self.save_state(now)

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                now = self._now()
                try:
                    await self.life_tick(now=now)
                except Exception as exc:
                    self.stats.errors.append(
                        f"life_tick: {type(exc).__name__}: {exc}"
                    )
                try:
                    await self._maybe_night_reflect(now)
                except Exception as exc:
                    self.stats.errors.append(
                        f"night_reflect: {type(exc).__name__}: {exc}"
                    )
                self._maybe_save(self._now())
                await asyncio.sleep(max(1.0, float(self.config.tick_seconds)))
        finally:
            await self.drain()
            self.save_state()

    def stop(self) -> None:
        self._running = False

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._running = False
        await self.drain()
        now = self._now()
        for uid in list(self.st.relations):
            try:
                await self._end_session(uid, now)
            except Exception as exc:
                self.stats.errors.append(
                    f"end_session: {type(exc).__name__}: {exc}"
                )
        await self.drain()
        self.save_state(now)
        self._closed = True

    async def close(self) -> None:
        await self.shutdown()
        try:
            self.mem.close()
        except Exception:
            pass
        if self.state_store is not None:
            try:
                self.state_store.close()
            except Exception:
                pass

    def report(self) -> dict[str, Any]:
        data = self.stats.summary()
        data["state"] = {
            "P": round(self.st.P, 4),
            "A": round(self.st.A, 4),
            "D": round(self.st.D, 4),
            "mood": self.st.mood,
            "drive": {k: round(v, 4) for k, v in self.st.drive.items()},
            "affinity": {
                uid: round(rel.affinity, 4)
                for uid, rel in self.st.relations.items()
            },
        }
        return data


__all__ = [
    "DigitalLife",
    "EngineConfig",
    "EngineStats",
    "MessageResult",
    "TickResult",
]
