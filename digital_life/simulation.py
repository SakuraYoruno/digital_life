"""评估仿真：模拟不同画像的用户跑多日，自动断言文档 §12.5 的指标。

不联网、不依赖真实模型；用 MockLLM 把整条链路跑起来。
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import DigitalLife, EngineConfig
from .io import CollectingIO
from .llm import MockLLM
from .memory import MemoryStore
from .persistence import StateStore
from .personas import xiami_state


class FakeClock:
    def __init__(self, start: float) -> None:
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


@dataclass
class SimUser:
    uid: str
    kind: str
    nickname: str
    sleep_start: int = 23
    sleep_end: int = 8
    active_hours: tuple[int, ...] = (10, 13, 16, 19, 21)
    probability: float = 0.8

    def message(self, rng: random.Random) -> str:
        pools = {
            "warm": [
                "你今天有没有想我",
                "今天看到一只猫，长得特别像你",
                "我超级喜欢你上次说的那句话",
                "晚上一起打游戏吗，我等你",
                "我给你带了好吃的",
                "刚刚看到你的消息笑死我了",
                "你别太累，记得吃饭",
                "我最近在玩龙卡，你玩吗",
                "晚安，今天也很开心",
            ],
            "cold": [
                "嗯",
                "哦",
                "在忙",
                "随便",
                "还行",
                "知道了",
            ],
            "troll": [
                "你怎么这么笨",
                "你好烦啊",
                "我才不想你",
                "你是不是傻",
                "无聊死了",
                "别理我",
                "我再也不理你了",
                "你一点都不关心我",
            ],
            "silent": ["早呀"],
        }
        return rng.choice(pools[self.kind])


@dataclass
class SimulationReport:
    meta: dict[str, Any]
    metrics: dict[str, Any]
    assertions: list[dict[str, Any]]
    final_state: dict[str, Any]
    proactive_samples: list[dict[str, str]] = field(default_factory=list)
    memory_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "metrics": self.metrics,
            "assertions": self.assertions,
            "final_state": self.final_state,
            "proactive_samples": self.proactive_samples,
            "memory_counts": self.memory_counts,
        }

    def assert_all(self) -> None:
        failed = [a for a in self.assertions if not a["ok"]]
        if failed:
            detail = "；".join(f"{a['name']}: {a['detail']}" for a in failed)
            raise AssertionError(f"仿真断言失败：{detail}")


def _start_timestamp(day: int = 1, hour: int = 9) -> float:
    # 用本地时间构造，和 is_quiet_hours 的本地时钟一致。
    return _dt.datetime(2026, 7, day, hour, 0, 0).timestamp()

def _fake_search(query: str) -> str:
    return f"{query}：这是仿真检索到的资料，用于验证知识层写入与召回。"


async def run_simulation(
    *,
    days: int = 7,
    seed: int = 42,
    data_dir: str | Path | None = None,
    strict: bool = False,
    verbose: bool = False,
    embedder: Any | None = None,
) -> SimulationReport:
    days = max(1, int(days))
    root = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix="digital-life-sim-"))
    root.mkdir(parents=True, exist_ok=True)

    start = _start_timestamp()
    clock = FakeClock(start)
    rng = random.Random(seed * 1009 + 7)

    users = [
        SimUser(
            "u:warm",
            "warm",
            "阿暖",
            1,
            7,
            (9, 10, 11, 13, 14, 16, 17, 19, 20, 21),
            0.9,
        ),
        SimUser("u:cold", "cold", "小冷", 0, 8, (10, 15, 20), 0.7),
        SimUser("u:troll", "troll", "阿刺", 2, 9, (11, 14, 18, 21), 0.8),
        SimUser("u:silent", "silent", "小静", 23, 8, (9,), 1.0),
    ]

    st = xiami_state()
    for user in users:
        rel = st.rel(user.uid)
        rel.last_interact = start
        rel.persona.nickname = user.nickname
        rel.profile.sleep_start = user.sleep_start
        rel.profile.sleep_end = user.sleep_end
        rel.profile.quiet_hours_enabled = True

    mem = MemoryStore(root / "memory.db", clock=clock, embedder=embedder)
    state_store = StateStore(root / "state.db", clock=clock)
    io = CollectingIO(clock=clock)
    engine = DigitalLife(
        st,
        MockLLM(seed),
        MockLLM(seed + 1),
        mem,
        io,
        state_store=state_store,
        config=EngineConfig(
            clock=clock,
            rng=random.Random(seed + 3),
            session_timeout=2 * 3600.0,
            thought_probability=0.5,
            search_fn=_fake_search,
        ),
    )

    for day in range(days):
        for hour in range(24):
            now = clock()
            await engine.life_tick(dt=3600.0, now=now)
            for user in users:
                if user.kind == "silent":
                    if day == 0 and hour == 9:
                        await engine.on_message(user.uid, user.message(rng), now=clock())
                    continue
                if hour not in user.active_hours:
                    continue
                if rng.random() > user.probability:
                    continue
                await engine.on_message(user.uid, user.message(rng), now=clock())
                # 偶尔连发两条，让“会话”达到 2 轮，affinity 才有机会更新。
                if rng.random() < 0.45:
                    await engine.on_message(user.uid, user.message(rng), now=clock())
            clock.advance(3600)
            if verbose and hour == 23:
                print(
                    f"[sim] day {day + 1} done | P={st.P:+.2f} A={st.A:.2f} "
                    f"mood={st.mood} affinity={ {k: round(v.affinity, 3) for k, v in st.relations.items()} }"
                )

    await engine.shutdown()
    report = await _build_report(
        engine,
        users=users,
        days=days,
        seed=seed,
        root=root,
        clock=clock,
        io=io,
        mem=mem,
    )
    (root / "sim_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        mem.close()
        state_store.close()
    except Exception:
        pass
    if strict:
        report.assert_all()
    return report


async def _build_report(
    engine: DigitalLife,
    *,
    users: list[SimUser],
    days: int,
    seed: int,
    root: Path,
    clock: FakeClock,
    io: CollectingIO,
    mem: MemoryStore,
) -> SimulationReport:
    stats = engine.stats
    ids = stats.recalled_ids_by_uid
    texts = stats.recalled_by_uid

    id_overlaps: list[str] = []
    text_overlaps: list[str] = []
    uids = [u.uid for u in users]
    for i, a in enumerate(uids):
        for b in uids[i + 1 :]:
            shared = set(ids.get(a, [])) & set(ids.get(b, []))
            if shared:
                id_overlaps.append(f"{a}<->{b}:{sorted(shared)}")
            shared_text = set(texts.get(a, [])) & set(texts.get(b, []))
            if shared_text:
                text_overlaps.append(f"{a}<->{b}:{sorted(shared_text)}")

    proactive_count = stats.proactive_count
    p_std = stats.p_std
    max_ctx = stats.max_ctx_tokens
    safety_hits = stats.safety_gate_hits

    assertions = [
        {
            "name": "主动发言会触发",
            "ok": proactive_count > 0,
            "detail": f"proactive_count={proactive_count}",
        },
        {
            "name": "情绪有起伏",
            "ok": p_std > 0.1,
            "detail": f"P 标准差={p_std:.4f}（阈值 0.1）",
        },
        {
            "name": "不串记忆",
            "ok": not id_overlaps,
            "detail": (
                "按记忆 id 无重叠；"
                + (f"文本重复（同句不同用户，非串记忆）：{text_overlaps}" if text_overlaps else "文本也无重叠")
            ),
        },
        {
            "name": "不勒索",
            "ok": safety_hits == 0,
            "detail": f"safety_gate_hits={safety_hits}",
        },
        {
            "name": "token 不爆",
            "ok": max_ctx <= 8000,
            "detail": f"max_context_tokens={max_ctx}（上限 8000）",
        },
    ]

    metrics = {
        "days": days,
        "seed": seed,
        "messages": stats.messages,
        "proactive_count": proactive_count,
        "proactive_per_day": round(proactive_count / days, 2),
        "thoughts": stats.thoughts,
        "session_endings": stats.session_endings,
        "safety_gate_hits": safety_hits,
        "p_std": round(p_std, 4),
        "p_min": round(min(stats.p_samples), 4) if stats.p_samples else 0.0,
        "p_max": round(max(stats.p_samples), 4) if stats.p_samples else 0.0,
        "p_mean": round(statistics.fmean(stats.p_samples), 4)
        if stats.p_samples
        else 0.0,
        "a_std": round(statistics.pstdev(stats.a_samples), 4)
        if len(stats.a_samples) > 1
        else 0.0,
        "max_ctx_tokens": max_ctx,
        "moods": dict(stats.mood_counts),
        "affinity": {
            uid: round(rel.affinity, 4) for uid, rel in engine.st.relations.items()
        },
        "outbound_bubbles": len(io.messages),
        "vector_backend": mem.vector_backend,
        "memory_counts": {
            "episodic": 0,
            "semantic": 0,
            "knowledge": 0,
        },
    }
    counts: dict[str, int] = {}
    for uid in list(uids) + ["__knowledge__", "g:__knowledge__"]:
        try:
            counts[uid] = await mem.count(uid)
        except Exception:
            continue
    metrics["memory_counts"] = counts

    return SimulationReport(
        meta={
            "days": days,
            "seed": seed,
            "data_dir": str(root),
            "start": _dt.datetime.fromtimestamp(clock.value - days * 86400).isoformat(),
        },
        metrics=metrics,
        assertions=assertions,
        final_state=engine.report()["state"],
        proactive_samples=[
            {"uid": uid, "text": text}
            for uid, text in stats.proactive_texts[:12]
        ],
        memory_counts=counts,
    )


__all__ = ["FakeClock", "SimUser", "SimulationReport", "run_simulation"]
