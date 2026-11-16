"""命令行入口。

    python -m digital_life chat --mock
    python -m digital_life sim --days 7 --seed 42 --strict
    python -m digital_life demo
    python -m digital_life reflect --gc
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from .embedding import make_embedder
from .env_loader import load_windows_environment
from .engine import DigitalLife, EngineConfig
from .io import CollectingIO, ConsoleIO
from .llm import llm_from_env
from .memory import MemoryStore
from .persona_evolution import memory_gc, nightly_reflect
from .persistence import ObsidianReflector, StateStore
from .personas import default_state, xiami_state
from .simulation import FakeClock, run_simulation


def _default_data_dir() -> Path:
    return Path(os.getenv("DIGITAL_LIFE_DATA_DIR", "data"))


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, data_dir / "memory.db", data_dir / "state.db"


def _reflector(args: argparse.Namespace) -> ObsidianReflector | None:
    vault = args.vault or os.getenv("DIGITAL_LIFE_VAULT")
    return ObsidianReflector(vault) if vault else None


def _make_embedder(args: argparse.Namespace, backend: str | None = None):
    """按 --embed / 环境变量构造 embedding；auto 不可用时回退哈希。"""
    try:
        return make_embedder(
            backend or getattr(args, "embed", "auto"),
            model=getattr(args, "embed_model", None),
            dim=getattr(args, "embed_dim", None),
            base_url=getattr(args, "ollama_url", None),
        )
    except Exception as exc:
        print(f"（embedding 初始化失败，回退内置哈希：{exc}）")
        return None


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


async def cmd_chat(args: argparse.Namespace) -> int:
    data_dir, mem_path, state_path = _paths(args)
    store = StateStore(state_path, clock=time.time)
    st = store.load()
    if st is None:
        st = default_state() if args.persona == "default" else xiami_state()
    small, big = llm_from_env(mock=args.mock, seed=args.seed)
    embedder = _make_embedder(args)
    mem = MemoryStore(mem_path, clock=time.time, embedder=embedder)
    io = ConsoleIO(prefix=st.core.name)
    reflector = _reflector(args)
    engine = DigitalLife(
        st,
        small,
        big,
        mem,
        io,
        state_store=store,
        config=EngineConfig(
            clock=time.time,
            rng=None,
            vault_path=str(reflector.vault) if reflector else None,
            search_fn=None,
        ),
        reflector=reflector,
    )

    print(f"—— 数字生命 · {st.core.name} ——")
    print(f"记忆向量后端：{mem.vector_backend}")
    print(
        f"对话模型：{getattr(big, 'model', type(big).__name__)} / "
        f"小模型：{getattr(small, 'model', type(small).__name__)}"
    )
    print("输入 /quit 退出，/mood 看当前状态，/thought 催一条念头，/tick 过一小时。")
    if not hasattr(small, "model") and not hasattr(big, "model"):
        print("（离线 MockLLM；设置 DIGITAL_LIFE_SMALL_MODEL / API key 可接真实模型）")
    print()

    while True:
        try:
            text = await asyncio.to_thread(input, "你：")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = text.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "退出"):
            break
        if text == "/mood":
            print(
                f"　P={st.P:+.3f} A={st.A:.3f} D={st.D:+.3f} "
                f"mood={st.mood} drive={ {k: round(v, 3) for k, v in st.drive.items()} }"
            )
            continue
        if text == "/tick":
            tick = await engine.life_tick(dt=3600, now=time.time())
            print(f"　[慢循环] dt={tick.dt:.0f}s thought={tick.thought.text if tick.thought else '—'}")
            for uid, msg in tick.sent:
                print(f"　[主动→{uid}] {msg}")
            continue
        if text == "/thought":
            tick = await engine.life_tick(dt=0, now=time.time())
            print(f"　[念头] {tick.thought.text if tick.thought else '（这轮没冒念头）'}")
            continue
        result = await engine.on_message(args.uid, text)
        if not result.reply:
            print("　（这句话被安全闸拦下了，没有发送）")
        if args.show_state:
            print(
                f"　[mood={result.mood} P={st.P:+.2f} A={st.A:.2f} "
                f"D={st.D:+.2f} tokens={result.ctx_tokens}]"
            )

    await engine.close()
    if embedder is not None:
        embedder.close()
    print(f"状态已保存到 {state_path}")
    return 0


# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------


async def cmd_sim(args: argparse.Namespace) -> int:
    # 仿真默认走哈希，保证可复现、且不会对 Ollama 发几百次请求；
    # 显式 --embed ollama 才启用真 embedding。
    backend = "hash" if args.embed == "auto" else args.embed
    embedder = _make_embedder(args, backend=backend)
    report = await run_simulation(
        days=args.days,
        seed=args.seed,
        data_dir=args.data_dir,
        strict=False,
        verbose=args.verbose,
        embedder=embedder,
    )
    if embedder is not None:
        embedder.close()
    print("\n=== 评估仿真 ===")
    print(json.dumps(report.metrics, ensure_ascii=False, indent=2))
    print("\n=== 断言 ===")
    exit_code = 0
    for item in report.assertions:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"[{mark}] {item['name']} — {item['detail']}")
        if not item["ok"]:
            exit_code = 1
    if args.strict and exit_code:
        print("\nstrict 模式：断言失败。")
    path = Path(report.meta["data_dir"]) / "sim_report.json"
    print(f"\n报告：{path}")
    return exit_code if args.strict else 0


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


async def cmd_demo(args: argparse.Namespace) -> int:
    from .llm import MockLLM

    clock = FakeClock(time.time())
    st = xiami_state()
    st.rel("demo").persona.nickname = "你"
    embedder = _make_embedder(args)
    mem = MemoryStore(":memory:", clock=clock, embedder=embedder)
    io = CollectingIO(clock=clock)
    engine = DigitalLife(
        st,
        MockLLM(args.seed),
        MockLLM(args.seed + 1),
        mem,
        io,
        config=EngineConfig(
            clock=clock,
            rng=random.Random(args.seed),
            search_fn=lambda q: f"{q}：这是演示资料。",
        ),
    )
    script = [
        ("demo", "今天好累啊，被老板骂了"),
        ("demo", "不过看到你我就好点了"),
        ("demo", "晚安，明天见"),
    ]
    for i, (uid, text) in enumerate(script):
        result = await engine.on_message(uid, text, now=clock())
        print(f"你：{text}")
        print(f"她：{result.reply or '（安全闸拦截）'}")
        print(
            f"  [mood={result.mood} P={st.P:+.2f} A={st.A:.2f} D={st.D:+.2f} "
            f"drive={ {k: round(v, 2) for k, v in st.drive.items()} }]"
        )
        clock.advance(2 * 3600 if i < 2 else 6 * 3600)
        tick = await engine.life_tick(dt=2 * 3600 if i < 2 else 6 * 3600, now=clock())
        if tick.thought:
            print(f"  [独处念头] {tick.thought.text}")
        for _, msg in tick.sent:
            print(f"  [她主动说] {msg}")
        print()
    print("=== 最终状态 ===")
    print(json.dumps(engine.report(), ensure_ascii=False, indent=2))
    await engine.close()
    if embedder is not None:
        embedder.close()
    return 0


# ---------------------------------------------------------------------------
# reflect / gc
# ---------------------------------------------------------------------------


async def cmd_reflect(args: argparse.Namespace) -> int:
    data_dir, mem_path, state_path = _paths(args)
    store = StateStore(state_path, clock=time.time)
    st = store.load()
    if st is None:
        print("没有找到状态。先 chat / sim 跑一点数据。")
        return 1
    small, _ = llm_from_env(mock=args.mock, seed=args.seed)
    embedder = _make_embedder(args)
    mem = MemoryStore(mem_path, clock=time.time, embedder=embedder)
    reflector = _reflector(args)
    result = await nightly_reflect(
        small,
        st,
        mem,
        now=time.time(),
        reflector=reflector,
        run_gc=args.gc or None,
    )
    store.save(st)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    mem.close()
    if embedder is not None:
        embedder.close()
    store.close()
    return 0


async def cmd_gc(args: argparse.Namespace) -> int:
    data_dir, mem_path, state_path = _paths(args)
    small, _ = llm_from_env(mock=True, seed=args.seed)
    embedder = _make_embedder(args)
    mem = MemoryStore(mem_path, clock=time.time, embedder=embedder)
    uid = args.uid
    stats = await memory_gc(small, mem, uid, now=time.time())
    print(json.dumps({uid: stats}, ensure_ascii=False, indent=2))
    mem.close()
    if embedder is not None:
        embedder.close()
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital_life",
        description="数字生命：情绪/驱力/记忆/人格的极简单体实现",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=str(_default_data_dir()))
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--mock", action="store_true", help="强制离线 MockLLM")
    common.add_argument("--vault", default=None, help="Obsidian 仓库路径")
    common.add_argument(
        "--persona", choices=("xiami", "default"), default="xiami"
    )
    common.add_argument(
        "--embed",
        choices=("auto", "hash", "ollama"),
        default="auto",
        help="记忆向量：auto=有 Ollama 模型就用，hash=内置哈希，ollama=强制 Ollama",
    )
    common.add_argument("--embed-model", default=None, help="Ollama embedding 模型，默认 bge-m3")
    common.add_argument("--embed-dim", type=int, default=None, help="embedding 维度，默认按模型推断")
    common.add_argument("--ollama-url", default=None, help="Ollama 地址，默认 http://127.0.0.1:11434")

    chat = sub.add_parser("chat", parents=[common], help="控制台聊天")
    chat.add_argument("--uid", default="owner")
    chat.add_argument("--show-state", action="store_true")
    chat.set_defaults(func=cmd_chat)

    sim = sub.add_parser("sim", parents=[common], help="多日评估仿真")
    sim.add_argument("--days", type=int, default=7)
    sim.add_argument("--strict", action="store_true", help="断言失败时返回非零")
    sim.add_argument("--verbose", action="store_true")
    sim.set_defaults(func=cmd_sim)

    demo = sub.add_parser("demo", parents=[common], help="一分钟演示")
    demo.set_defaults(func=cmd_demo)

    reflect = sub.add_parser("reflect", parents=[common], help="执行一次夜间反思")
    reflect.add_argument("--gc", action="store_true", help="强制跑记忆治理")
    reflect.set_defaults(func=cmd_reflect)

    gc = sub.add_parser("gc", parents=[common], help="对某个用户做记忆治理")
    gc.add_argument("--uid", required=True)
    gc.set_defaults(func=cmd_gc)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_windows_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
