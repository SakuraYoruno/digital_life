"""命令行入口。

    python -m digital_life chat --mock
    python -m digital_life sim --days 7 --seed 42 --strict
    python -m digital_life demo
    python -m digital_life reflect --gc
    python -m digital_life im --self-id 123456
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from .kb import import_markdown
from .llm import llm_from_env
from .memory import MemoryStore
from .onebot import OneBotBridge, OneBotClient, OneBotIO, config_from_env
from .persona_evolution import memory_gc, nightly_reflect
from .persistence import ObsidianReflector, StateStore
from .personas import default_state, xiami_state
from .simulation import FakeClock, run_simulation
from .view import export_map
from .view_webgl import export_webgl_map
from .vision import ImageDescriber
from .websearch import make_search_fn


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
    preset = default_state() if args.persona == "default" else xiami_state()
    st = store.load(persona=preset) or preset
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
    # 仿真默认写自己的目录：和 im 共用 data/ 会把假用户灌进线上状态。
    # 注意不能用 subparser.set_defaults 设 data_dir——argparse 的 parents 是
    # 按引用共享 action 的，一改会把 chat/im/reflect 的默认值也带跑（真踩过）。
    if str(args.data_dir) == str(_default_data_dir()):
        args.data_dir = os.getenv("DIGITAL_LIFE_SIM_DIR") or "data/sim"
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
    preset = default_state() if args.persona == "default" else xiami_state()
    st = store.load(persona=preset)
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
# im（OneBot 11 / QQ）
# ---------------------------------------------------------------------------


def _csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.replace(";", ",").replace(",", " ").split() if part)


def _kb_root(args: argparse.Namespace, data_dir: Path) -> str:
    """知识库目录：--kb > 环境变量 > <data_dir>/kb.path（UTF-8 一行）。

    为什么不只靠环境变量：路径里有中文时，cmd 的 codepage 会把它啃掉，
    `set DIGITAL_LIFE_KB=D:\\note\\digital life\\知识库` 到 Python 里就变成乱码，
    表现为"知识库一直没加载"。写文件最稳。
    """
    if getattr(args, "kb", None):
        return str(args.kb)
    env = os.getenv("DIGITAL_LIFE_KB")
    if env:
        return env
    marker = Path(data_dir) / "kb.path"
    if marker.exists():
        try:
            text = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return text
    return ""


async def cmd_im(args: argparse.Namespace) -> int:
    try:
        import websockets  # noqa: F401  可选依赖，先探一下
    except ImportError:
        print('接入 QQ 需要 websockets：pip install -e ".[im]"  或  pip install websockets')
        return 1

    data_dir, mem_path, state_path = _paths(args)
    store = StateStore(state_path, clock=time.time)
    preset = default_state() if args.persona == "default" else xiami_state()
    st = store.load(persona=preset) or preset
    small, big = llm_from_env(mock=args.mock, seed=args.seed)
    embedder = _make_embedder(args)
    mem = MemoryStore(mem_path, clock=time.time, embedder=embedder)
    reflector = _reflector(args)

    config = config_from_env()
    if args.ws:
        config.ws_url = args.ws
    if args.token:
        config.access_token = args.token
    if args.self_id:
        config.self_id = args.self_id
    if args.private_mode:
        config.private_mode = args.private_mode
    if args.private_allow is not None:
        config.private_allow = _csv(args.private_allow)
    if args.group_mode:
        config.group_mode = args.group_mode
    if args.group_allow is not None:
        config.group_allow = _csv(args.group_allow)
    if args.group_prob is not None:
        config.group_reply_prob = args.group_prob
    if args.group_cooldown is not None:
        config.group_cooldown = args.group_cooldown
    if args.single_message:
        config.split_bubbles = False
    if args.stickers:
        config.sticker_dir = args.stickers
    if args.sticker_prob is not None:
        config.sticker_prob = args.sticker_prob
    if args.sticker_only_prob is not None:
        config.sticker_only_prob = args.sticker_only_prob
    if args.no_vision:
        config.vision = False

    kb_root = _kb_root(args, data_dir)
    kb = None
    if kb_root:
        try:
            from .vault import VaultKB

            kb = VaultKB(kb_root)
            print(f"知识库：{kb.size} 块（{kb_root}，直接读文件）")
        except Exception as exc:  # noqa: BLE001 - 加载失败就当没有
            print(f"（知识库加载失败，跳过：{type(exc).__name__}: {exc}）")

    search_kind = (args.search or os.getenv("DIGITAL_LIFE_SEARCH") or "off").lower()
    search_fn = make_search_fn(
        search_kind, proxy=os.getenv("DIGITAL_LIFE_SEARCH_PROXY")
    )
    if search_fn is not None:
        print(f"联网检索：{search_kind}（知识库没命中时才会用）")

    client = OneBotClient(config)
    stickers = None
    if config.sticker_dir:
        try:
            from .sticker import StickerBook

            stickers = StickerBook(config.sticker_dir, metadata_path=config.sticker_meta or None)
            print(f"表情包库：{stickers.size} 张（{config.sticker_dir}）")
        except Exception as exc:  # noqa: BLE001 - 加载失败就当没有
            print(f"（表情包库加载失败，跳过：{type(exc).__name__}: {exc}）")
    io = OneBotIO(client, config, stickers=stickers)
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
            search_fn=search_fn,
            kb=kb,
            tick_seconds=config.tick_seconds,
            # 常驻进程会被强杀（更新/重启），10 分钟一存太冒险：关系一丢，
            # 她连「这个人昨天刚聊过」都不记得，夜间反思也没对象可反思。
            autosave_seconds=60.0,
        ),
        reflector=reflector,
    )
    describer = ImageDescriber(big) if config.vision else None
    if describer is not None and not describer.available:
        describer = None
    bridge = OneBotBridge(
        engine,
        config,
        describer=describer,
        sticker_sender=io.send_sticker if stickers is not None else None,
    )
    client.on_message = bridge.dispatch

    print(f"[启动] state={state_path} 关系 {len(st.relations)} 个：{list(st.relations)[:6]}", flush=True)
    me = config.self_id or "（等 NapCat 报上来）"
    print(f"—— 数字生命 · {st.core.name} · QQ 通道 ——")
    print(f"机器人 QQ：{me}")
    print(f"OneBot WS：{config.ws_url}")
    print(
        f"私聊：{config.private_mode}"
        + (f"（白名单 {', '.join(config.private_allow) or '空'}）" if config.private_mode == "allowlist" else "")
    )
    print(
        f"群聊：{config.group_mode}"
        + (f"（限定 {', '.join(config.group_allow)}）" if config.group_allow else "")
    )
    print(
        f"对话模型：{getattr(big, 'model', type(big).__name__)} / "
        f"小模型：{getattr(small, 'model', type(small).__name__)}"
    )
    print(f"记忆向量后端：{mem.vector_backend}")
    print("Ctrl+C 退出；退出前会把状态落盘。\n")

    ticker = asyncio.create_task(engine.run())
    try:
        await client.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        engine.stop()
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ticker
        await bridge.drain()
        await client.close()
        await engine.close()
        if embedder is not None:
            embedder.close()
    print(f"状态已保存到 {state_path}")
    return 0


# ---------------------------------------------------------------------------
# kb（把 Markdown 资料导入知识层）
# ---------------------------------------------------------------------------


async def cmd_kb(args: argparse.Namespace) -> int:
    data_dir, mem_path, state_path = _paths(args)
    embedder = _make_embedder(args)
    mem = MemoryStore(mem_path, clock=time.time, embedder=embedder)
    target = Path(args.dir).expanduser()
    if target.is_dir():
        files = sorted(target.rglob("*.md"))
    elif target.is_file():
        files = [target]
    else:
        print(f"找不到：{target}")
        return 1
    if not files:
        print(f"{target} 里没有 .md")
        return 1

    total = {"written": 0, "skipped": 0, "chunks": 0}
    for file_path in files:
        stats = await import_markdown(
            mem, file_path, uid=args.uid, max_chars=args.chunk, now=time.time()
        )
        for key in total:
            total[key] += stats[key]
        print(
            f"  {file_path.name}: 写入 {stats['written']} / 跳过 {stats['skipped']}"
            f"（切块 {stats['chunks']}）"
        )
    print(
        f"合计：文件 {len(files)}，写入 {total['written']} 块，"
        f"跳过 {total['skipped']} 块 → uid={args.uid}"
    )
    mem.close()
    if embedder is not None:
        embedder.close()
    return 0


# ---------------------------------------------------------------------------
# view（记忆星图）
# ---------------------------------------------------------------------------


async def cmd_view(args: argparse.Namespace) -> int:
    data_dir, mem_path, state_path = _paths(args)
    out = Path(args.out) if args.out else Path(data_dir) / "memory-map.html"
    engine = (args.engine or "webgl").lower()
    path = None
    if engine == "webgl":
        try:
            path = export_webgl_map(
                mem_path, state_path, out, cache_dir=Path(data_dir) / "_libs"
            )
            print("渲染：three.js + Bloom（单文件，依赖已内联）")
        except Exception as exc:  # noqa: BLE001 - 没网/没缓存时回退
            print(f"（WebGL 版生成失败，回退 canvas：{type(exc).__name__}: {exc}）")
    if path is None:
        path = export_map(mem_path, state_path, out)
        print("渲染：canvas 2D（降级版）")
    print(f"记忆星图已生成：{path}")
    print("双击就能看：拖拽旋转、滚轮缩放、悬停看记忆正文。")
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
    # 默认目录在 cmd_sim 里改。**不要**在这里 set_defaults(data_dir=...)：
    # argparse 的 parents 是按引用共享 action 的，改了会把 im/chat 的默认值一起带跑。
    sim.set_defaults(func=cmd_sim)

    demo = sub.add_parser("demo", parents=[common], help="一分钟演示")
    demo.set_defaults(func=cmd_demo)

    reflect = sub.add_parser("reflect", parents=[common], help="执行一次夜间反思")
    reflect.add_argument("--gc", action="store_true", help="强制跑记忆治理")
    reflect.set_defaults(func=cmd_reflect)

    gc = sub.add_parser("gc", parents=[common], help="对某个用户做记忆治理")
    gc.add_argument("--uid", required=True)
    gc.set_defaults(func=cmd_gc)

    im = sub.add_parser("im", parents=[common], help="接入 QQ（OneBot 11 / NapCat）")
    im.add_argument("--ws", default=None, help="OneBot 正向 WS 地址，默认 ws://127.0.0.1:3001")
    im.add_argument("--token", default=None, help="OneBot access_token")
    im.add_argument("--self-id", default=None, help="机器人自己的 QQ 号")
    im.add_argument(
        "--private-mode", choices=("off", "allowlist", "open"), default=None,
        help="私聊策略：off 不接 / allowlist 只接白名单 / open 谁都接",
    )
    im.add_argument("--private-allow", default=None, help="私聊白名单，逗号分隔的 QQ 号")
    im.add_argument(
        "--group-mode", choices=("off", "mention", "open", "ambient"), default=None,
        help="群聊策略：off 不接 / mention 被 @ 或带前缀才接 / open 什么都接 / ambient 偶尔自己插话",
    )
    im.add_argument("--group-allow", default=None, help="群白名单，逗号分隔的群号，留空=不限制")
    im.add_argument("--group-prob", type=float, default=None, help="ambient 模式下每条群消息的接话概率")
    im.add_argument("--group-cooldown", type=float, default=None, help="ambient 模式下两次接话的最小间隔（秒）")
    im.add_argument("--stickers", default=None, help="表情包目录，给了才会发表情包")
    im.add_argument("--sticker-prob", type=float, default=None, help="每条回复补一张表情包的概率")
    im.add_argument("--sticker-only-prob", type=float, default=None, help="群里不接话、只丢一张表情的概率")
    im.add_argument("--no-vision", action="store_true", help="关掉看图：收到的图只显示 [图片]/[表情包]")
    im.add_argument(
        "--kb",
        default=None,
        help="知识库目录或 .md 文件（直接读源文件，不往库里拷正文）",
    )
    im.add_argument(
        "--search", choices=("off", "bing"), default=None,
        help="知识库没有时是否联网检索（默认 off，也可用 DIGITAL_LIFE_SEARCH）",
    )
    im.add_argument(
        "--single-message",
        action="store_true",
        help="她一句话分两口气时并成一条发（默认按两条气泡发，更像真人）",
    )
    im.set_defaults(func=cmd_im)

    view = sub.add_parser("view", parents=[common], help="生成记忆星图（单文件 HTML）")
    view.add_argument("--out", default=None, help="输出路径，默认 <data_dir>/memory-map.html")
    view.add_argument("--engine", choices=("webgl", "canvas"), default="webgl",
                      help="webgl=three.js+Bloom（推荐）；canvas=无依赖降级版")
    view.set_defaults(func=cmd_view)

    kb = sub.add_parser("kb", parents=[common], help="把 Markdown 资料导入知识层")
    kb.add_argument("--dir", required=True, help="一个 .md 文件或目录（递归找 *.md）")
    kb.add_argument("--uid", default="__knowledge__", help="写进哪个知识分区")
    kb.add_argument("--chunk", type=int, default=700, help="单块最大字符数")
    kb.set_defaults(func=cmd_kb)
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
