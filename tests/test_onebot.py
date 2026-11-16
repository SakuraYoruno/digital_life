"""OneBot 11 接入层测试：解析、过滤、以及和假服务端的端到端收环。"""

from __future__ import annotations

import asyncio
import json
import random
import time

from digital_life.engine import DigitalLife, EngineConfig
from digital_life.io import split_bubbles
from digital_life.llm import MockLLM
from digital_life.memory import MemoryStore
from digital_life.onebot import (
    OneBotBridge,
    OneBotClient,
    OneBotConfig,
    OneBotIO,
    _target,
    config_from_env,
    parse_event,
    segments_to_text,
)
from digital_life.personas import xiami_state


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_segments_to_text_array():
    message = [
        {"type": "at", "data": {"qq": "111", "name": "阿暖"}},
        {"type": "text", "data": {"text": " 你看这个"}},
        {"type": "image", "data": {"file": "a.jpg"}},
        {"type": "face", "data": {"id": "1"}},
    ]
    assert segments_to_text(message) == "@阿暖 你看这个[图片][QQ表情]"


def test_segments_to_text_cq_string():
    raw = "[CQ:at,qq=111] 在吗 [CQ:image,file=a.jpg]"
    assert segments_to_text(raw) == "@111 在吗 [图片]"


def test_segments_to_text_ignores_reply_segment():
    message = [
        {"type": "reply", "data": {"id": "1"}},
        {"type": "text", "data": {"text": "嗯"}},
    ]
    assert segments_to_text(message) == "嗯"


def _private_event(text: str, user_id: str = "222", self_id: str = "999") -> dict:
    return {
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "self_id": int(self_id),
        "user_id": int(user_id),
        "message_id": 110,
        "message": [{"type": "text", "data": {"text": text}}],
        "sender": {"nickname": "阿暖"},
    }


def _group_event(text: str, *, at: str | None = None, group_id: str = "555") -> dict:
    message: list[dict] = [{"type": "text", "data": {"text": text}}]
    if at:
        message.insert(0, {"type": "at", "data": {"qq": at}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 999,
        "user_id": 222,
        "group_id": int(group_id),
        "message_id": 111,
        "message": message,
        "sender": {"card": "阿暖", "nickname": "阿暖"},
    }


def test_parse_private_event():
    msg = parse_event(_private_event("今天好累"), "999")
    assert msg is not None
    assert msg.is_group is False
    assert msg.user_id == "222"
    assert msg.text == "今天好累"
    assert msg.sender_name == "阿暖"


def test_parse_group_event_detects_at_me():
    msg = parse_event(_group_event("在吗", at="999"), "999")
    assert msg is not None
    assert msg.is_group is True
    assert msg.group_id == "555"
    assert msg.at_me is True

    other = parse_event(_group_event("在吗", at="111"), "999")
    assert other is not None and other.at_me is False


def test_parse_ignores_self_and_non_message():
    assert parse_event(_private_event("我自己说的", user_id="999"), "999") is None
    assert parse_event({"post_type": "meta_event", "self_id": 999}, "999") is None
    assert parse_event({"post_type": "notice", "user_id": 1}, "999") is None


def test_parse_ignores_empty_text():
    event = _private_event("")
    event["message"] = [{"type": "image", "data": {"file": "a.jpg"}}]
    msg = parse_event(event, "999")
    assert msg is not None and msg.text == "[图片]"

    event["message"] = []
    assert parse_event(event, "999") is None


# ---------------------------------------------------------------------------
# 过滤
# ---------------------------------------------------------------------------


class _DummyEngine:
    async def on_message(self, uid, text, group_id=None):  # pragma: no cover - 过滤测试用不到
        raise AssertionError("不该被调用")


def _bridge(**kwargs) -> OneBotBridge:
    config = OneBotConfig(**kwargs)
    return OneBotBridge(_DummyEngine(), config, log=lambda line: None)


def test_private_allowlist():
    bridge = _bridge(private_mode="allowlist", private_allow=("222",))
    assert bridge.should_reply(parse_event(_private_event("hi"), "999")) is True
    assert bridge.should_reply(parse_event(_private_event("hi", user_id="333"), "999")) is False


def test_private_off_and_open():
    assert _bridge(private_mode="off").should_reply(parse_event(_private_event("hi"), "999")) is False
    assert _bridge(private_mode="open").should_reply(parse_event(_private_event("hi"), "999")) is True


def test_group_mention_only():
    bridge = _bridge(group_mode="mention")
    assert bridge.should_reply(parse_event(_group_event("在吗", at="999"), "999")) is True
    assert bridge.should_reply(parse_event(_group_event("在吗"), "999")) is False
    assert bridge.should_reply(parse_event(_group_event("/她 在吗"), "999")) is True


def test_group_allowlist():
    bridge = _bridge(group_mode="open", group_allow=("777",))
    assert bridge.should_reply(parse_event(_group_event("hi", group_id="777"), "999")) is True
    assert bridge.should_reply(parse_event(_group_event("hi", group_id="888"), "999")) is False


def test_config_from_env():
    env = {
        "DIGITAL_LIFE_QQ_WS": "ws://127.0.0.1:4000",
        "DIGITAL_LIFE_QQ_TOKEN": "tk",
        "DIGITAL_LIFE_QQ_PRIVATE_ALLOW": "1,2 3",
        "DIGITAL_LIFE_QQ_GROUP_MODE": "open",
    }
    config = config_from_env(env)
    assert config.ws_url == "ws://127.0.0.1:4000"
    assert config.access_token == "tk"
    assert config.private_allow == ("1", "2", "3")
    assert config.group_mode == "open"
    assert config.self_id == ""


def test_split_bubbles_still_used_by_io():
    assert split_bubbles("在吗。在干嘛？", {"split_bubbles": True}) == ["在吗。", "在干嘛？"]


# ---------------------------------------------------------------------------
# 出站：两条气泡 vs 并成一条
# ---------------------------------------------------------------------------


class _FakeClient:
    """只记录发出去什么，不碰网络。"""

    def __init__(self, config: OneBotConfig) -> None:
        self.config = config
        self.sent: list[tuple] = []

    def log(self, line: str) -> None:
        pass

    async def send_private(self, uid, message):
        self.sent.append(("private", uid, message))
        return {}

    async def send_group(self, gid, message):
        self.sent.append(("group", gid, message))
        return {}


def test_io_splits_newlines_into_two_bubbles_by_default():
    config = OneBotConfig(bubble_interval=0)
    client = _FakeClient(config)
    io = OneBotIO(client, config)

    asyncio.run(io.send("222", "嗯哼，叫我干嘛\n这么突然，有事？", {}))

    assert len(client.sent) == 2
    assert client.sent[0][2][0]["data"]["text"] == "嗯哼，叫我干嘛"
    assert client.sent[1][2][0]["data"]["text"] == "这么突然，有事？"


def test_io_single_message_mode_folds_newlines():
    config = OneBotConfig(split_bubbles=False, bubble_interval=0)
    client = _FakeClient(config)
    io = OneBotIO(client, config)

    asyncio.run(io.send("222", "嗯哼，叫我干嘛\n这么突然，有事？", {}))

    assert len(client.sent) == 1
    assert client.sent[0][2][0]["data"]["text"] == "嗯哼，叫我干嘛，这么突然，有事？"


def test_config_split_flag_from_env():
    assert config_from_env({"DIGITAL_LIFE_QQ_SPLIT": "0"}).split_bubbles is False
    assert config_from_env({"DIGITAL_LIFE_QQ_SPLIT": "1"}).split_bubbles is True
    assert config_from_env({}).split_bubbles is True


# ---------------------------------------------------------------------------
# 端到端：假 OneBot 服务端 ↔ 客户端 ↔ 引擎
# ---------------------------------------------------------------------------


def _make_engine(io, *, state=None) -> DigitalLife:
    st = state or xiami_state()
    mem = MemoryStore(":memory:", clock=time.time)
    return DigitalLife(
        st,
        MockLLM(7),
        MockLLM(8),
        mem,
        io,
        config=EngineConfig(clock=time.time),
    )


async def _run_roundtrip(
    event: dict,
    config: OneBotConfig,
    *,
    expect_reply: bool = True,
    rng: random.Random | None = None,
) -> tuple[list[dict], DigitalLife]:
    from websockets.asyncio.server import serve

    received: list[dict] = []
    got_one = asyncio.Event()

    async def handler(ws):
        await ws.send(json.dumps(event, ensure_ascii=False))
        async for raw in ws:
            data = json.loads(raw)
            if "action" not in data:
                continue
            received.append(data)
            await ws.send(
                json.dumps(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"message_id": 1},
                        "echo": data["echo"],
                    }
                )
            )
            got_one.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        config.ws_url = f"ws://127.0.0.1:{port}"

        client = OneBotClient(config, log=lambda line: None)
        io = OneBotIO(client, config)
        engine = _make_engine(io)
        bridge = OneBotBridge(engine, config, log=lambda line: None, rng=rng)
        client.on_message = bridge.dispatch

        task = asyncio.create_task(client.run())
        try:
            if expect_reply:
                await asyncio.wait_for(got_one.wait(), timeout=20)
                await asyncio.sleep(0.4)  # 等可能的多气泡发完
            else:
                await asyncio.sleep(1.0)  # 留足时间让"本该沉默"的那条真的沉默
        finally:
            await client.close()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            engine.stop()
            await engine.drain()
            engine.mem.close()
    return received, engine


def test_e2e_private_message_gets_reply():
    config = OneBotConfig(self_id="999", private_mode="open")

    async def run():
        received, _engine = await _run_roundtrip(_private_event("我叫阿暖，今天被老板骂了"), config)
        assert received, "没有收到出站消息"
        actions = {item["action"] for item in received}
        assert actions == {"send_private_msg"}
        assert received[0]["params"]["user_id"] == 222
        text = "".join(
            seg.get("data", {}).get("text", "")
            for item in received
            for seg in item["params"]["message"]
            if seg.get("type") == "text"
        )
        assert text.strip(), "回复是空的"

    asyncio.run(run())


def test_e2e_group_reply_at_sender():
    config = OneBotConfig(self_id="999", group_mode="mention")

    async def run():
        received, _engine = await _run_roundtrip(_group_event("在吗", at="999"), config)
        assert received, "群里被 @ 了却没有回复"
        assert received[0]["action"] == "send_group_msg"
        assert received[0]["params"]["group_id"] == 555
        first_segments = received[0]["params"]["message"]
        assert first_segments[0]["type"] == "at"
        assert str(first_segments[0]["data"]["qq"]) == "222"

    asyncio.run(run())


def test_e2e_group_without_mention_is_silent():
    config = OneBotConfig(self_id="999", group_mode="mention")

    async def run():
        received, _engine = await _run_roundtrip(
            _group_event("大家好"), config, expect_reply=False
        )
        assert received == [], "没被 @ 也回复了"

    asyncio.run(run())


def test_e2e_two_turns_keep_replying():
    """连收两条（不同 message_id）都要回；相同 message_id 只回一次。"""

    async def run():
        from websockets.asyncio.server import serve

        replies: list[dict] = []
        all_done = asyncio.Event()

        first = _private_event("我叫阿暖")
        first["message_id"] = 201
        second = _private_event("你还记得我叫什么吗？")
        second["message_id"] = 202

        async def handler(ws):
            await ws.send(json.dumps(first, ensure_ascii=False))
            async for raw in ws:
                data = json.loads(raw)
                if "action" not in data:
                    continue
                replies.append(data)
                await ws.send(
                    json.dumps(
                        {"status": "ok", "retcode": 0, "data": {"message_id": 1}, "echo": data["echo"]}
                    )
                )
                if len(replies) == 1:
                    await ws.send(json.dumps(second, ensure_ascii=False))
                else:
                    all_done.set()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = OneBotConfig(
                ws_url=f"ws://127.0.0.1:{port}", self_id="999", private_mode="open"
            )
            client = OneBotClient(config, log=lambda line: None)
            engine = _make_engine(OneBotIO(client, config))
            bridge = OneBotBridge(engine, config, log=lambda line: None)
            client.on_message = bridge.dispatch

            task = asyncio.create_task(client.run())
            try:
                await asyncio.wait_for(all_done.wait(), timeout=25)
            finally:
                await client.close()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                engine.stop()
                await engine.drain()
                engine.mem.close()

        assert len(replies) >= 2, f"两轮只回了 {len(replies)} 次"

    asyncio.run(run())


def test_target_contextvar_is_isolated():
    async def run():
        assert _target.get() is None
        token = _target.set(("group", "555"))
        assert _target.get() == ("group", "555")
        _target.reset(token)
        assert _target.get() is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 群聊主动接话（ambient）
# ---------------------------------------------------------------------------


def test_group_chance_reacts_to_her_name():
    bridge = _bridge(group_mode="ambient", group_reply_prob=0.05)
    plain = bridge.group_chance("今天好热")
    named = bridge.group_chance("夏弥你觉得呢")
    assert named > plain
    assert named >= 0.45  # 被点名至少抬到 0.45（无状态时不再乘以情绪系数）

    # 疑问句比陈述句更容易被接
    assert bridge.group_chance("有人吗？") > plain


def test_ambient_cooldown_blocks_second_chime_in():
    bridge = OneBotBridge(
        _DummyEngine(),
        OneBotConfig(group_mode="ambient", group_reply_prob=0.9, group_cooldown=300),
        log=lambda line: None,
        rng=random.Random(1),
    )
    message = parse_event(_group_event("今天吃什么"), "999")
    assert message is not None

    assert bridge.should_chime_in(message, now=1000.0) is True
    bridge._last_group_reply["555"] = 1000.0  # 刚接过话
    assert bridge.should_chime_in(message, now=1100.0) is False  # 冷却中
    assert bridge.should_chime_in(message, now=1400.0) is True  # 冷却过了


def test_mention_mode_never_chimes_in():
    bridge = _bridge(group_mode="mention", group_reply_prob=0.9)
    message = parse_event(_group_event("今天吃什么"), "999")
    assert message is not None
    assert bridge.should_chime_in(message) is False


def test_engine_speak_false_only_listens():
    from digital_life.io import CollectingIO

    async def run():
        io = CollectingIO(clock=time.time)
        engine = _make_engine(io)
        result = await engine.on_message("222", "群里在聊火锅", "555", speak=False)
        assert result.reply == ""
        assert io.messages == []            # 一个字都没发
        assert engine.stats.messages == 0   # 不算"回过话"
        assert engine.st.rel("222").last_interact > 0  # 但她听见了
        await engine.drain()
        engine.mem.close()

    asyncio.run(run())


def test_e2e_ambient_group_observes_without_reply():
    config = OneBotConfig(
        self_id="999", group_mode="ambient", group_reply_prob=0.0, group_cooldown=0
    )

    async def run():
        received, engine = await _run_roundtrip(
            _group_event("今天好热啊"), config, expect_reply=False, rng=random.Random(1)
        )
        assert received == [], "概率 0 还是说话了"
        assert engine.stats.messages == 0

    asyncio.run(run())


def test_e2e_ambient_group_can_chime_in():
    config = OneBotConfig(
        self_id="999", group_mode="ambient", group_reply_prob=0.9, group_cooldown=0
    )

    async def run():
        received, _engine = await _run_roundtrip(
            _group_event("今天好热啊"), config, rng=random.Random(1)
        )
        assert received, "该插话的时候没插"
        assert received[0]["action"] == "send_group_msg"
        assert str(received[0]["params"]["message"][0]["data"]["qq"]) == "222"

    asyncio.run(run())
