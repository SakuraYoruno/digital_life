"""表情包与看图：挑图、发图、把图片换成一句人话。"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

from digital_life.onebot import (
    IncomingMessage,
    OneBotBridge,
    OneBotConfig,
    OneBotIO,
    parse_event,
    segments_to_text,
)
from digital_life.personas import xiami_state
from digital_life.sticker import StickerBook
from digital_life.vision import ImageDescriber


# ---------------------------------------------------------------------------
# 表情包库
# ---------------------------------------------------------------------------


def _make_library(tmp_path: Path) -> StickerBook:
    for name in ("a.jpg", "b.gif", "c.png"):
        (tmp_path / name).write_bytes(b"x")
    meta = [
        {"path": str(tmp_path / "a.jpg"), "summary": "开心比耶"},
        {"path": str(tmp_path / "b.gif"), "summary": "生无可恋躺平"},
        {"path": str(tmp_path / "c.png"), "summary": "问号小兔"},
    ]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return StickerBook(tmp_path, metadata_path=meta_path, rng=random.Random(0))


def test_sticker_book_loads_and_counts(tmp_path):
    book = _make_library(tmp_path)
    assert book.size == 3


def test_sticker_book_picks_by_mood(tmp_path):
    book = _make_library(tmp_path)
    picked = book.pick("低落")
    assert picked is not None
    assert "生无可恋" in picked.summary


def test_sticker_book_avoids_repeating(tmp_path):
    book = _make_library(tmp_path)
    first = book.pick("")   # 没有 mood 提示时从全库里挑，才谈得上"不重复"
    second = book.pick("")
    assert first.path != second.path


def test_sticker_book_handles_missing_dir(tmp_path):
    book = StickerBook(tmp_path / "nope")
    assert book.size == 0
    assert book.pick("开心") is None


def test_sticker_file_url_is_uri(tmp_path):
    book = _make_library(tmp_path)
    sticker = book.pick("开心")
    assert sticker is not None
    assert sticker.file_url.startswith("file:///")


# ---------------------------------------------------------------------------
# 入站：认出表情包、拿到图片地址
# ---------------------------------------------------------------------------


def _image_event(url: str, *, sub_type: int = 1) -> dict:
    return {
        "post_type": "message",
        "message_type": "private",
        "self_id": 999,
        "user_id": 222,
        "message_id": 1,
        "message": [
            {
                "type": "image",
                "data": {"url": url, "sub_type": sub_type, "summary": "[动画表情]"},
            }
        ],
        "sender": {"nickname": "阿暖"},
    }


def test_parse_event_labels_sticker_and_keeps_url():
    message = parse_event(_image_event("https://example.com/x.gif"), "999")
    assert message is not None
    assert message.text == "[表情包]"
    assert message.images == ("https://example.com/x.gif",)


def test_parse_event_labels_plain_image():
    event = _image_event("https://example.com/x.jpg", sub_type=0)
    event["message"][0]["data"]["summary"] = ""
    message = parse_event(event, "999")
    assert message is not None
    assert message.text == "[图片]"


def test_cq_string_marks_sticker():
    assert segments_to_text("[CQ:image,sub_type=1,file=a.gif]") == "[表情包]"


# ---------------------------------------------------------------------------
# 看图
# ---------------------------------------------------------------------------


class _FakeVisionLLM:
    def __init__(self, answer: str = "一只在打哈欠的猫") -> None:
        self.answer = answer
        self.calls = 0

    async def vision(self, prompt, image_url, **kwargs):
        self.calls += 1
        assert image_url.startswith("data:image/")
        return self.answer


def test_describer_returns_text_and_caches(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"pretend-jpeg")

    async def run():
        llm = _FakeVisionLLM()
        describer = ImageDescriber(llm)
        first = await describer.describe(str(image))
        second = await describer.describe(str(image))
        assert first == second == "一只在打哈欠的猫"
        assert llm.calls == 1  # 第二张同一张图走缓存

    asyncio.run(run())


def test_describer_without_vision_support_returns_empty(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")

    async def run():
        describer = ImageDescriber(object())  # 没有 vision 方法
        assert describer.available is False
        assert await describer.describe(str(image)) == ""

    asyncio.run(run())


def test_describer_survives_bad_source():
    async def run():
        describer = ImageDescriber(_FakeVisionLLM())
        assert await describer.describe("https://127.0.0.1:1/nope.jpg") == ""

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 发图
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    config = OneBotConfig()

    def log(self, line: str) -> None:
        pass

    async def send_private(self, uid, message):
        self.sent.append(("private", uid, message))
        return {}

    async def send_group(self, gid, message):
        self.sent.append(("group", gid, message))
        return {}


def test_io_appends_sticker(tmp_path):
    book = _make_library(tmp_path)
    config = OneBotConfig(bubble_interval=0, sticker_prob=1.0, sticker_delay=0)
    client = _FakeClient()
    io = OneBotIO(client, config, stickers=book, rng=random.Random(0))

    asyncio.run(io.send("222", "哈哈，笑死", {"mood": "开心"}))

    assert len(client.sent) == 2
    assert client.sent[0][2][0]["type"] == "text"
    image = client.sent[1][2][0]
    assert image["type"] == "image"
    assert image["data"]["sub_type"] == 1        # 1 = QQ 表情包，不是普通图片
    assert image["data"]["summary"] == "[动画表情]"
    assert image["data"]["file"].startswith("file:///")


def test_io_skips_sticker_when_unlucky(tmp_path):
    book = _make_library(tmp_path)
    config = OneBotConfig(bubble_interval=0, sticker_prob=0.0, sticker_delay=0)
    client = _FakeClient()
    io = OneBotIO(client, config, stickers=book, rng=random.Random(0))

    asyncio.run(io.send("222", "嗯", {"mood": "平静"}))
    assert len(client.sent) == 1


# ---------------------------------------------------------------------------
# 桥接：把 [图片] 换成"她看到的"
# ---------------------------------------------------------------------------


class _FakeDescriber:
    available = True

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def describe(self, source: str) -> str:
        self.calls += 1
        return self.text


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.st = xiami_state()

    async def on_message(self, uid, text, group_id=None, **kwargs):
        self.calls.append((uid, text, group_id, kwargs))
        return SimpleNamespace(reply="", blocked=False)


def test_bridge_describes_image_before_speaking():
    engine = _Recorder()
    describer = _FakeDescriber("一只在打哈欠的猫")
    bridge = OneBotBridge(
        engine,
        OneBotConfig(private_mode="open", vision=True),
        log=lambda line: None,
        describer=describer,
    )
    message = IncomingMessage(
        user_id="222", text="你看[表情包]", images=("https://example.com/x.gif",)
    )

    asyncio.run(bridge.handle(message))

    assert engine.calls, "没有被调用"
    assert "一只在打哈欠的猫" in engine.calls[0][1]
    assert "[表情包]" not in engine.calls[0][1]


def test_bridge_keeps_placeholder_when_vision_fails():
    engine = _Recorder()
    bridge = OneBotBridge(
        engine,
        OneBotConfig(private_mode="open", vision=True),
        log=lambda line: None,
        describer=_FakeDescriber(""),  # 看不了图
    )
    message = IncomingMessage(
        user_id="222", text="你看[图片]", images=("https://example.com/x.jpg",)
    )

    asyncio.run(bridge.handle(message))
    assert "[图片]" in engine.calls[0][1]


# ---------------------------------------------------------------------------
# 只用表情接话（群里不吭声，但丢一张图）
# ---------------------------------------------------------------------------


class _StickerSender:
    def __init__(self, ok: bool = True) -> None:
        self.calls: list[tuple] = []
        self.ok = ok

    async def __call__(self, uid, *, group_id=None, mood=""):
        self.calls.append((uid, group_id, mood))
        return self.ok


def _group_message(text: str = "今天好热") -> IncomingMessage:
    return IncomingMessage(user_id="222", text=text, group_id="555", at_me=False)


def test_sticker_only_respects_cooldown():
    sender = _StickerSender()
    bridge = OneBotBridge(
        _Recorder(),
        OneBotConfig(group_mode="ambient", sticker_only_prob=0.9, sticker_cooldown=300),
        log=lambda line: None,
        rng=random.Random(1),
        sticker_sender=sender,
    )
    message = _group_message()
    assert bridge.should_sticker_only(message, now=1000.0) is True
    bridge._last_sticker["555"] = 1000.0
    assert bridge.should_sticker_only(message, now=1100.0) is False
    assert bridge.should_sticker_only(message, now=1400.0) is True


def test_sticker_only_requires_sender_and_ambient():
    assert OneBotBridge(
        _Recorder(), OneBotConfig(group_mode="ambient", sticker_only_prob=1.0), log=lambda l: None
    ).should_sticker_only(_group_message()) is False

    sender = _StickerSender()
    assert OneBotBridge(
        _Recorder(),
        OneBotConfig(group_mode="mention", sticker_only_prob=1.0),
        log=lambda l: None,
        sticker_sender=sender,
    ).should_sticker_only(_group_message()) is False


def test_bridge_replies_with_sticker_instead_of_text():
    sender = _StickerSender()
    engine = _Recorder()
    bridge = OneBotBridge(
        engine,
        OneBotConfig(
            group_mode="ambient",
            group_reply_prob=0.0,     # 不接话
            sticker_only_prob=1.0,    # 但用表情回应
            sticker_cooldown=0,
        ),
        log=lambda line: None,
        rng=random.Random(1),
        sticker_sender=sender,
    )

    asyncio.run(bridge.handle(_group_message()))

    assert len(sender.calls) == 1
    assert sender.calls[0][1] == "555"
    assert engine.calls == []          # 没有生成文字回复
