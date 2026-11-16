"""OneBot 11 接入层：把 QQ 变成数字生命的前端。

链路（与 NapCat / go-cqhttp 的标准用法一致）：

    QQ 账号 → QQNT → NapCat → OneBot 11 正向 WebSocket
             → 本模块（作为客户端连过去）
             → DigitalLife.on_message
             → send_group_msg / send_private_msg → 发回 QQ

设计要点：

- **只依赖 websockets**，且是可选依赖；没装时 `im` 子命令明确报错，其它命令不受影响。
- 收消息与发消息共用一条连接：收事件进 `parse_event`，发消息按 `echo` 等响应。
- 群聊里"她"该不该接话由 `group_mode` 决定（默认被 @ 才接）；私聊默认白名单。
- 同一句话切几条气泡由 `io.split_bubbles` 决定，发送间隔可配，避免刷屏。
- 出站目标是**按当前这一轮**绑定的：同一个 QQ 号在私聊和群里说话，回复分别落到
  私聊和群里，不会串。用 contextvars 而不是全局变量，保证并发用户互不污染。
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote

from .io import split_bubbles

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = re.split(r"[,;\s]+", value.strip())
    return tuple(p for p in parts if p)


@dataclass
class OneBotConfig:
    """一条 QQ 通道的全部可调项。CLI 参数优先于这里的环境变量默认值。"""

    ws_url: str = "ws://127.0.0.1:3001"
    access_token: str = ""
    self_id: str = ""
    # private: off 不接私聊 / allowlist 只接白名单 / open 谁都接
    private_mode: str = "allowlist"
    private_allow: tuple[str, ...] = ()
    # group: off 不接群 / mention 被 @ 或带前缀才接 / open 群里什么都接
    group_mode: str = "mention"
    group_allow: tuple[str, ...] = ()
    command_prefix: str = "/"
    at_reply: bool = True
    bubble_interval: float = 0.6
    tick_seconds: float = 60.0
    concurrency: int = 4

    def allow_private(self, uid: str) -> bool:
        if self.private_mode == "off":
            return False
        if self.private_mode == "open":
            return True
        return str(uid) in self.private_allow

    def allow_group(self, group_id: str) -> bool:
        if self.group_mode == "off":
            return False
        if self.group_allow:
            return str(group_id) in self.group_allow
        return True


def config_from_env(env: dict[str, str] | None = None) -> OneBotConfig:
    """从 DIGITAL_LIFE_QQ_* 环境变量读配置；缺省值就是 NapCat 的默认监听地址。"""
    e = os.environ if env is None else env
    return OneBotConfig(
        ws_url=e.get("DIGITAL_LIFE_QQ_WS") or "ws://127.0.0.1:3001",
        access_token=e.get("DIGITAL_LIFE_QQ_TOKEN") or "",
        self_id=e.get("DIGITAL_LIFE_QQ_SELF_ID") or "",
        private_mode=(e.get("DIGITAL_LIFE_QQ_PRIVATE_MODE") or "allowlist").lower(),
        private_allow=_split_csv(e.get("DIGITAL_LIFE_QQ_PRIVATE_ALLOW")),
        group_mode=(e.get("DIGITAL_LIFE_QQ_GROUP_MODE") or "mention").lower(),
        group_allow=_split_csv(e.get("DIGITAL_LIFE_QQ_GROUP_ALLOW")),
        command_prefix=e.get("DIGITAL_LIFE_QQ_PREFIX") or "/",
        at_reply=(e.get("DIGITAL_LIFE_QQ_AT_REPLY") or "1") not in ("0", "false", "False"),
        bubble_interval=float(e.get("DIGITAL_LIFE_QQ_BUBBLE_INTERVAL") or 0.6),
        tick_seconds=float(e.get("DIGITAL_LIFE_QQ_TICK") or 60.0),
        concurrency=int(e.get("DIGITAL_LIFE_QQ_CONCURRENCY") or 4),
    )


# ---------------------------------------------------------------------------
# 消息解析
# ---------------------------------------------------------------------------


@dataclass
class IncomingMessage:
    """一条已经"翻译成人话"的入站消息。"""

    user_id: str
    text: str
    group_id: str | None = None
    message_id: str = ""
    at_me: bool = False
    sender_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_group(self) -> bool:
        return self.group_id is not None


_SEGMENT_LABELS = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[聊天记录]",
    "json": "[卡片]",
    "xml": "[卡片]",
    "markdown": "[卡片]",
    "music": "[音乐]",
    "reply": "",
    "poke": "",
    "shake": "",
}

_RE_CQ = re.compile(r"\[CQ:([a-zA-Z0-9_.\-]+)((?:,[^\]]*)?)\]")


def _cq_to_text(message: str) -> str:
    """把 CQ 码字符串转成可读文本（messagePostFormat=string 时用）。"""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        params = match.group(2) or ""
        if name == "at":
            qq = re.search(r"qq=([^,\]]+)", params)
            target = qq.group(1) if qq else ""
            return "@全体成员" if target == "all" else f"@{target}"
        return _SEGMENT_LABELS.get(name, "")

    return _RE_CQ.sub(repl, message)


def segments_to_text(message: Any) -> str:
    """OneBot 的 message 字段（数组段 / CQ 字符串）→ 纯文本。

    图片、语音、表情等非文字段用方括号占位，让模型知道"对方发了图"，
    而不是看到一条空消息。
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return _cq_to_text(message).strip()
    if not isinstance(message, list):
        return str(message).strip()

    pieces: list[str] = []
    for seg in message:
        if not isinstance(seg, dict):
            continue
        kind = str(seg.get("type") or "")
        data = seg.get("data") or {}
        if kind == "text":
            pieces.append(str(data.get("text") or ""))
        elif kind == "at":
            qq = str(data.get("qq") or "")
            if qq == "all":
                pieces.append("@全体成员")
            else:
                pieces.append(f"@{data.get('name') or qq}")
        else:
            label = _SEGMENT_LABELS.get(kind, f"[{kind}]")
            if label:
                pieces.append(label)
    return "".join(pieces).strip()


def _mentions(message: Any, self_id: str) -> bool:
    """这条消息有没有 @ 机器人自己。"""
    if not self_id:
        return False
    if isinstance(message, str):
        return f"[CQ:at,qq={self_id}]" in message
    if isinstance(message, list):
        for seg in message:
            if isinstance(seg, dict) and seg.get("type") == "at":
                if str((seg.get("data") or {}).get("qq") or "") == str(self_id):
                    return True
    return False


def parse_event(event: dict[str, Any], self_id: str = "") -> IncomingMessage | None:
    """OneBot 事件 → IncomingMessage；不是"有人说话"就返回 None。"""
    if not isinstance(event, dict) or event.get("post_type") != "message":
        return None

    user_id = str(event.get("user_id") or "")
    if not user_id:
        return None
    if self_id and user_id == str(self_id):
        return None
    sender = event.get("sender") or {}
    if sender.get("is_bot") if isinstance(sender, dict) else False:
        return None

    text = segments_to_text(event.get("message"))
    if not text:
        return None

    group_id = None
    if event.get("message_type") == "group" and event.get("group_id"):
        group_id = str(event["group_id"])
    elif event.get("message_type") == "guild":
        return None  # 频道消息暂不处理

    return IncomingMessage(
        user_id=user_id,
        text=text,
        group_id=group_id,
        message_id=str(event.get("message_id") or ""),
        at_me=_mentions(event.get("message"), self_id),
        sender_name=str(
            (sender.get("card") if isinstance(sender, dict) else "")
            or (sender.get("nickname") if isinstance(sender, dict) else "")
            or ""
        ),
        raw=event,
    )


# ---------------------------------------------------------------------------
# 出站目标（按当前这一轮绑定，并发安全）
# ---------------------------------------------------------------------------

# (kind, id)：kind 是 "private" 或 "group"。由 OneBotBridge 在调用
# engine.on_message 之前设置，engine 里的 io.send 在同一条协程上下文中读取。
_target: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "digital_life_onebot_target", default=None
)


# ---------------------------------------------------------------------------
# OneBot 客户端
# ---------------------------------------------------------------------------


class OneBotError(RuntimeError):
    """调用 OneBot 接口失败（没连上、超时、retcode 非 0）。"""


def _as_id(value: str | int) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


class OneBotClient:
    """OneBot 11 正向 WebSocket 客户端。

    连接断开按 1s→30s 指数退避重连；重连期间 `send` 会直接失败（上层记日志），
    不会把消息悄悄吞掉。
    """

    def __init__(
        self,
        config: OneBotConfig | None = None,
        *,
        on_message: Callable[[IncomingMessage], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or config_from_env()
        self.on_message = on_message
        self.log = log or (lambda line: print(line, flush=True))
        self.self_id = str(self.config.self_id or "")
        self._ws: Any = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._closed = False
        self._seen: deque[str] = deque(maxlen=200)

    # -- 状态 ---------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _url(self) -> str:
        url = self.config.ws_url
        if self.config.access_token and "access_token=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}access_token={quote(self.config.access_token)}"
        return url

    # -- 生命周期 -----------------------------------------------------
    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        """连上并一直跑，直到 stop 被 set 或 close() 被调用。"""
        delay = 1.0
        while not self._closed and not (stop is not None and stop.is_set()):
            try:
                await self._session()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 断线原因不限
                self.log(f"[onebot] 连接失败：{type(exc).__name__}: {exc}")
            if self._closed or (stop is not None and stop.is_set()):
                break
            self.log(f"[onebot] {delay:.0f}s 后重连 {self.config.ws_url}")
            await asyncio.sleep(delay)
            delay = min(30.0, delay * 2)

    async def _session(self) -> None:
        from websockets.asyncio.client import connect  # 可选依赖，延迟导入

        headers = {"Authorization": f"Bearer {self.config.access_token}"} if self.config.access_token else None
        async with connect(
            self._url(),
            additional_headers=headers,
            proxy=None,  # 本机回环，别走系统代理
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=16 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            self._connected.set()
            self.log(f"[onebot] 已连接 {self.config.ws_url}")
            try:
                async for raw in ws:
                    await self._handle_raw(raw)
            finally:
                self._connected.clear()
                self._ws = None
                self._fail_pending("连接已断开")

    async def close(self) -> None:
        self._closed = True
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - 关闭失败无所谓
                pass
        self._fail_pending("客户端已关闭")

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(OneBotError(reason))
        self._pending.clear()

    # -- 收发 ---------------------------------------------------------
    async def _handle_raw(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            self.log("[onebot] 收到非 JSON 帧，已忽略")
            return
        if not isinstance(data, dict):
            return

        echo = data.get("echo")
        if echo is not None:
            future = self._pending.pop(str(echo), None)
            if future is not None and not future.done():
                future.set_result(data)
            return

        sid = data.get("self_id")
        if sid:
            self.self_id = str(sid)

        post_type = data.get("post_type")
        if post_type == "meta_event":
            return
        if post_type != "message":
            return

        message = parse_event(data, self.self_id)
        if message is None:
            return
        if message.message_id and message.message_id in self._seen:
            return  # 断线重连后 NapCat 可能重推同一条
        if message.message_id:
            self._seen.append(message.message_id)
        if self.on_message is not None:
            self.on_message(message)

    async def call(
        self,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        """调一个 OneBot 接口并等它的响应。"""
        ws = self._ws
        if ws is None:
            raise OneBotError("还没连上 OneBot，消息没发出去")

        echo = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[echo] = future
        payload = {"action": action, "params": params or {}, "echo": echo}
        try:
            async with self._write_lock:
                await ws.send(json.dumps(payload, ensure_ascii=False))
            data = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise OneBotError(f"{action} 超时（{timeout:.0f}s）") from None
        finally:
            self._pending.pop(echo, None)

        retcode = data.get("retcode")
        if retcode not in (0, None):
            raise OneBotError(f"{action} 失败：retcode={retcode} {data.get('message') or data.get('wording') or ''}".strip())
        return data

    async def send_private(self, user_id: str, message: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.call("send_private_msg", {"user_id": _as_id(user_id), "message": message})

    async def send_group(self, group_id: str, message: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.call("send_group_msg", {"group_id": _as_id(group_id), "message": message})


# ---------------------------------------------------------------------------
# 出站 IO
# ---------------------------------------------------------------------------


class OneBotIO:
    """`io.OutboundIO` 的 QQ 实现：按 A 维旋钮切气泡，逐条发出。"""

    def __init__(self, client: OneBotClient, config: OneBotConfig | None = None) -> None:
        self.client = client
        self.config = config or client.config

    async def send(self, uid: str, text: str, knobs: dict[str, Any] | None = None) -> None:
        knobs = knobs or {}
        bubbles = split_bubbles(text, knobs)
        if not bubbles:
            return

        target = _target.get()
        group_id = target[1] if target and target[0] == "group" else None
        if group_id is None and not str(uid).isdigit():
            # 主动发言时可能拿着 "owner"、"demo" 这类非 QQ 号的 uid，发不出去就记一笔。
            self.client.log(f"[onebot] 跳过非 QQ uid 的出站消息：{uid!r}")
            return

        total = len(bubbles)
        for index, bubble in enumerate(bubbles):
            segments: list[dict[str, Any]] = [{"type": "text", "data": {"text": bubble}}]
            try:
                if group_id is not None:
                    if index == 0 and self.config.at_reply:
                        segments.insert(0, {"type": "at", "data": {"qq": str(uid)}})
                    await self.client.send_group(group_id, segments)
                else:
                    await self.client.send_private(uid, segments)
            except OneBotError as exc:
                self.client.log(f"[onebot] 发送失败：{exc}")
                return
            if index + 1 < total:
                await asyncio.sleep(max(0.0, self.config.bubble_interval))


# ---------------------------------------------------------------------------
# 桥接：过滤 → 引擎 → 出站
# ---------------------------------------------------------------------------


class OneBotBridge:
    """把 QQ 消息接进 `DigitalLife`。

    过滤规则（都是配置项，不写死）：

    - 私聊：`private_mode` 决定谁的话会被接。
    - 群聊：`group_mode="mention"` 时只有被 @ 或带 `command_prefix` 才会接话。
    - 自己发出去的消息、其他机器人的消息、空消息一律丢弃。
    """

    def __init__(
        self,
        engine: Any,
        config: OneBotConfig | None = None,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or config_from_env()
        self.log = log or (lambda line: print(line, flush=True))
        self._sem = asyncio.Semaphore(max(1, self.config.concurrency))
        self._tasks: set[asyncio.Task[Any]] = set()

    # -- 过滤 ---------------------------------------------------------
    def should_reply(self, message: IncomingMessage) -> bool:
        if message.is_group:
            if not self.config.allow_group(str(message.group_id)):
                return False
            if self.config.group_mode == "open":
                return True
            if message.at_me:
                return True
            prefix = self.config.command_prefix
            return bool(prefix) and message.text.startswith(prefix)
        if not self.config.allow_private(message.user_id):
            return False
        return True

    # -- 调度 ---------------------------------------------------------
    def dispatch(self, message: IncomingMessage) -> None:
        """客户端回调：不 await，交给后台任务，慢生成不挡下一条消息。"""
        task = asyncio.ensure_future(self.handle(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def handle(self, message: IncomingMessage) -> None:
        if not self.should_reply(message):
            return

        text = message.text
        prefix = self.config.command_prefix
        if message.is_group and not message.at_me and prefix and text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            if not text:
                return

        who = message.sender_name or message.user_id
        where = f"群{message.group_id}" if message.is_group else "私聊"
        self.log(f"[onebot] ← {who}（{where}）：{text}")
        async with self._sem:
            token = _target.set(
                ("group", message.group_id) if message.is_group else ("private", message.user_id)
            )
            try:
                result = await self.engine.on_message(
                    message.user_id, text, message.group_id
                )
            except Exception as exc:  # noqa: BLE001 - 单条消息出错不能拖垮整条链路
                self.log(f"[onebot] 处理 {who}（{where}）失败：{type(exc).__name__}: {exc}")
                return
            finally:
                _target.reset(token)

        if result.reply:
            self.log(f"[onebot] {who}（{where}）→ {result.reply}")
        elif result.blocked:
            self.log(f"[onebot] {who}（{where}）→ 被安全闸拦下，未发送")


__all__ = [
    "OneBotConfig",
    "IncomingMessage",
    "OneBotClient",
    "OneBotError",
    "OneBotIO",
    "OneBotBridge",
    "config_from_env",
    "parse_event",
    "segments_to_text",
    "split_bubbles",
]
