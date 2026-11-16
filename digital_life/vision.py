"""看图：把聊天里的图片 / 表情包交给多模态模型，换成一句"她看到的"。

调用方只需要 `await describe(url)`，拿回一句人话；拿不回就返回空字符串，
上层会退回 `[图片]` 这种占位——看图失败不该让整条消息处理失败。

同一张图只问一次模型：按图片内容的 sha256 缓存，聊天里表情包重复率很高。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_PROMPT = (
    "这是一张 QQ 聊天里的图片。用一句不超过 20 字的中文说清楚它画了什么、什么情绪，"
    "像人跟朋友转述那样随口说，不要客套、不要分析画面细节。只输出这句话。"
)


class ImageDescriber:
    """多模态客户端 + 内容哈希缓存。"""

    def __init__(
        self,
        llm: Any,
        *,
        timeout: float = 25.0,
        max_bytes: int = 8_000_000,
        cache_size: int = 256,
    ) -> None:
        self.llm = llm
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.failures = 0

    @property
    def available(self) -> bool:
        return callable(getattr(self.llm, "vision", None))

    async def describe(self, source: str) -> str:
        if not source or not self.available:
            return ""
        try:
            raw = await asyncio.to_thread(self._read, source)
        except Exception:  # noqa: BLE001 - 图挂了就退回占位
            self.failures += 1
            return ""
        if not raw or len(raw) > self.max_bytes:
            return ""

        key = hashlib.sha256(raw).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        mime = mimetypes.guess_type(source)[0] or "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        try:
            text = await asyncio.wait_for(
                self.llm.vision(_PROMPT, data_url), timeout=self.timeout
            )
        except Exception:  # noqa: BLE001 - 模型不支持看图/超时，都当作看不了
            self.failures += 1
            return ""

        text = " ".join((text or "").split())
        if len(text) > 40:
            text = text[:40]
        self._cache[key] = text
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return text

    # -- 取图 ---------------------------------------------------------
    def _read(self, source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            request = urllib.request.Request(
                source, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return resp.read(self.max_bytes + 1)
        if source.startswith("file://"):
            path = unquote(urlparse(source).path)
            # Windows: /D:/x/y.jpg -> D:/x/y.jpg
            if len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return Path(path).read_bytes()
        return Path(source).read_bytes()


__all__ = ["ImageDescriber"]
