"""联网检索：给知识层兜底用。

只实现"抓搜索结果的标题 + 摘要"这一件事——后面还有小模型把它压成 300 字再入知识层，
所以这里不需要精确解析，抓到几条有用的文本即可。

默认用 Bing 的 HTML 结果页：不需要 API key，国内可直连。
DuckDuckGo 直连不通，需要走代理，所以放在备选。
"""

from __future__ import annotations

import asyncio
import html
import re
import urllib.parse
import urllib.request
from typing import Any, Callable

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r'<li class="b_algo".*?</li>', re.S)
_LINK = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET = re.compile(r"<p[^>]*>(.*?)</p>", re.S)


def _clean(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).strip()


def bing_search(
    query: str,
    *,
    limit: int = 4,
    timeout: float = 12.0,
    proxy: str | None = None,
) -> str:
    """抓 Bing 结果页，返回编号列表文本；失败返回空串。"""
    url = "https://cn.bing.com/search?q=" + urllib.parse.quote(query)
    handlers = (
        [urllib.request.ProxyHandler({"http": proxy, "https": proxy})]
        if proxy
        else []
    )
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with opener.open(request, timeout=timeout) as resp:
            page = resp.read(600_000).decode("utf-8", "ignore")
    except Exception:
        return ""

    items: list[str] = []
    for block in _BLOCK.findall(page)[:limit]:
        link = _LINK.search(block)
        snippet = _SNIPPET.search(block)
        title = _clean(link.group(2)) if link else ""
        body = _clean(snippet.group(1)) if snippet else ""
        if not body:
            continue
        items.append(f"{len(items) + 1}. {title}：{body}")
    return "\n".join(items)


def make_search_fn(
    kind: str = "bing",
    *,
    limit: int = 4,
    proxy: str | None = None,
    timeout: float = 12.0,
) -> Callable[[str], Any] | None:
    """返回一个 async 的 search_fn；未知实现返回 None（保持"安静跳过"）。"""
    engine = (kind or "").lower()
    if engine in ("", "off", "none"):
        return None
    if engine not in ("bing", "web"):
        return None

    async def search(query: str) -> str:
        return await asyncio.to_thread(
            bing_search, query, limit=limit, proxy=proxy, timeout=timeout
        )

    return search


__all__ = ["bing_search", "make_search_fn"]
