"""知识库导入：把 Obsidian / Markdown 资料切块、建索引，写进知识层。

为什么不是"直接把 Obsidian 当知识库"：`ObsidianReflector` 只写不读，
运行时召回全部走 SQLite 的 `memories(kind='knowledge')`。所以 Obsidian 的正确角色是
**编写与版本管理的地方**，导入这一步把它变成可检索的索引。

切块策略：按二级标题切，单块超过 `max_chars` 再按段落切。标题会拼进正文，
这样"绘梨衣 阵容"这种查询能命中"## 绘梨衣阵容"下面的内容。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

_HEADING = re.compile(r"^#{1,3}\s+(.*)$", re.M)


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """返回 [(标题, 正文)]；没有标题就退化成整篇一段。"""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text.strip())]
    chunks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append((title, body))
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """按句号切不动时，退化成按长度硬切（保留标点附近的可读性）。"""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(ch) for ch in "。！？\n")
            if cut > max_chars // 2:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def _split_long(title: str, body: str, max_chars: int) -> Iterable[tuple[str, str]]:
    if len(body) <= max_chars:
        yield title, body
        return
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    buffer = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # 单个段落就超长（比如一整段攻略），先按句子/长度切开
            if buffer:
                yield title, buffer
                buffer = ""
            for piece in _hard_split(para, max_chars):
                yield title, piece
            continue
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}".strip()
            continue
        if buffer:
            yield title, buffer
        buffer = para
    if buffer:
        for piece in _hard_split(buffer, max_chars):
            yield title, piece


def chunk_markdown(text: str, *, max_chars: int = 700) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    for title, body in _split_by_heading(text):
        pieces.extend(_split_long(title, body, max_chars))
    return pieces


async def import_markdown(
    mem: Any,
    path: str | Path,
    *,
    uid: str = "__knowledge__",
    max_chars: int = 700,
    now: float | None = None,
) -> dict[str, int]:
    """把一个 .md 文件导入知识层，返回统计。"""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_markdown(text, max_chars=max_chars)
    written = skipped = 0
    for title, body in chunks:
        digest = hashlib.sha1(f"{uid}|{title}|{body}".encode("utf-8")).hexdigest()
        key = f"kb:{uid}:{digest}"
        if await mem.kv_get(key):
            skipped += 1
            continue
        content = f"{title}\n{body}" if title else body
        await mem.write(
            uid,
            "knowledge",
            content,
            importance=0.7,
            category=file_path.stem,
            tags=["kb", file_path.name],
            now=now,
        )
        await mem.kv_set(key, file_path.name, now=now)
        written += 1
    return {"written": written, "skipped": skipped, "chunks": len(chunks)}


__all__ = ["chunk_markdown", "import_markdown"]
