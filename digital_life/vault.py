"""知识库：直接读 Markdown 源文件，而不是把正文复制进数据库。

为什么这样做：把正文拷进 SQLite 就有了两个真相——你在 Obsidian 里改了，库里还是旧的；
删了文件，库里还留着。索引可以是副本，**正文必须读源文件**。

做法：
- 按 mtime/size 缓存解析结果，文件动了才重新切块
- 检索用中文 bigram + BM25：不依赖 embedding，也不需要重新建索引
- 命中后把**文件里当前的正文**返回给上层

embedding 那条路仍然保留（`kb` 子命令导入到 memories），但它是"可选加速"，
不是真相：文件改了，向量才会因为内容哈希变化而失效。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .kb import chunk_markdown

_LATIN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = re.compile(r"[\s，。！？；：、“”‘’（）()\[\]【】《》…—\-~!?,.;:\"'<>/\\|]+")

_STOP_BIGRAMS = {
    "什么", "怎么", "这个", "那个", "可以", "一下", "时候", "现在", "还是",
    "就是", "不是", "有没", "没有", "我们", "你们", "他们", "如何", "为什么",
}


def tokenize(text: str) -> list[str]:
    """中文按 bigram、英文按词；够用且不依赖分词库。"""
    lowered = text.lower()
    tokens: list[str] = _LATIN.findall(lowered)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if gram not in _STOP_BIGRAMS:
                tokens.append(gram)
    return tokens


@dataclass
class Chunk:
    path: str
    title: str
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        name = Path(self.path).stem
        return f"{name} · {self.title}" if self.title else name


class VaultKB:
    """一个目录（或单个 .md）的知识库，检索时读文件当前内容。"""

    def __init__(self, root: str | Path, *, chunk_chars: int = 700) -> None:
        self.root = Path(root)
        self.chunk_chars = chunk_chars
        self._files: dict[str, tuple[float, int]] = {}
        self._chunks: list[Chunk] = []
        self._df: dict[str, int] = {}
        self._avgdl = 1.0
        self.refresh()

    # -- 索引 ---------------------------------------------------------
    def _iter_files(self) -> Iterable[Path]:
        if self.root.is_file() and self.root.suffix.lower() == ".md":
            yield self.root
        elif self.root.is_dir():
            yield from sorted(self.root.rglob("*.md"))

    def refresh(self, *, force: bool = False) -> int:
        """按 mtime/size 增量重建；返回重新解析的文件数。"""
        changed = 0
        seen: set[str] = set()
        for path in self._iter_files():
            key = str(path)
            seen.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp = (stat.st_mtime, stat.st_size)
            if not force and self._files.get(key) == stamp:
                continue
            self._files[key] = stamp
            self._chunks = [c for c in self._chunks if c.path != key]
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for title, body in chunk_markdown(text, max_chars=self.chunk_chars):
                content = f"{title}\n{body}" if title else body
                self._chunks.append(
                    Chunk(
                        path=key,
                        title=title,
                        text=content,
                        tokens=tokenize(content),
                    )
                )
            changed += 1

        dropped = set(self._files) - seen
        if dropped:
            for key in dropped:
                self._files.pop(key, None)
            self._chunks = [c for c in self._chunks if c.path in seen]
            changed += len(dropped)

        if changed:
            self._rebuild_stats()
        return changed

    def _rebuild_stats(self) -> None:
        df: dict[str, int] = {}
        for chunk in self._chunks:
            for token in set(chunk.tokens):
                df[token] = df.get(token, 0) + 1
        self._df = df
        total = sum(len(c.tokens) for c in self._chunks)
        self._avgdl = (total / len(self._chunks)) if self._chunks else 1.0

    # -- 检索 ---------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(
        self, query: str, *, k: int = 2, min_score: float = 1.2
    ) -> list[Chunk]:
        if not query or not self._chunks:
            return []
        self.refresh()  # 文件可能刚被改过
        tokens = tokenize(query)
        if not tokens:
            return []

        n = len(self._chunks)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, Chunk]] = []
        for chunk in self._chunks:
            dl = len(chunk.tokens) or 1
            freq: dict[str, int] = {}
            for token in chunk.tokens:
                freq[token] = freq.get(token, 0) + 1
            score = 0.0
            title_tokens = set(tokenize(chunk.title))
            for token in tokens:
                tf = freq.get(token, 0)
                if not tf:
                    continue
                df = self._df.get(token, 0) or 1
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * dl / self._avgdl)
                score += idf * tf * (k1 + 1) / denom
                if token in title_tokens:
                    score += idf * 0.8  # 标题命中加权
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for score, chunk in scored[:k] if score >= min_score]

    def context(self, query: str, *, k: int = 2) -> str:
        """给 prompt 用的一段文本；没命中返回空串。"""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        blocks = [
            f"[资料: {chunk.label}]\n{chunk.text}" for chunk in hits
        ]
        return "\n\n".join(blocks)


__all__ = ["Chunk", "VaultKB", "tokenize"]
