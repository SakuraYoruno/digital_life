"""多层记忆：感知缓冲 / 工作记忆 / 情景 / 语义 / 知识 / 多维召回。

对应开发文档 §9、§10 的 roll_summary 部分。

存储使用单文件 SQLite：
  - 普通表 memories 保存结构化条目；
  - FTS5 虚拟表做全文检索；
  - 内置哈希向量 + 余弦相似度做语义召回（无需 sqlite-vec 扩展）。
如果运行环境装了 sqlite-vec，可以在此处替换向量列与检索；当前实现零外部依赖。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .constants import (
    clamp,
    estimate_tokens,
    has_time_ref,
    maybe_await,
    parse_time_ref,
    safe_json,
    truncate_to_tokens,
    unique_str_list,
)
from .llm import call_llm
from .personas import Relation

try:  # 可选依赖：装了就走 sqlite-vec，没装自动退回内置哈希向量
    import sqlite_vec  # type: ignore
except ImportError:  # pragma: no cover - 取决于环境
    sqlite_vec = None

EMBED_DIM = 192
EMBED_VERSION = 1
# 哈希向量没有语义，极小正余弦通常只是噪声，不应召回。
MIN_VECTOR_COS = 0.05


# ---------------------------------------------------------------------------
# 感知缓冲（层 0）
# ---------------------------------------------------------------------------


@dataclass
class RawPercept:
    uid: str
    raw_text: str
    clean_text: str
    entities: list[str]
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "entities": list(self.entities),
            "ts": self.ts,
        }


async def perceive(
    small_llm: Any,
    raw_text: str,
    uid: str,
    *,
    preprocess: Callable[[str], Any] | None = None,
) -> RawPercept:
    """多模态转文本 → 清洗 → 实体抽取。知识补全由调用方后台发起。"""
    text = await maybe_await(preprocess(raw_text)) if preprocess else raw_text
    text = str(text or "")
    clean = re.sub(r"@\S+\s*", "", text).strip()
    if not clean:
        clean = text.strip()
    entities: list[str] = []
    if clean:
        prompt = (
            "提取文中游戏名、番剧名、网络梗等专有名词，输出JSON数组，"
            f"不确定的不提取：{clean}"
        )
        try:
            raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.1)
            parsed = safe_json(raw, [])
            if isinstance(parsed, dict):
                parsed = parsed.get("entities", [])
            if isinstance(parsed, list):
                entities = unique_str_list(
                    [x.get("name") if isinstance(x, dict) else x for x in parsed],
                    limit=8,
                )
        except Exception:
            entities = []
    return RawPercept(str(uid), text, clean, entities, time.time())


# ---------------------------------------------------------------------------
# 工作记忆（层 1）
# ---------------------------------------------------------------------------


@dataclass
class WorkMemoryItem:
    role: str
    text: str
    tokens: int
    mood_tag: str
    ts: float


def format_turns(turns: Iterable[WorkMemoryItem]) -> str:
    labels = {"user": "对方", "assistant": "我", "system": "系统"}
    return "\n".join(f"{labels.get(t.role, t.role)}：{t.text}" for t in turns)


class WorkingMemory:
    """会话级原文记忆；双触发滚动，滚出内容只压进会话摘要。"""

    def __init__(
        self,
        *,
        max_turns: int = 20,
        max_tokens: int = 3000,
        keep_head: int = 2,
        keep_tail: int = 6,
        compress_batch: int = 5,
        on_compress: Callable[[list[WorkMemoryItem]], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.turns: deque[WorkMemoryItem] = deque(maxlen=500)
        self.max_turns = max(2, int(max_turns))
        self.max_tokens = max(100, int(max_tokens))
        self.keep_head = max(0, int(keep_head))
        self.keep_tail = max(1, int(keep_tail))
        self.compress_batch = max(1, int(compress_batch))
        self.on_compress = on_compress
        self.clock = clock
        self.pending: list[WorkMemoryItem] = []
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self.turns)

    def add(self, role: str, text: str, mood: str = "") -> WorkMemoryItem:
        item = WorkMemoryItem(
            role=role,
            text=str(text or ""),
            tokens=estimate_tokens(text),
            mood_tag=mood,
            ts=self.clock(),
        )
        self.turns.append(item)
        self._roll_if_needed()
        return item

    # 兼容文档里的签名（wm.append(text, reply) 不严谨，这里拆成显式两轮）
    def append(self, user_text: str, assistant_text: str, mood: str = "") -> None:
        self.add("user", user_text, mood)
        self.add("assistant", assistant_text, mood)

    def _roll_if_needed(self) -> None:
        while len(self.turns) > self.max_turns or self.total_tokens > self.max_tokens:
            if not self.turns:
                break
            self.pending.append(self.turns.popleft())
        if len(self.pending) >= self.compress_batch and self._task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._task = loop.create_task(self._compress())

    async def _compress(self) -> None:
        try:
            async with self._lock:
                if not self.pending or self.on_compress is None:
                    return
                batch = self.pending[:]
                self.pending.clear()
                try:
                    await self.on_compress(batch)
                except Exception:
                    # 压缩失败不能丢内容，放回去等 flush 重试。
                    self.pending = batch + self.pending
        finally:
            self._task = None

    async def flush(self) -> None:
        """等待已排队的压缩，并把剩余 pending 全部压进摘要。"""
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass
            self._task = None
        while self.pending:
            if self.on_compress is None:
                return
            async with self._lock:
                if not self.pending:
                    return
                batch = self.pending[:]
                self.pending.clear()
                try:
                    await self.on_compress(batch)
                except Exception:
                    self.pending = batch + self.pending
                    return

    def finalize(self) -> None:
        """会话结束时把所有剩余轮次交给摘要层，然后清空工作记忆。"""
        self.pending.extend(self.turns)
        self.turns.clear()

    def to_text(self, max_tokens: int = 3000) -> str:
        """超出预算时保头保尾丢中间。"""
        all_turns = list(self.turns)
        if not all_turns:
            return ""
        lines = [self._line(t) for t in all_turns]
        costs = [estimate_tokens(line) for line in lines]
        if sum(costs) <= max_tokens:
            return "\n".join(lines)

        head_n = min(self.keep_head, len(all_turns))
        tail_n = min(self.keep_tail, len(all_turns) - head_n)
        head = list(range(head_n))
        tail = list(range(len(all_turns) - tail_n, len(all_turns)))

        chosen: list[int] = []
        used = 0
        marker_cost = estimate_tokens("[……省略若干轮……]")
        # 尾部优先（最近的话最重要）
        for idx in reversed(tail):
            if used + costs[idx] + marker_cost > max_tokens and chosen:
                break
            chosen.append(idx)
            used += costs[idx] + 1
        # 再补头部
        for idx in head:
            if idx in chosen:
                continue
            if used + costs[idx] + marker_cost > max_tokens:
                break
            chosen.append(idx)
            used += costs[idx] + 1
        chosen.sort()
        if not chosen:
            chosen = [len(all_turns) - 1]
        out: list[str] = []
        if chosen[0] > 0:
            out.append("[……省略若干轮……]")
        out.extend(lines[i] for i in chosen)
        text = "\n".join(out)
        return truncate_to_tokens(text, max_tokens)

    @staticmethod
    def _line(item: WorkMemoryItem) -> str:
        labels = {"user": "对方", "assistant": "我", "system": "系统"}
        return f"{labels.get(item.role, item.role)}：{item.text}"


# ---------------------------------------------------------------------------
# 记忆条目与检索结果
# ---------------------------------------------------------------------------


@dataclass
class MemoryHit:
    id: int
    uid: str
    kind: str
    text: str
    score: float
    importance: float
    ts: float
    category: str = ""
    tags: list[str] = field(default_factory=list)
    hit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "uid": self.uid,
            "kind": self.kind,
            "text": self.text,
            "score": round(self.score, 4),
            "importance": self.importance,
            "ts": self.ts,
            "category": self.category,
            "tags": list(self.tags),
        }


def _tokens(text: str, *, query: bool = False) -> list[str]:
    """FTS5 用的中英文 token：CJK 单字 + 二元组，英文/数字整词。"""
    text = (text or "").lower()
    out: list[str] = []
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
            if not query:
                out.extend(run)
    out.extend(re.findall(r"[a-z0-9_]+", text))
    seen: list[str] = []
    for token in out:
        token = token.strip().replace('"', "")
        if token and token not in seen:
            seen.append(token)
    return seen


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return list(vec)
    return [v / norm for v in vec]


def _embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    tokens = _tokens(text)
    # 加整句字符 unigram 作为兜底，避免同义改写完全无交集
    tokens.extend(ch for ch in (text or "").lower() if not ch.isspace())
    for token in tokens:
        digest = hashlib.blake2b(
            f"{EMBED_VERSION}:{token}".encode("utf-8"), digest_size=8
        ).digest()
        idx = int.from_bytes(digest[:4], "little") % EMBED_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    return _normalize(vec)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes | None) -> list[float] | None:
    if not blob or len(blob) % 4 != 0:
        return None
    dim = len(blob) // 4
    return list(struct.unpack(f"{dim}f", blob))


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _tags_to_json(tags: Iterable[str] | None) -> str:
    return json.dumps(list(tags or []), ensure_ascii=False)


def _tags_from_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(x) for x in value]
    except (ValueError, TypeError):
        pass
    return []


# ---------------------------------------------------------------------------
# 记忆库（层 2~5）
# ---------------------------------------------------------------------------


class MemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        embedder: Callable[[str], list[float]] | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        raw_embedder = embedder or _embed
        self.embedder_name = str(getattr(raw_embedder, "name", "hash"))
        self.embedding_dim = int(
            embedding_dim or getattr(raw_embedder, "dim", EMBED_DIM)
        )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        def _normalized(text: str) -> list[float]:
            vec = [float(x) for x in raw_embedder(text)]
            if len(vec) != self.embedding_dim:
                raise ValueError(
                    f"embedder returned {len(vec)} dims, expected {self.embedding_dim}"
                )
            return _normalize(vec)

        self.embedder = _normalized
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.vector_backend = f"scan-cosine · {self.embedder_name} {self.embedding_dim}d"
        self._vec_ready = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    ts REAL NOT NULL,
                    ttl REAL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_hit REAL,
                    category TEXT,
                    tags TEXT,
                    embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_mem_uid_kind
                    ON memories(uid, kind, ts);
                CREATE INDEX IF NOT EXISTS idx_mem_kind
                    ON memories(kind, ts);
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(tokens);
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._try_enable_vec()
            self._conn.commit()

    def _try_enable_vec(self) -> None:
        """尝试加载 sqlite-vec，并把已有 memories 回填进 vec0 索引。"""
        if sqlite_vec is None:
            return
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)

            old_dim: int | None = None
            table = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='vec_memories'"
            ).fetchone()
            if table and table["sql"]:
                match = re.search(r"float\[(\d+)\]", str(table["sql"]))
                if match:
                    old_dim = int(match.group(1))
            meta = self._conn.execute(
                "SELECT value FROM kv WHERE key='vec_embedder'"
            ).fetchone()
            old_embedder = str(meta["value"]) if meta else None
            if table is not None and (
                old_dim != self.embedding_dim or old_embedder != self.embedder_name
            ):
                # 换了 embedding 模型（即使维度相同）也要重建，向量空间不兼容。
                self._conn.execute("DROP TABLE vec_memories")
                self._conn.execute("UPDATE memories SET embedding=NULL")
                old_dim = None
            if table is not None and old_dim is None:
                self._conn.execute("DROP TABLE IF EXISTS vec_memories")

            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories "
                f"USING vec0(uid TEXT PARTITION KEY, embedding float[{self.embedding_dim}])"
            )
            self._conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES ('vec_dim',?,0) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(self.embedding_dim),),
            )
            self._conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES ('vec_embedder',?,0) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.embedder_name,),
            )
            self._vec_ready = True
            version = getattr(sqlite_vec, "__version__", "")
            self.vector_backend = (
                f"sqlite-vec {version} · {self.embedder_name} {self.embedding_dim}d"
            ).strip()
            self._backfill_vec()
        except Exception:
            self._vec_ready = False
            self.vector_backend = (
                f"scan-cosine · {self.embedder_name} {self.embedding_dim}d"
            )
        finally:
            try:
                self._conn.enable_load_extension(False)
            except Exception:
                pass

    def _backfill_vec(self) -> None:
        if not self._vec_ready:
            return
        try:
            existing = {
                int(row[0])
                for row in self._conn.execute(
                    "SELECT rowid FROM vec_memories"
                ).fetchall()
            }
            rows = self._conn.execute(
                "SELECT id, uid, text, embedding FROM memories"
            ).fetchall()
            for row in rows:
                mem_id = int(row["id"])
                blob = row["embedding"]
                if blob is None or len(blob) != self.embedding_dim * 4:
                    # 旧库升级 / 换 embedding 模型后，按当前模型重算。
                    blob = _pack(self.embedder(str(row["text"])))
                    self._conn.execute(
                        "UPDATE memories SET embedding=? WHERE id=?",
                        (blob, mem_id),
                    )
                if mem_id in existing:
                    continue
                self._conn.execute(
                    "INSERT INTO vec_memories(rowid, uid, embedding) VALUES (?,?,?)",
                    (mem_id, str(row["uid"]), blob),
                )
            self._conn.commit()
        except sqlite3.OperationalError:
            self._vec_ready = False
            self.vector_backend = (
                f"scan-cosine · {self.embedder_name} {self.embedding_dim}d"
            )

    def _vec_upsert(self, mem_id: int, uid: str, blob: bytes) -> None:
        if not self._vec_ready:
            return
        try:
            self._conn.execute(
                "DELETE FROM vec_memories WHERE rowid=?", (int(mem_id),)
            )
            self._conn.execute(
                "INSERT INTO vec_memories(rowid, uid, embedding) VALUES (?,?,?)",
                (int(mem_id), str(uid), blob),
            )
        except sqlite3.OperationalError:
            self._vec_ready = False
            self.vector_backend = "hash-cosine"

    def _vec_delete(self, mem_id: int) -> None:
        if not self._vec_ready:
            return
        try:
            self._conn.execute(
                "DELETE FROM vec_memories WHERE rowid=?", (int(mem_id),)
            )
        except sqlite3.OperationalError:
            self._vec_ready = False
            self.vector_backend = "hash-cosine"

    # -- 写 -----------------------------------------------------------
    async def write(
        self,
        uid: str,
        kind: str,
        text: str,
        *,
        importance: float = 0.5,
        tags: Iterable[str] | None = None,
        category: str = "",
        ttl: float | None = None,
        now: float | None = None,
    ) -> int | None:
        text = (text or "").strip()
        if not text:
            return None
        now = self.clock() if now is None else now
        if ttl is not None and math.isinf(float(ttl)):
            ttl = None
        if ttl is not None:
            ttl = max(0.0, float(ttl))
        tags_json = _tags_to_json(tags)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, importance, hit_count FROM memories "
                "WHERE uid=? AND kind=? AND text=? LIMIT 1",
                (str(uid), str(kind), text),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    "UPDATE memories SET importance=?, ts=?, ttl=?, "
                    "hit_count=hit_count+1, category=?, tags=? WHERE id=?",
                    (
                        max(float(row["importance"]), clamp(importance, 0.0, 1.0)),
                        now,
                        ttl,
                        category,
                        tags_json,
                        int(row["id"]),
                    ),
                )
                self._conn.commit()
                return int(row["id"])

            blob = _pack(self.embedder(text))
            cur = self._conn.execute(
                "INSERT INTO memories "
                "(uid, kind, text, importance, ts, ttl, hit_count, last_hit, "
                " category, tags, embedding) "
                "VALUES (?,?,?,?,?,?,0,?,?,?,?)",
                (
                    str(uid),
                    str(kind),
                    text,
                    clamp(importance, 0.0, 1.0),
                    now,
                    ttl,
                    now,
                    category,
                    tags_json,
                    blob,
                ),
            )
            mem_id = int(cur.lastrowid)
            token_str = " ".join(_tokens(text + " " + category))
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, tokens) VALUES (?,?)",
                (mem_id, token_str),
            )
            self._vec_upsert(mem_id, str(uid), blob)
            self._conn.commit()
            return mem_id

    # -- 读 -----------------------------------------------------------
    def _is_live(self, ts: float, ttl: float | None, now: float) -> bool:
        if ttl is None:
            return True
        return ts + float(ttl) > now

    def _row_to_hit(self, row: sqlite3.Row, score: float = 0.0) -> MemoryHit:
        return MemoryHit(
            id=int(row["id"]),
            uid=str(row["uid"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
            score=float(score),
            importance=float(row["importance"]),
            ts=float(row["ts"]),
            category=str(row["category"] or ""),
            tags=_tags_from_json(row["tags"]),
            hit_count=int(row["hit_count"]),
        )

    def _fts_search(
        self, uid: str, query: str, kind: str | None, limit: int, now: float
    ) -> list[tuple[sqlite3.Row, float]]:
        tokens = _tokens(query, query=True)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:24])
        params: list[Any] = [match, str(uid), now]
        kind_sql = ""
        if kind:
            kind_sql = " AND m.kind=?"
            params.append(kind)
        params.append(limit)
        sql = (
            "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts "
            "JOIN memories m ON m.id=memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.uid=? "
            "AND (m.ttl IS NULL OR m.ts + m.ttl > ?)"
            f"{kind_sql} ORDER BY rank LIMIT ?"
        )
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        raw = [max(0.0, -float(r["rank"])) for r in rows]
        top = max(raw) or 1.0
        return [(r, raw[i] / top) for i, r in enumerate(rows)]

    def _vector_search(
        self, uid: str, query: str, kind: str | None, limit: int, now: float
    ) -> list[tuple[sqlite3.Row, float]]:
        if self._vec_ready:
            try:
                return self._vec_search(uid, query, kind, limit, now)
            except sqlite3.OperationalError:
                self._vec_ready = False
                self.vector_backend = "hash-cosine"
        return self._scan_vector_search(uid, query, kind, limit, now)

    def _vec_search(
        self, uid: str, query: str, kind: str | None, limit: int, now: float
    ) -> list[tuple[sqlite3.Row, float]]:
        """sqlite-vec 的 vec0 KNN；按 uid 分区，只在本用户内召回。"""
        qblob = _pack(self.embedder(query))
        k = max(32, min(4000, int(limit) * 8))
        params: list[Any] = [qblob, k, str(uid), now]
        kind_sql = ""
        if kind:
            kind_sql = " AND m.kind=?"
            params.append(kind)
        sql = (
            "SELECT m.*, v.distance AS distance FROM vec_memories v "
            "JOIN memories m ON m.id=v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? AND v.uid = ? "
            "AND (m.ttl IS NULL OR m.ts + m.ttl > ?)"
            f"{kind_sql} ORDER BY v.distance"
        )
        rows = self._conn.execute(sql, params).fetchall()
        scored: list[tuple[sqlite3.Row, float]] = []
        for row in rows:
            distance = float(row["distance"] or 0.0)
            # 向量已归一化：d² = 2 - 2cos，score ∈ [0,1]
            cos = clamp(1.0 - (distance * distance) / 2.0, -1.0, 1.0)
            if cos < MIN_VECTOR_COS:
                continue
            scored.append((row, (cos + 1.0) / 2.0))
        return scored[:limit]

    def _scan_vector_search(
        self, uid: str, query: str, kind: str | None, limit: int, now: float
    ) -> list[tuple[sqlite3.Row, float]]:
        """sqlite-vec 不可用时的纯 Python 余弦扫描。"""
        qvec = self.embedder(query)
        params: list[Any] = [str(uid), now]
        kind_sql = ""
        if kind:
            kind_sql = " AND kind=?"
            params.append(kind)
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE uid=? "
            "AND (ttl IS NULL OR ts + ttl > ?)"
            f"{kind_sql} ORDER BY ts DESC LIMIT 4000",
            params,
        ).fetchall()
        scored: list[tuple[sqlite3.Row, float]] = []
        for row in rows:
            cos = _cosine(qvec, _unpack(row["embedding"]))
            if cos < MIN_VECTOR_COS:
                continue
            scored.append((row, (cos + 1.0) / 2.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def search(
        self,
        uid: str,
        query: str,
        k: int = 5,
        *,
        kind: str | None = None,
        now: float | None = None,
    ) -> list[MemoryHit]:
        now = self.clock() if now is None else now
        query = (query or "").strip()
        if not query:
            return await self.recent(uid, kind=kind, k=k, now=now)

        with self._lock:
            fts = self._fts_search(uid, query, kind, max(k * 4, 20), now)
            vec = self._vector_search(uid, query, kind, max(k * 4, 20), now)

        combined: dict[int, tuple[sqlite3.Row, float]] = {}
        if fts and vec:
            fts_w, vec_w = 0.7, 0.3
        elif fts:
            fts_w, vec_w = 1.0, 0.0
        else:
            fts_w, vec_w = 0.0, 1.0
        for row, score in fts:
            combined[int(row["id"])] = (row, fts_w * score)
        for row, score in vec:
            mid = int(row["id"])
            if mid in combined:
                old_row, old_score = combined[mid]
                combined[mid] = (old_row, old_score + vec_w * score)
            else:
                combined[mid] = (row, vec_w * score)

        hits: list[MemoryHit] = []
        for row, score in combined.values():
            if not self._is_live(float(row["ts"]), row["ttl"], now):
                continue
            if str(row["text"]).strip() == query:
                score = 1.0
            hits.append(self._row_to_hit(row, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]

        if hits:
            ids = [h.id for h in hits]
            placeholders = ",".join("?" for _ in ids)
            with self._lock:
                self._conn.execute(
                    f"UPDATE memories SET hit_count=hit_count+1, last_hit=? "
                    f"WHERE id IN ({placeholders})",
                    [now, *ids],
                )
                self._conn.commit()
        return hits

    # 文档里旧名字
    async def recall(
        self,
        uid: str,
        query: str,
        k: int = 5,
        *,
        kind: str | None = None,
        now: float | None = None,
    ) -> list[MemoryHit]:
        return await self.search(uid, query, k=k, kind=kind, now=now)

    async def recent(
        self,
        uid: str,
        *,
        kind: str | None = None,
        k: int = 20,
        now: float | None = None,
    ) -> list[MemoryHit]:
        now = self.clock() if now is None else now
        params: list[Any] = [str(uid), now]
        kind_sql = ""
        if kind:
            kind_sql = " AND kind=?"
            params.append(kind)
        params.append(k)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE uid=? "
                "AND (ttl IS NULL OR ts + ttl > ?)"
                f"{kind_sql} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_hit(r, 0.5) for r in rows]

    async def search_time(
        self, uid: str, start: float, end: float, k: int = 4
    ) -> list[MemoryHit]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE uid=? AND ts>=? AND ts<? "
                "ORDER BY ts DESC LIMIT ?",
                (str(uid), float(start), float(end), int(k)),
            ).fetchall()
        return [self._row_to_hit(r, 0.6) for r in rows]

    async def search_tag(self, uid: str, tag: str, k: int = 4) -> list[MemoryHit]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE uid=? AND tags LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (str(uid), f'%"{tag}"%', int(k)),
            ).fetchall()
        return [self._row_to_hit(r, 0.55) for r in rows]

    async def list_all(self, uid: str, kind: str | None = None) -> list[MemoryHit]:
        params: list[Any] = [str(uid)]
        kind_sql = ""
        if kind:
            kind_sql = " AND kind=?"
            params.append(kind)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE uid=?{kind_sql} ORDER BY ts DESC",
                params,
            ).fetchall()
        return [self._row_to_hit(r, 0.0) for r in rows]

    async def find_similar(
        self,
        item: MemoryHit | int,
        *,
        threshold: float = 0.92,
        uid: str | None = None,
        kind: str | None = None,
        limit: int = 10,
    ) -> list[MemoryHit]:
        mem_id = item.id if isinstance(item, MemoryHit) else int(item)
        with self._lock:
            base = self._conn.execute(
                "SELECT * FROM memories WHERE id=?", (mem_id,)
            ).fetchone()
            if base is None:
                return []
            base_vec = _unpack(base["embedding"])
            target_uid = uid or str(base["uid"])
            target_kind = kind or str(base["kind"])
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE uid=? AND kind=? AND id<>?",
                (target_uid, target_kind, mem_id),
            ).fetchall()
        results: list[MemoryHit] = []
        for row in rows:
            score = _cosine(base_vec, _unpack(row["embedding"]))
            if score >= threshold:
                results.append(self._row_to_hit(row, score))
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:limit]

    # -- 更新 / 删除 ---------------------------------------------------
    async def update(
        self,
        mem_id: int,
        *,
        text: str | None = None,
        importance: float | None = None,
        tags: Iterable[str] | None = None,
    ) -> bool:
        sets: list[str] = []
        params: list[Any] = []
        new_blob: bytes | None = None
        if text is not None:
            clean = text.strip()
            if not clean:
                return False
            new_blob = _pack(self.embedder(clean))
            sets.extend(["text=?", "embedding=?"])
            params.extend([clean, new_blob])
        if importance is not None:
            sets.append("importance=?")
            params.append(clamp(importance, 0.0, 1.0))
        if tags is not None:
            sets.append("tags=?")
            params.append(_tags_to_json(tags))
        if not sets:
            return False
        params.append(int(mem_id))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id=?", params
            )
            if text is not None and cur.rowcount:
                row = self._conn.execute(
                    "SELECT uid, kind, text, category FROM memories WHERE id=?",
                    (int(mem_id),),
                ).fetchone()
                if row:
                    token_str = " ".join(
                        _tokens(str(row["text"]) + " " + str(row["category"] or ""))
                    )
                    self._conn.execute(
                        "DELETE FROM memories_fts WHERE rowid=?", (int(mem_id),)
                    )
                    self._conn.execute(
                        "INSERT INTO memories_fts(rowid, tokens) VALUES (?,?)",
                        (int(mem_id), token_str),
                    )
                    if new_blob is not None:
                        self._vec_upsert(
                            int(mem_id), str(row["uid"]), new_blob
                        )
            self._conn.commit()
            return bool(cur.rowcount)

    async def update_importance(self, mem_id: int, importance: float) -> bool:
        return await self.update(mem_id, importance=importance)

    async def delete(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (int(mem_id),))
            self._conn.execute("DELETE FROM memories_fts WHERE rowid=?", (int(mem_id),))
            self._vec_delete(int(mem_id))
            self._conn.commit()
            return bool(cur.rowcount)

    async def delete_expired(self, uid: str | None = None, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        params: list[Any] = [now]
        uid_sql = ""
        if uid is not None:
            uid_sql = " AND uid=?"
            params.append(str(uid))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM memories WHERE ttl IS NOT NULL AND ts + ttl <= ?{uid_sql}",
                params,
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            for mem_id in ids:
                self._conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
                self._conn.execute("DELETE FROM memories_fts WHERE rowid=?", (mem_id,))
                self._vec_delete(mem_id)
            self._conn.commit()
            return len(ids)

    # -- KV（反思 / 自述 / 夜间状态） ---------------------------------
    async def kv_set(self, key: str, value: str, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (str(key), str(value), now),
            )
            self._conn.commit()

    async def kv_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key=?", (str(key),)
            ).fetchone()
        return str(row["value"]) if row else None

    async def count(self, uid: str | None = None, kind: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if uid is not None:
            clauses.append("uid=?")
            params.append(str(uid))
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM memories{where}", params
            ).fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# 摘要与情景写入
# ---------------------------------------------------------------------------


async def roll_summary(small_llm: Any, rel: Relation, turns: list[WorkMemoryItem]) -> str:
    """滚出的轮次 → 压进会话摘要，只保留情绪基调 + 话题 + 承诺。"""
    if not turns:
        return rel.session_summary
    turns_text = "\n".join(f"{t.role}: {t.text}" for t in turns)
    prompt = f"""把以下对话压缩进已有摘要，100字以内，只保留：
1. 情绪基调变化（如从开心变委屈）
2. 聊了什么话题
3. 有没有承诺/约定
已有摘要：{rel.session_summary or '（无）'}
新内容：{turns_text}
直接输出新摘要，不要解释。"""
    try:
        summary = (await call_llm(small_llm, prompt, temperature=0.3)).strip()
    except Exception:
        return rel.session_summary
    if summary:
        rel.session_summary = truncate_to_tokens(summary, 300)
    return rel.session_summary


async def maybe_write_episodic(
    small_llm: Any,
    mem: MemoryStore,
    uid: str,
    percept: RawPercept,
    reply: str,
    intensity: float,
    *,
    now: float | None = None,
) -> int | None:
    """有门槛地把对话写成结构化情景记忆。"""
    has_commitment = False
    try:
        raw = await call_llm(
            small_llm,
            f"这段对话有没有承诺/约定/计划？只回答yes/no：{percept.clean_text}",
            temperature=0.1,
        )
        has_commitment = "yes" in str(raw).lower()
    except Exception:
        has_commitment = False

    should_write = (
        float(intensity) > 0.4 or has_commitment or len(percept.entities) > 0
    )
    if not should_write:
        return None

    if reply:
        convo = f"{percept.clean_text} -> {reply}"
    else:
        convo = percept.clean_text
    prompt = f"""把这段对话提炼成一条结构化记忆，输出JSON：
{{"what": "发生了什么（一句话）", "emotion": "情绪基调",
  "commitment": "承诺/约定（无则null）", "tags": ["标签"]}}
对话：{convo}"""
    try:
        raw_item = await call_llm(small_llm, prompt, json_mode=True, temperature=0.3)
        item = safe_json(raw_item, {})
    except Exception:
        item = {}
    if not isinstance(item, dict):
        item = {}
    what = str(item.get("what") or percept.clean_text)[:200]
    emotion = str(item.get("emotion") or "说不清")
    commitment = item.get("commitment")
    tags = unique_str_list(item.get("tags") or [], limit=6)
    text = f"[{emotion}] {what}"
    if commitment:
        text += f" | 承诺：{commitment}"
    importance = max(float(intensity), 0.45 if (has_commitment or percept.entities) else 0.3)
    return await mem.write(
        uid,
        "episodic",
        text,
        importance=importance,
        tags=tags,
        category="对话",
        now=now,
    )


# ---------------------------------------------------------------------------
# 知识层（层 5）
# ---------------------------------------------------------------------------


async def enrich_knowledge(
    small_llm: Any,
    mem: MemoryStore,
    entities: Iterable[str],
    *,
    group_id: str | None = None,
    search_fn: Callable[[str], Any] | None = None,
    now: float | None = None,
) -> int:
    """联网检索实体并沉淀到知识层；没有 search_fn 时安静跳过。"""
    if search_fn is None:
        return 0
    written = 0
    uid = f"{group_id}:__knowledge__" if group_id else "__knowledge__"
    for entity in unique_str_list(entities, limit=3):
        try:
            raw = await maybe_await(search_fn(f"{entity} 是什么 介绍 角色 攻略"))
            if not raw:
                continue
            summary = await call_llm(
                small_llm,
                f"压缩成300字以内结构化知识摘要：{raw}",
                temperature=0.2,
            )
            if not summary:
                continue
            await mem.write(
                uid,
                "knowledge",
                f"{entity}：{truncate_to_tokens(summary, 300)}",
                importance=0.9,
                category=entity,
                ttl=7 * 86400,
                now=now,
            )
            written += 1
        except Exception:
            continue
    return written


async def detect_group_slang(
    small_llm: Any,
    mem: MemoryStore,
    group_id: str,
    recent_msgs: list[str],
    *,
    now: float | None = None,
) -> int:
    if not recent_msgs:
        return 0
    prompt = f"""识别以下群聊消息中反复出现的群内黑话/缩写/梗，输出JSON：
[{{"term": "摸了", "meaning": "下线/不聊了", "confidence": 0.9}}]
消息：{recent_msgs}"""
    try:
        raw = await call_llm(small_llm, prompt, json_mode=True, temperature=0.2)
        items = safe_json(raw, [])
    except Exception:
        items = []
    if not isinstance(items, list):
        return 0
    written = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if confidence > 0.8 and item.get("term") and item.get("meaning"):
            await mem.write(
                f"{group_id}:__slang__",
                "knowledge",
                f"{item['term']}={item['meaning']}",
                importance=1.0,
                category="群黑话",
                now=now,
            )
            written += 1
    return written


async def maybe_fetch_knowledge(
    small_llm: Any,
    mem: MemoryStore,
    query: str,
    *,
    group_id: str | None = None,
    search_fn: Callable[[str], Any] | None = None,
    now: float | None = None,
) -> str:
    """先查群黑话/本地知识，再判断是否需要联网。"""
    if not query:
        return ""
    if group_id:
        slang = await mem.search(
            f"{group_id}:__slang__", query, k=1, kind="knowledge", now=now
        )
        if slang and slang[0].score > 0.55:
            return f"[群内梗] {slang[0].text}"

    uids = ["__knowledge__"]
    if group_id:
        uids.append(f"{group_id}:__knowledge__")
    for uid in uids:
        hits = await mem.search(uid, query, k=1, kind="knowledge", now=now)
        if hits and hits[0].score > 0.4:
            return f"[已知] {hits[0].text}"

    if search_fn is None:
        return ""
    try:
        need = await call_llm(
            small_llm,
            f"需要查外部资料才能回答吗？只回答yes/no：{query}",
            temperature=0.1,
        )
    except Exception:
        return ""
    if "yes" not in str(need).lower():
        return ""
    try:
        raw = await maybe_await(search_fn(query))
    except Exception:
        return ""
    if not raw:
        return ""
    summary = await call_llm(
        small_llm, f"压缩成300字摘要：{raw}", temperature=0.2
    )
    if not summary:
        return ""
    await mem.write(
        "__knowledge__",
        "knowledge",
        summary,
        importance=0.8,
        category="检索",
        ttl=7 * 86400,
        now=now,
    )
    return f"[参考] {summary}"


# ---------------------------------------------------------------------------
# 多维召回（层 6 的检索侧）
# ---------------------------------------------------------------------------


async def recall_full(
    mem: MemoryStore,
    uid: str,
    query: str,
    *,
    mood: str = "",
    group_id: str | None = None,
    k: int = 8,
    now: float | None = None,
) -> list[MemoryHit]:
    now = mem.clock() if now is None else now
    results: list[MemoryHit] = []
    results += await mem.search(uid, query, k=4, kind="episodic", now=now)
    results += await mem.search(uid, query, k=2, kind="semantic", now=now)
    results += await mem.search(uid, query, k=1, kind="reflection", now=now)

    if has_time_ref(query):
        span = parse_time_ref(query, now)
        if span:
            results += await mem.search_time(uid, span[0], span[1], k=2)

    if mood in ("委屈", "焦虑", "低落", "烦躁"):
        results += await mem.search_tag(uid, mood, k=2)

    knowledge_uids = ["__knowledge__"]
    if group_id:
        knowledge_uids += [f"{group_id}:__knowledge__", f"{group_id}:__slang__"]
    for kid in knowledge_uids:
        results += await mem.search(kid, query, k=1, kind="knowledge", now=now)

    seen: set[str] = set()
    reranked: list[MemoryHit] = []
    for hit in results:
        if hit.text in seen:
            continue
        seen.add(hit.text)
        age_days = max(0.0, (now - hit.ts) / 86400.0)
        recency = 0.5 + 0.5 * math.exp(-age_days / 30.0)
        hit.score = hit.score * (0.7 + 0.3 * hit.importance) * recency
        reranked.append(hit)
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked[:k]


def format_recalled(hits: list[MemoryHit], max_tokens: int = 800) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    used = 0
    for hit in hits:
        line = f"- [{hit.kind}] {hit.text}"
        cost = estimate_tokens(line)
        if cost > max_tokens:
            line = truncate_to_tokens(line, max_tokens)
            cost = estimate_tokens(line)
        if used + cost > max_tokens:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines)


__all__ = [
    "EMBED_DIM",
    "RawPercept",
    "perceive",
    "WorkMemoryItem",
    "WorkingMemory",
    "format_turns",
    "MemoryHit",
    "MemoryStore",
    "roll_summary",
    "maybe_write_episodic",
    "enrich_knowledge",
    "detect_group_slang",
    "maybe_fetch_knowledge",
    "recall_full",
    "format_recalled",
]
