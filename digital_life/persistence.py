"""状态持久化与 Obsidian 反思落盘。

对应开发文档 §1.1、§12.1、§15.1。
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .constants import safe_json
from .personas import LifeState


def state_to_dict(st: LifeState) -> dict[str, Any]:
    return st.to_dict()


def state_from_dict(data: dict[str, Any] | None) -> LifeState:
    return LifeState.from_dict(data)


class StateStore:
    """LifeState 快照，单行 KV。重启后恢复身份连续性。"""

    def __init__(self, path: str | Path, *, clock=time.time) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS life_state ("
            "id INTEGER PRIMARY KEY CHECK (id=1), snapshot TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        self._conn.commit()

    def save(self, st: LifeState, *, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        payload = json.dumps(state_to_dict(st), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO life_state(id,snapshot,updated_at) VALUES (1,?,?) "
                "ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot, "
                "updated_at=excluded.updated_at",
                (payload, now),
            )
            self._conn.commit()

    def load(self) -> LifeState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot FROM life_state WHERE id=1"
            ).fetchone()
        if not row:
            return None
        data = safe_json(row["snapshot"], None)
        if not isinstance(data, dict):
            return None
        return state_from_dict(data)

    def updated_at(self) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT updated_at FROM life_state WHERE id=1"
            ).fetchone()
        return float(row["updated_at"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _safe_token(uid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in uid)[:80]


class ObsidianReflector:
    """把反思、自我叙事、关系图谱写进 Obsidian 仓库。"""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault = Path(vault_path)
        self.reflect_dir = self.vault / "数字生命" / "反思"
        self.identity_dir = self.vault / "数字生命"
        self.reflect_dir.mkdir(parents=True, exist_ok=True)

    def _date(self, now: float) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(now)

    def write_reflection(
        self,
        uid: str,
        reflection: dict[str, Any],
        *,
        rel: Any | None = None,
        now: float | None = None,
    ) -> Path:
        now = time.time() if now is None else now
        stamp = self._date(now)
        path = self.reflect_dir / f"{stamp:%Y-%m-%d}.md"
        lines: list[str] = []
        if not path.exists():
            lines.append(f"# {stamp:%Y-%m-%d} 反思\n")
        lines.append(f"\n## {stamp:%H:%M} · `{uid}`\n")
        insight = str(reflection.get("insight") or "").strip()
        boundary = str(reflection.get("boundary") or "").strip()
        if insight:
            lines.append(f"- 感悟：{insight}\n")
        if boundary:
            lines.append(f"- 边界：{boundary}\n")
        if rel is not None:
            lines.append(
                f"- 亲密度：{rel.affinity:.2f}；策略：{rel.persona.strategy or '（暂无）'}\n"
            )
        with path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write("".join(lines))
        return path

    def write_self_narrative(self, st: LifeState) -> Path:
        path = self.identity_dir / "自我叙事.md"
        text = st.self_narrative.strip() or "（还没形成稳定的自述）"
        path.write_text(
            f"# 我是谁\n\n{text}\n\n> 由 nightly_reflect 自动更新。\n",
            encoding="utf-8",
        )
        return path

    def write_relation_graph(self, st: LifeState) -> Path:
        path = self.identity_dir / "关系图谱.md"
        rows = [
            "| 用户 | 亲密度 | 称呼 | 策略 | 边界 |",
            "|---|---:|---|---|---|",
        ]
        for uid, rel in st.relations.items():
            strategy = (rel.persona.strategy or "").replace("|", "／")
            boundaries = "；".join(rel.persona.boundaries).replace("|", "／")
            rows.append(
                f"| `{uid}` | {rel.affinity:.2f} | {rel.persona.nickname or '—'} | "
                f"{strategy or '—'} | {boundaries or '—'} |"
            )
        path.write_text(
            "# 关系图谱\n\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )
        return path


__all__ = ["StateStore", "ObsidianReflector", "state_to_dict", "state_from_dict"]
