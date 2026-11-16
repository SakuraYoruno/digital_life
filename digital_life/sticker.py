"""表情包库：按情绪从本地文件夹里挑一张发出去。

不靠给每张图打标签——`表情包收藏.json` 里那些人工写的一句话（"挠头懵逼"、
"等不到回复"）本身就是最好的标签，拿 mood 关键词去撞它就行。
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# mood → 关键词。撞不上就随便挑一张，聊胜于无。
_MOOD_HINTS: dict[str, tuple[str, ...]] = {
    "开心": ("笑", "哈", "得意", "坏笑", "耶", "比心", "眨眼"),
    "兴奋": ("激动", "冲", "燃", "耶", "起飞", "兴奋"),
    "平静": ("盯", "看", "嗯", "点头", "发呆"),
    "无聊": ("无语", "白眼", "发呆", "无聊", "瘫"),
    "低落": ("丧", "躺", "失魂", "生无可恋", "失望", "哭"),
    "委屈": ("委屈", "可怜", "哭", "泪"),
    "焦虑": ("慌", "紧张", "冷汗", "问号"),
    "烦躁": ("烦", "气", "白眼", "无语", "嫌弃"),
}


@dataclass(frozen=True)
class Sticker:
    path: str
    summary: str = ""

    @property
    def file_url(self) -> str:
        # NapCat 吃 file:/// 的绝对路径；中文目录不做 percent-encoding，
        # 否则部分版本不会解码，图会发成空文件。
        return "file:///" + self.path.replace("\\", "/").lstrip("/")


class StickerBook:
    """一个文件夹 + 可选的描述文件。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        metadata_path: str | Path | None = None,
        rng: random.Random | None = None,
        recent: int = 8,
    ) -> None:
        self.directory = Path(directory)
        self.rng = rng or random.Random()
        self._recent: deque[str] = deque(maxlen=max(1, recent))
        summaries = self._load_metadata(metadata_path)
        self.stickers: list[Sticker] = []
        if self.directory.is_dir():
            for path in sorted(self.directory.iterdir()):
                if path.suffix.lower() not in _EXTS:
                    continue
                self.stickers.append(
                    Sticker(str(path), summaries.get(str(path).lower(), ""))
                )

    @property
    def size(self) -> int:
        return len(self.stickers)

    def _load_metadata(self, metadata_path: str | Path | None) -> dict[str, str]:
        if not metadata_path:
            return {}
        try:
            data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 描述文件坏了就当没有
            return {}
        mapping: dict[str, str] = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("path"):
                    mapping[str(item["path"]).lower()] = str(item.get("summary") or "")
        return mapping

    # -- 挑图 ---------------------------------------------------------
    def pick(self, mood: str = "", text: str = "") -> Sticker | None:
        if not self.stickers:
            return None
        hints = _MOOD_HINTS.get(mood, ())
        scored = [(self._score(s, hints), s) for s in self.stickers]
        best = max(score for score, _ in scored)
        pool: Iterable[Sticker]
        if best > 0:
            pool = [s for score, s in scored if score == best]
        else:
            pool = self.stickers
        fresh = [s for s in pool if s.path not in self._recent]
        chosen = self.rng.choice(fresh or list(pool))
        self._recent.append(chosen.path)
        return chosen

    @staticmethod
    def _score(sticker: Sticker, hints: Iterable[str]) -> int:
        summary = sticker.summary
        if not summary:
            return 0
        return sum(1 for hint in hints if hint and hint in summary)


__all__ = ["Sticker", "StickerBook"]
