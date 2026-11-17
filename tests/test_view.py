"""记忆星图：数据组装与 HTML 生成。"""

from __future__ import annotations

import asyncio

from digital_life.memory import MemoryStore
from digital_life.persistence import StateStore
from digital_life.personas import xiami_state
from digital_life.view import build_payload, export_map, render_html


def _seed(tmp_path):
    mem = MemoryStore(tmp_path / "m.db", embedder=lambda t: [0.1] * 8, embedding_dim=8)

    async def write():
        await mem.write("111", "episodic", "[开心] 她说今天很热", importance=0.8)
        await mem.write("222", "episodic", "[低落] 被老板骂了", importance=0.6)
        await mem.write("__knowledge__", "knowledge", "龙族卡塞尔之门 攻略", importance=0.9)

    asyncio.run(write())
    mem.close()

    store = StateStore(tmp_path / "s.db")
    state = xiami_state()
    state.rel("111").persona.nickname = "小虾米"
    store.save(state)
    store.close()
    return str(tmp_path / "m.db"), str(tmp_path / "s.db")


def test_payload_has_nodes_and_rings(tmp_path):
    mem_path, state_path = _seed(tmp_path)
    payload = build_payload(mem_path, state_path)
    assert len(payload["nodes"]) == 3
    assert payload["core"]["mood"] == "平静"
    labels = {ring["who"] for ring in payload["rings"]}
    assert "小虾米" in labels and "知识" in labels
    assert all("color" in node for node in payload["nodes"])


def test_render_embeds_data_inline(tmp_path):
    mem_path, state_path = _seed(tmp_path)
    html = render_html(build_payload(mem_path, state_path))
    assert "意识海" in html
    assert "她说今天很热" in html
    assert "__DATA__" not in html          # 占位符被替换掉了
    assert "http://" not in html           # 自包含，不引用外部资源


def test_export_writes_file(tmp_path):
    mem_path, state_path = _seed(tmp_path)
    out = tmp_path / "map.html"
    path = export_map(mem_path, state_path, out)
    assert path.exists() and path.stat().st_size > 5000


def test_empty_memory_still_renders(tmp_path):
    mem = MemoryStore(tmp_path / "m.db", embedder=lambda t: [0.1] * 8, embedding_dim=8)
    mem.close()
    store = StateStore(tmp_path / "s.db")
    store.save(xiami_state())
    store.close()

    payload = build_payload(str(tmp_path / "m.db"), str(tmp_path / "s.db"))
    assert payload["nodes"] == []
    html = render_html(payload)
    assert "还没有记忆" in html
