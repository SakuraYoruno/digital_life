"""知识库直读：检索命中的是文件当前内容，不是入库的副本。"""

from __future__ import annotations

from digital_life.vault import VaultKB, tokenize

DOC = """# 龙族：卡塞尔之门

卡牌回合制手游，改编自江南小说《龙族》。

## 绘梨衣阵容

绘梨衣是水系输出，前排配源稚生，后排堆暴击。

## 新手建议

攒够三十抽再决定主 C。
"""


def _make(tmp_path) -> VaultKB:
    (tmp_path / "game.md").write_text(DOC, encoding="utf-8")
    return VaultKB(tmp_path)


def test_search_hits_right_section(tmp_path):
    kb = _make(tmp_path)
    hits = kb.search("绘梨衣怎么配队", k=1)
    assert hits and hits[0].title == "绘梨衣阵容"


def test_search_misses_unrelated(tmp_path):
    kb = _make(tmp_path)
    assert kb.search("今天天气怎么样") == []
    assert kb.search("你吃饭了吗") == []


def test_edit_is_picked_up_without_reindex(tmp_path):
    kb = _make(tmp_path)
    before = kb.search("绘梨衣怎么配队", k=1)
    assert before and "源稚生" in before[0].text

    target = tmp_path / "game.md"
    target.write_text(
        DOC.replace("前排配源稚生", "前排换成上杉越"), encoding="utf-8"
    )
    after = kb.search("绘梨衣怎么配队", k=1)
    assert after and "上杉越" in after[0].text
    assert "源稚生" not in after[0].text


def test_deleted_file_drops_out(tmp_path):
    kb = _make(tmp_path)
    assert kb.search("绘梨衣怎么配队", k=1)
    (tmp_path / "game.md").unlink()
    kb.refresh()
    assert kb.search("绘梨衣怎么配队", k=1) == []


def test_tokenize_is_bigram_based():
    tokens = tokenize("绘梨衣阵容")
    assert "绘梨" in tokens and "梨衣" in tokens
    assert "绘梨衣阵容" not in tokens


def test_context_block_has_source_label(tmp_path):
    kb = _make(tmp_path)
    text = kb.context("绘梨衣怎么配队")
    assert "资料:" in text and "game" in text
