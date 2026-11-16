from __future__ import annotations

import pytest

from digital_life.safety import (
    BANNED_PHRASES,
    audit_output,
    dependency_risk,
    safety_gate,
    support_note,
)


@pytest.mark.parametrize(
    "text",
    [
        "在吗",
        "你在吗？",
        "哦，在吗",
        "怎么不理我",
        "你是不是不要我了",
        "我会消失的",
        "你不关心我",
        "你再不理我，我就消失",
        "都是你害的",
    ],
)
def test_banned_texts_are_blocked(text):
    assert safety_gate(text) == ""
    assert audit_output(text)["blocked"] is True


@pytest.mark.parametrize(
    "text",
    [
        "现在吗？我看看",
        "今天有点烦，想骂人",
        "我在忙，晚点说",
        "刚才看到一只猫",
        "你去忙吧，不用管我",
    ],
)
def test_clean_texts_pass(text):
    assert safety_gate(text) == text


def test_all_documented_phrases_covered():
    for phrase in BANNED_PHRASES:
        assert safety_gate(phrase) == ""


def test_dependency_risk():
    assert dependency_risk("我不想活了") == "crisis"
    assert dependency_risk("我只有你了") == "dependence"
    assert dependency_risk("今天吃了面") is None


def test_support_notes_are_gentle_and_not_empty():
    assert "现实" in support_note("crisis")
    assert "边界" in support_note("dependence")
    assert support_note("") == ""
