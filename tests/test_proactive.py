from __future__ import annotations

import asyncio
import datetime as _dt
from collections import deque

import pytest

from digital_life.llm import MockLLM
from digital_life.personas import Relation, xiami_state
from digital_life.proactive import (
    PENDING_TTL,
    can_generate_proactive,
    decay_affinity,
    gen_proactive,
    is_quiet_hours,
    pop_due_pending,
    proactive_score,
    should_proactive,
    update_affinity,
)


NOW = 1_800_000_000.0


def test_score_formula_and_threshold():
    st = xiami_state()
    rel = Relation()
    rel.last_interact = NOW - 5 * 3600
    rel.affinity = 0.2
    st.drive["bond"] = 0.6
    score = proactive_score(st, rel, NOW)
    assert score == pytest.approx(0.27)
    assert should_proactive(st, rel, NOW)

    rel.last_interact = NOW - 4 * 3600
    assert not should_proactive(st, rel, NOW)

    st.drive["bond"] = 0.1
    score = proactive_score(st, rel, NOW)
    assert score > 0.5


def test_quiet_hours():
    rel = Relation()
    rel.profile.timezone = "UTC"
    rel.profile.sleep_start = 23
    rel.profile.sleep_end = 8
    tz = _dt.timezone.utc
    two_am = _dt.datetime(2026, 7, 1, 2, 0, tzinfo=tz).timestamp()
    noon = _dt.datetime(2026, 7, 1, 12, 0, tzinfo=tz).timestamp()
    assert is_quiet_hours(rel, two_am)
    assert not is_quiet_hours(rel, noon)
    rel.profile.quiet_hours_enabled = False
    assert not is_quiet_hours(rel, two_am)


def test_gen_proactive_then_cooldown():
    async def run():
        st = xiami_state()
        rel = Relation()
        rel.last_interact = NOW - 6 * 3600
        rel.affinity = 0.4
        st.drive["bond"] = 0.2
        llm = MockLLM(3)
        text = await gen_proactive(
            llm, st, rel, NOW, deque(["刚才想到一件事"])
        )
        assert text
        assert rel.pending
        assert rel.last_proactive_at == NOW
        second = await gen_proactive(llm, st, rel, NOW + 60, deque())
        assert second is None
        popped = pop_due_pending(rel, NOW + 60)
        assert popped == text
        assert not rel.pending

    asyncio.run(run())


def test_awaiting_reply_expires_and_backs_off():
    rel = Relation()
    rel.proactive_awaiting_reply = True
    rel.proactive_sent_at = NOW - PENDING_TTL - 1
    rel.last_proactive_at = NOW - PENDING_TTL - 1
    rel.proactive_backoff = 1.0
    assert not can_generate_proactive(rel, NOW)  # 退避后冷却还没到
    assert rel.proactive_backoff > 1.0
    assert not rel.proactive_awaiting_reply

    rel.last_proactive_at = NOW - 24 * 3600
    assert can_generate_proactive(rel, NOW)


def test_pending_expiry_and_max():
    rel = Relation()
    rel.pending = [
        {"text": "a", "urgency": 0.3, "expire": NOW - 1},
        {"text": "b", "urgency": 0.9, "expire": NOW + 100},
        {"text": "c", "urgency": 0.8, "expire": NOW + 100},
    ]
    assert pop_due_pending(rel, NOW) == "b"
    assert [m["text"] for m in rel.pending] == ["c"]


def test_long_silence_cools_affinity_slowly():
    rel = Relation()
    rel.affinity = 0.5
    rel.last_interact = NOW - 2 * 86400
    assert decay_affinity(rel, NOW, 86400) == 0.5  # 3 天宽限期内不扣

    rel.last_interact = NOW - 4 * 86400
    assert decay_affinity(rel, NOW, 86400) == pytest.approx(0.49)
    assert decay_affinity(rel, NOW, 10 * 86400) == 0.39


def test_update_affinity_caps_and_skips_short_sessions():
    rel = Relation()
    rel.affinity = 0.2
    short = {
        "n_turns": 1,
        "avg_intensity": 1.0,
        "avg_valence": 1.0,
        "user_initiated": True,
        "pending_responded": True,
    }
    assert update_affinity(rel, short) == 0.2

    warm = {
        "n_turns": 10,
        "avg_intensity": 0.9,
        "avg_valence": 1.0,
        "user_initiated": True,
        "pending_responded": True,
    }
    new = update_affinity(rel, warm)
    assert new == 0.25

    cold = {
        "n_turns": 10,
        "avg_intensity": 0.9,
        "avg_valence": -1.0,
        "user_initiated": False,
        "pending_responded": False,
    }
    assert update_affinity(rel, cold) == 0.2
