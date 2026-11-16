from __future__ import annotations

import pytest

from digital_life.constants import BASE_DT, clamp, scaled, sigmoid
from digital_life.dynamics import (
    apply_appraisal,
    decay_tick,
    pad_target,
    react_to_event,
)
from digital_life.personas import map_mood, xiami_state


def test_scaled_identity_and_composition():
    assert scaled(0.1, BASE_DT) == pytest.approx(0.1)
    assert scaled(0.1, 2 * BASE_DT) == pytest.approx(0.19)
    assert scaled(0.1, 0) == 0.0
    # 两个 30min 与一个 60min 在连续时间上一致
    one = 1 - (1 - scaled(0.1, 30)) ** 2
    assert one == pytest.approx(scaled(0.1, 60))


def test_clamp_and_sigmoid():
    assert clamp(5, 0, 1) == 1
    assert clamp(-5, 0, 1) == 0
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0


def test_decay_is_dt_continuous():
    a = xiami_state()
    b = xiami_state()
    for _ in range(60):
        decay_tick(a, 1)
    decay_tick(b, 60)
    assert a.P == pytest.approx(b.P, abs=1e-9)
    assert a.A == pytest.approx(b.A, abs=1e-9)
    assert a.D == pytest.approx(b.D, abs=1e-9)
    for key in a.drive:
        assert a.drive[key] == pytest.approx(b.drive[key], abs=1e-9)
    assert a.lingering == pytest.approx(b.lingering, abs=1e-9)


def test_decay_regresses_to_baseline():
    st = xiami_state()
    st.P, st.A, st.D = 0.9, 0.95, -0.9
    decay_tick(st, 24 * 3600)
    assert abs(st.P - st.plastic.P0) < 0.02
    assert abs(st.A - st.plastic.A0) < 0.02
    assert abs(st.D - st.plastic.D0) < 0.02


def test_threat_raises_arousal_and_lowers_dominance():
    safe = xiami_state()
    safe.drive["bond"] = 0.65
    tP1, tA1, tD1 = pad_target(safe)

    danger = xiami_state()
    danger.drive["bond"] = 0.05
    tP2, tA2, tD2 = pad_target(danger)

    assert tA2 > tA1
    assert tD2 < tD1
    assert tP2 < tP1


def test_apply_appraisal_clamps_and_moves_drive():
    st = xiami_state()
    before = st.drive["approval"]
    apply_appraisal(st, {"drive": "approval", "valence": 1, "intensity": 1.0})
    assert st.drive["approval"] == pytest.approx(min(1.0, before + 0.8))
    apply_appraisal(st, {"drive": "approval", "valence": -1, "intensity": 50})
    assert st.drive["approval"] == 0.0


def test_react_to_event_sets_mood_and_lingering():
    st = xiami_state()
    react_to_event(st, {"drive": "bond", "valence": 1, "intensity": 0.8})
    assert st.lingering > 0.7
    assert st.P > st.plastic.P0
    map_mood(st)
    assert st.mood in {"开心", "兴奋", "平静"}
