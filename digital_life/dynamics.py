"""驱力 → PAD 的确定性链路，以及慢循环的时间演化。

对应开发文档 §2.3、§3、§16。
"""

from __future__ import annotations

from .constants import clamp, scaled, sigmoid
from .personas import LifeState, PlasticPersona


def drive_baseline(p: PlasticPersona) -> dict[str, float]:
    return {
        "bond": p.bond0,
        "approval": p.approval0,
        "curiosity": p.curiosity0,
        "coherence": p.coherence0,
    }


def apply_appraisal(st: LifeState, a: dict) -> None:
    """drive[d] += sens[d] * valence * intensity，并 clamp 到 [0,1]。"""
    drive = a["drive"]
    if drive not in st.drive:
        return
    p = st.plastic
    st.drive[drive] = clamp(
        st.drive[drive]
        + p.sens.get(drive, 1.0) * float(a["valence"]) * float(a["intensity"]),
        0.0,
        1.0,
    )


def pad_target(
    st: LifeState, intensity: float | None = None
) -> tuple[float, float, float]:
    """缺口 + 本次事件强度 → 目标 PAD。threat 是求生欲/焦虑的数学出口。"""
    p = st.plastic
    base = drive_baseline(p)
    gap = {k: st.drive[k] - base[k] for k in base}

    target_P = clamp(
        p.P0 + sum(p.wP.get(k, 0.0) * gap[k] for k in gap) * 2.0,
        -1.0,
        1.0,
    )

    # bond 越接近（甚至低于）警戒线，threat 越接近 1。
    threat = sigmoid((p.guard_line - st.drive["bond"]) * p.steepness)
    # 文档写的是 abs(mean(gap))；这里用 mean(abs(gap))，更符合“波动量”的语义：
    # 正负缺口同时存在时不会互相抵消，A 才会对混合事件有反应。
    change = sum(abs(g) for g in gap.values()) / max(1, len(gap))
    # 累积缺口只能解释"长期状态"，解释不了"这句话让我一激灵"。
    # 所以再叠一条事件强度通道：强度高于 impulse_floor 抬高 A，
    # 低于它（敷衍、无聊的一句话）压一点 A，让"无聊"这种低唤醒状态也能出现。
    impulse = 0.0
    if intensity is not None:
        impulse = clamp(float(intensity) - p.impulse_floor, -0.5, 0.75)
    target_A = clamp(
        p.A0
        + p.w_A_threat * threat
        + p.w_A_change * change
        + p.w_A_impulse * impulse,
        0.0,
        1.0,
    )

    target_D = clamp(
        p.D0 + p.w_D * (gap["approval"] + gap["coherence"]) - 0.5 * threat,
        -1.0,
        1.0,
    )
    return target_P, target_A, target_D


def approach_target(st: LifeState, tP: float, tA: float, tD: float) -> None:
    """情绪向本次事件目标惯性逼近。"""
    p = st.plastic
    st.P = clamp(st.P + p.k_P * (tP - st.P), -1.0, 1.0)
    st.A = clamp(st.A + p.k_A * (tA - st.A), 0.0, 1.0)
    st.D = clamp(st.D + p.k_D * (tD - st.D), -1.0, 1.0)


def idle_target(st: LifeState) -> tuple[float, float, float]:
    """没人说话时"回落到哪"。

    固定回落点会让角色永远回到"有点开心"——被忽视再久也不会无聊/低落，
    因为基线是常数。这里让回落点随"驱力亏空"下移：亏空越大，越无趣、越蔫。
    亏空 = 各驱力低于基线的比例均值（0 表示都满足，1 表示全空）。
    """
    p = st.plastic
    base = drive_baseline(p)
    deficit = 0.0
    for key, anchor in base.items():
        if anchor <= 0:
            continue
        missing = max(0.0, anchor - st.drive[key]) / anchor
        deficit += missing
    deficit = clamp(deficit / max(1, len(base)), 0.0, 1.0)

    return (
        clamp(p.P0 - 0.35 * deficit, -1.0, 1.0),
        clamp(p.A0 - 0.25 * deficit, 0.0, 1.0),
        clamp(p.D0 - 0.20 * deficit, -1.0, 1.0),
    )


def decay_tick(st: LifeState, dt: float) -> None:
    """没人说话时：驱力流失、情绪回归基线、余味衰减。"""
    p = st.plastic
    dt = max(0.0, float(dt))
    for key in list(st.drive):
        loss = scaled(p.lam.get(key, 0.0), dt) + scaled(p.alpha_idle, dt)
        st.drive[key] = clamp(st.drive[key] - loss, 0.0, 1.0)

    tP, tA, tD = idle_target(st)
    st.P += scaled(p.rho_P, dt) * (tP - st.P)
    st.A += scaled(p.rho_A, dt) * (tA - st.A)
    st.D += scaled(p.rho_D, dt) * (tD - st.D)
    st.P = clamp(st.P, -1.0, 1.0)
    st.A = clamp(st.A, 0.0, 1.0)
    st.D = clamp(st.D, -1.0, 1.0)

    st.lingering = clamp(st.lingering * (1.0 - scaled(0.1, dt)), 0.0, 1.0)


def react_to_event(st: LifeState, appraisal: dict) -> tuple[float, float, float]:
    """快循环的标准顺序：改驱力 → 算目标 → 逼近 → 累余味。"""
    apply_appraisal(st, appraisal)
    target = pad_target(st, appraisal.get("intensity"))
    approach_target(st, *target)
    st.lingering = clamp(
        st.lingering + float(appraisal.get("intensity", 0.0)), 0.0, 1.0
    )
    return target
