from __future__ import annotations

import random

import pytest

from digital_life.personas import (
    CorePersona,
    LifeState,
    Relation,
    XIAMI_CORE,
    arousal_knobs,
    build_system_prompt,
    inner_monologue,
    map_mood,
    user_model_text,
    xiami_state,
)


def test_mood_anchor_exact():
    st = xiami_state()
    st.P, st.A, st.D = 0.6, 0.6, 0.3
    st.mood = "平静"
    assert map_mood(st) == "开心"


def test_mood_hysteresis_prevents_jitter():
    st = xiami_state()
    st.mood = "平静"
    # 已经更靠近“无聊”，但差距不到 0.12，不切换
    st.P, st.A, st.D = -0.02, 0.225, 0.0
    assert map_mood(st) == "平静"
    # 差距足够大才切换
    st.P, st.A, st.D = -0.3, 0.15, 0.0
    assert map_mood(st) == "无聊"


def test_inner_monologue_intensity_and_dominance():
    st = xiami_state()
    st.mood = "兴奋"          # 用能接副词的 state_desc（"来劲"）
    st.P, st.D = 0.8, 0.5
    text = inner_monologue(st)
    assert "很来劲" in text
    assert "想带节奏" in text

    st.P, st.D = 0.3, -0.5
    text = inner_monologue(st)
    assert "有点来劲" in text
    assert "有点缩着" in text


def test_inner_monologue_skips_adverb_for_phrases():
    """「心情不错」「状态平平」「委屈但绷着」是短语，前面加副词读不通。"""
    st = xiami_state()
    for mood, p in (("开心", 0.9), ("平静", 0.9), ("委屈", -0.9)):
        st.mood = mood
        st.P, st.D = p, 0.0
        text = inner_monologue(st).split("\n")[0]
        assert "很心情" not in text
        assert "很状态" not in text
        assert "很委屈但" not in text
        assert "有点心情" not in text


def test_inner_monologue_does_not_double_prefix():
    st = xiami_state()
    st.mood = "低落"
    st.P, st.D = -0.8, 0.0
    text = inner_monologue(st)
    assert "很蔫" in text
    assert "有点蔫" not in text
    assert "很有" not in text

    st.P = -0.3
    text = inner_monologue(st)
    assert "有点蔫" in text
    assert "有点有点" not in text


def test_arousal_knobs_ranges():
    st = xiami_state()
    for _ in range(30):
        k = arousal_knobs(st, random.Random(1))
        assert 0.5 <= k["temperature"] <= 1.1
        assert k["max_sentences"] in (1, 2, 3)
        assert isinstance(k["split_bubbles"], bool)
        assert isinstance(k["sticker"], bool)


def test_sticker_knob_scales_with_arousal():
    """表情包频率必须跟着 A 走：兴奋时爱甩图，蔫下去就懒得发。"""
    st = xiami_state()
    rng = random.Random(7)

    st.A = 0.9
    hot = sum(1 for _ in range(400) if arousal_knobs(st, rng)["sticker"])
    st.A = 0.05
    cold = sum(1 for _ in range(400) if arousal_knobs(st, rng)["sticker"])

    assert hot > cold * 1.5, f"A=0.9 出图 {hot} 次，A=0.05 出图 {cold} 次"


def test_build_system_prompt_has_four_layers_and_no_pad_numbers():
    st = xiami_state()
    rel = Relation()
    rel.persona.nickname = "姐姐"
    rel.persona.strategy = "别追问，先接话"
    knobs = {
        "temperature": 0.8,
        "max_sentences": 2,
        "split_bubbles": True,
        "exclaim": True,
        "emoji": False,
    }
    prompt, returned = build_system_prompt(st, rel, knobs=knobs)
    assert returned is knobs
    for block in ("【不变的你】", "【此刻的状态】", "【对这个人】", "【表达形式】"):
        assert block in prompt
    assert "姐姐" in prompt
    assert "别追问，先接话" in prompt
    assert "没心情发表情" in prompt
    assert "P=" not in prompt and "A=" not in prompt and "D=" not in prompt
    assert "不解释情绪" in prompt


def test_xiami_core_is_locked_identity():
    assert XIAMI_CORE.name == "夏弥"
    assert "不勒索" in XIAMI_CORE.values
    assert "在吗" in XIAMI_CORE.taboos
    # 龙族设定还在：来历、关系、口头禅规则都得在，否则又会演成空壳
    assert "耶梦加得" in XIAMI_CORE.background
    assert "风王之瞳" in XIAMI_CORE.background
    assert "楚子航" in XIAMI_CORE.relationships
    assert any("你猜" in rule for rule in XIAMI_CORE.voice_rules)
    assert len(XIAMI_CORE.speech_examples) >= 3
    assert set(XIAMI_CORE.mood_behaviors) == {
        "开心",
        "兴奋",
        "平静",
        "无聊",
        "低落",
        "焦虑",
        "委屈",
        "烦躁",
    }


def test_default_core_has_generic_behaviors():
    core = CorePersona()
    assert core.mood_behaviors
    st = LifeState(core=core)
    assert "这会儿——" in inner_monologue(st)


def test_user_model_text_is_lightweight():
    rel = Relation()
    rel.persona.nickname = "阿暖"
    rel.profile.preferred_name = "暖暖"
    rel.profile.notes = {"雷区": "家人", "在意的事": "龙卡"}
    text = user_model_text(rel)
    assert "阿暖" in text
    assert "暖暖" in text
    assert "家人" in text
    assert "龙卡" in text


def test_persona_blocks_reach_the_prompt():
    st = xiami_state()
    rel = Relation()
    prompt, _k = build_system_prompt(st, rel)
    assert "耶梦加得" in prompt          # background
    assert "楚子航" in prompt            # relationships
    assert "你猜" in prompt              # voice_rules
    assert "【和谁】" in prompt


def test_saved_state_takes_core_from_code(tmp_path):
    """旧快照里的人格不该盖掉代码里改过的人设。"""
    from digital_life.persistence import StateStore

    store = StateStore(tmp_path / "state.db")
    old = xiami_state()
    old.core.background = "旧人设：什么都不知道"
    old.core.speech_examples = []
    old.P = 0.42
    store.save(old)
    store.close()

    reopened = StateStore(tmp_path / "state.db")
    loaded = reopened.load(persona=xiami_state())
    assert loaded is not None
    assert "耶梦加得" in loaded.core.background   # 用代码里的新人格
    assert abs(loaded.P - 0.42) < 1e-9            # 情绪状态照样持久化
    reopened.close()


def test_load_without_persona_keeps_snapshot_core(tmp_path):
    from digital_life.persistence import StateStore

    store = StateStore(tmp_path / "state.db")
    old = xiami_state()
    old.core.background = "旧人设"
    store.save(old)
    store.close()

    reopened = StateStore(tmp_path / "state.db")
    loaded = reopened.load()
    assert loaded is not None and loaded.core.background == "旧人设"
    reopened.close()


def test_saved_state_takes_dynamics_constants_from_code(tmp_path):
    """改动力学常数（λ/k/w）要立刻生效，别被旧快照锁住；养成值仍要保留。"""
    from digital_life.persistence import StateStore

    store = StateStore(tmp_path / "state.db")
    old = xiami_state()
    old.plastic.lam["bond"] = 1e-9   # 旧常数
    old.plastic.P0 = 0.35            # 养成出来的值
    store.save(old)
    store.close()

    reopened = StateStore(tmp_path / "state.db")
    loaded = reopened.load(persona=xiami_state())
    assert loaded is not None
    assert loaded.plastic.lam["bond"] == pytest.approx(0.006)   # 用代码里的新常数
    assert loaded.plastic.P0 == pytest.approx(0.35)             # 养成的值保留
    reopened.close()


def test_speaker_reaches_the_prompt():
    """群聊里必须知道"谁在说话"，否则她只能对着一个匿名的"你"。"""
    st = xiami_state()
    rel = Relation()
    prompt, _ = build_system_prompt(st, rel, speaker="小虾米")
    assert "小虾米" in prompt
    assert "这条消息来自「小虾米」" in prompt

    blank, _ = build_system_prompt(st, rel)
    assert "这条消息来自" not in blank


def test_refusal_block_follows_dominance():
    """她得能说"不"：拒绝通道在，且底气跟着 D 维走。"""
    st = xiami_state()
    rel = Relation()

    st.D = 0.5
    high, _ = build_system_prompt(st, rel)
    assert "【接不接】" in high
    assert "占着上风" in high
    assert "别再反问" in high          # 拒绝时不许用反问把活儿接回去

    st.D = -0.5
    low, _ = build_system_prompt(st, rel)
    assert "【接不接】" in low
    assert "缩着" in low
