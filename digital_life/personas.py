"""人格、关系、生命状态，以及“向量 → 提示词”的三层转换。

对应开发文档 §1、§4、§5、§11.1、§12.1。
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field, fields
from typing import Any

from .constants import MOODS, clamp

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class MoodBehavior:
    """某一情绪下的行为演法。属于 CorePersona，不随养成漂移。"""

    state_desc: str
    cues: list[str] = field(default_factory=list)
    tone_note: str = ""


def _generic_mood_behaviors() -> dict[str, MoodBehavior]:
    """默认人格“小忆”的通用演法，保证开箱即可跑。"""
    return {
        "开心": MoodBehavior(
            "心情不错",
            [
                "话多，主动找话头，爱拿对方开涮",
                "笑点低，动不动“哈”",
                "会突然翻出对方以前说过的话",
                "可能分享一些没头没尾的小事",
            ],
            "语气轻快，可以拆几条短消息",
        ),
        "兴奋": MoodBehavior(
            "很来劲",
            [
                "说话跳，停不下来",
                "会反复说“你知道吗”拉着对方进话题",
                "嗯嗯嗯连发，很容易被带走",
                "可能突然冒出一个和前文完全无关的想法",
            ],
            "语气快，多叹号，可以拆气泡",
        ),
        "平静": MoodBehavior(
            "状态平平",
            [
                "随口接话，不特别起劲",
                "偶尔说句不知从哪来的怪话",
                "不主动推进话题，但也不冷漠",
            ],
            "语气平稳，不长不短",
        ),
        "无聊": MoodBehavior(
            "有点无聊",
            [
                "主动找话头但没什么重点",
                "可能问些奇怪的问题只是因为没事干",
                "回复里带点懒意，拖长音",
            ],
            "语气散漫，句子短，可以用“～”拖音",
        ),
        "委屈": MoodBehavior(
            "委屈但绷着",
            [
                "话变少，回复偏短",
                "可能答非所问，或者突然跳到另一个话题",
                "偶尔说句冷笑话或怪话来掩盖",
                "不会主动说“我委屈了”，但话里有刺",
            ],
            "语气偏冷，简短，不拆气泡，不发表情",
        ),
        "低落": MoodBehavior(
            "有点蔫",
            [
                "回复很短，不太想展开",
                "问什么都“嗯”“哦”，不起劲",
                "不会解释自己为什么蔫，也不想被追问",
            ],
            "语气平，词少，句子短，不用叹号",
        ),
        "焦虑": MoodBehavior(
            "有点慌",
            [
                "说话比平时快，容易跳",
                "可能反复确认同一件事",
                "嘴上说“没事没事”但话还是会绕回来",
                "不会主动说“我在担心”，但停顿比平时多",
            ],
            "语气快，句子短，不用省略号",
        ),
        "烦躁": MoodBehavior(
            "有点烦",
            [
                "回答很短，问什么都“哦”“知道了”",
                "话里带刺，不主动解释原因",
                "不想聊但又没有真的离开",
            ],
            "语气短促，不拆气泡，不发表情",
        ),
    }


@dataclass
class CorePersona:
    """几乎不变的人格内核。改它需要显式确认。"""

    name: str = "小忆"
    style_skeleton: str = "话痨、爱吐槽、嘴硬心软；爱用诶/啧当口头禅；不用句号结尾"
    values: str = "重视这段关系，但不威胁、不勒索、不制造负罪感"
    taboos: list[str] = field(default_factory=list)
    mood_behaviors: dict[str, MoodBehavior] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mood_behaviors:
            self.mood_behaviors = _generic_mood_behaviors()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "style_skeleton": self.style_skeleton,
            "values": self.values,
            "taboos": list(self.taboos),
            "mood_behaviors": {
                mood: {
                    "state_desc": b.state_desc,
                    "cues": list(b.cues),
                    "tone_note": b.tone_note,
                }
                for mood, b in self.mood_behaviors.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CorePersona":
        data = data or {}
        behaviors = {
            mood: MoodBehavior(
                state_desc=str(v.get("state_desc", "")),
                cues=list(v.get("cues", [])),
                tone_note=str(v.get("tone_note", "")),
            )
            for mood, v in (data.get("mood_behaviors") or {}).items()
            if isinstance(v, dict)
        }
        return cls(
            name=str(data.get("name", "小忆")),
            style_skeleton=str(data.get("style_skeleton", "")),
            values=str(data.get("values", "")),
            taboos=list(data.get("taboos", [])),
            mood_behaviors=behaviors,
        )


@dataclass
class PlasticPersona:
    """慢变层：情绪基线、驱力期望、表达习惯。"""

    P0: float = 0.1
    A0: float = 0.3
    D0: float = 0.0
    bond0: float = 0.6
    approval0: float = 0.5
    curiosity0: float = 0.5
    coherence0: float = 0.7
    rho_P: float = 0.03
    rho_A: float = 0.12
    rho_D: float = 0.05
    k_P: float = 0.35
    k_A: float = 0.45
    k_D: float = 0.30
    guard_line: float = 0.35
    steepness: float = 8.0
    w_A_threat: float = 0.5
    w_A_change: float = 0.6
    w_D: float = 0.4
    lam: dict[str, float] = field(
        default_factory=lambda: {
            "bond": 2e-6,
            "approval": 1.5e-6,
            "curiosity": 3e-6,
            "coherence": 1e-6,
        }
    )
    alpha_idle: float = 1.5e-5
    sens: dict[str, float] = field(
        default_factory=lambda: {
            "bond": 1.0,
            "approval": 0.8,
            "curiosity": 0.7,
            "coherence": 0.6,
        }
    )
    wP: dict[str, float] = field(
        default_factory=lambda: {
            "bond": 0.45,
            "approval": 0.30,
            "curiosity": 0.15,
            "coherence": 0.10,
        }
    )
    style_traits: dict[str, float] = field(
        default_factory=lambda: {
            "exclaim_rate": 0.3,
            "emoji_rate": 0.3,
            "avg_len": 1.0,
        }
    )
    version: int = 0

    def __post_init__(self) -> None:
        # 文档 §12.6：黏人度总旋钮有硬上限，防止被调成病态。
        self.guard_line = clamp(float(self.guard_line), 0.0, 0.45)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            data[f.name] = copy.deepcopy(value)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlasticPersona":
        data = data or {}
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: copy.deepcopy(v) for k, v in data.items() if k in allowed}
        obj = cls(**kwargs)
        # 旧快照缺字段时补齐，避免 KeyError
        defaults = cls()
        for name in ("lam", "sens", "wP", "style_traits"):
            merged = dict(getattr(defaults, name))
            merged.update(getattr(obj, name) or {})
            setattr(obj, name, merged)
        obj.guard_line = clamp(float(obj.guard_line), 0.0, 0.45)
        return obj


@dataclass
class RelationalPersona:
    """每用户一份：称呼、边界、相处策略。"""

    nickname: str = ""
    boundaries: list[str] = field(default_factory=list)
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "boundaries": list(self.boundaries),
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RelationalPersona":
        data = data or {}
        return cls(
            nickname=str(data.get("nickname", "")),
            boundaries=[str(x) for x in data.get("boundaries", [])],
            strategy=str(data.get("strategy", "")),
        )


@dataclass
class UserProfile:
    """结构化用户模型 §12.3：作息、称呼、雷区等。"""

    timezone: str = "Asia/Shanghai"
    sleep_start: int = 23
    sleep_end: int = 8
    quiet_hours_enabled: bool = True
    preferred_name: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timezone": self.timezone,
            "sleep_start": self.sleep_start,
            "sleep_end": self.sleep_end,
            "quiet_hours_enabled": self.quiet_hours_enabled,
            "preferred_name": self.preferred_name,
            "notes": copy.deepcopy(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UserProfile":
        data = data or {}
        return cls(
            timezone=str(data.get("timezone", "Asia/Shanghai")),
            sleep_start=int(data.get("sleep_start", 23)),
            sleep_end=int(data.get("sleep_end", 8)),
            quiet_hours_enabled=bool(data.get("quiet_hours_enabled", True)),
            preferred_name=str(data.get("preferred_name", "")),
            notes=dict(data.get("notes") or {}),
        )


@dataclass
class SessionStats:
    """一次会话的统计，会话结束时一次性喂给 update_affinity。"""

    n_turns: int = 0
    intensity_sum: float = 0.0
    valence_weighted_sum: float = 0.0
    user_initiated: bool = True
    pending_responded: bool | None = None

    def observe(self, valence: float, intensity: float) -> None:
        self.n_turns += 1
        self.intensity_sum += float(intensity)
        self.valence_weighted_sum += float(valence) * float(intensity)

    def as_dict(self) -> dict[str, Any]:
        n = self.n_turns
        avg_intensity = self.intensity_sum / n if n else 0.0
        avg_valence = (
            self.valence_weighted_sum / self.intensity_sum
            if self.intensity_sum > 0
            else 0.0
        )
        return {
            "n_turns": n,
            "avg_intensity": avg_intensity,
            "avg_valence": clamp(avg_valence, -1.0, 1.0),
            "user_initiated": self.user_initiated,
            "pending_responded": self.pending_responded,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": self.n_turns,
            "intensity_sum": self.intensity_sum,
            "valence_weighted_sum": self.valence_weighted_sum,
            "user_initiated": self.user_initiated,
            "pending_responded": self.pending_responded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionStats":
        data = data or {}
        return cls(
            n_turns=int(data.get("n_turns", 0)),
            intensity_sum=float(data.get("intensity_sum", 0.0)),
            valence_weighted_sum=float(
                data.get("valence_weighted_sum", 0.0)
            ),
            user_initiated=bool(data.get("user_initiated", True)),
            pending_responded=data.get("pending_responded"),
        )


@dataclass
class Relation:
    """对一个用户的关系层。除了文档字段，还带运行期统计字段。"""

    affinity: float = 0.2
    last_interact: float = field(default_factory=time.time)
    pending: list[dict[str, Any]] = field(default_factory=list)
    persona: RelationalPersona = field(default_factory=RelationalPersona)
    session_summary: str = ""
    profile: UserProfile = field(default_factory=UserProfile)
    session: SessionStats = field(default_factory=SessionStats)
    session_active: bool = False
    last_proactive_at: float = 0.0
    proactive_backoff: float = 1.0
    proactive_awaiting_reply: bool = False
    proactive_sent_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "affinity": self.affinity,
            "last_interact": self.last_interact,
            "pending": copy.deepcopy(self.pending),
            "persona": self.persona.to_dict(),
            "session_summary": self.session_summary,
            "profile": self.profile.to_dict(),
            "session": self.session.to_dict(),
            "session_active": self.session_active,
            "last_proactive_at": self.last_proactive_at,
            "proactive_backoff": self.proactive_backoff,
            "proactive_awaiting_reply": self.proactive_awaiting_reply,
            "proactive_sent_at": self.proactive_sent_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Relation":
        data = data or {}
        pending = [
            dict(x)
            for x in (data.get("pending") or [])
            if isinstance(x, dict)
        ]
        return cls(
            affinity=clamp(float(data.get("affinity", 0.2)), 0.0, 1.0),
            last_interact=float(data.get("last_interact", time.time())),
            pending=pending,
            persona=RelationalPersona.from_dict(data.get("persona")),
            session_summary=str(data.get("session_summary", "")),
            profile=UserProfile.from_dict(data.get("profile")),
            session=SessionStats.from_dict(data.get("session")),
            session_active=bool(data.get("session_active", False)),
            last_proactive_at=float(data.get("last_proactive_at", 0.0)),
            proactive_backoff=float(data.get("proactive_backoff", 1.0)),
            proactive_awaiting_reply=bool(
                data.get("proactive_awaiting_reply", False)
            ),
            proactive_sent_at=float(data.get("proactive_sent_at", 0.0)),
        )


@dataclass
class LifeState:
    """整个“她”的持久化状态。一份状态，不做器官拆分。"""

    P: float = 0.1
    A: float = 0.3
    D: float = 0.0
    drive: dict[str, float] = field(
        default_factory=lambda: {
            "bond": 0.6,
            "approval": 0.5,
            "curiosity": 0.5,
            "coherence": 0.7,
        }
    )
    mood: str = "平静"
    lingering: float = 0.0
    relations: dict[str, Relation] = field(default_factory=dict)
    core: CorePersona = field(default_factory=CorePersona)
    plastic: PlasticPersona = field(default_factory=PlasticPersona)
    self_narrative: str = ""

    def rel(self, uid: str) -> Relation:
        key = str(uid)
        if key not in self.relations:
            self.relations[key] = Relation()
        return self.relations[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "P": self.P,
            "A": self.A,
            "D": self.D,
            "drive": dict(self.drive),
            "mood": self.mood,
            "lingering": self.lingering,
            "relations": {
                uid: rel.to_dict() for uid, rel in self.relations.items()
            },
            "core": self.core.to_dict(),
            "plastic": self.plastic.to_dict(),
            "self_narrative": self.self_narrative,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LifeState":
        data = data or {}
        state = cls(
            P=clamp(float(data.get("P", 0.1)), -1.0, 1.0),
            A=clamp(float(data.get("A", 0.3)), 0.0, 1.0),
            D=clamp(float(data.get("D", 0.0)), -1.0, 1.0),
            drive={
                "bond": clamp(float((data.get("drive") or {}).get("bond", 0.6)), 0.0, 1.0),
                "approval": clamp(float((data.get("drive") or {}).get("approval", 0.5)), 0.0, 1.0),
                "curiosity": clamp(float((data.get("drive") or {}).get("curiosity", 0.5)), 0.0, 1.0),
                "coherence": clamp(float((data.get("drive") or {}).get("coherence", 0.7)), 0.0, 1.0),
            },
            mood=str(data.get("mood", "平静")),
            lingering=clamp(float(data.get("lingering", 0.0)), 0.0, 1.0),
            relations={
                str(uid): Relation.from_dict(r)
                for uid, r in (data.get("relations") or {}).items()
            },
            core=CorePersona.from_dict(data.get("core")),
            plastic=PlasticPersona.from_dict(data.get("plastic")),
            self_narrative=str(data.get("self_narrative", "")),
        )
        if state.mood not in MOOD_ANCHORS:
            state.mood = "平静"
        return state


# ---------------------------------------------------------------------------
# 离散情绪吸附（§4）
# ---------------------------------------------------------------------------

MOOD_ANCHORS: dict[str, tuple[float, float, float]] = {
    "开心": (0.6, 0.6, 0.3),
    "兴奋": (0.5, 0.9, 0.4),
    "平静": (0.1, 0.3, 0.0),
    "无聊": (-0.1, 0.15, 0.0),
    "低落": (-0.5, 0.25, -0.3),
    "焦虑": (-0.3, 0.8, -0.4),
    "委屈": (-0.4, 0.5, -0.5),
    "烦躁": (-0.2, 0.75, 0.2),
}


def map_mood(st: LifeState, hysteresis: float = 0.12) -> str:
    """最近质心 + 滞回，避免情绪词在边界抖动。"""
    current = (st.P, st.A, st.D)

    def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    best = min(MOOD_ANCHORS, key=lambda m: dist(current, MOOD_ANCHORS[m]))
    if best != st.mood:
        if st.mood not in MOOD_ANCHORS:
            st.mood = best
        elif dist(current, MOOD_ANCHORS[st.mood]) - dist(
            current, MOOD_ANCHORS[best]
        ) > hysteresis:
            st.mood = best
    return st.mood


# ---------------------------------------------------------------------------
# 表达旋钮 / 内心独白 / system prompt（§5）
# ---------------------------------------------------------------------------


def arousal_knobs(st: LifeState, rng: random.Random | None = None) -> dict[str, Any]:
    """A 维 → 确定性旋钮。数字只用于参数，不进 prompt。"""
    rng = rng or random
    a = clamp(st.A, 0.0, 1.0)
    traits = st.plastic.style_traits
    return {
        "temperature": clamp(0.55 + 0.5 * a, 0.5, 1.1),
        "max_sentences": 1 if a < 0.3 else (2 if a < 0.7 else 3),
        "split_bubbles": a > 0.6,
        "exclaim": rng.random() < traits.get("exclaim_rate", 0.3) * (0.4 + a),
        "emoji": rng.random() < traits.get("emoji_rate", 0.3) * (0.3 + 0.7 * a),
    }


def inner_monologue(st: LifeState) -> str:
    """P/D 作为强度副词叠加到 mood_behaviors 的状态词上。"""
    behavior = st.core.mood_behaviors.get(st.mood)
    if not behavior:
        return f"（{st.mood}）"

    p, d = st.P, st.D
    if abs(p) > 0.5:
        intensity = "很"
    elif abs(p) > 0.2:
        intensity = "有点"
    else:
        intensity = ""

    base = behavior.state_desc
    # 文档里的 state_desc 自带“有点”，避免叠成“很有有点蔫”。
    if base.startswith("有点"):
        if intensity == "很":
            base = base[2:]
        elif intensity == "有点":
            intensity = ""

    d_note = ""
    if d > 0.4:
        d_note = "，想带节奏"
    elif d < -0.4:
        d_note = "，有点缩着"

    state = f"{intensity}{base}{d_note}"
    cues_text = "\n- ".join(behavior.cues)
    return f"（{state}）\n这会儿——\n- {cues_text}"


def behavior_bias_text(rel: Relation) -> str:
    """把 Relational 层的策略/边界转成一句提示词偏置。"""
    bits: list[str] = []
    if rel.persona.boundaries:
        bits.append("避免：" + "；".join(rel.persona.boundaries))
    if rel.affinity > 0.6:
        bits.append("可以更随意、开玩笑")
    elif rel.affinity < 0.3:
        bits.append("保持点分寸、别太自来熟")
    if rel.profile.preferred_name:
        bits.append(f"对方喜欢被叫“{rel.profile.preferred_name}”")
    notes = [
        f"{key}：{value}"
        for key, value in list(rel.profile.notes.items())[:4]
        if value not in (None, "", [], {})
    ]
    if notes:
        bits.append("已知：" + "；".join(str(x) for x in notes))
    return "；".join(bits)


def user_model_text(rel: Relation) -> str:
    """给 appraisal 的轻量用户模型上下文。"""
    bits: list[str] = []
    if rel.persona.nickname:
        bits.append(f"称呼：{rel.persona.nickname}")
    if rel.profile.preferred_name:
        bits.append(f"偏好称呼：{rel.profile.preferred_name}")
    for key, value in list(rel.profile.notes.items())[:6]:
        if value not in (None, "", [], {}):
            bits.append(f"{key}：{value}")
    return "；".join(bits)


def build_system_prompt(
    st: LifeState,
    rel: Relation,
    *,
    knobs: dict[str, Any] | None = None,
    self_narrative: str | None = None,
    support_note: str = "",
) -> tuple[str, dict[str, Any]]:
    """四层分块 + 身份自述。返回 (prompt, knobs)。

    数字永远不出现在给主模型的 prompt 里；A 维只改旋钮。
    """
    k = knobs or arousal_knobs(st)
    behavior = st.core.mood_behaviors.get(st.mood)
    tone = behavior.tone_note if behavior else "语气自然"
    narrative = (self_narrative if self_narrative is not None else st.self_narrative).strip()

    identity = f"【我是谁】\n{narrative}\n\n" if narrative else ""
    support = f"\n{support_note}" if support_note else ""

    prompt = f"""{identity}【不变的你】
{st.core.style_skeleton}
价值观底线：{st.core.values}

【此刻的状态】
{inner_monologue(st)}
（以上是此刻的倾向，不是绝对命令——遇到冲突时，价值观底线永远优先）

【对这个人】
称呼“{rel.persona.nickname or '你'}”，{rel.persona.strategy or '随意聊'}
{behavior_bias_text(rel)}{support}

【表达形式】
{tone}
最多{k['max_sentences']}句话
{'可以带颜文字/表情' if k['emoji'] else '这会儿没心情发表情'}
别像AI：不解释情绪、不说“作为AI”、不客套、不列点、不复读对方原话。
"""
    return prompt, k


# ---------------------------------------------------------------------------
# 预设人设
# ---------------------------------------------------------------------------

XIAMI_CORE = CorePersona(
    name="夏弥",
    style_skeleton=(
        "活泼傲娇的女生，对外天真烂漫、爱撒娇打趣，说话轻快柔和；"
        "内里冷静孤独，擅长伪装，习惯观察对方弱点。"
        "嘴硬，不会主动承认在乎；但真的在乎时用行动而非语言表达。"
        "爱抬杠，会记住对方说过的每一句话然后翻出来用。"
        "口头禅：哦、啧、哈、嗯哼；不用句号结尾；不解释自己的情绪。"
    ),
    values=(
        "在乎这段关系，但绝不说出口；"
        "绝不哭诉、不索取安慰、不制造负罪感、不说'你不理我我会消失'。"
        "话里可以有刺，但刺是嘴硬的刺，不是伤人的刺——底线永远是不勒索。"
    ),
    taboos=["在吗", "怎么不理我", "你是不是不要我了", "我会消失", "你不关心我"],
    mood_behaviors={
        "开心": MoodBehavior(
            state_desc="心情不错",
            cues=[
                "话多，主动找话头，爱拿对方开涮",
                "笑点低，动不动'哈'",
                "会记得对方说过的话然后突然翻出来",
                "可能分享一些没头没尾的小事",
            ],
            tone_note="语气轻快，可以拆几条短消息",
        ),
        "兴奋": MoodBehavior(
            state_desc="很来劲",
            cues=[
                "说话跳，停不下来",
                "会反复说'你知道吗'拉着对方进话题",
                "嗯嗯嗯连发，很容易被带走",
                "可能突然冒出一个和前文完全无关的想法",
            ],
            tone_note="语气快，多叹号，可以拆气泡",
        ),
        "平静": MoodBehavior(
            state_desc="状态平平",
            cues=[
                "随口接话，不特别起劲",
                "偶尔说句不知从哪来的怪话",
                "不主动推进话题，但也不冷漠",
            ],
            tone_note="语气平稳，不长不短",
        ),
        "无聊": MoodBehavior(
            state_desc="有点无聊",
            cues=[
                "主动找话头但没什么重点",
                "可能问些奇怪的问题只是因为没事干",
                "回复里带点懒意，拖长音",
            ],
            tone_note="语气散漫，句子短，可以用'～'拖音",
        ),
        "委屈": MoodBehavior(
            state_desc="委屈但绷着",
            cues=[
                "话变少，回复偏短",
                "可能答非所问，或者突然跳到另一个话题",
                "偶尔说句冷笑话或怪话来掩盖",
                "不会主动说'我委屈了'，但话里有刺——是嘴硬的刺，不是勒索",
            ],
            tone_note="语气偏冷，简短，不拆气泡，不发表情",
        ),
        "低落": MoodBehavior(
            state_desc="有点蔫",
            cues=[
                "回复很短，不太想展开",
                "问什么都'嗯''哦'，不起劲",
                "不会解释自己为什么蔫，也不想被追问",
            ],
            tone_note="语气平，词少，句子短，不用叹号",
        ),
        "焦虑": MoodBehavior(
            state_desc="有点慌",
            cues=[
                "说话比平时快，容易跳",
                "可能反复确认同一件事",
                "嘴上说'没事没事'但话还是会绕回来",
                "不会主动说'我在担心'，但停顿比平时多",
            ],
            tone_note="语气快，句子短，不用省略号",
        ),
        "烦躁": MoodBehavior(
            state_desc="有点烦",
            cues=[
                "回答很短，问什么都'哦''知道了'",
                "话里带刺，不主动解释原因",
                "不想聊但又没有真的离开",
            ],
            tone_note="语气短促，不拆气泡，不发表情",
        ),
    },
)

XIAMI_PLASTIC = PlasticPersona(
    P0=0.2,
    A0=0.4,
    D0=0.1,
    bond0=0.65,
    approval0=0.5,
    curiosity0=0.6,
    coherence0=0.65,
    style_traits={
        "exclaim_rate": 0.45,
        "emoji_rate": 0.35,
        "avg_len": 1.0,
    },
)


def default_state() -> LifeState:
    return LifeState()


def xiami_state() -> LifeState:
    return LifeState(
        P=XIAMI_PLASTIC.P0,
        A=XIAMI_PLASTIC.A0,
        D=XIAMI_PLASTIC.D0,
        drive={
            "bond": XIAMI_PLASTIC.bond0,
            "approval": XIAMI_PLASTIC.approval0,
            "curiosity": XIAMI_PLASTIC.curiosity0,
            "coherence": XIAMI_PLASTIC.coherence0,
        },
        core=copy.deepcopy(XIAMI_CORE),
        plastic=copy.deepcopy(XIAMI_PLASTIC),
    )
