"""数字生命：情绪 / 驱力 / 记忆 / 人格的极简单体实现。

快速入口：
    from digital_life import xiami_state, MockLLM, MemoryStore, DigitalLife
"""

from __future__ import annotations

from .appraisal import llm_appraise, parse_appraisal
from .constants import BASE_DT, SEND_THRESHOLD, clamp, scaled, sigmoid
from .context import BUDGET, ContextAssembler, ContextBundle
from .dynamics import apply_appraisal, decay_tick, pad_target, react_to_event
from .embedding import OllamaEmbedder, make_embedder
from .engine import DigitalLife, EngineConfig, EngineStats, MessageResult, TickResult
from .llm import MockLLM, OpenAICompatibleLLM, call_llm, llm_from_env
from .memory import (
    MemoryHit,
    MemoryStore,
    RawPercept,
    WorkingMemory,
    maybe_write_episodic,
    perceive,
    recall_full,
)
from .personas import (
    CorePersona,
    LifeState,
    MoodBehavior,
    PlasticPersona,
    Relation,
    RelationalPersona,
    XIAMI_CORE,
    XIAMI_PLASTIC,
    build_system_prompt,
    default_state,
    inner_monologue,
    map_mood,
    xiami_state,
)
from .proactive import proactive_score, update_affinity
from .safety import safety_gate
from .thoughts import thought_tick

__version__ = "0.1.0"

__all__ = [
    "BASE_DT",
    "SEND_THRESHOLD",
    "BUDGET",
    "clamp",
    "sigmoid",
    "scaled",
    "LifeState",
    "CorePersona",
    "PlasticPersona",
    "RelationalPersona",
    "Relation",
    "MoodBehavior",
    "XIAMI_CORE",
    "XIAMI_PLASTIC",
    "default_state",
    "xiami_state",
    "map_mood",
    "inner_monologue",
    "build_system_prompt",
    "llm_appraise",
    "parse_appraisal",
    "apply_appraisal",
    "pad_target",
    "react_to_event",
    "decay_tick",
    "proactive_score",
    "update_affinity",
    "safety_gate",
    "thought_tick",
    "perceive",
    "RawPercept",
    "WorkingMemory",
    "MemoryStore",
    "MemoryHit",
    "maybe_write_episodic",
    "recall_full",
    "ContextAssembler",
    "ContextBundle",
    "DigitalLife",
    "EngineConfig",
    "EngineStats",
    "MessageResult",
    "TickResult",
    "MockLLM",
    "OpenAICompatibleLLM",
    "call_llm",
    "llm_from_env",
    "OllamaEmbedder",
    "make_embedder",
]
