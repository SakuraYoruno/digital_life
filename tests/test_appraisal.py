from __future__ import annotations

import asyncio
import json

import pytest

from digital_life.appraisal import (
    APPRAISE_FALLBACK,
    llm_appraise,
    parse_appraisal,
)
from digital_life.llm import MockLLM


def test_parse_valid():
    data = parse_appraisal(
        json.dumps(
            {"drive": "approval", "valence": 3, "intensity": 1.7, "reason": "被夸"}
        )
    )
    assert data["drive"] == "approval"
    assert data["valence"] == 1
    assert data["intensity"] == 1.0


def test_parse_negative_and_fenced():
    raw = '```json\n{"drive":"bond","valence":-2,"intensity":-1,"reason":"x"}\n```'
    data = parse_appraisal(raw)
    assert data["valence"] == -1
    assert data["intensity"] == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"drive": "unknown", "valence": 1, "intensity": 0.5, "reason": ""}),
        json.dumps({"drive": "bond", "valence": 1}),
        json.dumps({"drive": "bond", "valence": "x", "intensity": "y", "reason": ""}),
    ],
)
def test_parse_fallback(raw):
    assert parse_appraisal(raw) == APPRAISE_FALLBACK


def test_llm_appraise_with_mock():
    async def run():
        llm = MockLLM(1)
        data = await llm_appraise(llm, "你今天真可爱，我超级喜欢你", "刚认识")
        assert data["valence"] == 1
        assert data["drive"] in {"bond", "approval"}
        assert 0.0 <= data["intensity"] <= 1.0
        neg = await llm_appraise(llm, "你真烦，滚开", "")
        assert neg["valence"] == -1

    asyncio.run(run())
