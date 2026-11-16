from __future__ import annotations

import asyncio
from types import SimpleNamespace

from digital_life.llm import (
    MockLLM,
    OpenAICompatibleLLM,
    call_llm,
    llm_from_env,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  hello  "))]
        )


def test_openai_adapter_builds_messages_and_json_mode():
    async def run():
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        llm = OpenAICompatibleLLM("test-model", client=client)
        text = await llm.complete(
            "hi", system="sys", json_mode=True, temperature=0.7
        )
        assert text == "hello"
        assert completions.kwargs["model"] == "test-model"
        assert completions.kwargs["messages"][0] == {
            "role": "system",
            "content": "sys",
        }
        assert completions.kwargs["messages"][1]["content"] == "hi"
        assert completions.kwargs["response_format"] == {"type": "json_object"}
        assert completions.kwargs["temperature"] == 0.7

    asyncio.run(run())


def test_call_llm_accepts_plain_callable():
    async def fake(prompt, json_mode=False):
        return f"{prompt}:{json_mode}"

    assert asyncio.run(call_llm(fake, "x", json_mode=True)) == "x:True"


def test_mock_is_deterministic_for_same_seed():
    async def run():
        a = MockLLM(7)
        b = MockLLM(7)
        ra = await a.complete("用一句内心独白表达'放空'，口吻是随便，当前平静")
        rb = await b.complete("用一句内心独白表达'放空'，口吻是随便，当前平静")
        assert ra == rb

    asyncio.run(run())


def test_llm_from_env_qwen3_small_and_big_fallback(monkeypatch):
    for key in (
        "DIGITAL_LIFE_MOCK",
        "DIGITAL_LIFE_API_KEY",
        "OPENAI_API_KEY",
        "DIGITAL_LIFE_BASE_URL",
        "DIGITAL_LIFE_BIG_MODEL",
        "DIGITAL_LIFE_SMALL_MODEL",
        "DIGITAL_LIFE_SMALL_BASE_URL",
        "DIGITAL_LIFE_SMALL_API_KEY",
        "DIGITAL_LIFE_BIG_API_KEY",
        "DIGITAL_LIFE_SMALL_REASONING_EFFORT",
        "DIGITAL_LIFE_BIG_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DIGITAL_LIFE_SMALL_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("DIGITAL_LIFE_SMALL_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("DIGITAL_LIFE_SMALL_API_KEY", "ollama")

    small, big = llm_from_env()
    assert small.model == "qwen3:1.7b"
    assert big.model == "qwen3:1.7b"
    assert small.reasoning_effort == "none"
    assert big.reasoning_effort == "none"
    assert small.base_url == "http://127.0.0.1:11434/v1"
