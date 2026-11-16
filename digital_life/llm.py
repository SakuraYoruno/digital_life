"""LLM 适配层。

生产环境用 OpenAI 兼容接口；测试/仿真用确定性的 MockLLM。
文档里的 small_llm / big_llm 既可以是实现了 complete() 的对象，
也可以是 `async def f(prompt, json_mode=...)` 形式的可调用对象。
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Protocol, runtime_checkable

from .constants import clamp, truncate_to_tokens


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str: ...


async def call_llm(
    llm: Any,
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
) -> str:
    """统一调用入口，兼容对象式与函数式 LLM。"""
    if llm is None:
        raise ValueError("LLM client is None")
    complete = getattr(llm, "complete", None)
    if callable(complete):
        return await complete(
            prompt,
            system=system,
            json_mode=json_mode,
            temperature=temperature,
        )
    if callable(llm):
        try:
            result = llm(
                prompt,
                system=system,
                json_mode=json_mode,
                temperature=temperature,
            )
        except TypeError:
            result = llm(prompt, json_mode=json_mode)
        if hasattr(result, "__await__"):
            return await result
        return result
    raise TypeError(f"Unsupported LLM object: {type(llm)!r}")


class OpenAICompatibleLLM:
    """异步 OpenAI 兼容客户端（DeepSeek / OpenAI / 本地 vLLM 均可）。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        default_temperature: float = 0.8,
        max_tokens: int = 800,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.base_url = (
            base_url
            or os.getenv("DIGITAL_LIFE_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - 取决于环境
                raise RuntimeError(
                    "未安装 openai。请 `pip install openai`，或使用 --mock 运行。"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=api_key or os.getenv("DIGITAL_LIFE_API_KEY") or "not-needed",
                base_url=self.base_url,
                timeout=60.0,
            )

    async def vision(
        self,
        prompt: str,
        image_url: str,
        *,
        max_tokens: int = 120,
        temperature: float = 0.3,
    ) -> str:
        """让模型看一眼图，返回一段描述（OpenAI 兼容的多模态 content 数组）。

        DeepSeek / GPT / 本地 vLLM 的多模态端点都吃这个形状；
        纯文本模型会返回 400，调用方按"看不了图"处理即可。
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = await self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature
            if temperature is None
            else clamp(float(temperature), 0.0, 1.5),
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            if not json_mode:
                raise
            # 有些兼容端点不支持 response_format；退化为提示词约束。
            kwargs.pop("response_format", None)
            messages[-1]["content"] += "\n\n只输出一个 JSON，不要任何解释或代码块。"
            resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if not content:
            # Qwen3 等思考模型在思考模式下可能把正文放在 reasoning；优先
            # 关闭思考（reasoning_effort="none"），这里只做兜底。
            message = resp.choices[0].message
            content = (
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
                or ""
            )
        return (content or "").strip()


# ---------------------------------------------------------------------------
# 离线 MockLLM：用于 demo / 仿真 / 测试，不联网
# ---------------------------------------------------------------------------


_POSITIVE = (
    "喜欢",
    "开心",
    "厉害",
    "可爱",
    "谢谢",
    "陪你",
    "好棒",
    "爱你",
    "想你",
    "早上好",
    "晚安",
    "一起",
    "哈哈",
    "夸",
    "真棒",
    "好看",
    "有意思",
    "不错",
    "乖",
    "抱抱",
    "辛苦",
    "等你",
)
_NEGATIVE = (
    "滚",
    "烦",
    "笨",
    "丑",
    "讨厌",
    "不理",
    "骂",
    "气",
    "难过",
    "哭",
    "无聊",
    "傻",
    "差劲",
    "闭嘴",
    "恶心",
    "敷衍",
    "随便",
    "烦死",
    "别烦",
)
_RELATION = (
    "离开",
    "消失",
    "不要",
    "分手",
    "不理",
    "冷落",
    "忘了",
    "拉黑",
    "在吗",
    "忙",
    "没空",
    "走了",
)
_CURIOSITY = (
    "游戏",
    "番",
    "为什么",
    "怎么",
    "什么",
    "吗",
    "?",
    "？",
    "新",
)


def _after_marker(text: str, marker: str) -> str:
    pos = text.find(marker)
    if pos < 0:
        return text
    return text[pos + len(marker) :].strip()


def _mood_in(prompt: str) -> str:
    for mood in ("开心", "兴奋", "平静", "无聊", "低落", "焦虑", "委屈", "烦躁"):
        if mood in prompt:
            return mood
    return "平静"


class MockLLM:
    """规则驱动的假模型。

    它不追求“像人”，只保证链路能跑通、输出稳定、可复现，
    并覆盖 appraisal / 念头 / 摘要 / 反思 / 主动发言等所有 prompt 类型。
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.prompts: list[str] = []
        self.calls = 0

    # -- 辅助 ---------------------------------------------------------
    def _sentiment(self, text: str) -> tuple[int, float]:
        neg = sum(1 for w in _NEGATIVE if w in text)
        pos = sum(1 for w in _POSITIVE if w in text)
        valence = -1 if neg > pos else 1
        intensity = 0.35 + min(0.25, 0.08 * neg) + min(0.2, 0.06 * pos)
        intensity += min(0.15, len(text) / 160)
        intensity += self.rng.random() * 0.12
        if "!" in text or "！" in text:
            intensity += 0.08
        return valence, clamp(intensity, 0.2, 0.95)

    def _drive(self, text: str) -> str:
        if any(w in text for w in _RELATION):
            return "bond"
        if any(w in text for w in ("夸", "厉害", "谢谢", "喜欢", "可爱", "棒")):
            return "approval"
        if any(w in text for w in _CURIOSITY):
            return "curiosity"
        if any(w in text for w in ("记错", "说错", "不是这样", "矛盾")):
            return "coherence"
        return "bond"

    # -- 主入口 -------------------------------------------------------
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        p = prompt

        if "你在评估一条消息对" in p:
            return self._appraise(p)
        if "用一句内心独白表达" in p:
            return self._thought(p)
        if "你想主动找对方说句话" in p:
            return self._proactive(p)
        if "把以下对话压缩进已有摘要" in p:
            return self._summary(p)
        if "这段对话有没有承诺" in p:
            convo = _after_marker(p, "：")
            return "yes" if any(w in convo for w in ("约", "答应", "明天", "下次", "说好", "记得")) else "no"
        if "把这段对话提炼成一条结构化记忆" in p:
            return self._episodic(p)
        if "从以下情景记忆提炼关于该用户的稳定事实" in p:
            return "[]"
        if "提取文中游戏名" in p:
            text = _after_marker(p, "：")
            entities = [w for w in ("龙卡", "原神", "明日方舟", "星穹铁道", "夏弥") if w in text]
            return json.dumps(entities[:4], ensure_ascii=False)
        if "识别以下群聊消息" in p:
            return "[]"
        if "回顾与该用户最近互动" in p:
            return self._reflection(p)
        if "更新你对自己的自述" in p or "第一人称自述" in p:
            name = "夏弥" if "夏弥" in p else "她"
            return f"我是{name}，嘴硬，记性却很好。来路不想多讲，反正现在在这里。"
        if "压缩成300字" in p:
            body = p.split("：", 1)[-1]
            return truncate_to_tokens(body, 280)
        if "需要查外部资料" in p:
            return "no"
        if "把这几条相似记忆合并" in p:
            body = p.split("：", 1)[-1]
            return truncate_to_tokens(body, 80)
        return self._chat(p)

    # -- 各分支 -------------------------------------------------------
    def _appraise(self, prompt: str) -> str:
        text = _after_marker(prompt, "消息：").split("上下文：")[0].strip()
        valence, intensity = self._sentiment(text)
        drive = self._drive(text)
        reasons = {
            "bond": "关系存续欲被触动",
            "approval": "被认可欲被触动",
            "curiosity": "好奇/表达欲被触动",
            "coherence": "自我一致欲被触动",
        }
        return json.dumps(
            {
                "drive": drive,
                "valence": valence,
                "intensity": round(intensity, 3),
                "reason": reasons[drive] + "（mock）",
            },
            ensure_ascii=False,
        )

    def _thought(self, prompt: str) -> str:
        m = re.search(r"表达'([^']+)'", prompt)
        kind = m.group(1) if m else "放空"
        mood = _mood_in(prompt)
        templates = {
            "想念对方": [
                "她今天怎么还没来…算了，我也不是很在意",
                "刚想跟她说件小事，又觉得没什么好说的",
            ],
            "回味刚才": [
                "刚才那句还挺好笑的，现在想起来还想笑",
                "她那句话是不是有点别的意思",
            ],
            "好奇发散": [
                "那个东西到底是怎么弄的，改天问问",
                "突然想知道她平时都在干嘛",
            ],
            "自我怀疑": [
                "刚是不是说太过了",
                "我是不是又嘴硬了",
            ],
            "期待分享": [
                "这个得跟她说，她肯定要笑",
                "等会儿要跟她讲那个事",
            ],
            "放空": ["……好像也没什么", "发呆中"],
        }
        choices = templates.get(kind, ["……"])
        text = self.rng.choice(choices)
        if mood in ("委屈", "低落"):
            text += "，不过算了"
        return text

    def _proactive(self, prompt: str) -> str:
        mood = _mood_in(prompt)
        nickname = ""
        m = re.search(r"称呼：([^\n]+)", prompt)
        if m:
            candidate = m.group(1).strip().rstrip("。，,、 ")
            if candidate and candidate not in ("对方", "你"):
                nickname = candidate
        lead = f"{nickname}，" if nickname else ""
        pools = {
            "开心": ["刚看到个东西，笑死", "喂，今天有没有什么好玩的事"],
            "兴奋": ["诶诶，我跟你说个事", "你猜我刚才想到什么了"],
            "平静": ["刚发了一会儿呆", "突然想起你上次说的那个"],
            "无聊": ["好无聊啊，你在干嘛～", "随便问一句，你吃饭了没"],
            "低落": ["今天有点没劲", "没什么事，就是有点闷"],
            "焦虑": ["你那边还好吧", "刚才有点心慌，现在好点了"],
            "委屈": ["哦，没什么", "刚才想找你说话来着"],
            "烦躁": ["烦死了，今天什么破事都有", "不想说话，但也不是针对你"],
        }
        return lead + self.rng.choice(pools.get(mood, pools["平静"]))

    def _summary(self, prompt: str) -> str:
        new_part = _after_marker(prompt, "新内容：").split("直接输出")[0]
        neg = sum(1 for w in _NEGATIVE if w in new_part)
        pos = sum(1 for w in _POSITIVE if w in new_part)
        if neg:
            mood = "情绪有点低"
        elif pos:
            mood = "情绪不错"
        else:
            mood = "情绪平稳"
        topic = "最近的事"
        for word in ("游戏", "工作", "学校", "考试", "吃饭", "电影", "番", "猫", "狗"):
            if word in new_part:
                topic = word
                break
        commit = "有约定" if any(w in new_part for w in ("约", "答应", "下次", "说好")) else "没有明确承诺"
        return f"{mood}；聊了{topic}；{commit}。"[:100]

    def _episodic(self, prompt: str) -> str:
        convo = _after_marker(prompt, "对话：").split("\n", 1)[0]
        user_part = convo.split("->")[0].strip()
        valence, _ = self._sentiment(user_part)
        emotion = "开心" if valence > 0 else "不太高兴"
        commitment = None
        if any(w in convo for w in ("约", "答应", "下次", "说好")):
            commitment = "约好了下次再聊"
        what = truncate_to_tokens(user_part, 40) or "一次普通对话"
        return json.dumps(
            {
                "what": what,
                "emotion": emotion,
                "commitment": commitment,
                "tags": ["日常"],
            },
            ensure_ascii=False,
        )

    def _reflection(self, prompt: str) -> str:
        if "委屈" in prompt or "烦躁" in prompt:
            insight = "她不高兴的时候不要追问，先接住话头"
            boundary = "别在她情绪低的时候讲大道理"
        else:
            insight = "她喜欢被认真接话，而不是被敷衍"
            boundary = ""
        return json.dumps(
            {
                "insight": insight,
                "boundary": boundary,
                "style_drift": {
                    "exclaim_rate": round(self.rng.uniform(-0.03, 0.05), 3),
                    "emoji_rate": round(self.rng.uniform(-0.03, 0.04), 3),
                    "avg_len": 0.0,
                },
            },
            ensure_ascii=False,
        )

    def _chat(self, prompt: str) -> str:
        mood = _mood_in(prompt)
        replies = {
            "开心": ["哈，你这人怎么这样", "哦？继续说，我爱听", "行啊你，还挺会"],
            "兴奋": ["诶诶诶，真的假的！", "你再说一遍，我没听够", "等下等下，这也太离谱了"],
            "平静": ["嗯，然后呢", "哦，这样啊", "也还行吧"],
            "无聊": ["那你说点有意思的呗～", "好闲啊，找点事做", "随便聊聊吧～"],
            "低落": ["嗯。", "哦。", "不太想说话"],
            "焦虑": ["没事没事，就是……算了", "你那边真没事吧", "等下，我再确认一下"],
            "委屈": ["哦。", "随你吧", "没什么"],
            "烦躁": ["知道了。", "哦。", "烦"],
        }
        reply = self.rng.choice(replies.get(mood, replies["平静"]))
        # 偶尔加一句嘴硬的尾巴，增加变化但不越安全底线
        if self.rng.random() < 0.25 and mood not in ("低落", "烦躁", "委屈"):
            reply += self.rng.choice(["，啧", "，别多想", "，哈"])
        return reply


def _reasoning_default(model: str) -> str | None:
    # Qwen3 默认开启 thinking，会把正文放进 reasoning、content 为空；
    # 小模型做 JSON/摘要时应该关掉。
    return "none" if "qwen3" in (model or "").lower() else None


def _ollama_has_model(model: str, base_url: str) -> bool:
    """探测本机 Ollama 是否已有模型；失败不影响主流程。"""
    import json
    import urllib.request

    base = (base_url or "http://127.0.0.1:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=0.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = {
            str(item.get("name") or item.get("model") or "").split(":")[0]
            for item in data.get("models", [])
            if isinstance(item, dict)
        }
        return model.split(":")[0] in names
    except Exception:
        return False


def llm_from_env(*, mock: bool = False, seed: int = 0) -> tuple[Any, Any]:
    """按环境变量构造 (small_llm, big_llm)。

    小模型与大模型可以指向不同端点，例如：
      DIGITAL_LIFE_SMALL_MODEL=qwen3:1.7b
      DIGITAL_LIFE_SMALL_BASE_URL=http://127.0.0.1:11434/v1
      DIGITAL_LIFE_SMALL_API_KEY=ollama
      DIGITAL_LIFE_BIG_MODEL=deepseek-chat
      DIGITAL_LIFE_BASE_URL=https://api.deepseek.com/v1
      DIGITAL_LIFE_API_KEY=sk-...

    没有单独配置 big 时，big 复用 small（适合全本地跑）。
    """
    use_mock = mock or os.getenv("DIGITAL_LIFE_MOCK") == "1"
    env = os.environ
    global_key = env.get("DIGITAL_LIFE_API_KEY") or env.get("OPENAI_API_KEY")
    explicit_small = env.get("DIGITAL_LIFE_SMALL_MODEL")
    explicit_big = env.get("DIGITAL_LIFE_BIG_MODEL")
    small_base = env.get("DIGITAL_LIFE_SMALL_BASE_URL") or env.get("DIGITAL_LIFE_BASE_URL")
    big_base = env.get("DIGITAL_LIFE_BASE_URL") or small_base
    small_key = env.get("DIGITAL_LIFE_SMALL_API_KEY") or global_key
    big_key = env.get("DIGITAL_LIFE_BIG_API_KEY") or global_key
    ollama_base = (env.get("DIGITAL_LIFE_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
    local_openai_base = f"{ollama_base}/v1"

    # 完全没配置时，如果本机 Ollama 已有 Qwen3，就自动用全本地模型；
    # 否则才退回 Mock。这样 `ollama pull qwen3:1.7b` 之后开箱即可聊天。
    if not use_mock and not (
        explicit_small or explicit_big or small_base or big_base or global_key
    ):
        if _ollama_has_model("qwen3:1.7b", ollama_base):
            explicit_small = explicit_big = "qwen3:1.7b"
            small_base = big_base = local_openai_base
            small_key = big_key = "ollama"
        else:
            use_mock = True

    if use_mock:
        return MockLLM(seed), MockLLM(seed + 1)

    small_model = explicit_small or "deepseek-chat"
    big_model = explicit_big or small_model
    # 显式写了本地 qwen3 但没给地址时，也补上 Ollama 的 OpenAI 兼容地址。
    if not small_base and "qwen3" in small_model.lower() and _ollama_has_model(
        small_model, ollama_base
    ):
        small_base = local_openai_base
        small_key = small_key or "ollama"
    if not big_base and "qwen3" in big_model.lower() and _ollama_has_model(
        big_model, ollama_base
    ):
        big_base = local_openai_base
        big_key = big_key or "ollama"
    if not big_base:
        big_base = small_base

    if not use_mock and not (small_key or small_base or big_key or big_base):
        use_mock = True
    if use_mock:
        return MockLLM(seed), MockLLM(seed + 1)

    small_effort = env.get("DIGITAL_LIFE_SMALL_REASONING_EFFORT") or _reasoning_default(
        small_model
    )
    big_effort = env.get("DIGITAL_LIFE_BIG_REASONING_EFFORT") or _reasoning_default(
        big_model
    )
    small = OpenAICompatibleLLM(
        small_model,
        base_url=small_base,
        api_key=small_key,
        max_tokens=512,
        reasoning_effort=small_effort,
    )
    big = OpenAICompatibleLLM(
        big_model,
        base_url=big_base,
        api_key=big_key,
        max_tokens=800,
        reasoning_effort=big_effort,
    )
    return small, big


__all__ = [
    "LLMClient",
    "call_llm",
    "OpenAICompatibleLLM",
    "MockLLM",
    "llm_from_env",
]
