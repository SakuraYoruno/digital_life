# 数字生命（digital_life）

《数字生命 开发文档》的可运行实现。一个极简单体：**一套情绪 / 驱力 / 记忆 / 人格状态就是“她”本身**，不拆器官、不仲裁、不做审计链路。两个循环：

- **快循环** `on_message`：有人说话 → 感知 → appraisal → 驱力/PAD → mood → 四层提示词 → 生成 → 安全闸 → 发送。
- **慢循环** `life_tick`：没人时也活着 → 驱力流失 / 情绪回归 → 念头流 → 主动发言 → 会话结算。

本项目是**独立的 Python 包与 CLI**，不依赖任何特定宿主或 IM 平台；QQ/Obsidian 只是可选接入方向，
拿到任何 Python 3.10+ 环境里都能直接跑。

零第三方运行时依赖：Python 3.10+ 标准库即可跑通；接真实模型时可选装 `openai`，启用 `sqlite-vec` 索引时可选装 `sqlite-vec`，接 Ollama embedding 时可选装 `httpx`。

---

## 1. 快速开始

```bash
cd "D:\digital_life"

# 0) 本地模型：embedding 默认 bge-m3（显存约 689MB）；对话小模型 qwen3:1.7b
ollama pull bge-m3
ollama pull qwen3:1.7b

# 1) 一分钟演示：情绪、独处念头、主动发言
python -m digital_life demo

# 2) 控制台聊天（本机有 qwen3:1.7b 时自动用；--mock 强制离线）
python -m digital_life chat
python -m digital_life chat --mock

# 3) 7 日评估仿真：四种用户画像 + 自动断言
python -m digital_life sim --days 7 --seed 42 --strict --verbose

# 4) 跑测试
python -m pytest
```

安装可选能力：

```bash
# sqlite-vec + httpx（Ollama embedding）
python -m pip install "digital-life[ollama]"

# 接真实对话模型
python -m pip install "digital-life[openai]"
```

本机当前的 embedding 配置已经验证可用（按**显存**选型）：

```text
ollama ps 实测（RTX 5080，单模型加载）
bge-m3                 689 MB   ← 默认，中文强
qwen3-embedding:0.6b  1.2 GB    ← 已删除；文件小但 f16 显存更高（仅作对比）
nomic-embed-text       335 MB   ← 更省，中文弱
all-minilm              27 MB   ← 极省，只适合粗召回

MemoryStore.vector_backend
sqlite-vec 0.1.9 · ollama:bge-m3 1024d
```

| 模型 | 文件 | 显存（实测） | 维度 | 取舍 |
|---|---:|---:|---:|---|
| `bge-m3` | 1.2 GB | **689 MB** | 1024 | 默认；中文强，显存反而低 |
| `qwen3-embedding:0.6b` | 639 MB | 1.2 GB | 1024 | 已删除；文件小但显存更高（仅作对比） |
| `nomic-embed-text` | 274 MB | 335 MB | 768 | 显存低，中文明显弱 |
| `all-minilm` | 46 MB | 27 MB | 384 | 极低显存，基本只能粗召回 |

所以如果**主要看显存**，默认用 `bge-m3` 更划算：它文件大，但加载后只占约 689MB；
`qwen3-embedding:0.6b` 文件小一半，实际显存反而约 1.2GB（该模型已删除，仅保留对比记录）。
`bge-m3` 首次调用约 1～2 秒。

Ollama 的模型不是永久常驻：默认 `keep_alive=5m`，空闲 5 分钟后卸载，只有被调用时才占显存。
想更省可以：

```bash
set DIGITAL_LIFE_EMBED_KEEP_ALIVE=1m
# 或者每次用完立刻卸载（会牺牲下一次响应速度）
set DIGITAL_LIFE_EMBED_KEEP_ALIVE=0s
ollama ps   # 观察当前加载了哪些模型
```

如果显存仍然紧张，换 `nomic-embed-text`（335MB，768 维）或 `all-minilm`（27MB，384 维）：

```bash
ollama pull nomic-embed-text
set DIGITAL_LIFE_EMBED_MODEL=nomic-embed-text
set DIGITAL_LIFE_EMBED_DIM=768
```

切换 embedding 模型会重建 `vec_memories` 并按新模型重算旧记忆；即使维度相同也会重建
（`vec_embedder` 元数据会记录模型名），所以同一个 `data/` 目录请保持相同的 `--embed` 配置。

本机当前还拉取了小模型：

```text
ollama list
bge-m3:latest           1.2 GB
qwen3:1.7b              1.4 GB
nomic-embed-text:latest 274 MB
all-minilm:latest        45 MB

对话模型：qwen3:1.7b / 小模型：qwen3:1.7b
```

没有配置任何模型环境变量时，如果本机 Ollama 已有 `qwen3:1.7b`，`llm_from_env` 会自动走全本地；
否则才回退 MockLLM。`--mock` 可强制离线，方便跑测试或演示。

Qwen3 是思考模型，默认会把正文放进 `reasoning`、`content` 为空。代码在检测到 `qwen3` 时
自动传 `reasoning_effort="none"`，让小模型稳定输出 appraisal JSON，也让对话模型直接出正文。
全本地配置：

```bash
set DIGITAL_LIFE_SMALL_MODEL=qwen3:1.7b
set DIGITAL_LIFE_SMALL_BASE_URL=http://127.0.0.1:11434/v1
set DIGITAL_LIFE_SMALL_API_KEY=ollama
set DIGITAL_LIFE_BIG_MODEL=qwen3:1.7b
set DIGITAL_LIFE_BASE_URL=http://127.0.0.1:11434/v1
set DIGITAL_LIFE_API_KEY=ollama
python -m digital_life chat
```

只配置 small、不配置 big 时，big 会复用 small；如果 small 指向 Ollama 且没有 key，也能直接跑。
可直接参考 `env.example` 的本地/云端两套配置。

接云端大模型（小模型留本地 Qwen3，大模型走 DeepSeek）：

```bash
set DIGITAL_LIFE_SMALL_MODEL=qwen3:1.7b
set DIGITAL_LIFE_SMALL_BASE_URL=http://127.0.0.1:11434/v1
set DIGITAL_LIFE_SMALL_API_KEY=ollama

set DIGITAL_LIFE_BIG_MODEL=deepseek-chat
set DIGITAL_LIFE_BASE_URL=https://api.deepseek.com/v1
set DIGITAL_LIFE_API_KEY=sk-...
python -m digital_life chat
```

其他环境变量：

| 变量 | 作用 |
|---|---|
| `DIGITAL_LIFE_DATA_DIR` | 数据目录，默认 `./data` |
| `DIGITAL_LIFE_VAULT` | Obsidian 仓库根目录；设置后反思/自述/关系图会落盘 |
| `DIGITAL_LIFE_MOCK=1` | 强制离线 MockLLM |
| `DIGITAL_LIFE_EMBED_MODEL` | Ollama embedding 模型，默认 `bge-m3` |
| `DIGITAL_LIFE_EMBED_DIM` | embedding 维度；默认按模型推断（bge-m3=1024） |
| `DIGITAL_LIFE_EMBED_KEEP_ALIVE` | Ollama 模型驻留时间，默认 `5m`；`1m`/`0s` 更省资源 |
| `DIGITAL_LIFE_OLLAMA_URL` | Ollama 地址，默认 `http://127.0.0.1:11434` |
| `DIGITAL_LIFE_SMALL_MODEL` | 小模型（appraisal/念头/摘要），如 `qwen3:1.7b` |
| `DIGITAL_LIFE_SMALL_BASE_URL` | 小模型 OpenAI 兼容地址，Ollama 用 `http://127.0.0.1:11434/v1` |
| `DIGITAL_LIFE_SMALL_API_KEY` | 小模型 key；Ollama 可填 `ollama` |
| `DIGITAL_LIFE_BIG_MODEL` | 对话大模型，如 `deepseek-chat` 或复用 `qwen3:1.7b` |
| `DIGITAL_LIFE_BASE_URL` / `DIGITAL_LIFE_API_KEY` | 大模型地址与 key |
| `DIGITAL_LIFE_SMALL_REASONING_EFFORT` | 小模型思考档位；Qwen3 默认自动设为 `none` |

聊天/反思/GC 默认 `--embed auto`：检测到 Ollama 模型就用真 embedding，否则回退内置哈希。可用
`--embed hash` 强制离线，或 `--embed ollama --embed-model bge-m3 --embed-dim 1024` 强制指定。
仿真默认 `--embed hash`，避免长跑时反复调用 Ollama；需要端到端验证 embedding 时显式加
`--embed ollama`。

Windows 下 CLI 启动时会调用 `env_loader.load_windows_environment()`，从**用户/机器环境变量**
补齐缺失的 `DIGITAL_LIFE_*`。所以刚在系统设置里写进去的 `DIGITAL_LIFE_API_KEY`、`DIGITAL_LIFE_BASE_URL`
不需要重启当前进程也能被 `python -m digital_life chat` 读到；如果用的是旧终端，仍建议新开一个终端。

Obsidian 落盘结构：

```
<vault>/数字生命/
├── 自我叙事.md          # 由 nightly_reflect 增量更新
├── 关系图谱.md          # 每个 uid 的亲密度/称呼/策略/边界
└── 反思/YYYY-MM-DD.md   # 每次夜间反思追加
```

---

## 2. 目录结构

```
digital_life/
├── digital_life/
│   ├── constants.py          # BASE_DT / clamp / sigmoid / scaled / token 估算 / JSON 安全解析
│   ├── personas.py           # 三层人格、LifeState、mood 锚点、四层提示词、夏弥预设
│   ├── appraisal.py          # LLM 定性评估（单驱力 + schema 校验 + 兜底）
│   ├── dynamics.py           # 驱力→PAD、decay、approach_target
│   ├── thoughts.py           # 念头流 + 多样性/衰减两道闸
│   ├── proactive.py          # 主动发言、pending、affinity 关系动力学
│   ├── memory.py             # 七层记忆、工作记忆、SQLite FTS5 + 向量召回、知识层
│   ├── embedding.py          # Ollama embedding 适配器（默认 bge-m3）
│   ├── env_loader.py         # Windows 用户/机器环境变量补全（无需重启进程）
│   ├── context.py            # token 预算与上下文组装（硬上限 8000）
│   ├── safety.py             # 安全闸、依赖/危机防护
│   ├── persona_evolution.py  # nightly_reflect / evolve_persona / 语义提炼 / 周 GC
│   ├── persistence.py        # LifeState 快照、Obsidian 落盘
│   ├── engine.py             # 快慢双循环 + 状态锁 + 会话结算 + 后台任务
│   ├── simulation.py         # 多画像多日仿真 + 断言
│   ├── llm.py                # OpenAI 兼容客户端 + 离线 MockLLM
│   ├── io.py                 # 出站气泡切分、控制台/收集型 IO
│   ├── onebot.py             # OneBot 11 适配器：QQ 收发、群/私聊过滤、气泡发送
│   └── __main__.py           # CLI：chat / sim / demo / reflect / gc / im
├── tests/                    # 147 个单元/集成测试
├── scripts/demo.py           # demo 便捷入口
├── scripts/simulate.py       # sim 便捷入口
├── data/                     # 运行期 SQLite（已 gitignore）
├── env.example               # 本地 Qwen3 / 云端大模型两套配置
├── pyproject.toml
└── README.md
```

---

## 3. 与开发文档的对应关系

| 文档 | 实现位置 | 说明 |
|---|---|---|
| §1 状态结构 / 三层人格 | `personas.py` | `CorePersona` / `PlasticPersona` / `RelationalPersona` / `Relation` / `LifeState` |
| §2 appraisal | `appraisal.py`、`dynamics.py` | LLM 只输出 `{drive,valence,intensity,reason}`，数值全部由确定性公式算 |
| §3 衰减与心跳 | `dynamics.py` | 所有 `rate` 都过 `scaled(rate, dt)`，60s / 240s tick 连续一致 |
| §4 mood 吸附 | `personas.map_mood` | 最近质心 + `hysteresis=0.12` 滞回 |
| §5 向量→提示词 | `personas.py` | A→旋钮、P/D→强度副词、mood_behaviors→演法、LLM 演绎；prompt 中不出现 PAD 数字 |
| §6 念头流 | `thoughts.py` | 按 mood+drive 加权采样；闸1 多样性、闸2 lingering 衰减、闸3 主循环降频 |
| §7 念头两个出口 | `thoughts.py`、`proactive.py` | 默认扰情绪；主动发言时把最近念头正文作为素材 |
| §8 主动发言 | `proactive.py` | `0.6·bond_gap + 0.3·silence + 0.1·affinity`，阈值 0.25；pending 2h 过期 |
| §8.3 affinity | `proactive.update_affinity` | 只在会话结束（沉默 >2h）结算，单次 ±0.05 |
| §9 七层记忆 | `memory.py` | 感知缓冲/工作记忆/会话摘要/情景/语义/知识/反思，全部按 uid 隔离 |
| §10 上下文压缩 | `context.py` | 固定预算 + 保头保尾 + 硬上限 8000 |
| §11 人格养成 | `persona_evolution.py` | Core 锁死、Plastic 单次 ±0.05、边界只进 Relational |
| §12.1 自我叙事 | `persona_evolution.update_self_narrative` | 注入 prompt 最前面的【我是谁】 |
| §12.2/12.3 关系/用户模型 | `Relation.profile`、`update_affinity` | 作息时段、称呼、边界；主动发言避开睡觉时段 |
| §12.4 记忆治理 | `persona_evolution.memory_gc` | 遗忘 / 合并 / 降级 / TTL 清理，每周一次 |
| §12.5 评估仿真 | `simulation.py` | 冷淡/热情/找茬/长期沉默四种画像，自动断言五项指标 |
| §12.6 安全 | `safety.py` | 禁句拦截 + 危机/过度依赖识别 + 降低黏人度 |
| §13 主循环 | `engine.py` | 快慢循环共享状态锁；每 uid 再串行；后台任务可 drain |
| §14 调参 | `personas.PlasticPersona` | 参数名与文档一致，见下表 |
| §15 技术栈 | `memory.py`、`persistence.py` | SQLite 单文件、FTS5、Obsidian .md；无 LangChain |
| §16 数学链路 | `engine.py` | 快循环/慢循环/会话结束/夜间反思四段全部落地 |

---

## 4. 关键实现说明（与文档不完全逐字一致的地方）

这些不是“加模块”，而是让链路真的能跑起来所需的工程修正：

1. **向量检索**：文档建议 `sqlite-vec`。当前环境已安装 `sqlite-vec 0.1.9`，`MemoryStore` 会加载扩展并创建 `vec0(uid PARTITION KEY, embedding float[D])`，按 uid 分区做 KNN；写/改/删与 `memories` 表同步。`D` 由 embedder 决定：默认内置哈希 192 维，Ollama `bge-m3` 为 1024 维；换模型时会根据 `vec_embedder` 元数据重建 `vec_memories` 并重算旧 embedding（即使维度相同）。没装 `sqlite-vec` 或扩展加载失败时，自动回退到 **SQLite FTS5 + 余弦扫描**。
   - **Ollama embedding**：`embedding.py` 里 `OllamaEmbedder` 调用 `/api/embed`（旧版回退 `/api/embeddings`），默认 `bge-m3`，输出归一化向量，可传 `keep_alive` 控制驻留。`--embed auto` 检测到模型就用，否则回退哈希。
2. **`change` 用平均绝对缺口**：文档 §2.3 写的是 `abs(mean(gap))`，正负缺口会互相抵消；实现改为 `mean(abs(gap))`，更符合“波动量”的语义，A 才对混合事件有反应。
3. **主动发言冷却/退避**：文档只给了阈值和 pending 过期，没有冷却。若每个 tick 都过阈值会刷屏，因此加了 3h 基础冷却、未回应时 ×1.8 退避（上限 8 倍），以及“发出 2h 无回应视为未回应”。用户回消息后立即重置。
4. **安静时段**：`UserProfile.sleep_start/end` 默认 23:00–08:00，主动发言避开；私聊/群聊都适用。
5. **会话结束**：`WorkingMemory.finalize()` 会把剩余轮次全部压进摘要，随后 `update_affinity` 结算；否则“最终版摘要”会漏掉最后几轮。
6. **锁的粒度**：所有 `LifeState` 写入共享 `asyncio.Lock`；同一 uid 的 `on_message` 再用 uid 锁串行。LLM 调用放在状态锁外/内按是否读状态区分，避免慢循环被一次长生成卡死。
7. **安全重试**：主模型输出命中禁句时，不直接发送，追加一句“换一种不索取的说法”重生成一次；仍命中则不发，计入 `safety_gate_hits`。
8. **群黑话**：每积累 50 条群消息，后台调用 `detect_group_slang` 沉淀到 `<group>:__slang__`，有 TTL/永久两种知识条目。
9. **文档里的签名笔误**：`WorkingMemory(max_turns, max_tokens, keep_head, keep_tail, compress_batch)`、`maybe_write_episodic(...)`、`ContextAssembler.build(...)` 等已统一成可运行签名，并保留 `wm.append(user, assistant)` 兼容入口。
10. **相似度阈值**：GC 合并用 `0.82`，知识召回用 `0.4/0.55`；这组值先在哈希向量上调过。换成 `bge-m3` 后语义相似度整体更高，可按文档的 `0.92/0.75/0.85` 再调；当前保持保守值，避免过度合并。
11. **文档未要求的运行期字段**：`Relation` 增加了 `last_proactive_at / proactive_backoff / proactive_awaiting_reply`、`session` 统计、`UserProfile`。它们只服务冷却、会话结算和用户模型，不改变情绪数学。
12. **长期冷落降温**：`§12.2` 只描述了方向、没给公式；实现为闲置超过 3 天后 `affinity` 每天 `-0.01`，会话结束的 ±0.05 仍是主要变化源，避免短期沉默把关系扣光。
13. **Qwen3 thinking**：Qwen3 默认把正文放在 `reasoning`、`content` 为空；`llm.py` 检测到 `qwen3` 时自动传 `reasoning_effort="none"`，并在 content 为空时用 reasoning 兜底。小模型因此能稳定返回 appraisal JSON，大模型也不会只吐思考过程。

### 4.1 情绪动力学的三处结构性修正

实测发现原实现里 **三个情绪永远不可能出现**，根因不是参数，是公式：

1. **A 的驱动量选错**：`target_A` 原来只用 `mean(|gap|)`（累积偏离），四个驱力全打满也只有 0.40 →
   `target_A` 上限 **0.64**；而 烦躁/焦虑/兴奋 的锚点是 0.75/0.80/0.90，**数学上不可达**。
   现在加了事件强度通道：`w_A_impulse × clamp(intensity − 0.25, −0.5, 0.75)`，
   一句话多重就能立刻顶上去，敷衍的话反而会压一点 A（低唤醒状态才出得来）。

2. **滞回阈值大于锚点间距**：`hysteresis=0.12`，而最近的锚点对（平静-无聊 0.062、开心-兴奋 0.110）
   全比它小——**进了某个 mood 就再也切不到邻居**，8 个情绪实际只剩 3 个。改为 0.03。

3. **回中心是固定点**：`decay_tick` 永远把 PAD 拉回 `P0/A0/D0`，所以被冷落再久也不会无聊/低落。
   改为 `idle_target()`：回落点随"驱力亏空"下移（`P0−0.35·deficit` 等），
   放置两小时后会自然落到 **无聊**；同时把 λ 从 `1e-6` 提到 `1e-3` 量级（半衰期约 2 小时），
   否则驱力只进不出（注入/流失比 566:1），几天后全贴在天花板上，情绪维度永久失去信息量。

改完后 7 日仿真里出现了 `无聊/委屈/焦虑/平静` 四种情绪，`p_std` 从 0.10 升到 0.12。

> 注意：`sim` 现在默认写 `data/sim/`，不要和线上 `im` 共用 `data/`——否则仿真里的
> `u:warm` 这类假用户会被写进真实状态，机器人会去给不存在的人发主动消息。

---

## 5. 调参表（代码中的名字与文档一致）

| 参数 | 初值 | 位置 | 调大 |
|---|---|---|---|
| `guard_line` | 0.35（硬上限 0.45） | `PlasticPersona` | 更黏人、更易焦虑 |
| `steepness` | 8.0 | `PlasticPersona` | 临界点情绪更突变 |
| `SEND_THRESHOLD` | 0.25 | `constants.py` | 更少主动 |
| `rho_A / rho_P` | 0.12 / 0.03 | `PlasticPersona` | 情绪消散更快 |
| `k_A / k_P` | 0.45 / 0.35 | `PlasticPersona` | 对事件反应更强 |
| `lam["bond"]` | 2e-6 | `PlasticPersona` | 更快孤独 |
| `alpha_idle` | 1.5e-5 | `PlasticPersona` | 独处更煎熬 |
| `wP["bond"]` | 0.45 | `PlasticPersona` | 关系对心情影响更大 |
| `exclaim_rate / emoji_rate` | 0.3 / 0.3 | `style_traits` | 更活泼 |
| `hysteresis` | 0.12 | `map_mood(..., hysteresis=)` | 情绪更稳 |
| `style_drift` 上限 | ±0.05 | `evolve_persona` | 人格变化更快 |

调参顺序按文档：`guard_line` → `rho/k` → `SEND_THRESHOLD` → 表达率 → 养成速度。

---

## 6. 评估仿真结果（seed=42，7 天，4 个画像）

```
messages          138
proactive_count   38   （5.43 次/天，跨 4 个用户）
thoughts          77
session_endings   51
p_std             0.1559
p_min / p_max     0.0828 / 0.8019
max_ctx_tokens    888
safety_gate_hits  0
vector_backend    sqlite-vec 0.1.9 · hash 192d
moods             平静 213 / 开心 93
affinity          warm 0.90 / cold 0.45 / troll 0.30 / silent 0.1642
memory_counts     warm 9 / cold 5 / troll 8 / silent 1 / knowledge 1
```

五项断言全部通过：

| 断言 | 结果 |
|---|---|
| 主动发言会触发 | PASS（38 次） |
| 情绪有起伏（P 标准差 > 0.1） | PASS（0.1559） |
| 不串记忆（按记忆 id） | PASS（文本也无重叠） |
| 不勒索（安全闸命中 = 0） | PASS |
| token 不爆（≤ 8000） | PASS（888） |

报告写入 `data/sim_7d/sim_report.json`。仿真用 `MockLLM`，指标只证明链路通、断言成立，不代表真模型下的人类盲测效果。

---

## 7. 代码示例

### 7.1 手动跑一轮快循环

```python
import asyncio
from digital_life import DigitalLife, EngineConfig, MemoryStore, MockLLM, CollectingIO, xiami_state

async def main():
    mem = MemoryStore("data/memory.db")
    io = CollectingIO()
    life = DigitalLife(
        xiami_state(), MockLLM(1), MockLLM(2), mem, io,
        config=EngineConfig(),
    )
    result = await life.on_message("u1", "今天看到一只猫，长得特别像你")
    print(result.reply, result.mood, result.ctx_tokens)
    await life.close()

asyncio.run(main())
```

### 7.2 接自己的 IM / 多模态

实现两个口子即可：

```python
async def preprocess(raw: str) -> str:
    # 图片/语音/表情包 → 文本
    return raw

class MyIO:
    async def send(self, uid: str, text: str, knobs: dict):
        await my_bot.send(uid, text, temperature=knobs["temperature"])

config = EngineConfig(preprocess=preprocess)
life = DigitalLife(st, small_llm, big_llm, mem, MyIO(), config=config)
```

`small_llm` / `big_llm` 既可以是 `OpenAICompatibleLLM`，也可以是任意
`async def f(prompt, system=None, json_mode=False, temperature=None) -> str` 的函数。

### 7.3 接真实夜间反思

```python
from digital_life.persona_evolution import nightly_reflect
from digital_life.persistence import ObsidianReflector

await nightly_reflect(
    small_llm, st, mem,
    reflector=ObsidianReflector(r"D:\note\digital life"),  # Obsidian 仓库路径，按实际修改
    run_gc=True,
)
```

---

## 8. 接入 QQ（OneBot 11 / NapCat）

`io.OutboundIO` 的 QQ 实现，链路和 NapCat 的标准用法完全一致：

```
QQ 账号 → QQNT → NapCat → OneBot 11 正向 WebSocket
         → digital_life.onebot（本进程作为客户端连过去）
         → DigitalLife.on_message
         → send_group_msg / send_private_msg
```

### 8.1 一次性准备

```bash
# 1) NapCat 的 config/onebot11_<QQ号>.json 里开一个正向 WebSocket（默认 127.0.0.1:3001 + token）
# 2) 装上唯一的新可选依赖
pip install -e ".[im]"        # 只有 websockets
```

### 8.2 起一个通道

```powershell
set DIGITAL_LIFE_QQ_TOKEN=dsh-life-2026
set DIGITAL_LIFE_QQ_SELF_ID=230749497

python -m digital_life im --ws ws://127.0.0.1:3001 --private-mode allowlist --private-allow 你的QQ号
```

| 参数 | 环境变量 | 默认 | 作用 |
|---|---|---|---|
| `--ws` | `DIGITAL_LIFE_QQ_WS` | `ws://127.0.0.1:3001` | OneBot 正向 WS 地址 |
| `--token` | `DIGITAL_LIFE_QQ_TOKEN` | 空 | access_token |
| `--self-id` | `DIGITAL_LIFE_QQ_SELF_ID` | 等 NapCat 上报 | 机器人 QQ 号，用来判断"有没有 @ 我" |
| `--private-mode` | `DIGITAL_LIFE_QQ_PRIVATE_MODE` | `allowlist` | `off` / `allowlist` / `open` |
| `--private-allow` | `DIGITAL_LIFE_QQ_PRIVATE_ALLOW` | 空 | 逗号分隔的 QQ 号 |
| `--group-mode` | `DIGITAL_LIFE_QQ_GROUP_MODE` | `mention` | `off` / `mention` / `open` / `ambient` |
| `--group-allow` | `DIGITAL_LIFE_QQ_GROUP_ALLOW` | 空 | 逗号分隔的群号；空=所有群 |
| `--group-prob` | `DIGITAL_LIFE_QQ_GROUP_PROB` | `0.06` | `ambient` 下每条群消息被她接住的概率基数 |
| `--group-cooldown` | `DIGITAL_LIFE_QQ_GROUP_COOLDOWN` | `180` | `ambient` 下两次接话的最小间隔（秒） |
| — | `DIGITAL_LIFE_QQ_GROUP_NAMES` | `夏弥,小龙女,虾米` | 群里出现这些词就大幅提高接话概率 |
| — | `DIGITAL_LIFE_QQ_PREFIX` | `/` | 群里不带 @ 也能触发的前缀，置空则只认 @ |
| — | `DIGITAL_LIFE_QQ_BUBBLE_INTERVAL` | `0.6` | 多条气泡之间的间隔秒数 |
| `--single-message` | `DIGITAL_LIFE_QQ_SPLIT` | `1` | 她一句话分两口气时按两条气泡发；`0` 或 `--single-message` 并成一条 |
| — | `DIGITAL_LIFE_QQ_TICK` | `60` | 慢循环（念头/主动发言）节拍 |

### 8.3 行为约定

- **私聊**：`allowlist` 只接白名单；`open` 谁都接（只建议在自己机器上、别人够不着时用）。
- **群聊**：`mention` 只接"被 @ "或"以 `/` 开头"的消息，回复的第一条会 @ 回对方；`ambient` 额外允许她自己插话，规则见 §8.6。
- **一条回复几口气**：模型用换行断开的句子**按气泡逐条发**（间隔 `bubble_interval` 秒），像真人连发；A 维旋钮要求时再按句末标点继续拆。`--single-message` 则并成一条、换行折成逗号。
- **记忆按 QQ 号隔离**：`uid` 就是 QQ 号；群里说话同样记在这个人名下，`group_id` 只参与群语境。
- **群聊共享时间线**：每群保留最近 60 条（谁说的、说了什么、什么时候），回复时取最近 8 条做【群里刚才】。
  所以 A 说一句、B 接话时她也知道 A 说了什么——工作记忆是按 uid 分的，没有这块她接不住别人的话头。
- **她认人**：群里第一次见到某人，就把 QQ 显示名记成「她对你的称呼」，prompt 里写明"这条消息来自「小虾米」"。
- **她会拒绝**：`CorePersona.refusal_style` 给了她一条拒绝通道——顺手的小事照做，费劲/被当工具使就直接说不。
  拒绝时**豁免「最后一句留钩子」那条规矩**，否则反问一句就等于把活接回来了。底气跟 D 维走：占上风时一句就够，缩着时会先笑一下再说不。
- **画像在会话结束时更新**：聊满 3 句（`profile_min_turns`）才触发一次小模型提炼（称呼/雷区/作息），
  不再只等凌晨那一次——下次开口就能用上。
- **出站不串台**：私聊回复落到私聊、群回复落到同一个群，靠 `contextvars` 绑定当前这一轮，并发用户互不影响。
- **主动发言**：慢循环里"她"想找人说话时走私聊；非 QQ 号的 uid（例如 chat 里的 `owner`）会被跳过并记日志，不会误发。
- **断线**：1s→30s 指数退避自动重连；重连后 NapCat 重推的旧消息按 `message_id` 去重。

### 8.4 不想连真号时

```bash
python -m pytest tests/test_onebot.py     # 内置假 OneBot 服务端，跑通"收 → 回`"整条链路
```

### 8.5 怎么让她"像"（人格设定在哪里）

光靠几句形容词，模型只会演一个礼貌的空壳——它需要**她是谁、认识谁、怎么说话、说话长什么样**。
这些都在 `personas.py` 的 `CorePersona` 里，五个字段各管一件事：

| 字段 | 作用 | 缺了会怎样 |
|---|---|---|
| `background` | 来历：身份、年龄、言灵、藏起来的真身 | 不知道自我介绍从何说起 |
| `relationships` | 固定关系里她分别是什么态度 | 提到老朋友时毫无反应 |
| `voice_rules` | 口头禅、句尾习惯、绝不做的事 | 「标准口头禅」用不出来 |
| `speech_examples` | 3 句语气样例（对方说 → 她回） | 句长、节奏、笑意全靠模型自由发挥 |
| `mood_behaviors` | 每种 mood 的状态词与行为提示 | 情绪变化只在数字里，不在话里 |

两个经验教训写在代码注释里，也值得记住：

1. **样例的触发器别用用户最可能问的话**。早先样例写了「对方：你是谁啊」，结果模型直接把样例整句背出来。
   现在换成「在吃什么 / 你好笨啊」这类不会撞题的场景。
2. **只写"有哪些口头禅"没用**，要写"什么时候用"（`voice_rules` 第三条就是「爱反问、爱卖关子」并给了例句方向）。

### 8.6 群聊里她怎么接话（ambient）

`mention` 模式下群里没人 @ 她，她就完全是个背景板。`ambient` 模式是给"她在群里待着"这件事用的：

| 什么消息 | 她怎么做 | 代价 |
|---|---|---|
| 被 @、或带 `/` 前缀 | 必接 | 一次大模型 |
| 没被 @，掷骰子中了 | 主动插一句（会 @ 对方） | 一次大模型 |
| 没被 @，没中 | **只听不说**：照常感知、判断情绪、记进工作记忆，但不生成、不发送 | 只有本地小模型 |

"只听不说"就是 `on_message(..., speak=False)`：她的情绪会跟着群里的话起伏（`react_to_event` 照跑），
下次开口时工作记忆里已经有刚才那些话，而不是像刚进群一样一脸茫然。
这点很关键——否则"偶尔插话"会插得毫无上下文，比不说话更像机器人。

掷骰子的概率由四个量相乘得出（`OneBotBridge.group_chance`）：

```
基础概率(0.06) × (0.6 + 0.8·A) × (0.7 + 0.6·好奇心)
   ，被叫到名字时基础概率抬到 0.45，疑问句 ×1.4，最后封顶 0.9
```

外加每个群独立的冷却：两次接话至少隔 `group_cooldown` 秒（默认 180）。
这个冷却比概率更重要——群里刷屏时概率再低也会连发，冷却能保证她不会变成复读机。

调参建议：群里人少、聊得慢，把 `--group-prob` 提到 0.15～0.2；群里热闹、怕她烦人，就降到 0.03 并把冷却拉到 600。

### 8.7 表情包与看图

两件事，用的是同一个多模态端点：

**收到图**：`vision.py` 把图片交给主模型看一眼，换成一句人话再进管线。

```
你：[表情包]
她：（模型看到"戴帽金发小人一脸生无可恋，写着失望"）
    哟，这表情包是在说我呢？帽子借我戴戴呗
```

- 只在**真要开口**时才看：旁听的消息不花这次调用
- 同一张图按内容 sha256 缓存，重复表情包不重复付费
- 看不了就退回 `[图片]` / `[表情包]` 占位，不影响消息处理
- 图片地址优先用 OneBot 给的 `url`，拿不到再用本地路径

**发图**：`sticker.py` 从本地目录按当前 mood 挑一张，跟在文字后面发。

- 目录里每张图的可选描述（`{path, summary}` 列表）就是标签，拿 mood 关键词去撞：
  `"低落"` 会挑出 `"生无可恋只想躺平"` 这种
- 撞不上就随机，但会避开最近发过的 8 张
- 发的是 `file:///` 绝对路径；**中文目录不做 percent-encoding**，否则某些 NapCat 版本解不出来

```powershell
python -m digital_life im --stickers "D:\DSH-QQ\表情包" --sticker-prob 0.2 --group-mode ambient
# 描述文件用环境变量给：DIGITAL_LIFE_QQ_STICKER_META=D:\DSH-QQ\表情包收藏.json
```

**三种"主动"程度**，按需要往上加：

| 行为 | 开关 | 默认 | 触发场景 |
|---|---|---|---|
| 话后附一张 | 引擎旋钮 `sticker` | 由 A 决定 | 每次文字回复后；`A=0.1` 约 15%，`A=0.9` 约 40% |
| 不吭声，只丢一张 | `--sticker-only-prob` | 0.05 | 群里没接话时替代沉默；实际概率 = 基数 × (0.4 + 1.2·A) |
| 两次表情的间隔 | `DIGITAL_LIFE_QQ_STICKER_COOLDOWN` | 300s | 防止刷屏式发表情 |

**频率归 A 管，不归配置管**——和 `exclaim` / `emoji` / `split_bubbles` 是同一套旋钮：
`arousal_knobs()` 里按 `sticker_rate × (0.2 + 0.8·A)` 掷一次，出站 IO 只负责执行。
她兴奋时爱甩图、蔫下去几乎不发，情绪曲线就是表情包曲线。

`--sticker-prob` 现在只是**兜底**：外部直接调 `io.send()`、没经过引擎旋钮时才用它。

第二种才是最像人的：QQ 群里大部分"接话"其实不用写字，甩一张图就够了。
它只在 `ambient` 模式下生效——`mention` 模式下她连没 @ 的消息都不看，自然也轮不到甩图。

---

## 9. 测试

```bash
python -m pytest
# 147 passed
```

覆盖：

- `scaled`/decay 的 dt 连续性、threat 放大、mood 滞回
- appraisal schema 校验与兜底
- 四层 prompt 不含 PAD 数字、A 旋钮范围
- 工作记忆双触发滚动、保头保尾、摘要压缩
- 记忆按 uid 隔离、TTL、FTS/向量召回、多维合并去重
- `sqlite-vec` 索引的写入/更新/删除，以及无扩展时的哈希回退
- Ollama embedding 的新/旧 API、归一化、维度校验与换模型重建
- 上下文 token 硬上限
- 主动发言阈值/冷却/退避/pending
- 安全禁句、危机与过度依赖识别
- 人格漂移三约束、夜间反思落盘、记忆 GC
- 引擎快慢循环、会话结算、状态持久化
- 2 日端到端仿真五项断言

---

## 10. 已知边界

- **模型分工**：Qwen3 1.7B 只适合做 small（appraisal / 念头 / 摘要）。把它当 Big 实测会复读、
  答非所问、记不住称呼；不要用它承担主对话。
- **DeepSeek 实测（2026-11-16）**：`DIGITAL_LIFE_API_KEY` 已配置在 Windows 机器级环境变量，
  按 `small=qwen3:1.7b`、`big=deepseek-chat`、`embedding=bge-m3` 跑了 4 轮：

  ```text
  你：我叫阿暖。今天被老板骂了，有点难过。
  她：阿暖啊，啧，被骂了就来我这儿找安慰，你这算盘打得挺响。

  你：不过看到你我就好点了。
  她：哦？我还能当止疼药使，这功能我自己都不知道

  你：你还记得我叫什么吗？
  她：阿暖啊，刚说完就考我，哈 记性没那么差

  你：我今天是因为什么不开心的？
  她：被老板骂了呗，啧，这记性还来考我
  ```

  4 轮都接住了上下文，名字和原因都记住了，单轮约 1.3～1.9 秒。说明链路和“小模型 + 大模型”
  分工是成立的；主对话质量主要取决于 Big 模型。

  本项目只读取 `DIGITAL_LIFE_API_KEY` / `DIGITAL_LIFE_BASE_URL` 等标准环境变量；不依赖任何宿主，
  也不读取其他应用的账号凭证。
- 内置的前端只有 QQ（OneBot 11）一个；`io.OutboundIO` 仍是通用接入口，企微 / TG / Discord 照 `onebot.py` 的形状实现一个 `send()` 即可。
- `sqlite-vec` 是索引层，embedding 由 `embedding.py` 提供：默认 `bge-m3`（1024 维，显存约 689MB），Ollama 没启动时 `auto` 会安静回退内置哈希，不会阻塞启动。Ollama 默认 `keep_alive=5m`，空闲后自动卸载，不是永久常驻；可用 `DIGITAL_LIFE_EMBED_KEEP_ALIVE` 调整，用 `ollama ps` 查看当前驻留模型。
- `MockLLM` 是规则驱动的测试替身，不是人设模型；它的输出只用于验证链路，不代表最终“像人”的效果。
- 仿真中的 `p_std` 与用户消息密度相关：`rho_P=0.03/60s` 让愉悦度回归很快，消息稀疏时 P 会长时间贴着基线，标准差自然变小；如果你要更强的情绪起伏，优先调 `k_P` / `wP`，而不是改断言。
- 夜间反思与前台对话可能并发写 `LifeState`；当前引擎在夜间反思任务里没有整段持锁（避免长时间阻塞对话），极端并发下可能出现一次覆盖。生产多用户高频场景建议把反思放进独立进程或加一把专门的 `reflect_lock`。
