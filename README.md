# VOXPET · 个性化语音克隆桌宠

> **Personalized Voice-Clone Desktop Pet** — 对着它说话，它用「你朋友的声音和语气」回答你。

一个端到端的本地语音对话系统：基于 **RAG** 学习目标人物的说话风格、基于 **GPT-SoVITS** 克隆其声音音色，串联 **ESP32 硬件**与本地 **STT → RAG → LLM → TTS** 链路，实现「用朋友的声音陪你聊天」的实时语音交互。

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white">
<img alt="RAG" src="https://img.shields.io/badge/RAG-ChromaDB-5A29E4">
<img alt="LLM" src="https://img.shields.io/badge/LLM-GPT--4o-412991?logo=openai&logoColor=white">
<img alt="TTS" src="https://img.shields.io/badge/Voice%20Clone-GPT--SoVITS-FF6F61">
<img alt="MCU" src="https://img.shields.io/badge/MCU-ESP32--S3-E7352C">
</p>

> 本仓库为该项目的 **本地电脑版**（硬件 + 网页双端）。涉及的真实聊天语料、音色权重、API 密钥均不随仓库公开（见 [隐私与脱敏](#-隐私与脱敏说明)）。

---

## ✨ 功能特性

- 🎙️ **语音进 / 语音出**：对着 ESP32 桌宠按键说话，几秒后听到「克隆音色」的回答；也可在网页端用麦克风或文字聊天。
- 🧠 **像那个人，不像 AI**：用 RAG 检索目标人物的真实历史发言，作为 few-shot 注入对话上下文，让回复带上 TA 的口头禅、句式与语气。
- 🗣️ **克隆 TA 的音色**：基于 GPT-SoVITS 少样本音色克隆，几分钟语料即可训练出专属音色。
- 🧩 **可扩展多角色**：新增一个可对话角色只需在一个配置文件里加一段，训练 / 检索 / 对话 / 合成全流程复用（[详见](#-可扩展多角色架构)）。
- 🔒 **隐私优先**：聊天语料、音色权重、密钥全部本地化，不入仓库、不外传。

---

## 🏗️ 系统架构

采用「**瘦客户端 + 重处理后端**」设计，把算力需求与交互终端解耦：

```
┌────────────────┐   WebSocket    ┌──────────────────────────────────────┐
│  ESP32-S3 终端  │ ◄────────────► │              本地 PC 后端               │
│ · PDM 麦克风录音 │   PCM 音频流    │  STT(Whisper) → RAG 检索 → LLM(GPT-4o)  │
│ · 功放喇叭播放   │                │     → 音色克隆 TTS(GPT-SoVITS)          │
│ · 按键触发 PTT  │                │        → 32k 重采样 16k 回传            │
└────────────────┘                └──────────────────────────────────────┘
        ▲                                          ▲
        └── 也可纯网页端：浏览器麦克风 / 文字 ──────┘  (app.py)
```

- **轻量协议**：文本控制帧 `REC_START / REC_END`（上行）与 `TTS_START / TTS_END`（下行）+ 二进制 PCM 流，清晰区分录音上行与合成下行。
- **采样率适配**：终端 16kHz 采集，TTS 模型输出 32kHz，后端重采样对齐终端播放采样率，避免变调。
- **两种入口**：`desk_pet_server.py`（硬件端 WebSocket 服务）与 `app.py`（本地网页版，浏览器录音 / 文字）。

---

## 🧱 技术栈

| 模块 | 技术 |
|------|------|
| 语音识别 STT | faster-whisper（本地 CPU，int8 量化） |
| 个性化检索 RAG | ChromaDB + sentence-transformers（`paraphrase-multilingual-MiniLM-L12-v2`） |
| 对话生成 LLM | OpenAI GPT-4o（Prompt 工程 + few-shot 注入） |
| 语音克隆 TTS | GPT-SoVITS v2Pro（音色微调训练 + 推理 API） |
| 后端 / 通信 | Python · FastAPI · WebSocket · asyncio |
| 嵌入式 | ESP32-S3（XIAO ESP32S3 Sense）· Arduino · I2S 音频 · PDM 麦克风 |

---

## 📂 项目结构

```
voxpet-deskpet/
├── app.py                  # 本地网页版后端 (FastAPI)：浏览器麦克风/文字 → 语音回复
├── index.html              # 网页前端 (微信风格 UI，支持多角色下拉切换)
├── desk_pet_server.py      # 硬件端 WebSocket 服务：ESP32 按键说话 → 语音回复
├── friends.py              # ★ 多角色花名册 (脱敏示例)：新增角色只改这里
├── build_rag.py            # 从聊天记录 CSV 为每个角色构建向量检索库
├── start_sovits_api.py     # 启动 GPT-SoVITS 推理 API 并加载训练好的音色权重
├── requirements.txt
├── tools/                  # 语音训练数据工程工具
│   ├── collect_voices.py   #   从微信导出里按说话人提取语音素材
│   ├── sort_voices.py      #   按 talker 归类语音文件
│   ├── pick_ref.py         #   按时长/响度打分推荐参考音频候选
│   ├── check_duration.py   #   语音时长分布分析
│   └── filter_short.py     #   按时长区间筛选训练片段
├── firmware/               # ESP32-S3 固件 (Arduino)
│   ├── desk_pet_esp32_v2/  #   桌宠主固件：按住说话(PTT) → 上传/播放
│   └── test_speaker_440/   #   喇叭自检：440Hz 正弦波
└── docs/
    ├── 项目详解.md          # 完整项目文档 (难点拆解 / 面试讲解参考)
    └── chat_format_example.csv  # 聊天记录 CSV 字段示例
```

---

## 🚀 快速开始

### 0. 环境准备

```bash
pip install -r requirements.txt
# 首次运行会自动下载多语言 embedding 模型 (~400MB)
```

配置 API key（从环境变量读，**不写进代码**）：

```bash
# 复制 .env.example 为 .env 并填入你的 key
cp .env.example .env
# .env 内容: OPENAI_API_KEY=sk-...
```

> 语音克隆需要本地装好 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2Pro，并在 `friends.py` / `start_sovits_api.py` 顶部把路径改成你本机的。GPT-SoVITS 推理建议有 NVIDIA GPU。

### 1. 构建角色检索库

把聊天记录导出成 CSV（字段见 `docs/chat_format_example.csv`），在 `friends.py` 配好角色的 `rag_names`，然后：

```bash
python build_rag.py            # 给花名册里所有角色建库
python build_rag.py buddy      # 只给某个角色建库
python build_rag.py --list     # 列出 CSV 里所有说话人 (用来发现昵称)
```

### 2A. 网页版（最快体验，不用硬件）

```bash
python start_sovits_api.py     # 终端 1：启动音色克隆 API (端口 9880)
python app.py                  # 终端 2：启动网页后端 → http://localhost:8000
```

浏览器打开 `http://localhost:8000`：可文字聊天 / 麦克风说话，右上角下拉切换角色（音色 + 风格 + 知识库一起切换）。

### 2B. 硬件版（ESP32 桌宠）

1. 用 Arduino IDE / `arduino-cli` 烧录 `firmware/desk_pet_esp32_v2/`（板子选 `XIAO_ESP32S3`），先把固件顶部的 `WIFI_SSID / WIFI_PASS / WS_HOST` 改成你的；
2. PC 端运行 `start_sovits_api.py` + `desk_pet_server.py`（WebSocket 监听 8765）；
3. 让 ESP32 与 PC 在同一 2.4GHz 局域网，**按住按键说话、松开发送**，几秒后从喇叭听到克隆音色的回答。

> 喇叭接线异常时可先烧 `firmware/test_speaker_440/` 自检（应听到稳定的 440Hz 嘟嘟声）。

---

## 🧩 可扩展多角色架构

新增一个可对话角色，**只需在 `friends.py` 的 `FRIENDS` 字典里加一段**，其它代码一律不动：

```python
"buddy": {
    "display_name": "老王",
    "gpt_model":    "buddy-e15.ckpt",       # 训练得到的 GPT 权重
    "sovits_model": "buddy_e8_s392.pth",    # 训练得到的 SoVITS 权重
    "ref_audio":    ".../slicer_buddy/ref.wav",
    "ref_text":     "参考音频说的那句话(一字不差)",
    "rag_names":    ["老王", "王哥"],         # 在聊天记录里的昵称
    "persona":      "# 老王的性格\n损友型, 阴阳怪气, 短句懒散...",
},
```

之后 `python build_rag.py buddy` 建库、重启服务，网页下拉即可选到新角色 —— 训练、检索、对话、TTS 全流程复用。

---

## 🔬 工程亮点 / 难点拆解

> 完整版见 [`docs/项目详解.md`](docs/项目详解.md)，这里摘要。

- **让 LLM「说人话」且「像那个人」**：检索目标人物历史发言，以 `assistant` 历史发言形式注入 messages 做 few-shot，比文字描述风格更有效；配合强约束 system prompt（短句、口语、无 markdown/emoji、不主动总结反问）。
- **音色训练流水线落地**：系统性排查并修复 GPT-SoVITS v2Pro 命令行训练中的 **8+ 处缺陷**（版本判定、分片命名、SV 特征缺失、超参归属、目录预创建、按字符串而非数值排序选错 epoch 等），封装为一键自动化训练脚本。
- **短文本合成不稳定的一次完整定位**：长句正常、短句时有时无 → 用 **音频 RMS 能量** 做量化诊断（静音样本 RMS≈148 vs 正常≈2875）→ 排除前端 / 采样参数 → **控制变量法** 锁定根因是参考音频，把 4 秒短参考换成 7–8 秒韵律完整的陈述句后稳定。
- **数据工程**：从混乱的群聊 / 私聊导出里按 `talker` 精准提取目标人语音，按时长分布筛选训练最优区间（3–10 秒），增量补料重训显著提升音色相似度。

---

## 🔒 隐私与脱敏说明

本项目以真实社交对话作为训练 / 检索语料，公开仓库中已做脱敏：

- **不包含** 任何真实聊天记录、向量库、音色权重、音频缓存或 API 密钥；
- `friends.py` 中的角色为**示例占位**，真实人设 / 参考文本不公开；
- 固件中的 WiFi 凭据为占位符，请替换为自己的；
- 密钥一律从环境变量读取（`.env`，已被 `.gitignore` 忽略）。

---

## 📄 License

[MIT](LICENSE)
