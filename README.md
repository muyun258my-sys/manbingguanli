# 慢病健康管理助理系统

> 基于 **LangGraph + RAG** 架构的慢病管理 AI 助手，支持症状分析、用药咨询、就医导诊与饮食咨询，面向高血压、糖尿病、冠心病等慢性病患者及家属。

---

## 项目背景

慢性病患者在日常管理中面临三类高频痛点：不知道当前症状是否需要就医、不确定多种药物能否同时服用、不清楚应该挂哪个科室。传统搜索引擎给出的信息零散且缺乏个性化，本项目通过 Multi-Agent 架构将不同诉求交由专属 Agent 处理，并结合用户健康档案与医学指南知识库提供上下文感知的回答。

---

## 功能特性

| 能力 | 说明 |
|------|------|
| 症状分析 | 结合健康档案与医学指南，评估症状风险、提示缺失信息 |
| 用药咨询 | 核对过敏史、当前用药、禁忌与相互作用，不做处方决策 |
| 就医导诊 | 严重度分级（🔴 立即就医 / 🟡 建议近期就诊 / 🟢 可自行观察）+ 科室建议 |
| 饮食咨询 | 结合慢病类型给出忌口与限量建议（高盐、高糖、高嘌呤、高脂等） |
| 档案管理 | 用户健康档案（病史 / 用药清单 / 过敏史）的查看与更新，每轮对话自动注入 |
| 多意图混合 | 一次提问包含多个诉求时，自动拆解并按「症状 → 用药」串行处理再合并 |
| 高风险 Short-circuit | 双重检测命中即停止一切医疗建议生成，强制输出急救提示 |

---

## 系统架构

```
用户输入
   │
   ▼
Streamlit 前端 (8501) ──HTTP──> FastAPI 后端 (8080)
   │                                   │
   │                              /chat · /profile · /health
   │                                   ▼
   │                        LangGraphOrchestrator（编排层）
   │                                   │
   │                  LangGraphWorkflow（StateGraph 状态机）
   │                                   │
   │        safety ──> prepare ──> retrieve ──> route ──> agent ──> aggregate
   │           │                                                         │
   │        emergency（短路）                                   合并结果（取最保守）
   │                                   │
   │                 ┌─────────────────┼───────────────────┐
   │                 ▼                 ▼                   ▼
   │        症状分析 Agent     用药管理 Agent          导诊 Agent
   │        饮食咨询 Agent     General Agent（兜底）
   │                 │                 │                   │
   │                 └─────────────────┴───────────────────┘
   │                                   │
   │             ┌─────────────────────┼──────────────────┐
   │             ▼                     ▼                  ▼
   │      医学知识库              用户健康档案         药品数据库
   │    ChromaDB 向量库         SQLite / MySQL        drugs.json
   │   （指南 PDF 切片）        （病史·用药·过敏史）
   │             │
   │             ▼
   │      DeepSeek LLM（推理 / 生成，可选，失败自动降级本地规则）
   └── 离线 Ingestion 管道（独立于在线链路）
       shujuku/*.pdf ──> 解析 ──> 切片 ──> bge-small-zh-v1.5 向量化 ──> ChromaDB
```

### 分层职责

- **表现层（Streamlit）**：多轮对话界面 + 健康档案管理面板，仅做展示与交互，所有数据经接口层获取
- **接口层（FastAPI）**：请求校验、会话/用户标识管理、统一响应包络，不承载业务决策
- **编排层（LangGraphOrchestrator）**：以 LangGraph `StateGraph` 串联「安全网关 → 意图分类 → 检索 → 路由 → Agent 执行 → 结果聚合」，多意图走「症状 → 用药」串行链路
- **能力层（5 个 Agent）**：症状分析、用药管理、导诊、饮食咨询、General 兜底；LLM 可用时走 Prompt 链，不可用/失败时自动降级为本地规则回复
- **服务层**：档案存储（SQLite/MySQL）、向量检索（ChromaDB）、会话记忆（最近 5 轮）
- **数据层**：`vector_db/`（指南向量库）、`data/profiles.db`（档案）、`shujuku/drugs/drugs.json`（药品结构化数据）

### Agent 执行策略

每个 Agent 有两条路径：

1. **LLM 路径**：`LangChainRuntime.invoke_agent()` 组装 System Prompt + 档案 + 检索片段，调用 DeepSeek（OpenAI 兼容）生成回答
2. **降级路径**：LLM 未配置、调用失败或超时时，回退到内置的本地规则回复（仍能完成意图应答，内容更模板化）

---

## 接口层 API 契约

所有响应统一包络：`{ code, message, data, disclaimer }`

### `POST /chat`

| 字段 | 方向 | 类型 | 说明 |
|------|------|------|------|
| `session_id` | 入 | string | 会话标识，缺失时由服务端生成 |
| `user_id` | 入 | string | 用户标识，关联健康档案 |
| `message` | 入 | string | 用户自然语言输入 |
| `reply` | 出 | string | 最终自然语言回答 |
| `intent` | 出 | enum | 命中的意图类型（8 类，见下） |
| `severity` | 出 | enum\|null | 🟢/🟡/🔴，仅导诊相关时有值 |
| `emergency` | 出 | bool | 是否触发 short-circuit |
| `sources` | 出 | array | RAG 引用的文档来源（标题/出处） |

意图类型：`symptom_analysis`（症状）、`medication_query`（用药）、`diagnosis_query`（导诊）、`diet_query`（饮食）、`profile_management`（档案）、`mixed_query`（多意图）、`high_risk_input`（高风险短路）、`general_health`（兜底）。

### `GET /profile/{user_id}` · `PUT /profile/{user_id}`

| 字段 | 类型 | 说明 |
|------|------|------|
| `condition_description` | string | 当前病情描述 |
| `conditions` | array | 已知病史（如高血压、糖尿病） |
| `medications` | array | 当前用药清单 |
| `allergies` | array | 过敏史 |
| `updated_at` | datetime | 最近更新时间 |

### `GET /health`

返回服务存活与依赖（档案库 / LLM / 向量库）连通性，供部署平台探活。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 编排 | LangGraph | `StateGraph` 状态机编排多 Agent 流程 |
| Agent 运行时 | LangChain / LangChain Runtime | Prompt 链、消息历史、Runnable 调用 |
| LLM | DeepSeek API（OpenAI 兼容） | 推理生成；Agent 配置见 `.agents/*.json` |
| 向量数据库 | ChromaDB（本地持久化） | 医学指南知识库存储与检索，位于 `vector_db/` |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文医学文本向量化，缓存于 `models/embedding` |
| 后端接口 | FastAPI + Uvicorn | RESTful API（8080 端口） |
| 前端展示 | Streamlit | 多轮对话 + 健康档案管理（8501 端口） |
| 数据存储 | SQLite（默认）/ MySQL（可选） | 用户健康档案、药品结构化数据 |
| 测试 | pytest | 114 个用例，覆盖 API / 服务层 / Agent / 检索管道 |

---

## 目录结构

```
.
├── app/                    # 后端核心包
│   ├── __init__.py         # 进程最早执行点（强制 HF 离线模式，见“常见问题”）
│   ├── __main__.py         # python -m app 启动入口（uvicorn，8080）
│   ├── app.py              # FastAPI 路由定义
│   ├── services.py         # Orchestrator（LangGraphOrchestrator）、5 个 Agent、档案/安全/意图等服务
│   ├── langgraph_workflow.py   # LangGraph StateGraph 工作流定义
│   ├── langgraph_runtime.py    # LangChain 运行时：LLM 封装、Chroma 检索、会话历史
│   ├── agent_runtime.py        # Agent 配置加载（.agents/*.json）、OpenAI 兼容客户端
│   ├── ingestion.py            # 离线管道：PDF 解析、切片、向量库构建与检索
│   └── models.py               # 数据模型（Profile / ChatRequest / SourceRef 等）
├── frontend/app.py         # Streamlit 前端
├── tests/                  # 114 个 pytest 用例
├── shujuku/                # 知识库原始材料
│   ├── guidelines/         #   国家卫健委慢病管理指南 PDF（高血压/糖尿病/高尿酸/高脂血）
│   └── drugs/drugs.json    #   常用慢性病药品结构化数据
├── vector_db/              # ChromaDB 向量库（构建产物，manifest.json 记录元信息）
├── models/embedding/       # bge-small-zh-v1.5 模型本地缓存
├── .agents/                # Agent 配置（DeepSeek 端点、system prompt 等）
├── data/                   # SQLite 档案库（profiles.db）
├── docs/                   # 补充文档
├── start.bat               # Windows 一键启动（后端就绪等待后起前端）
├── start.sh                # Git Bash / WSL / macOS / Linux 一键启动
└── requirements.txt
```

---

## 快速开始

### 环境要求

- Python 3.10+（代码使用 `Path | str` 联合类型语法）
- 依赖安装：`pip install -r requirements.txt`

### 1. 一键启动（推荐）

```bash
pip install -r requirements.txt
```

- **Windows**：双击 `start.bat`，或在 CMD 中运行 `start.bat`
- **Git Bash / WSL / macOS / Linux**：`bash start.sh`

脚本会先启动后端（8080），**轮询 `/health` 等待就绪**（最长 90 秒）后再启动前端（8501），前端会自动打开浏览器。前端侧边栏的"后端地址"默认指向 `http://127.0.0.1:8080`，无需修改。

> `start.bat` 在前端关闭后，后端窗口（标题 `app-backend`）仍会保留，手动关闭即可；`start.sh` 在前端退出时自动关闭后端。

### 2. 手动启动

```bash
# 终端 A：后端（FastAPI，8080）
python -m app
# 或: python -m uvicorn app.app:app --host 127.0.0.1 --port 8080

# 终端 B：前端（Streamlit，8501）
python -m streamlit run frontend/app.py --server.port 8501
```

启动后访问 `http://127.0.0.1:8080/docs` 查看交互式 API 文档。

### 3. 配置 LLM（可选）

不配置也能运行——后端会降级到本地规则回复。需要真实 LLM 回答时：

1. 在项目根目录创建 `.env`，设置 `DEEPSEEK_API_KEY=sk-xxx`
2. 可按需覆盖 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_TEMPERATURE` 等（见 `app/langgraph_runtime.py`）
3. 重启后端

Agent 的系统提示词与模型参数位于 `.agents/*.json`，可直接编辑。

### 4. 构建知识库（首次或更新指南时）

`vector_db/` 已包含构建好的向量库，可直接使用。若需重建：

```bash
python -c "from app.ingestion import build_vector_store; build_vector_store()"
```

该命令会解析 `shujuku/guidelines/*.pdf`，切片后用本地 embedding 模型向量化并写入 ChromaDB。**需要 `models/embedding` 中存在模型缓存**（仓库已内置）。

---

## 运行测试

```bash
python -m pytest tests/ -v
```

共 **114 个用例**，覆盖：API 端点（/chat、/profile、/health）、意图分类、安全网关、档案存储（SQLite/MySQL）、会话记忆、LangGraph 工作流（紧急短路、多意图串行）、5 个 Agent 行为、Agent 配置加载与 OpenAI 客户端、PDF 解析与分块管道。

> 测试环境自动隔离：关闭 LLM、档案库与向量库指向临时目录，不触碰真实数据，也不调用外部 API。

---

## 安全边界设计

医疗 AI 的核心风险是"过度自信"。本项目通过以下机制控制风险：

1. **免责声明**：所有输出附加"本系统仅供参考，不构成医疗诊断或治疗建议"
2. **高风险 Short-circuit**：安全网关用正则高危词库匹配（胸痛、昏迷、意识障碍、单侧肢体无力、口角歪斜、大出血等），命中即停止一切医疗建议生成，仅输出急救提示
3. **严重度置顶**：导诊结果为 🔴 时，前端强制置顶显示就医建议
4. **拒绝诊断**：系统不对疾病做确诊性表述，仅输出"可能相关""建议咨询医生确认"等措辞
5. **降级兜底**：LLM 不可用时自动回退本地规则，避免"空回复"

---

## 数据来源

| 数据类型 | 来源 |
|----------|------|
| 慢病管理指南 | 国家卫健委发布的慢性病管理指南 PDF（高血压、糖尿病、高尿酸、高脂血） |
| 药品信息 | `shujuku/drugs/drugs.json`（常用慢性病药品结构化数据） |
| 用户数据 | 本地模拟生成（demo 演示用） |

---

## 常见问题（FAQ）

### 前端报"无法读取档案 / 连接被拒绝 (WinError 10061)"

后端尚未就绪。`start.bat` / `start.sh` 已改为**轮询 `/health` 就绪后才启动前端**；若仍遇到，检查后端窗口是否存活，或手动先启动后端（`python -m app`）再刷新页面。

### 对话请求超时（Read timed out）

旧版本有两个已知诱因，均已修复：

1. **HuggingFace 联网检查**：sentence-transformers 加载本地模型时会向 `huggingface.co` 发起联网检查，国内网络连接超时会把模型加载拖到 60s+。现在 `app/__init__.py` 在进程最早期强制设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（注意：`huggingface_hub` 在 import 时会快照该变量，因此必须在任何第三方库导入之前设置）。
2. **后端未就绪**：见上一条。

首次对话约 10~15s（含 embedding 模型加载），后续每轮 10~20s 主要取决于 DeepSeek API 响应速度，属正常现象；前端超时上限为 30s。

### 回答偏模板化 / 不像"AI 生成"

未配置 `DEEPSEEK_API_KEY` 或调用失败时，系统会降级到本地规则回复。检查 `.env` 配置后重启后端即可。

### 模型文件缺失导致构建/检索失败

`models/embedding` 中缓存了 `bge-small-zh-v1.5`；若被清理，需在可访问 HuggingFace 的网络环境中下载一次（或设置镜像 `HF_ENDPOINT=https://hf-mirror.com` 后执行构建命令）。
