# Local RAG Knowledge Assistant (本地化个人知识库助手)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![Vue3](https://img.shields.io/badge/Vue-3-4FC08D?logo=vue.js&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Community-green?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--R1-purple)

**Local RAG Knowledge Assistant** 是一个基于 **DeepSeek-R1** 大模型与 **ChromaDB** 向量数据库构建的**完全本地化**个人知识库系统。

上传私有文档（PDF, DOCX, TXT, MD, CSV），自动构建向量索引，通过自然语言与文档进行对话。**数据不出本地，隐私绝对安全。**

---

## ✨ 核心功能

- **🔒 100% 本地化运行** — 依托 LM Studio 运行 DeepSeek 模型，所有推理与存储均在本地完成
- **📚 多格式文档** — 支持 PDF, Word (.docx), Markdown, TXT, CSV 等格式的解析与入库
- **🧠 智能 RAG 引擎**
  - 父子文档索引策略（`Parent-Child Chunking`），提升检索上下文完整度
  - `Flash` 模式：纯向量检索，速度优先
  - `Pro` 模式：向量检索 + `bge-reranker-base` 精排，精度优先
- **🎯 意图路由** — 自动判断用户输入是知识检索还是闲聊，分别处理
- **💬 多轮会话** — SQLite 持久化存储会话历史，支持新建/切换/删除对话
- **⚡ 流式输出** — 实时展示 AI 的思考过程（`<think>` 标签）和最终回答
- **🖥️ 双前端**
  - **Vue3 + Naive UI**：现代化 Web 界面（推荐）
  - **Streamlit**：轻量级 Python 界面

---

## 🛠️ 技术栈

| 层级 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **LLM** | **LM Studio** + DeepSeek-R1 | 本地大模型推理服务 (端口 1234) |
| **后端 API** | **FastAPI** + Uvicorn | RESTful API，流式响应，异步桥接 |
| **前端 (推荐)** | **Vue 3** + Vite + Naive UI | 现代化单页应用，支持 Markdown 渲染 |
| **前端 (备选)** | **Streamlit** | 轻量级 Python Web 界面 |
| **RAG 引擎** | **LangChain** + Requests | 文档加载/切分 + 手写 LLM 直连（绕过代理） |
| **向量数据库** | **ChromaDB** | 轻量级本地向量库，无需 Server |
| **Embedding** | **all-MiniLM-L6-v2** | Sentence-Transformers (CPU) |
| **Reranker** | **bge-reranker-base** | Pro 模式精排 (可选) |
| **会话存储** | **SQLite** | 对话历史持久化 |

---

## 🏗️ 系统架构

```text
┌─────────────────────────────────────────────────────┐
│                    用户浏览器                         │
│  ┌───────────────┐         ┌─────────────────────┐  │
│  │ Vue3 + Vite   │   或    │    Streamlit UI     │  │
│  │ (localhost:5173)│        │  (localhost:8501)    │  │
│  └───────┬───────┘         └─────────┬───────────┘  │
└──────────┼───────────────────────────┼──────────────┘
           │ /api/*                    │ 直接调用
           ▼                           ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI 后端 (server.py :8000)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ 会话管理  │ │ 文件管理  │ │   流式聊天端点       │ │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────────┘ │
└───────┼────────────┼────────────────┼───────────────┘
        │            │                │
        ▼            ▼                ▼
┌────────────┐ ┌──────────┐ ┌─────────────────────────┐
│  SQLite    │ │ data/    │ │  RAG Engine (rag_core02) │
│ (chat_     │ │  docs/   │ │  ┌─────────────────────┐ │
│  history   │ │          │ │  │ 1. 意图路由 (CHAT/   │ │
│  .db)      │ │          │ │  │    SEARCH)           │ │
└────────────┘ └────┬─────┘ │  │ 2. 向量检索          │ │
                    │       │  │ 3. Reranker 精排     │ │
                    ▼       │  │ 4. Prompt 构建       │ │
              ┌──────────┐  │  │ 5. LLM 流式调用     │ │
              │ ingest.py│  │  └──────────┬──────────┘ │
              │ 文档入库  │  └─────────────┼───────────┘
              └────┬─────┘                │
                   ▼                      ▼
              ┌──────────┐         ┌─────────────┐
              │ ChromaDB │         │  LM Studio  │
              │ 向量数据库 │         │  :1234/v1   │
              └──────────┘         └─────────────┘
```

---

## 📂 项目结构

```text
Local_RAG_Assistant/
├── server.py                # FastAPI 后端 (API 入口)
├── app.py                   # Streamlit 前端 (备选)
├── requirements.txt         # Python 依赖
├── download_models.py       # Reranker 模型下载脚本
│
├── src/
│   ├── rag_core02.py        # RAG 引擎 (意图路由/检索/Rerank/LLM)
│   ├── ingest.py            # 文档加载/切分/入库 (父子索引)
│   └── database.py          # SQLite 会话管理
│
├── frontend/                # Vue3 前端
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatLayout.vue       # 主界面
│   │   ├── composables/
│   │   │   ├── useChat.ts           # 聊天逻辑
│   │   │   ├── useChatSessions.ts   # 会话管理
│   │   │   └── useKnowledgeBase.ts  # 知识库操作
│   │   └── types/chat.ts            # TS 类型定义
│   ├── package.json
│   └── vite.config.ts               # Vite 配置 (API 代理)
│
├── data/                    # [自动生成]
│   ├── docs/                # 上传的原始文档
│   └── chroma_db/           # 向量数据库 + parent_map.json
│
└── model_cache/             # [自动生成] Reranker 模型缓存
```

---

## 🚀 快速开始

### 1. 环境准备

> ⚠️ **推荐 Python 3.11**。3.14 存在 Pydantic v1 兼容性问题。

```bash
# 克隆项目
git clone https://github.com/your-username/local-rag-assistant.git
cd local-rag-assistant

# 创建虚拟环境 (Windows 指定 3.11)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 安装 Python 依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. 配置 LM Studio

1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 搜索并下载模型：`DeepSeek-R1-Distill-Qwen-14B`（推荐）或 `7B` 版本
3. 点击左侧 **Local Server (↔)** 图标
4. 重要设置：
   - 加载模型
   - 确保 Server Port 为 **`1234`**
   - 开启 **CORS**
   - 点击 **Start Server**

### 3. 下载 Reranker 模型（可选，Pro 模式需要）

```bash
.venv\Scripts\python.exe download_models.py
```

### 4. 启动

**终端 1 — 后端：**

```bash
.venv\Scripts\python.exe server.py
```

**终端 2 — 前端（Vue）：**

```bash
cd frontend
npm run dev
```

打开浏览器访问 `http://localhost:5173`。

---

## 📖 使用指南

1. **上传知识** — 在左侧侧边栏上传文档（PDF/DOCX/TXT/MD/CSV），点击 **"构建知识库"**
2. **选择模式** — `Flash`（快速向量检索）或 `Pro`（+ Reranker 精排）
3. **开始对话** — 输入问题，AI 自动判断是否需要检索知识库
4. **会话管理** — 新建/切换/删除对话，历史自动保存
5. **恢复出厂** — "重置" 一键清空所有文件与数据库（慎用）

---

## 🔧 常见问题

**Q: Python 3.14 报 Pydantic 兼容性错误？**
> 使用 `py -3.11 -m venv .venv` 创建 Python 3.11 虚拟环境即可解决。

**Q: 点击"重建"时报错 `WinError 32`？**
> Windows 文件锁问题，当前版本已修复（通过 ChromaDB API 逻辑清空，而非物理删除）。

**Q: 报错 `Connection error` 或 `APIConnectionError`？**
> 1. 检查 LM Studio Server 是否已启动
> 2. 确认端口为 1234
> 3. 代码已内置绕过系统代理逻辑，如仍有问题检查 VPN 设置

**Q: 上传大文件卡死？**
> Embedding 运行在 CPU 上，大文件建议切分后上传，或在代码中启用 GPU 加速。

---

## 🗓️ 未来规划

- [ ] GPU 加速支持（Embedding CUDA 加速）
- [ ] 表格深度解析（LlamaParse 优化复杂 PDF 表格）
- [ ] 混合检索（BM25 关键词 + 向量语义）
- [ ] 多用户支持

---

> **Note**: 本项目是一个学习型项目，旨在展示 Local RAG 的核心原理与工程实践。欢迎 Fork 和 Star！🌟