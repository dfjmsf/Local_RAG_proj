# 🚀 智能 CRM 商业数据洞察平台 (Enterprise Edition)

> **基于 Vue3 + Flask + MySQL + 本地 AI 大模型的全栈数据分析系统**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Vue](https://img.shields.io/badge/Vue.js-3.x-green.svg)
![Python](https://img.shields.io/badge/Python-3.13-yellow.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 📖 项目简介

本项目是一款现代化的 **BI (商业智能) 数据分析平台**。它区别于传统的静态报表，集成了 **数据挖掘**、**机器学习预测**、**3D 沉浸式可视化** 以及 **生成式 AI 对话** 等前沿技术。

系统底层采用 MySQL 进行数据持久化，通过 Python 后端实现数据清洗与特征工程，前端采用赛博朋克风格的 UI 设计，支持多模态语音交互，旨在为企业提供深度的数据洞察能力。

---

## 🛠️ 核心技术栈 (Tech Stack)

### 前端 (Frontend)
*   **框架**: Vue 3 (Composition API) + Vite
*   **UI 库**: Element Plus (深度定制深色主题)
*   **可视化**: ECharts 5 + **ECharts GL (3D)** + LiquidFill (水球图)
*   **交互**: Web Speech API (原生语音识别与合成)
*   **特效**: HTML5 Canvas 粒子背景

### 后端 (Backend)
*   **框架**: Python 3.13 + Flask
*   **数据处理**: Pandas (清洗/聚合), NumPy
*   **AI 算法**: Scikit-learn (K-Means 聚类 / 逻辑回归预测)
*   **数据库 ORM**: SQLAlchemy + PyMySQL
*   **API 通信**: RESTful API + Axios

### 基础设施 & AI
*   **数据库**: MySQL 8.0
*   **容器化**: Docker + Docker Compose
*   **大模型**: **DeepSeek-R1** (via LM Studio / Ollama)

---

## ✨ 系统功能亮点

1.  **📊 高级可视化大屏**：集成桑基图、3D 散点图、动态水球图、雷达图等高级图表。
2.  **🎨 赛博朋克 UI**：支持 **Cyber / Light** 双主题一键热切换，包含霓虹呼吸光效与粒子背景。
3.  **🤖 ChatBI 智能助手**：
    *   基于本地部署的大模型 (DeepSeek-R1)。
    *   支持 **语音交互**：按住说话提问，AI 分析后自动语音播报。
    *   能读取实时数据库指标进行分析。
4.  **🧊 AI 客户分群 (3D)**：利用 K-Means 算法对客户进行无监督聚类，并在 3D 空间中可视化展示。
5.  **🚀 购买行为预测**：基于逻辑回归模型，预测单个客户的购买转化率。
6.  **🎛️ 全局多维透视**：侧边栏集成地区/性别/设备三级联动筛选，一键穿透所有页面数据。
7.  **💾 完整数据流**：支持从 CSV 迁移数据到 MySQL，支持数据的增删改查 (CRUD)。

---

## 💻 软件安装与环境要求

在运行项目之前，请确保安装以下软件：

### 必选软件
1.  **Docker Desktop** (推荐方式，一键运行)
    *   *下载地址:https://www.docker.com/products/docker-desktop/
    *   *注意：Windows 用户安装时请勾选 "Use WSL 2 instead of Hyper-V"*。
2.  **LM Studio** (用于运行本地 AI 模型)
    *   下载地址: [https://lmstudio.ai/](https://lmstudio.ai/)

### 可选软件 (如果你想本地源码开发)
1.  **Node.js**: v18 或 v20+(最好20+,不然可能会遇到兼容性问题)
2.  **Python**: 3.10+
3.  **MySQL**: 8.0 (如果不使用 Docker 内置数据库)
4.  **IDE**: PyCharm 或 VS Code

---

## 🚀 快速启动指南 (Docker 方式 - 推荐)

这是最简单的运行方式，无需配置本地 Python 和 Node 环境。

### 1. 准备 AI 模型 (LM Studio)
1.  打开 **LM Studio**。
2.  搜索并下载模型：`DeepSeek-R1-Distill-Qwen-14B` (推荐 Q6_K 或 Q8_0 量化版本)。
    *   *AMD 显卡用户请在设置中开启 ROCm 或 Vulkan 加速。*
3.  点击左侧 **Local Server (<->)** 图标。
4.  选择模型，确保 GPU Offload 拉到最大。
5.  点击 **Start Server**，确保端口为 `1234`。

### 2. 启动项目容器
在项目根目录下打开终端 (CMD / PowerShell)，运行：

```bash
docker-compose up --build
```
等待几分钟，直到看到控制台输出 Running on http://0.0.0.0:5000 和 Local: http://localhost:5173。
3. 初始化数据库 (仅首次运行需要)
保持上面的终端开启，新建一个终端窗口，运行以下命令将初始数据导入 MySQL：
```bash
docker-compose exec backend python migrate_csv_to_db.py
```
看到 "成功！...表已创建" 字样即完成。
4. 访问系统
打开浏览器访问：http://localhost:5173
## 🔧 本地源码开发指南 (非 Docker)
如果你需要修改代码并进行调试：
1. 数据库配置
在本地 MySQL 中新建数据库 crm_db。
修改 database.py 中的 DB_URI，填入你的 MySQL 账号密码。
2. 后端启动
```bash
# 安装依赖
pip install -r requirements.txt

# 初始化数据
python migrate_csv_to_db.py

# 启动 Flask
python app.py
```
3. 前端启动
```bash
cd crm-frontend
npm install
npm run dev
```
## ⚠️ 运行注意事项 (Troubleshooting)
1. AI 助手无法连接？
Docker 模式下：项目通过 host.docker.internal:1234 访问宿主机。请确保 LM Studio 的 Server 已经启动 (绿色按钮)。
本地模式下：项目通过 localhost:1234 访问。
防火墙：请确保 Windows 防火墙允许 LM Studio 接受连接。
2. 语音交互没反应？
浏览器限制：Web Speech API 仅在 Chrome、Edge 或 Safari 上支持良好。Firefox 可能不支持。
麦克风权限：首次点击麦克风时，请允许浏览器访问麦克风。
无声音：请检查系统是否安装了中文语音包 (Microsoft Huihui / Google 普通话)。
3. 数据库报错？
如果在 Docker 中启动报错 Access denied 或 Can't connect，请尝试运行 docker-compose down -v 清理卷后重试。
确保 3306 端口没有被本地 MySQL 占用，Docker 配置默认映射到了 3307。
4. 3D 图表显示不全？
这是 ECharts GL 的视口问题，已在代码中通过 resize 强制修复。如果仍有问题，尝试缩放浏览器窗口即可触发重绘。
## 📂 项目结构说明

```
Project_Root/
├── app.py                  # Flask 后端核心入口 (API, AI接口)
├── database.py             # 数据库连接与 ORM 模型定义
├── enhance_data.py         # 数据增强脚本 (生成模拟数据)
├── migrate_csv_to_db.py    # 数据迁移脚本 (CSV -> MySQL)
├── docker-compose.yml      # Docker 编排配置
├── Dockerfile.backend      # 后端镜像构建文件
├── Dockerfile.frontend     # 前端镜像构建文件
├── requirements.txt        # Python 依赖清单
├── datactv/                # 原始数据文件夹
└── crm-frontend/           # Vue3 前端项目
    ├── src/
    │   ├── api/            # Axios 封装
    │   ├── assets/         # 静态资源 (theme.css 在这里)
    │   ├── components/     # 公共组件 (ChatAssistant, ParticleBackground)
    │   ├── layout/         # 布局文件 (MainLayout - 侧边栏与头部)
    │   ├── utils/          # 工具类 (filterState - 全局状态)
    │   └── views/          # 页面视图 (Home, Customer, Cluster...)
    └── ...
```
## 👨‍💻 作者与版权
**Designed & Developed by qin-shun-yang** \
**Copyright © 2025. All Rights Reserved.**
