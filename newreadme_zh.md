# AutoWebWorld

<div align="center">

[English](./README.md) | [简体中文](./README_zh.md)

</div>

## 📖 项目介绍

<div align="center">
  <img src="./assets/main_figure.png" alt="AutoWebWorld Main Figure" width="800"/>
</div>

AutoWebWorld 是一个用于**自动化 Web 环境生成与 GUI Agent 训练**的开源框架。给定一个 Web 应用主题，框架自动生成有限状态机（FSM）来编码所有页面、操作和交互逻辑，再将其编译为可运行的 React Web 应用、通过 Playwright 回放产生的具体轨迹，以及用于训练视觉语言模型的训练数据。

核心理念：**数据生成是编译问题，而非标注问题**。FSM 是唯一的真实来源，其他一切都从它派生。

## 📰 更新日志

- **[2026-03-19]** 公开论文中使用的训练数据：[HuggingFace Dataset](https://huggingface.co/datasets/Evanwu50020/travel_media_commerce_communication_productivity/settings)
- **[2026-03-18]** 新增 `autowebworld/` Web 合成模块：通过多轮 coding agent 将 FSM 编译为 React UI
- **[2026-02-10]** 新增 GRPO 训练模块，支持视觉语言模型训练
- **[2026-02-10]** 完成 BFS 遍历模块迁移
- **[2026-02-10]** 完成 FSM 生成器模块迁移
- **[2026-02-06]** 项目初始化，发布 v0.1.0

## 🏗️ 系统架构

```
主题输入（如"电商平台"）
      │
      ▼
┌─────────────────────────────────────────┐
│  第一层：FSM 生成                        │  trajectory/fsm/
│  LLM 生成 FSM → 验证 → 迭代优化          │
└─────────────────────────┬───────────────┘
                          │  FSM (JSON)
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  第二层：             │  │  第三层：Web 合成              │
│  BFS 遍历            │  │  FSM → React/Vite Web 应用     │
│  → 抽象轨迹          │  │  通过多轮 coding agent 生成     │
│                      │  └──────────────┬────────────────┘
└──────────┬───────────┘                 │
           │  抽象轨迹                   │  运行中的 Web 应用
           └──────────────┬─────────────┘
                          ▼
┌─────────────────────────────────────────┐
│  第四层：轨迹回放                        │  autowebworld/whole_pipeline/
│  Playwright 在真实 Web 应用上            │
│  执行抽象轨迹，逐步截图                  │
└─────────────────────────┬───────────────┘
                          │  （截图，操作）数据对
                          ▼
┌─────────────────────────────────────────┐
│  第五层：Agent 训练                      │  training/
│  基于截图 + 操作序列的 GRPO 训练         │
└─────────────────────────────────────────┘
```

## ✨ 模块详解

### 🎯 第一层 — FSM 生成（`trajectory/fsm/`）

使用 LLM 自动生成高质量的 Web 应用有限状态机。流程分三步：`FSMGeneratorAgent` 并行生成 4–16 个候选 FSM，`FSMValidatorAgent` 通过 BFS 可达性分析对每个候选打分，`FSMImproveAgent` 迭代优化最优候选直至达到目标分数。

复杂度由 `trajectory/fsm/generator/profiles/` 中的配置文件控制：
- **easy**：10–15 个页面，3–5 个操作/页，基础导航
- **medium**：15–25 个页面，4–6 个操作/页，含过滤器、权限、搜索
- **hard**：25–35 个页面，5–7 个操作/页，多路径导航、拦截器

每个 FSM 包含：带 `signature_schema`（页面数据状态）的页面定义，带 `preconditions`（前置条件）和 `effects`（效果）的操作定义，以及指定底层浏览器操作（点击、输入、悬停、拖拽）的 `gui_procedure`。

```bash
# 使用默认配置运行（medium 档，主题：blog_platform-medium）
bash trajectory/fsm/generator/run.sh

# 自定义主题、输出目录、并发数和复杂度配置
FSM_THEME="ecommerce-shopify" \
bash trajectory/fsm/generator/run.sh \
    trajectory/fsm/generator/outputs/my_fsm \
    16 \
    trajectory/fsm/generator/profiles/hard.json
```

### 🔄 第二层 — BFS 遍历（`trajectory/bfs/`）

枚举 FSM 中所有合法的交互路径，并将其实例化为抽象轨迹。FSM 首先被标准化为边列表格式，然后 BFS 从 HOME 页出发遍历至所有 terminal 页面，通过 `(page_id, signature_hash)` 去重。`{ITEM_ANY}`、`{search_query}` 等占位符被填充为具体值，生成可执行的操作序列。过滤器组合通过 `split_filters.py` 单独枚举。

```bash
cd autowebworld/whole_pipeline/env_generator/bfs_generator

FSM=path/to/fsm.json

python split_filters.py --input $FSM            --output fsm_split.json
python normalize.py     --input fsm_split.json  --output fsm_norm.json
python bfs_action.py    --fsm fsm_split.json    --norm fsm_norm.json --out allshortest.json
python gui_mapping.py   --fsm fsm_split.json    --bfs allshortest.json --out bfs_mapping/
```

### 🌐 第三层 — Web 合成（`autowebworld/`）

通过多轮 LLM coding agent 将 FSM 编译为可运行的 React/Vite Web 应用。以基础 Vite+React 模板为起点，`react_coding_agent.py` 反复修改代码库以匹配 FSM，并在每轮之间将 agent 状态持久化到 `memory.json`。生成的应用保存在 `autowebworld/web_outputs/` 下。

```bash
# 设置 FSM 路径并启动 coding agent
FSM_PATH=trajectory/fsm/generator/outputs/my_fsm/fsm.json \
bash autowebworld/run_react_coding_agent.sh
```

关键环境变量：`FSM_PATH`、`MODEL`（默认：`gemini-3.1-pro-preview`）、`MAX_TURNS`（默认：50）、`OPENAI_API_KEY`。

### 🎬 第四层 — 轨迹回放（`autowebworld/whole_pipeline/`）

使用 Playwright 在真实 Web 应用上执行抽象 BFS 轨迹，逐步截图，生成具体的 `(观察, 操作)` 数据对作为训练数据。同时处理数据项扩展（LLM 增强 mock 数据）、视觉查询生成和 QA 数据对生成。

```bash
export OPENAI_API_KEY="your-key-here"

cd autowebworld/whole_pipeline
bash whole_run.sh
```

通过编辑 `whole_run.sh` 顶部的 `PROJECT_LISTS` 来选择要运行的项目。输出写入 `whole_pipeline/new_outputs/<category>/<project>/`。

### 🤖 第五层 — Agent 训练（`training/`）

使用 **GRPO**（广义奖励策略优化）在收集的轨迹上训练视觉语言模型。默认模型为 Qwen2VL，奖励函数同时评估答案准确性和输出格式。通过 DeepSpeed Zero-3 支持多卡分布式训练。

```bash
# 将训练数据放置在 data/train_data/ 下：
#   train.json       — 轨迹数据
#   train_imgs/      — 截图图片

# 将基础模型 checkpoint 放置在 models/Qwen2.5-VL-3B/ 下

bash training/scripts/train.sh
```

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目并安装 Python 依赖
git clone https://github.com/your-username/AutoWebWorld.git
cd AutoWebWorld
pip install -r requirements.txt

# 安装 pnpm（Web 项目必需）
npm install -g pnpm

# 安装 Playwright 浏览器
playwright install chromium
```

### 使用示例项目运行完整 Pipeline

项目内置 **30 个预构建 Web 应用**，覆盖 5 个领域，开箱即用：

| 分类 | 项目 |
|------|------|
| 电商与金融 | aliexpress, JD_COM, klaviyo, revolut, shopify, walmart |
| 通讯与社交 | facebook, medium, microsoft_teams, outlook, quora, signal, slack, twitter, zoom |
| 媒体、学习与健康 | coursera, headspace, health, Spotify, tumblr, youtube |
| 生产力与开发运维 | airtable, asana, bitbucket, freshdesk, github, onenote, Optimizely |
| 旅行与房产 | skyscanner |

**第一步 — 设置 API Key：**
```bash
export OPENAI_API_KEY="your-key-here"
```

**第二步 — 选择要运行的分类。** 打开 `autowebworld/whole_pipeline/whole_run.sh`，设置 `PROJECT_LISTS`：

```bash
PROJECT_LISTS=(
    # "travel_real_estate_projects.txt"
    # "media_learning_wellness_projects.txt"
    # "commerce_finance_projects.txt"
    # "communication_social_projects.txt"
    "productivity_projects.txt"     # ← 当前激活
)
```

**第三步 — 运行：**
```bash
cd autowebworld/whole_pipeline
bash whole_run.sh
```

Pipeline 对每个项目依次执行 6 个阶段：自动修补组件、页面截图、BFS 遍历与数据项扩展、Playwright 轨迹回放、视觉查询生成、QA 数据对生成。输出写入 `autowebworld/whole_pipeline/new_outputs/<category>/<project>/`。

## 📂 项目结构

```
AutoWebWorld/
├── trajectory/
│   ├── fsm/
│   │   ├── generator/
│   │   │   ├── fsm.py                  # 编排器（生成 → 验证 → 优化）
│   │   │   ├── base_agent.py           # LLM Agent 基类
│   │   │   ├── fsm_generator_agent.py  # 初始 FSM 生成
│   │   │   ├── fsm_validator_agent.py  # 打分与验证
│   │   │   ├── fsm_improve_agent.py    # 迭代优化
│   │   │   ├── profiles/               # easy / medium / hard 配置
│   │   │   └── run.sh                  # CLI 入口
│   │   └── prompts/                    # LLM 提示词模板
│   └── bfs/
│       ├── bfs.py                      # BFS 核心算法
│       ├── bfs_action.py               # 轨迹实例化
│       ├── normalize.py                # FSM 标准化
│       ├── gui_mapping.py              # 操作 → GUI 映射
│       └── split_filters.py            # 过滤器组合枚举
├── autowebworld/
│   ├── fsm_to_web_agent.py             # FSM → React 应用入口
│   ├── react_coding_agent.py           # 多轮 coding agent
│   ├── run_react_coding_agent.sh       # Web 合成 CLI 入口
│   ├── memory_store.py                 # Agent 状态持久化
│   ├── template/react_template/        # 基础 Vite+React 模板
│   ├── web_outputs/                    # 生成的 Web 应用
│   └── whole_pipeline/                 # Playwright 回放 + 数据增强
│       ├── whole_run.sh                # 端到端 Pipeline 入口
│       ├── env_generator/              # BFS 生成器 + 数据增强
│       └── web_extractor/              # Grounding 提取（Playwright）
├── training/
│   ├── grpo_train.py                   # GRPO 训练入口
│   ├── trainer/                        # 自定义 GRPOTrainer
│   ├── scripts/train.sh                # 训练启动脚本
│   └── configs/zero3.json              # DeepSpeed Zero-3 配置
├── data/                               # 训练数据与输出
├── requirements.txt
└── README.md
```

## 🛠️ 技术栈

- **LLM API**：OpenAI 兼容接口（GPT、Claude、Gemini、DeepSeek）
- **Web**：React、Vite、Playwright
- **训练**：PyTorch、Transformers、TRL、DeepSpeed Zero-3
- **优化**：Flash Attention 2、Liger Kernel、bitsandbytes
- **实验追踪**：Weights & Biases、TensorBoardX
- **工具库**：Pydantic、PyYAML、Rich、python-dotenv

## 🚧 TODO

- [ ] 评测框架：面向 Web 自动化 Agent 的基准测试与评估指标
- [ ] 公开数据集：发布跨领域的预生成 FSM 与轨迹数据

## 🤝 贡献指南

欢迎贡献代码！请先提 Issue 说明你的改动方向，再提交 Pull Request。

## 📄 开源协议

[MIT License](./LICENSE)

## 📖 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{autowebworld2026,
  title  = {AutoWebWorld: An Open Framework for Web Environment Generation and GUI Agent Training},
  year   = {2026},
  url    = {https://github.com/your-username/AutoWebWorld}
}
```
