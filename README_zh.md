# AutoWebWorld

<div align="center">

[English](./README.md) | [简体中文](./README_zh.md)

</div>

## 📖 简介

AutoWebWorld 是一个用于自动化Web应用建模、遍历和智能Agent训练的开源框架。本项目提供了从有限状态机(FSM)生成到Agent训练的完整工具链，帮助研究者和开发者构建和评估Web自动化Agent。

## 📰 最新动态

- **[2026-02]** 🎉 项目初始化，发布 v0.1.0 版本
- **[2026-02]** ✨ 完成 FSM 生成器核心功能
- **[2026-02]** 🚀 添加 BFS 遍历模块
- **[2026-02]** 🤖 集成 Agent 训练 pipeline

> 💡 **提示**: 关注本项目获取最新更新！

## ✨ 核心功能

### 🔄 FSM生成器
- 基于主题自动生成Web应用的有限状态机
- 支持复杂的页面状态和动作建模
- 内置验证和改进机制，确保FSM质量

### 🤖 Agent训练
- 提供完整的Agent训练pipeline
- 支持多种训练策略和算法
- 可扩展的训练框架

### 🌐 BFS遍历与Web示例
- 基于BFS算法遍历FSM生成轨迹
- 包含多个真实Web应用示例
- 支持轨迹可视化和分析

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/[your-username]/AutoWebWorld.git
cd AutoWebWorld
pip install -r requirements.txt
```

### 基本使用

#### 1. 生成FSM
```bash
cd fsm_generator
python -m fsm_generator.fsm --theme "Your_Theme" --model "gpt-4" --output_dir "outputs"
```

#### 2. BFS遍历
```bash
cd bfs_traversal
python normalize.py --input fsm.json --output fsm_norm.json
python bfs_action.py --fsm fsm.json --norm fsm_norm.json --out trajectories.json
```

#### 3. 训练Agent
```bash
cd agent_training
python train.py --config config.yaml
```

## 📂 项目结构

```
AutoWebWorld/
├── fsm_generator/      # FSM生成模块
├── agent_training/     # Agent训练模块
├── bfs_traversal/      # BFS遍历模块
└── examples/           # Web应用示例
```

## 📚 详细文档

- [FSM生成器文档](./fsm_generator/README.md)
- [Agent训练文档](./agent_training/README.md)
- [BFS遍历文档](./bfs_traversal/README.md)
- [示例说明](./examples/README.md)

## 🛠️ 技术栈

- Python 3.8+
- OpenAI API / 其他LLM API
- Playwright (用于Web自动化)
- Vue.js (Web示例)

## 📊 示例应用

项目包含多个领域的Web应用示例：
- 电商平台 (Amazon, AliExpress)
- 生产力工具 (Asana, Notion)
- 社交媒体 (Discord, Twitter)
- 旅游预订 (Booking, Skyscanner)
- 更多...

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 [MIT License](./LICENSE)

## 📧 联系方式

- 项目主页: [GitHub链接]
- 问题反馈: [Issues链接]
- 邮箱: [your-email]

## 📖 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{autowebworld2024,
  title={AutoWebWorld: An Open Framework for Web Automation and Agent Training},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/AutoWebWorld}
}
```

## 🙏 致谢

[感谢的人或项目]

