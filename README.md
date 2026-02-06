# AutoWebWorld

<div align="center">

[English](./README.md) | [简体中文](./README_zh.md)

</div>

## 📖 Introduction

AutoWebWorld is an open-source framework for automated web application modeling, traversal, and intelligent agent training. This project provides a complete toolchain from Finite State Machine (FSM) generation to agent training, helping researchers and developers build and evaluate web automation agents.

## 📰 News
- **[2026-02-06]** ✨ Completed FSM generator core functionality
- **[2026-02-06]** 🎉 Project initialized, released v0.1.0. We put the FSM generation pipeline of autowebworld with the

> 💡 **Tip**: Follow this project for the latest updates!

## ✨ Core Features

### 🔄 FSM Generator
- Automatically generate FSMs for web applications based on themes
- Support complex page state and action modeling
- Built-in validation and improvement mechanisms to ensure FSM quality

### 🤖 Agent Training
- Complete agent training pipeline
- Support multiple training strategies and algorithms
- Extensible training framework

### 🌐 BFS Traversal & Web Examples
- Traverse FSM using BFS algorithm to generate trajectories
- Include multiple real-world web application examples
- Support trajectory visualization and analysis

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/[your-username]/AutoWebWorld.git
cd AutoWebWorld
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate FSM
```bash
cd fsm_generator
python -m fsm_generator.fsm --theme "Your_Theme" --model "gpt-4" --output_dir "outputs"
```

#### 2. BFS Traversal
```bash
cd bfs_traversal
python normalize.py --input fsm.json --output fsm_norm.json
python bfs_action.py --fsm fsm.json --norm fsm_norm.json --out trajectories.json
```

#### 3. Train Agent
```bash
cd agent_training
python train.py --config config.yaml
```

## 📂 Project Structure

```
AutoWebWorld/
├── fsm_generator/      # FSM generation module
├── agent_training/     # Agent training module
├── bfs_traversal/      # BFS traversal module
└── examples/           # Web application examples
```

## 📚 Documentation

- [FSM Generator Documentation](./fsm_generator/README.md)
- [Agent Training Documentation](./agent_training/README.md)
- [BFS Traversal Documentation](./bfs_traversal/README.md)
- [Examples Documentation](./examples/README.md)

## 🛠️ Tech Stack

- Python 3.8+
- OpenAI API / Other LLM APIs
- Playwright (for web automation)
- Vue.js (web examples)

## 📊 Example Applications

The project includes web application examples from multiple domains:
- E-commerce platforms (Amazon, AliExpress)
- Productivity tools (Asana, Notion)
- Social media (Discord, Twitter)
- Travel booking (Booking, Skyscanner)
- And more...

## 🤝 Contributing

Contributions are welcome! Please check [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the [MIT License](./LICENSE)

## 📧 Contact

- Project Homepage: [GitHub Link]
- Issue Tracker: [Issues Link]
- Email: [your-email]

## 📖 Citation

If this project helps your research, please cite:

```bibtex
@software{autowebworld2024,
  title={AutoWebWorld: An Open Framework for Web Automation and Agent Training},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/AutoWebWorld}
}
```

## 🙏 Acknowledgments

[People or projects to thank]