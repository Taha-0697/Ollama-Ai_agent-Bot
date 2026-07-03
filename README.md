# 🤖 Ollama AI Agent Bot

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Framework-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**Ollama AI Agent Bot** is a fully autonomous multi-agent AI system built with Python that uses **local LLMs (Ollama)** to convert natural language prompts into real working software projects.

It uses a **Manager + Sub-Agent architecture** to break tasks into steps, generate code, and automatically create or modify files inside a workspace.

This project is designed for **AI-assisted development, automation, and intelligent coding workflows**, fully running offline.

---

## 🚀 Key Capabilities

### 🧠 AI Intelligence
- Understands natural language prompts
- Converts requests into structured development plans
- Uses local LLMs (no API required)
- Multi-step reasoning via Manager agent

### 📁 Automation
- Creates folder structures automatically
- Generates full source code files
- Edits existing code intelligently
- Maintains workspace-aware context

### ⚙️ Multi-Agent System
- Manager Agent → task planning
- Sub-Agents → file/code execution
- Parallel execution support
- Follow-up task continuation

### 💾 Optimization
- Workspace caching system
- Token usage tracking
- Efficient context reuse
- Incremental planning system

### 💤 System Integration
- Prevents system sleep during execution
- OS notifications (Windows/macOS/Linux)
- Safe background processing

---

## 🖥️ User Interface (TUI - Textual)

### Features
- 📂 File explorer for workspace
- 🧭 Live agent execution dashboard
- 📜 Real-time logs
- ⚙️ Task lifecycle tracking
- 📊 Token usage monitoring
- 💾 Cache performance tracking
- 🔔 Notifications system
- 🧵 Multi-agent execution view

### Workflow

Queued → Planning → Execution → Review → Completed


### Controls
- Approve & Close task
- Add more work (continue task)
- Switch workspace
- Live prompt input

---

## 🧱 Architecture


User Prompt
↓
Manager Agent (LLM)
↓
Task Decomposition (JSON Plan)
↓
Sub-Agent Executor
↓
File System (Create/Edit)
↓
Cache Layer
↓
TUI Dashboard


---

## 🛠️ Tech Stack

- Python 3.10+
- Ollama (Local LLMs)
- LangChain
- Textual (TUI framework)
- AsyncIO
- OS system APIs
- hashlib (caching)


---

## ⚡ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Ollama-AI-Agent-Bot.git
cd Ollama-AI-Agent-Bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate environment:
- Linux / macOS
```bash
source venv/bin/activate
```
- Windows
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
🤖 Install Ollama

Download and install Ollama:

👉 https://ollama.com

Pull the required model:
```bash
ollama pull qwen2.5-coder:7b
```
▶️ Run the Project
```bash 
  python main.py
```

## 💡 Example Usage

### Input Prompt
```text
Create a React login page with authentication and routing
System Output
Project folders created automatically
React UI components generated
Routing configured
Code files written to workspace
Task waits for user approval
🔥 Features Summary
🧠 Fully autonomous AI coding agent
🏗️ Automated project scaffolding
⚡ Multi-agent execution system
📁 Workspace-aware file operations
💾 Smart caching for performance
🔄 Follow-up task support
💤 System sleep prevention
📊 Live execution monitoring
🔔 OS + terminal notifications
✅ Human-in-the-loop approval system
🔮 Future Improvements
🌐 Web-based dashboard (React / Next.js)
🔌 Plugin system (Git, Docker, APIs)
🧠 Long-term memory system
☁️ Cloud workspace synchronization
🎤 Voice command interface
🤝 Multi-user collaboration mode
📦 Dockerized deployment
📊 Advanced analytics dashboard
📁 Project Structure
src/
 ├── agents/
 ├── core/
 ├── ui/
 ├── workspace_cache.py
 ├── manager.py
 ├── sub_agents.py
 ├── main.py
⚠️ Notes
Optimized for Ollama models like qwen2.5-coder
Runs fully offline (no API keys required)
Requires sufficient RAM for local LLM execution
Best performance on SSD-based systems
📄 License

This project is licensed under the MIT License.
