# ⚖️ NyayaSetu — Justice Bridge

**NyayaSetu** is a multimodal legal intelligence platform that bridges the gap between **static legal statutes** (IPC / BNS) and **dynamic judicial reality** (Supreme Court judgments, stays, and amendments).

> 🚀 **Built for Convolve 4.0 - Pan IIT AI/ML Hackathon**

🟢 **Live Deployment:** [https://legal-advisory-counsel.streamlit.app/](https://legal-advisory-counsel.streamlit.app/)

---

## 📚 Documentation

We have comprehensive documentation available for developers and evaluators:

### 1️⃣ [Replication & Setup Guide](./REPLICATION.md)
*   **Step-by-step installation** instructions.
*   **Database setup** (Qdrant).
*   **Running the application** locally.

### 2️⃣ [System Architecture](./ARCHITECTURE.md)
*   **Inner workings** of the Multi-Agent System.
*   **Agent Roles**: Legal Clerk, Amendment Watchdog, Evidence Auditor, Senior Counsel.
*   **Data Flow** diagrams.

---

## 🏗️ Quick Overview

NyayaSetu runs on a **LangGraph state machine**, orchestrating four specialized agents to solve the problem of hallucinations in Legal AI.

![Agentic Orchestrator](./Agentic%20Orchestrator.png)

| Agent | Function | Technology |
| :--- | :--- | :--- |
| **Legal Clerk** | Retrieves Static Laws | Qdrant (Hybrid Search) |
| **Amendment Watchdog** | Verifies Live Judgments | DuckDuckGo Search |
| **Evidence Auditor** | Analyzes Video/Audio Evidence | Gemini 2.5 Flash |
| **Senior Counsel** | Synthesizes Final Advice | Gemini 2.5 Flash |

---

## 🚀 Key Features

*   **✅ Truth-Grounded**: Prioritizes Supreme Court judgments over outdated statutes.
*   **📹 Multimodal**: Can watch crime footage (video) and listen to audio recordings.
*   **⚡ Real-Time**: Checks the web for laws changed *today*.
*   **🛡️ Privacy**: Optional Incognito Mode (no data caching).

---

## 🛠️ Tech Stack

*   **Orchestration**: LangGraph, LangChain
*   **Frontend**: Streamlit
*   **Vector Database**: Qdrant (Dense + Sparse embeddings)
*   **LLM / VLM**: Google Gemini 2.5 Flash
*   **Search**: DuckDuckGo (DDGS)

---

## 🤝 Project Structure

```text
LEGAL-ADVISORY-MULTI-AGENT-SYSTEM/
├── src/
│   ├── agents.py           # Core Logic & Agents
│   ├── app.py              # Frontend UI
│   ├── ingest_data.py      # Dataset Loader
│   └── setup_qdrant.py     # DB Initializer
├── REPLICATION.md          # 👈 Start Here
├── ARCHITECTURE.md         # 👈 Technical Details
└── requirements.txt
```

---

