# CSC Two-Agents Demo (LangGraph + Tools)

> Public demo repo by **LadiosSopp** · Two collaborating LLM agents (research→chart) with LangGraph.


A compact, production-friendly demo that orchestrates **two collaborating agents** to research a topic and produce an accompanying **chart**.

- **research_agent** uses web search (Tavily) to collect factual time-series data.
- **chart_agent** turns the facts into **Python code** (via a Python REPL tool) and **saves** a chart as `chart.png`.

> This repo is a sanitized and modernized rework of an internal prototype. Secrets have been removed and the code has been documented and parameterized.

## Features
- 🕵️ Research with [Tavily](https://tavily.com) (set `TAVILY_API_KEY`)
- 🧠 LLM (Azure OpenAI preferred; falls back to OpenAI)
- 🧰 Orchestration with [LangGraph](https://github.com/langchain-ai/langgraph)
- 📈 Automatic matplotlib chart saved to `chart.png`
- 🔒 No hard-coded secrets; `.env` support for local dev

## Quickstart

### 1) Clone and install
```bash
git clone https://github.com/LadiosSopp/csc-two-agents-demo.git
cd csc-two-agents-demo
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Configure environment
Create a `.env` from the template and fill in your keys:
```bash
cp .env.sample .env
# open .env and set values
```

**Azure OpenAI (preferred):**
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=YOUR_DEPLOYMENT
OPENAI_MODEL_NAME=gpt-4o
```

**or OpenAI:**
```
OPENAI_API_KEY=...
OPENAI_MODEL_NAME=gpt-4o-mini
```

**Search:**
```
TAVILY_API_KEY=...
```

### 3) Run
```bash
python app.py "China Steel Corporation Production Volume 2010–2020; generate and save a line chart."
# The chart will be saved to chart.png
```

> Tip: You can also set `QUESTION="..."` in `.env` and run without CLI args.

## How it works
- The graph starts at **research_agent** → may call Tavily to gather factual data.
- It passes a concise, structured summary to **chart_agent**.
- **chart_agent** writes Python code for pandas + matplotlib, executes it via a Python REPL tool, and **saves** `chart.png`.
- The run finishes when an agent replies with **`FINAL ANSWER`**.

## Security Notes
- Do **not** commit your `.env`. Use GitHub Actions secrets or your platform's secret manager.
- The Python REPL tool executes code; never expose it to untrusted users in production.

## License
MIT — see [LICENSE](LICENSE).
