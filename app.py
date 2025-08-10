#!/usr/bin/env python3
"""
CSC Two-Agents Demo (LangGraph + Tools)
---------------------------------------
A minimal, production-friendly rework of your original demo that:
- Removes hard-coded secrets (use .env or environment variables)
- Adds clear docstrings and inline comments
- Improves structure and error handling
- Makes the "prompt" configurable via CLI arg or environment variable
- Instructs the chart agent to *save* a PNG (chart.png) so it can be shared

Agents:
1) research_agent  — can browse (Tavily) to collect facts.
2) chart_agent     — uses a Python REPL tool to generate a chart based on the facts.

Orchestration:
- A simple two-node graph that loops until "FINAL ANSWER" appears.
"""
import os
import sys
import operator
import functools
from typing import Annotated, Sequence, TypedDict, Literal

# Optional: load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


# -------------------------
# Model factory (Azure or OpenAI)
# -------------------------
def make_chat_model():
    """
    Prefer Azure OpenAI if AZURE_* envs exist; otherwise use OpenAI.
    Required envs (Azure):
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_DEPLOYMENT_NAME
    Fallback (OpenAI):
        OPENAI_API_KEY
        OPENAI_MODEL_NAME (e.g., "gpt-4o-mini" or "gpt-4o")
    """
    azure_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    if all(os.getenv(k) for k in azure_vars):
        # Azure
        deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        return AzureChatOpenAI(
            deployment_name=deployment_name,
            model_name=model_name,
            temperature=temperature,
        )
    else:
        # OpenAI
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        return ChatOpenAI(model_name=model_name, temperature=temperature)


# -------------------------
# LangGraph State
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# -------------------------
# Tools
# -------------------------
# Search tool (make sure TAVILY_API_KEY is set in env)
tavily_tool = TavilySearchResults(max_results=5)

# Warning: Executes code locally; DO NOT expose to untrusted inputs in production
repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The Python code to execute to generate a chart. Always save your chart as 'chart.png' and print 'SAVED: chart.png' on success."]
):
    """Execute Python code in a sandboxed REPL.

    If you want to see the value of something, you must print it.
    Always save charts to 'chart.png' and print 'SAVED: chart.png' when done.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\\n```python\\n{code}\\n```\\nStdout: {result}"
    return result_str + "\\n\\nIf you have completed all tasks, respond with FINAL ANSWER."


# -------------------------
# Agent builders
# -------------------------
def create_agent_with_tools(llm, tools, system_message: str):
    """Create an agent that can call tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant collaborating with another assistant.\n"
                    "Use the provided tools when beneficial. If you have the final answer or deliverable, "
                    "prefix your message with 'FINAL ANSWER'. You have access to: {tool_names}.\n"
                    "{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


def create_agent_no_tools(llm, system_message: str):
    """Create an agent with NO tools (rare in this demo, kept for completeness)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant collaborating with another assistant.\n"
                    "If you have the final answer or deliverable, prefix your message with 'FINAL ANSWER'.\n"
                    "{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm


def agent_node(state, agent, name):
    """Wrap an agent to a LangGraph node."""
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """Simple router: call tools if requested; stop if 'FINAL ANSWER' present."""
    last = state["messages"][-1]
    if last.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last.content:
        return "__end__"
    return "continue"


def build_graph():
    """Build the two-agent LangGraph workflow."""
    llm = make_chat_model()

    # Agent 1: research_agent — can browse for facts (Tavily)
    research_agent = create_agent_with_tools(
        llm,
        [tavily_tool],
        system_message=(
            "Research facts needed to answer the user's question. "
            "Cite specific numbers/ranges and identify trustworthy sources. "
            "Summarize key data points that the charting agent can plot."
        ),
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="research_agent")

    # Agent 2: chart_agent — uses Python REPL to produce a chart (chart.png)
    chart_agent = create_agent_with_tools(
        llm,
        [python_repl],
        system_message=(
            "Transform the research summary into executable Python code that:\n"
            "1) Builds a pandas DataFrame with the key time series data\n"
            "2) Plots a clear matplotlib line chart\n"
            "3) Saves it as 'chart.png' (no display)\n"
            "4) Prints 'SAVED: chart.png' when done\n"
            "After successfully saving, reply with 'FINAL ANSWER' and a one-paragraph caption."
        ),
    )
    chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_agent")

    # Tools node
    tools_node = ToolNode([tavily_tool, python_repl])

    # Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("chart_agent", chart_node)
    workflow.add_node("call_tool", tools_node)

    workflow.add_conditional_edges(
        "research_agent",
        router,
        {"continue": "chart_agent", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "chart_agent",
        router,
        {"continue": "research_agent", "call_tool": "call_tool", "__end__": END},
    )
    workflow.set_entry_point("research_agent")
    return workflow.compile()


def run(question: str):
    """Run the graph on the provided user question."""
    graph = build_graph()
    events = graph.stream(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": int(os.getenv("RECURSION_LIMIT", "80"))},
    )
    final_text = None
    for step in events:
        # Uncomment to debug each event:
        # print(step, "\\n----")
        # Capture final answer if present
        msg = step.get("messages", [None])[-1] if step else None
        if msg and hasattr(msg, "content") and isinstance(msg.content, str) and "FINAL ANSWER" in msg.content:
            final_text = msg.content

    if final_text:
        print("\\n=== FINAL ANSWER ===")
        # Strip the leading tag for clean printing
        print(final_text.replace("FINAL ANSWER", "").strip())
    else:
        print("Completed without an explicit FINAL ANSWER tag.")


if __name__ == "__main__":
    default_q = (
        os.getenv("QUESTION")
        or "China Steel Corporation Production Volume 2010–2020; generate and save a line chart."
    )
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_q
    run(q)
