"""
AI Data Analyst Agent
LangGraph pipeline: Question → Plan → Generate Code → Execute → Visualize → Explain
Takes natural language questions about a dataset and answers them with analysis + charts.
"""

import os
import re
import json
import sys
import io
import contextlib
import traceback
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2


# ─── State ────────────────────────────────────────────────────────────────────

class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    dataset_path: str
    df_schema: Optional[dict]
    plan: Optional[dict]
    generated_code: Optional[str]
    execution_result: Optional[dict]
    chart_path: Optional[str]
    needs_chart: bool
    answer: Optional[str]
    retry_count: int
    errors: List[str]
    status: str


# ─── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.1):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=4096,
        api_key=api_key,
    )


def call_llm_json(system: str, user: str, temperature=0.1) -> dict:
    llm = get_llm(temperature)
    response = llm.invoke([
        SystemMessage(content=system + "\n\nRESPOND ONLY WITH VALID JSON. No markdown, no backticks."),
        HumanMessage(content=user),
    ])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def call_llm_text(system: str, user: str, temperature=0.2) -> str:
    llm = get_llm(temperature)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


# ─── Node 1: Load schema ──────────────────────────────────────────────────────

_df_cache = {}

def get_df(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        ext = path.split(".")[-1].lower()
        if ext == "csv":
            _df_cache[path] = pd.read_csv(path)
        elif ext in ("xlsx", "xls"):
            _df_cache[path] = pd.read_excel(path)
        elif ext == "json":
            _df_cache[path] = pd.read_json(path)
        elif ext == "parquet":
            _df_cache[path] = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    return _df_cache[path]


def load_schema_node(state: AnalystState) -> AnalystState:
    logger.info("📥 Loading dataset schema...")
    try:
        df = get_df(state["dataset_path"])
        schema = {
            "shape": list(df.shape),
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "sample_values": df[col].dropna().head(3).tolist(),
                    "nunique": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                }
                for col in df.columns
            ],
        }
        state["df_schema"] = schema
        state["status"] = "schema_loaded"
        state["messages"].append(AIMessage(content=f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns"))
    except Exception as e:
        state["errors"].append(f"Schema load error: {e}")
        state["status"] = "error"
    return state


# ─── Node 2: Plan the analysis ────────────────────────────────────────────────

def plan_node(state: AnalystState) -> AnalystState:
    logger.info(f"🧠 Planning analysis for: {state['question']}")
    schema = state["df_schema"]

    cols_desc = "\n".join(
        f"- {c['name']} ({c['dtype']}, {c['nunique']} unique values, sample: {c['sample_values']})"
        for c in schema["columns"]
    )

    try:
        plan = call_llm_json(
            system="""You are a senior data analyst planning how to answer a question about a dataset using pandas.
Think about what computation is needed and whether a chart would help communicate the answer.""",
            user=f"""Dataset has {schema['shape'][0]} rows and these columns:
{cols_desc}

QUESTION: {state['question']}

Return JSON:
{{
  "interpretation": "restate what the user is asking in analytical terms",
  "approach": "brief description of the pandas operations needed (groupby, filter, agg, etc.)",
  "needs_chart": true/false,
  "chart_type": "bar|line|scatter|pie|histogram|none",
  "relevant_columns": ["col1", "col2"]
}}""",
        )
        state["plan"] = plan
        state["needs_chart"] = plan.get("needs_chart", False)
        state["status"] = "planned"
        state["messages"].append(AIMessage(content=f"🧠 Plan: {plan.get('approach', '')}"))
    except Exception as e:
        state["errors"].append(f"Planning error: {e}")
        state["plan"] = {"interpretation": state["question"], "approach": "direct pandas analysis", "needs_chart": False, "chart_type": "none", "relevant_columns": []}
        state["needs_chart"] = False
        state["status"] = "planned"
    return state


# ─── Node 3: Generate pandas code ─────────────────────────────────────────────

def generate_code_node(state: AnalystState) -> AnalystState:
    logger.info("✍️ Generating analysis code...")
    schema = state["df_schema"]
    plan = state["plan"]
    is_retry = state["retry_count"] > 0

    cols_desc = "\n".join(f"- {c['name']} ({c['dtype']})" for c in schema["columns"])

    retry_context = ""
    if is_retry and state.get("execution_result"):
        retry_context = f"""
PREVIOUS ATTEMPT FAILED WITH:
{state['execution_result'].get('error', '')}

PREVIOUS CODE:
{state.get('generated_code', '')}

Fix the issue and write corrected code.
"""

    chart_instruction = ""
    if state["needs_chart"]:
        chart_instruction = f"""
ALSO create a {plan.get('chart_type','bar')} chart using matplotlib:
- Save it to 'chart.png' using plt.savefig('chart.png', dpi=150, bbox_inches='tight')
- Use a dark theme: fig.patch.set_facecolor('#0f1117'), ax.set_facecolor('#1e2130')
- Set text colors to '#e0e0e0' for readability
- Call plt.close() after saving
"""

    prompt = f"""The dataframe is already loaded as variable `df`. Columns:
{cols_desc}

QUESTION: {state['question']}
ANALYSIS APPROACH: {plan.get('approach', '')}
{chart_instruction}
{retry_context}

Write Python code that:
1. Performs the analysis using `df` (already loaded, don't reload it)
2. Stores the final answer in a variable called `result` (can be a number, string, DataFrame, or Series)
3. Prints `result` at the end
4. Uses only pandas, numpy, and matplotlib (already imported as pd, np, plt)
5. Handles potential issues like missing values, division by zero

Return ONLY the Python code in a ```python block."""

    try:
        response = call_llm_text(
            system="You are an expert Python/pandas developer. Write clean, correct, minimal code. Return ONLY code in markdown fences.",
            user=prompt,
            temperature=0.05,
        )
        code = extract_code(response)
        state["generated_code"] = code
        state["status"] = "code_generated"
        state["messages"].append(AIMessage(content=f"✍️ Code generated ({'retry' if is_retry else 'initial'})"))
    except Exception as e:
        state["errors"].append(f"Code generation error: {e}")
        state["generated_code"] = "result = df.describe()\nprint(result)"
        state["status"] = "code_generated"
    return state


# ─── Node 4: Execute code safely ──────────────────────────────────────────────

SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "sorted": sorted, "list": list, "dict": dict,
    "str": str, "int": int, "float": float, "bool": bool, "enumerate": enumerate,
    "zip": zip, "print": print, "isinstance": isinstance, "type": type,
}


def execute_code_node(state: AnalystState) -> AnalystState:
    logger.info("▶️ Executing analysis code...")
    code = state["generated_code"]
    df = get_df(state["dataset_path"]).copy()

    exec_globals = {
        "df": df, "pd": pd, "np": np, "plt": plt,
        "__builtins__": SAFE_BUILTINS,
    }

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals)

        result = exec_globals.get("result", None)
        output_str = stdout_capture.getvalue()

        # Convert result to a serializable representation
        if isinstance(result, pd.DataFrame):
            result_repr = result.head(20).to_dict(orient="records")
            result_type = "dataframe"
        elif isinstance(result, pd.Series):
            result_repr = result.head(20).to_dict()
            result_type = "series"
        elif isinstance(result, (np.integer, np.floating)):
            result_repr = float(result)
            result_type = "number"
        else:
            result_repr = result
            result_type = type(result).__name__

        state["execution_result"] = {
            "success": True,
            "result": result_repr,
            "result_type": result_type,
            "stdout": output_str,
            "error": None,
        }
        state["status"] = "executed"
        state["messages"].append(AIMessage(content="✅ Code executed successfully"))

        # Check if a chart was created
        if state["needs_chart"] and os.path.exists("chart.png"):
            state["chart_path"] = "chart.png"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        state["execution_result"] = {
            "success": False,
            "result": None,
            "result_type": None,
            "stdout": stdout_capture.getvalue(),
            "error": error_msg,
        }
        state["status"] = "execution_failed"
        state["messages"].append(AIMessage(content=f"❌ Execution failed: {error_msg}"))
        logger.warning(f"Execution failed: {error_msg}")

    return state


# ─── Node 5: Check & route ────────────────────────────────────────────────────

def check_retry_node(state: AnalystState) -> AnalystState:
    exec_result = state["execution_result"]
    if exec_result["success"]:
        state["status"] = "success"
    elif state["retry_count"] >= MAX_RETRIES:
        state["status"] = "failed_final"
    else:
        state["retry_count"] += 1
        state["status"] = "retry"
    return state


def route_after_check(state: AnalystState) -> str:
    if state["status"] == "retry":
        return "generate_code"
    return "explain"


# ─── Node 6: Explain the answer ───────────────────────────────────────────────

def explain_node(state: AnalystState) -> AnalystState:
    logger.info("📝 Generating explanation...")
    exec_result = state["execution_result"]

    if not exec_result["success"]:
        state["answer"] = (
            f"I wasn't able to complete this analysis after {state['retry_count']} attempt(s). "
            f"Error: {exec_result['error']}\n\nTry rephrasing your question or check that the relevant columns exist."
        )
        return state

    try:
        explanation = call_llm_text(
            system="You are a data analyst explaining results to a business stakeholder. Be clear, concise, and lead with the direct answer.",
            user=f"""The user asked: "{state['question']}"

The analysis produced this result:
{json.dumps(exec_result['result'], indent=2, default=str)[:2000]}

Console output (if any):
{exec_result['stdout'][:500]}

Write a clear, direct answer (2-4 sentences) that:
1. Directly answers their question with the specific numbers/findings
2. Adds brief context or a notable insight if relevant
3. Avoids restating the question back

Be conversational but precise with numbers.""",
            temperature=0.3,
        )
        state["answer"] = explanation
    except Exception as e:
        state["answer"] = f"Analysis complete. Result: {exec_result['result']}"

    state["status"] = "complete"
    state["messages"].append(AIMessage(content="✅ Explanation generated"))
    return state


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_analyst_graph():
    graph = StateGraph(AnalystState)
    graph.add_node("load_schema", load_schema_node)
    graph.add_node("plan", plan_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute", execute_code_node)
    graph.add_node("check_retry", check_retry_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("load_schema")
    graph.add_edge("load_schema", "plan")
    graph.add_edge("plan", "generate_code")
    graph.add_edge("generate_code", "execute")
    graph.add_edge("execute", "check_retry")
    graph.add_conditional_edges("check_retry", route_after_check, {
        "generate_code": "generate_code",
        "explain": "explain",
    })
    graph.add_edge("explain", END)

    return graph.compile()


def run_analyst(question: str, dataset_path: str) -> AnalystState:
    # Clean up old chart
    if os.path.exists("chart.png"):
        os.remove("chart.png")

    graph = build_analyst_graph()
    initial: AnalystState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "dataset_path": dataset_path,
        "df_schema": None,
        "plan": None,
        "generated_code": None,
        "execution_result": None,
        "chart_path": None,
        "needs_chart": False,
        "answer": None,
        "retry_count": 0,
        "errors": [],
        "status": "start",
    }
    return graph.invoke(initial)
