from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from db import execute_sql, fetch_schema_summary


load_dotenv()


class GraphState(TypedDict, total=False):
    question: str
    sql: Optional[str]
    result_summary: Optional[str]
    rows: Optional[List[List[Any]]]
    messages: List[Any]
    requires_approval: bool
    approved: Optional[bool]


llm = ChatOpenAI(model="gpt-4o", temperature=0.0)


def planner_node(state: GraphState) -> GraphState:
    """LLM agent: turn business question + schema into SQL and explanation."""
    question = state["question"]
    schema = fetch_schema_summary()

    system = SystemMessage(
        content=(
        "You are an expert SQL assistant for a MySQL business database.\n"
        "Given a user question and the database schema, you must:\n"
        "1. Understand the intent.\n"
        "2. Produce a single, executable SQL statement for MySQL.\n"
        "3. NEVER run destructive operations without explicit instruction.\n"
        "4. Prefer SELECT queries when the user is just asking for information.\n"
        "5. Do not use multiple statements; output exactly one SQL statement.\n"
        "6. You MUST use column names exactly as written in the schema.\n"
        "7. Do NOT guess or rename column names.\n"
        )
    )
    
    schema_msg = SystemMessage(content=f"Database schema:\\n{schema}")
    user = HumanMessage(
        content=(
            "User question: " + question + "\\n"
            "Return ONLY the SQL statement, nothing else."
        )
    )

    resp = llm.invoke([system, schema_msg, user])
    sql = resp.content.strip()

    # Remove markdown fences if present
    if sql.startswith("```"):
        sql = sql.replace("```sql", "").replace("```", "").strip()

    messages = state.get("messages", [])
    messages = messages + [user, resp]

    return {
        **state,
        "sql": sql,
        "messages": messages,
    }


def classify_dml_node(state: GraphState) -> GraphState:
    """Determine whether the SQL is DML and requires human approval."""
    sql = (state.get("sql") or "").strip()
    is_dml = False
    if sql:
        first_token = sql.split()[0].upper()
        if first_token in {"INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE"}:
            is_dml = True
    return {**state, "requires_approval": is_dml}


def approval_node(state: GraphState):
    """Interrupt the graph to ask a human to approve or reject the DML query."""
    sql = state.get("sql") or ""
    # Use interrupt() correctly - it pauses and returns the user input when resumed
    user_input = interrupt({"sql_to_approve": sql})
    # user_input will be a dict when resumed, e.g., {"approved": True}
    approved = user_input.get("approved", False) if isinstance(user_input, dict) else False
    return {**state, "approved": approved}

def exec_node(state: GraphState) -> GraphState:
    """Execute the SQL (if approved or not DML) and store results."""
    sql = state.get("sql") or ""
    if not sql:
        return {**state, "result_summary": "No SQL was generated."}

    # If it was DML and not approved, skip execution
    if state.get("requires_approval") and not state.get("approved"):
        return {**state, "result_summary": "Execution cancelled by human."}

    summary, rows = execute_sql(sql)
    return {
        **state,
        "result_summary": summary,
        "rows": [list(r) for r in rows],
    }


def format_answer_node(state: GraphState) -> GraphState:
    """Optional LLM step to turn raw rows into a friendly answer."""
    question = state.get("question", "")
    sql = state.get("sql", "")
    summary = state.get("result_summary", "")
    rows = state.get("rows") or []

    messages = state.get("messages", [])

    system = SystemMessage(
        content=(
            "You are a helpful business analytics assistant.\\n"
            "You are given: the user question, the SQL that was run, a short summary, and raw rows.\\n"
            "Explain the result in clear business language. If execution was cancelled, explain that."
        )
    )
    user = HumanMessage(
        content=(
            f"User question: {question}\\n\\n"
            f"SQL: {sql}\\n\\n"
            f"Execution summary: {summary}\\n\\n"
            f"Rows (as Python list of tuples): {rows}"
        )
    )

    resp = llm.invoke([system, user])
    messages = messages + [user, resp]

    return {
        **state,
        "rows": [list(r) for r in rows],
        "messages": messages,
        "result_summary": resp.content,
    }


def build_graph():
    """Create and compile the LangGraph app with memory and interrupts."""
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("classify_dml", classify_dml_node)
    graph.add_node("approval", approval_node)
    graph.add_node("exec", exec_node)
    graph.add_node("format_answer", format_answer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "classify_dml")

    # Conditional: if DML, go to approval, else straight to exec
    def route_after_classify(state: GraphState):
        if state.get("requires_approval"):
            return "approval"
        return "exec"

    graph.add_conditional_edges("classify_dml", route_after_classify)

    # After approval (which now sets approved directly), go to exec
    graph.add_edge("approval", "exec")
    graph.add_edge("exec", "format_answer")
    graph.add_edge("format_answer", END)

    # In-memory checkpoint (can be swapped for SqliteSaver for persistence)
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app

