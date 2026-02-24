from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from main import run_agent_step

app = FastAPI(
    title="Business MySQL Agent API",
    description=(
        "API for a LangGraph-powered MySQL assistant with human-in-the-loop\n"
        "approval for DML queries. Use /ask to send a question; if a DML\n"
        "statement is generated, the API will return a 'needs_approval' "
        "response. Then call /approve to approve or reject execution."
    ),
    version="0.1.0",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc", # ReDoc UI (optional)
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


class AskRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post(
    "/ask",
    summary="Ask a business question",
    tags=["agent"],
)
async def ask(req: AskRequest):
    """
    Start/continue a conversation for a given thread_id.
    If the graph hits a DML approval interrupt, return the SQL to the client.
    """
    thread_id = req.thread_id or "thread_" + "default"

    result = await run_agent_step(
        question=req.question,
        thread_id=thread_id,
        resume=None,
    )
    state = result["state"]
    interrupt = result["interrupt"]

    if interrupt is not None:
        # Human approval required
        sql = interrupt.get("sql_to_approve", "")
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "sql_to_approve": sql,
        }

    # No interrupt: finished normally
    if state:
        summary = state.get("result_summary", "No answer produced.")
        rows = state.get("rows", [])
        return {
            "status": "completed",
            "thread_id": thread_id,
            "summary": summary,
            "rows": rows,
        }

    return {"status": "error", "thread_id": thread_id, "message": "No response generated."}


@app.post(
    "/approve",
    summary="Approve or reject a pending DML query",
    tags=["agent"],
)
async def approve(req: ApproveRequest):
    """
    Resume a previously interrupted run with human approval or rejection.
    """
    result = await run_agent_step(
        question=None,
        thread_id=req.thread_id,
        resume={"approved": req.approved},
    )
    state = result["state"]

    if state:
        summary = state.get("result_summary", "No answer produced.")
        rows = state.get("rows", [])
        return {
            "status": "completed",
            "thread_id": req.thread_id,
            "summary": summary,
            "rows": rows,
        }

    return {"status": "error", "thread_id": req.thread_id, "message": "No response generated after approval."}