import asyncio
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.types import Command

from graph import build_graph


load_dotenv()

# Build a single app instance to reuse across requests
_app = build_graph()


async def run_agent_step(
    question: Optional[str],
    thread_id: str,
    resume: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run one step of the LangGraph agent.

    - If `resume` is None, this starts/continues the graph from a user question.
    - If `resume` is provided (e.g. {"approved": True}), it resumes from an interrupt.

    Returns a dict with:
      - "state": final graph state (or None)
      - "interrupt": interrupt payload if the graph paused for human approval, else None
    """
    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    if resume is not None:
        # Resume from a human-in-the-loop interrupt
        input_data: Any = Command(resume=resume)
    else:
        # Start / continue from a user question
        if not question:
            raise ValueError("question is required when resume is None")
        input_data = {"question": question, "messages": []}

    final_state: Optional[Dict[str, Any]] = None
    interrupt_payload: Optional[Dict[str, Any]] = None

    async for event in _app.astream(input_data, config=config):
        # Handle interrupts surfaced by LangGraph
        if "__interrupt__" in event:
            interrupt_payload = event["__interrupt__"][0].value

        # Each non-interrupt key is a node_name -> node_state
        for node_name, node_state in event.items():
            if node_name == "__interrupt__":
                continue
            final_state = node_state  # last node_state wins

    return {"state": final_state, "interrupt": interrupt_payload}


# Optional: simple CLI for debugging without FastAPI
async def _cli() -> None:
    thread_id = "cli_session"
    print("Business MySQL Agent (LangGraph + Human-in-the-loop via CLI)")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = await run_agent_step(question=question, thread_id=thread_id)
        state = result["state"]
        interrupt = result["interrupt"]

        # If interrupt, ask for approval then resume
        if interrupt is not None:
            sql = interrupt.get("sql_to_approve", "")
            print(f"\nDML Query requires approval:\n{sql}\n")
            approved_input = input("Approve this query? (y/n): ").strip().lower()
            approved = approved_input in {"y", "yes"}

            resume_result = await run_agent_step(
                question=None,
                thread_id=thread_id,
                resume={"approved": approved},
            )
            state = resume_result["state"]

        if state:
            summary = state.get("result_summary", "No answer produced.")
            rows = state.get("rows", [])
            if rows:
                output = f"Data Found:\n{rows}\n\nData Summary:\n{summary}"
            else:
                output = f"Data Summary:\n{summary}"
            print(f"\nAgent:\n{output}\n")
        else:
            print("\nAgent:\nNo response generated.\n")


if __name__ == "__main__":
    asyncio.run(_cli())


