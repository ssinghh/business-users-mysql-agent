import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langgraph.types import Command

from graph import build_graph


load_dotenv()

async def run_conversation():
    app = build_graph()
    thread_id = "business_mysql_agent"
    config = {"configurable": {"thread_id": thread_id}}

    print("Business MySQL Agent (LangGraph + Human-in-the-loop)")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Initial state
        input_data = {"question": question, "messages": []}
        
        # Stream events and handle interrupts
        final_state = None
        async for event in app.astream(input_data, config=config):
            # Check for interrupts
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"][0].value
                sql_to_approve = interrupt_data.get("sql_to_approve", "")
                
                print(f"\nDML Query requires approval:\n{sql_to_approve}\n")
                approved_input = input("Approve this query? (y/n): ").strip().lower()
                approved = approved_input in {"y", "yes"}
                
                # Resume with approval as dictionary
                resume_command = Command(resume={"approved": approved})
                async for event in app.astream(resume_command, config=config):
                    # Collect final state from the last event
                    for node_name, node_state in event.items():
                        final_state = node_state
                break
            else:
                # No interrupt - collect state from each event
                for node_name, node_state in event.items():
                    final_state = node_state
        
        # Print the final result
        if final_state:
            # 1. Get the summary text
            summary = final_state.get("result_summary", "No answer produced.")    
            # 2. Get the data rows (default to an empty list if missing)
            rows = final_state.get("rows", [])           
            # 3. Combine them into the result variable 
            if rows:
                result = f"Data Found:\n{rows}\n\nData Summary:\n{summary}"
            else:
                result = f"Data Summary:\n{summary}"

            print(f"\nAgent:\n{result}\n")
        else:
            print("\nAgent:\nNo response generated.\n") 

if __name__ == "__main__":
    asyncio.run(run_conversation())

