## Business User MySQL Multi‑Agent (LangGraph + LangSmith)

This project is a **LangGraph-based multi‑agent system** that lets business users ask questions about a MySQL database in natural language.  
All **DML queries (INSERT/UPDATE/DELETE)** trigger a **human‑in‑the‑loop approval step** before they are executed.  
The graph uses **LangGraph SDK**, **LangChain + OpenAI**, **LangSmith tracing**, and **LangGraph memory**.

### Features

- **Natural‑language to SQL**: LLM agent generates SQL from business questions using DB schema context.
- **Execution agent**: Safely runs approved SQL against a MySQL database.
- **Human‑in‑the‑loop for DML**: Any non‑SELECT query is paused and requires explicit human approval.
- **Stateful memory**: Uses LangGraph’s `MemorySaver` checkpointer so conversations can be resumed and audited.
- **LangSmith integration**: End‑to‑end tracing, run history, and debugging.

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file (same directory as `main.py`) with at least:

```bash
OPENAI_API_KEY=your-openai-key
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=business-mysql-agent

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_DATABASE=your_database
```

### 3. Run the CLI agent

```bash
python main.py
```

Then type business questions, for example:

- “Show me the total sales by month for the last 6 months.”
- “Update the status of order 12345 to ‘shipped’.”

For **SELECT** queries, the agent will execute them directly and print results.  
For **INSERT/UPDATE/DELETE** or other DML, the agent will:

1. Show you the generated SQL.
2. Ask: `Approve this query? (y/n):`
3. Only execute if you answer `y`.

### 4. Files
- `web_app.py` – FASTAPI endpoints for ask queries and approve for DML.
- `main.py` – CLI entrypoint and run loop with human‑in‑the‑loop handling.
- `graph.py` – LangGraph `StateGraph` definition, multi‑agent nodes, and memory.
- `db.py` – MySQL connection and query helpers.
- `requirements.txt` – Python dependencies.

### 5. Run as FAST API backend app

`
uvicorn web_app:app --reload
`

Swagger UI   
Open in browser
http://127.0.0.1:8000/

### 6. Switching to persistent memory (optional)

By default, the graph uses in‑memory checkpoints.  
To persist state, install the SQLite checkpointer and swap the saver:

```bash
pip install \"langgraph-checkpoint-sqlite\"
```

Then in `graph.py`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string(\"sqlite:///memory.db\")
app = graph.compile(checkpointer=checkpointer)
```

### 7. LangSmith integration

With `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY` set, all LLM/tool calls made via LangChain + LangGraph will be traced to your LangSmith project (`LANGSMITH_PROJECT`).  
You can then:

- Inspect each run and step in the graph
- See the exact SQL that was generated and approved
- Replay or debug failed or interrupted runs

