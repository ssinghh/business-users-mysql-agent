import os
from contextlib import contextmanager
from typing import Any, Iterable, List, Tuple

import mysql.connector
from mysql.connector import MySQLConnection
from dotenv import load_dotenv


load_dotenv()


def get_mysql_connection() -> MySQLConnection:
    """Create a new MySQL connection from environment variables."""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE")

    if not database:
        raise ValueError("MYSQL_DATABASE is not set in environment variables")

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


@contextmanager
def mysql_cursor():
    """Context manager that yields a cursor and ensures connection cleanup."""
    conn = get_mysql_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def fetch_schema_summary() -> str:
    """Return a human-readable summary of tables and columns in the DB."""
    with mysql_cursor() as cur:
        cur.execute(
            """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
        )
        rows: List[Tuple[Any, ...]] = cur.fetchall()

    schema_lines: List[str] = []
    current_table: str | None = None
    for table_name, column_name, data_type, is_nullable in rows:
        if table_name != current_table:
            current_table = table_name
            schema_lines.append(f"\\nTable {table_name}:")
        schema_lines.append(f"  - {column_name} ({data_type} {is_nullable})")

    return "\\n".join(schema_lines).strip()


def _is_potentially_unsafe_sql(sql: str) -> bool:
    normalized = sql.strip().lower()

    # Remove exactly one trailing semicolon
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()

    # Disallow additional semicolons (multi-statement)
    if ";" in normalized:
        return True

    # Block comment injection
    if "--" in normalized or "/*" in normalized or "*/" in normalized:
        return True

    # Block destructive commands entirely
    dangerous = (" drop ", " truncate ", " alter ", " grant ", " revoke ", " shutdown ")
    if any(token in f" {normalized} " for token in dangerous):
        return True

    return False



def execute_sql(sql: str) -> Tuple[str, List[Tuple[Any, ...]]]:
    """Execute the given SQL and return a (summary, rows) tuple.

    - For SELECT queries, returns fetched rows.
    - For DML (INSERT/UPDATE/DELETE), returns affected row count and no rows.

    A lightweight SQL injection / safety guard is applied before execution.
    """
    sql_stripped = sql.strip().rstrip(";")

    print(sql_stripped)
    if _is_potentially_unsafe_sql(sql_stripped):
        # Do NOT execute obviously unsafe statements
        return (
            "Blocked potentially unsafe or multi-statement SQL. "
            "The query was not executed. Please rephrase your request.",
            [],
        )

    with mysql_cursor() as cur:
        cur.execute(sql_stripped)
        command = sql_stripped.split()[0].upper()
        if command == "SELECT":
            rows = cur.fetchall()
            return "Returned rows", rows
        else:
            affected = cur.rowcount if cur.rowcount is not None else 0
            return (f"{command} affected {affected} row(s)", [])