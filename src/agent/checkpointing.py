"""
SQLite checkpointer factory for the agent graph.

Purpose
-------
Keeps the SQLite checkpointer construction in one place and abstracts away the small
``langgraph-checkpoint-sqlite`` API surface differences (the saver uses
``from_conn_string`` as a context manager, which doesn't fit the long-lived
"compile once" model used by ``service``). We open a single shared connection and
hand the saver a plain ``sqlite3.Connection``.

Reference
---------
- ``langgraph.checkpoint.sqlite.SqliteSaver`` (constructor takes ``sqlite3.Connection``).
- LangGraph persistence guide:
  https://langchain-ai.github.io/langgraph/how-tos/persistence/

Sample
------
>>> from agent.checkpointing import build_checkpointer
>>> # saver = build_checkpointer(":memory:")
>>> # type(saver).__name__
'SqliteSaver'
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger("agent.checkpointing")


def build_checkpointer(db_path: str) -> Any:
    """Construct a ``SqliteSaver`` backed by ``db_path`` (file path or ``:memory:``).

    The connection has ``check_same_thread=False`` so the checkpointer can be used
    from Flask request handlers (different threads in the same process).
    """
    from langgraph.checkpoint.sqlite import SqliteSaver  # local import to keep deps optional

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    logger.info("agent_checkpointer_ready db=%r", db_path)
    return saver


__all__ = ["build_checkpointer"]
