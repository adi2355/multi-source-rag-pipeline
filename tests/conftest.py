"""
pytest bootstrap for the agent test suite.

Adds ``src/`` to ``sys.path`` so tests can ``from agent.* import ...`` the same way
production code does (the project's run mode is ``cd src && python ...``). Also
forces an ephemeral SQLite checkpointer for every test by setting
``AGENT_CHECKPOINT_DB=:memory:`` before any ``AgentService`` is constructed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("AGENT_CHECKPOINT_DB", ":memory:")
os.environ.setdefault("AGENT_MAX_REFINEMENT_LOOPS", "1")
os.environ.setdefault("AGENT_MAX_REGENERATE_LOOPS", "1")
