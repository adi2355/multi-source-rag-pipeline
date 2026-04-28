"""
LLM "chains" — pure functions that map (inputs, llm) -> validated Pydantic output.

Chains are stateless w.r.t. the graph: they do not read or write ``AgentState``.
Each chain owns one LLM-driven concern. Nodes in ``src/agent/nodes/`` are the only
place that touches state.

V1 chains
---------
- :mod:`agent.chains.router`              -- route decision
- :mod:`agent.chains.evidence_grader`     -- per-evidence relevance grade
- :mod:`agent.chains.generator`           -- final grounded answer
- :mod:`agent.chains.hallucination_grader`-- groundedness verdict
- :mod:`agent.chains.answer_grader`       -- answer-quality verdict
- :mod:`agent.chains.refiner`             -- regeneration directive

V2A chains (deep_research path)
--------------------------------
- :mod:`agent.chains.orchestrator`        -- query -> OrchestrationPlan
- :mod:`agent.chains.worker_analyst`      -- task + hits -> WorkerStructuredOutput
- :mod:`agent.chains.aggregator`          -- worker outputs -> GeneratedAnswer

This separation mirrors ``02-cognito-crag/graph/chains/`` and
``06-langgraph-orchestration/app/agents/`` in the reference repos.
"""
