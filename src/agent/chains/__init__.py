"""
LLM "chains" — pure functions that map (inputs, llm) -> validated Pydantic output.

Chains are stateless w.r.t. the graph: they do not read or write ``AgentState``.
Each chain owns one LLM-driven concern (router, evidence grader, generator,
hallucination grader, answer grader, refiner). Nodes in ``src/agent/nodes/`` are the
only place that touches state.

This separation mirrors ``02-cognito-crag/graph/chains/`` and
``06-langgraph-orchestration/app/agents/`` in the reference repos.
"""
