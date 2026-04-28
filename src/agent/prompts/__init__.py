"""
Prompt templates colocated with their chains.

Kept as plain Python strings (not Jinja or external files) so prompt changes show up
clearly in code review. Each prompt module exports ``SYSTEM_PROMPT`` and a function
``build_user_prompt(...)`` that takes typed inputs and returns the rendered user-side
text. Prompts NEVER take raw state dicts; the chain is responsible for selecting and
formatting the relevant fields.
"""
