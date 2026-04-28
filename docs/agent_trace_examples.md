# Agent Trace Examples

Illustrative end-to-end traces for the LangGraph agent (`POST /api/v1/agent/answer`). Each example shows the route the agent took, the per-node trace, and the final response payload. The traces below were captured with the LLM/retrieval stubs from `tests/agent/test_graph_smoke.py` and `tests/agent/test_graph_v2.py`; in production, durations and rationales come from the real LLM and the SQLite/Vector Search backends.

For a deeper architectural overview, see the **Agentic Orchestration** section in `README.md` and `docs/agent_v2_architecture.md` (V2A deep_research path).

---

## 1. Happy path — FAST route

**Query:** `what is GraphRAG?`

The router classifies the question as a single-source factual lookup, fast retrieval finds two relevant chunks, the per-evidence grader keeps both, the generator drafts an answer with citations, and both the hallucination grader and answer grader pass on the first attempt.

```text
[router]          ok    12.3 ms  detail=fast: single-concept lookup
[fast_retrieve]   ok    84.5 ms  detail=2 hits
[grade_evidence]  ok    38.1 ms  detail=kept=2 dropped=0 grader_failures=0
[generate]        ok   612.0 ms  detail=draft_len=412 citations=[0, 1] regen=0
[evaluate]        ok    71.4 ms  detail=grounded=True answers_question=True
[finalize]        ok     0.6 ms  detail=final_len=412
[service]         ok   819.7 ms  detail=route=fast
```

**Response (truncated):**

```json
{
  "answer": "GraphRAG augments retrieval with a knowledge graph...",
  "route": "fast",
  "evidence_used": [{"content_id": "...", "source_type": "research_paper", "...": "..."}],
  "citations": [0, 1],
  "grounded": true,
  "answers_question": true,
  "refinement_iterations": 0,
  "insufficient_evidence": false,
  "trace_id": "..."
}
```

---

## 2. DEEP route — fast retrieval + KG worker

**Query:** `how does Vector Search relate to BM25 in this index?`

The router picks DEEP because the question is comparative and benefits from both document chunks and graph relationships. After `fast_retrieve`, the conditional edge dispatches to `kg_worker` (sequential in V1; parallelized in V2), and only then does `grade_evidence` run.

```text
[router]          ok    18.2 ms  detail=deep: comparative; benefits from KG + docs
[fast_retrieve]   ok    91.3 ms  detail=6 hits
[kg_worker]       ok    44.0 ms  detail=4 concepts
[grade_evidence]  ok   102.6 ms  detail=kept=5 dropped=1 grader_failures=0
[generate]        ok   888.5 ms  detail=draft_len=672 citations=[0, 2, 4] regen=0
[evaluate]        ok    78.9 ms  detail=grounded=True answers_question=True
[finalize]        ok     0.5 ms  detail=final_len=672
```

---

## 3. KG_ONLY route — definition lookup

**Query:** `what is the "Adaptive Search Weighting" concept?`

The router decides the question is a pure definition / one-hop lookup, so it skips retrieval entirely and goes straight to `kg_worker`. `grade_evidence` runs over an empty evidence list but `kg_findings` is non-empty, so the routing edge sends the graph onward to `generate`.

```text
[router]          ok    11.8 ms  detail=kg_only: definitional; KG sufficient
[kg_worker]       ok    33.7 ms  detail=2 concepts
[grade_evidence]  empty  6.1 ms  detail=kept=0 dropped=0 grader_failures=0
[generate]        ok   504.0 ms  detail=draft_len=298 citations=[]
[evaluate]        ok    62.4 ms  detail=grounded=True answers_question=True
[finalize]        ok     0.5 ms  detail=final_len=298
```

---

## 4. Out-of-scope — router fallback

**Query:** `what's the weather in NYC tomorrow?`

The router recognizes that the indexed corpus (Instagram / ArXiv / GitHub) cannot answer this and short-circuits to `fallback` without spending compute on retrieval or generation. The response carries `insufficient_evidence: true` and a structured help message.

```text
[router]          ok     9.4 ms  detail=fallback: out-of-scope (real-time data)
[fallback]        ok     1.2 ms  detail=insufficient_evidence
```

---

## 5. Empty retrieval — graceful fallback

**Query:** `what is "Frobnicator-7000"?` (term not in the corpus)

The router picks FAST, but `fast_retrieve` returns `status="empty"` (zero hits, no error). With no evidence and no KG findings, `grade_evidence`'s routing edge sends the graph to `fallback` instead of generating a hallucinated answer.

```text
[router]          ok    10.1 ms  detail=fast: single-concept lookup
[fast_retrieve]   empty 27.3 ms  detail=0 hits
[grade_evidence]  empty  0.4 ms  detail=kept=0 dropped=0 grader_failures=0
[fallback]        ok     1.0 ms  detail=insufficient_evidence
```

---

## 6. Refine loop — grounded but not useful

**Query:** `compare A and B`

The first draft is grounded in the retrieved chunks but only discusses A. The answer grader returns `answers_question=false`. Within budget (`AGENT_MAX_REFINEMENT_LOOPS=1`), `refine` produces a sharper revised query, clears the previous draft and evidence, and the workers re-run. The second pass passes both grades.

```text
[router]          ok    12.0 ms  detail=fast
[fast_retrieve]   ok    81.0 ms  detail=4 hits
[grade_evidence]  ok    48.0 ms  detail=kept=3 dropped=1 grader_failures=0
[generate]        ok   601.0 ms  detail=draft_len=388 citations=[0, 1] regen=0
[evaluate]        ok    74.0 ms  detail=grounded=True answers_question=False
[refine]          ok   210.0 ms  detail=revised_query='comparison of A and B side by side' iter=1
[fast_retrieve]   ok    79.0 ms  detail=5 hits  (re-run with refined query)
[grade_evidence]  ok    49.0 ms  detail=kept=4 dropped=1 grader_failures=0
[generate]        ok   620.0 ms  detail=draft_len=521 citations=[0, 2, 3]
[evaluate]        ok    73.0 ms  detail=grounded=True answers_question=True
[finalize]        ok     0.5 ms  detail=final_len=521
```

`refinement_iterations: 1` in the response confirms the loop ran exactly once.

---

## 7. Regenerate loop — not grounded

**Query:** `summarize the GraphRAG paper`

The first draft introduces details not present in the retrieved chunks (the hallucination grader returns `grounded=false`). Within budget (`AGENT_MAX_REGENERATE_LOOPS=1`), the graph regenerates against the same evidence with a stricter system prompt. If the second draft is still ungrounded, the graph routes to `fallback` rather than emit a hallucination.

```text
[router]          ok    11.5 ms  detail=fast
[fast_retrieve]   ok    82.6 ms  detail=6 hits
[grade_evidence]  ok    58.0 ms  detail=kept=4 dropped=2 grader_failures=0
[generate]        ok   598.0 ms  detail=draft_len=410 citations=[0, 1] regen=0
[evaluate]        ok    77.0 ms  detail=grounded=False answers_question=True
[generate]        ok   612.0 ms  detail=draft_len=395 citations=[0, 2] regen=1
[evaluate]        ok    76.0 ms  detail=grounded=False answers_question=True
[fallback]        ok     1.1 ms  detail=insufficient_evidence  (regenerate budget exhausted)
```

---

## 8. DEEP_RESEARCH route — orchestrator + parallel workers + aggregator (V2A)

**Query:** `compare GraphRAG and HippoRAG using both papers and the github repos`

The router (or its heuristic pre-filter) recognizes a multi-source comparison and dispatches to `orchestrate`. The orchestrator decomposes the question into three per-source tasks; LangGraph's `Send` primitive fans them out to the same `worker` node in parallel. Each worker retrieves with its source-specific filter, calls `worker_analyst` to produce a structured per-source analysis, and writes a `WorkerResult` to the parallel-safe `worker_results` channel. The `aggregate` node de-duplicates evidence, calls the aggregator chain to synthesize the workers' analyses, and writes the draft into the V1 channels so the existing evaluator/refiner runs unchanged.

```text
[router]            ok    14.2 ms  detail=deep_research: heuristic: comparison/multi-source cue
[orchestrate]       ok   442.0 ms  detail=3 task(s): [paper,github,kg]
[worker:paper]      ok   721.4 ms  detail=task_id=t1 type=paper evidence=5 kg=0 status=ok
[worker:github]     ok   698.1 ms  detail=task_id=t2 type=github evidence=4 kg=0 status=ok
[worker:kg]         ok    91.7 ms  detail=task_id=t3 type=kg evidence=0 kg=3 status=ok
[aggregate]         ok   971.3 ms  detail=workers=3 evidence=8 kg=3 citations=4
[evaluate]          ok    78.6 ms  detail=grounded=True answers_question=True
[finalize]          ok     0.5 ms  detail=final_len=842
[service]           ok  3018.2 ms  detail=route=deep_research
```

**Response (truncated, with `include_plan=true` and `include_workers=true`):**

```json
{
  "answer": "GraphRAG indexes a knowledge graph alongside chunks; HippoRAG ...",
  "route": "deep_research",
  "agent_version": "v2",
  "external_used": false,
  "plan": {
    "summary": "compare GraphRAG and HippoRAG across papers and code",
    "decomposition_rationale": "split per source so each worker stays focused",
    "tasks": [
      {"task_id": "t1", "worker_type": "paper",  "query": "GraphRAG vs HippoRAG benchmarks", "...": "..."},
      {"task_id": "t2", "worker_type": "github", "query": "GraphRAG vs HippoRAG code/usage", "...": "..."},
      {"task_id": "t3", "worker_type": "kg",     "query": "GraphRAG concept neighbors",      "...": "..."}
    ]
  },
  "worker_results": [
    {"task_id": "t1", "worker_type": "paper",  "status": "ok", "output": {"confidence": "high",   "...": "..."}},
    {"task_id": "t2", "worker_type": "github", "status": "ok", "output": {"confidence": "medium", "...": "..."}},
    {"task_id": "t3", "worker_type": "kg",     "status": "ok", "output": {"confidence": "medium", "...": "..."}}
  ],
  "citations": [0, 2, 4, 7],
  "grounded": true,
  "answers_question": true,
  "trace_id": "..."
}
```

The `evidence_used` list is the de-duplicated union across workers, sorted by `combined_score` desc; citations index into that list.

---

## How to capture your own traces

```bash
cd src/
AGENT_CHECKPOINT_DB=:memory: python run_agent.py --query "your question" --pretty

# Force the deep_research path explicitly + show plan and per-worker outputs:
AGENT_CHECKPOINT_DB=:memory: python run_agent.py \
    --query "compare GraphRAG and HippoRAG using papers and github" \
    --mode deep_research --include-plan --include-workers --pretty
```

Or via the API:

```bash
curl -s -X POST http://localhost:5000/api/v1/agent/answer \
     -H 'Content-Type: application/json' \
     -H 'X-Request-Id: dev-001' \
     -d '{"query":"your question","mode":"auto"}' \
   | jq '.trace[] | {node, status, duration_ms, detail}'
```
