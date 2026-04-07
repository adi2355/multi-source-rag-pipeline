<div align="center">
  <img src="terminal-top-panel.svg" width="100%" alt="Terminal header">
</div>

## Overview

A multi-source Retrieval-Augmented Generation pipeline that ingests AI/ML knowledge from Instagram video transcripts, ArXiv research papers, and GitHub repositories, then organizes it into a searchable knowledge system with concept-level understanding. The pipeline spans seven layers: collection, processing, storage, knowledge extraction, vector embedding, hybrid retrieval, and LLM-powered answer generation.

The architecture treats each content source as a first-class data stream with its own ingestion, processing, and normalization path, converging into a unified SQLite store with full-text search, vector embeddings, and a concept knowledge graph. Retrieval combines cosine-similarity vector search with FTS5 keyword matching through an adaptive weighting system that classifies queries and adjusts strategy in real time.

---

## Technology Stack

<table>
  <tr>
    <td><strong>Languages</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
      <img src="https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white" alt="SQL">
      <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript">
    </td>
  </tr>
  <tr>
    <td><strong>AI / LLM</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Anthropic_Claude-D4A574?style=flat-square&logo=anthropic&logoColor=white" alt="Claude">
      <img src="https://img.shields.io/badge/Mistral_AI-FF7000?style=flat-square&logoColor=white" alt="Mistral">
      <img src="https://img.shields.io/badge/OpenAI_Whisper-412991?style=flat-square&logo=openai&logoColor=white" alt="Whisper">
    </td>
  </tr>
  <tr>
    <td><strong>ML / Embeddings</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Sentence_Transformers-FF6F00?style=flat-square&logo=huggingface&logoColor=white" alt="Sentence Transformers">
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
      <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
      <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
    </td>
  </tr>
  <tr>
    <td><strong>Data & Storage</strong></td>
    <td>
      <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
      <img src="https://img.shields.io/badge/FTS5-003B57?style=flat-square&logoColor=white" alt="FTS5">
      <img src="https://img.shields.io/badge/NetworkX-4C8CBF?style=flat-square&logoColor=white" alt="NetworkX">
    </td>
  </tr>
  <tr>
    <td><strong>Web & API</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
      <img src="https://img.shields.io/badge/Swagger-85EA2D?style=flat-square&logo=swagger&logoColor=black" alt="Swagger">
      <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    </td>
  </tr>
</table>

<br>

## Engineering Principles

### 1. Source-Agnostic Convergence

Every content source -- Instagram video transcripts, ArXiv research papers, GitHub repositories -- enters through its own specialized ingestion path but converges into a single unified schema (`ai_content`) with normalized metadata. Downstream systems (embedding, search, knowledge extraction) operate on this unified representation without knowledge of the original source.

> **Goal:** Add a new content source by implementing one collector and one normalizer, with zero changes to retrieval, embedding, or knowledge graph logic.

### 2. Hybrid Retrieval Over Single-Strategy Search

Pure vector search misses exact-match terminology. Pure keyword search misses semantic similarity. The retrieval layer combines both through an adaptive weighting system that classifies each query (code, factual, conceptual) and adjusts the vector-to-keyword balance in real time. A feedback loop learns optimal weights from user search interactions.

> **Goal:** Every query type -- exact code snippets, broad conceptual questions, specific factual lookups -- returns relevant results without manual tuning.

### 3. Knowledge-First Architecture

Raw documents are not just stored and embedded -- they are distilled into a structured concept graph using LLM-powered extraction. Concepts, their categories (algorithm, model, technique, framework), and weighted relationships form a NetworkX graph that supports centrality analysis, community detection, and concept-aware retrieval.

> **Goal:** Answer questions about relationships between ideas, not just questions about individual documents.

### 4. Measurable Retrieval Quality

The system includes a full evaluation framework computing precision, recall, F1, NDCG, and MRR across search strategies. Test queries are generated programmatically from knowledge graph concepts, ensuring evaluation coverage tracks the actual knowledge base.

> **Goal:** Every retrieval change is measured against a reproducible benchmark, not validated by subjective impression.

### 5. Cost-Efficient LLM Integration

Summarization uses the Claude Message Batches API (up to 100 items per batch, asynchronous polling), achieving approximately 50% cost reduction compared to sequential API calls. OCR uses Mistral AI for PDF text extraction with automatic chunking for large documents and PyPDF2 fallback for robustness.

> **Goal:** Process large document collections at scale without proportional cost scaling.

---

## Pipeline Architecture

The system operates as a seven-layer pipeline. Data flows from collection through processing, storage, knowledge extraction, and embedding before reaching the retrieval and generation layers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Data Collection                               │
│   Instagram (instaloader)  ·  ArXiv (arxiv API)  ·  GitHub (REST API)  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                          Data Processing                                │
│   Whisper Transcription  ·  Mistral OCR  ·  Claude Batch Summaries     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                        SQLite Unified Store                             │
│   ai_content  ·  source-specific tables  ·  FTS5 virtual tables        │
└───────────┬────────────────────┬────────────────────┬───────────────────┘
            │                    │                    │
┌───────────▼──────┐  ┌─────────▼──────────┐  ┌─────▼─────────────────┐
│  Knowledge       │  │  Vector             │  │  Hybrid Retrieval     │
│  Extraction      │  │  Embedding          │  │  Vector + FTS5 fusion │
│  Concepts +      │  │  768-dim, overlap   │  │  Adaptive weighting   │
│  Graph           │  │  chunking           │  │  Feedback learning    │
└───────────┬──────┘  └─────────┬──────────┘  └─────┬─────────────────┘
            │                   │                    │
┌───────────▼───────────────────▼────────────────────▼────────────────────┐
│                      LLM Response Generation                            │
│   Context selection  ·  Source citations  ·  Streaming responses        │
└─────────────────────────────────────────────────────────────────────────┘
```

<br>

## Multi-Source Ingestion

### Instagram Video Pipeline

The scraper uses instaloader with proxy rotation, account credential cycling, and rate-limit detection with configurable cooldown periods. Account state is tracked persistently in JSON files. Audio is extracted and transcribed locally using OpenAI Whisper, producing timestamped transcript segments.

### ArXiv Research Papers

Papers are collected via the arxiv API with configurable search queries and date ranges. PDF text is extracted using the Mistral AI OCR API with automatic chunking for large documents. A PyPDF2 fallback ensures extraction succeeds when the OCR API is unavailable. Papers enter a download-only mode for batch collection, followed by a separate processing phase.

### GitHub Repositories

Public repositories are collected via the GitHub REST API. Repository metadata, README content, file structure, and primary language information are normalized into the unified content schema.

> **Guarantee:** Each source operates independently -- an Instagram rate-limit event does not block ArXiv paper processing or GitHub collection.

---

## Knowledge Graph Engine

The concept extraction pipeline uses Claude to identify concepts from processed content, classify them by category (algorithm, model, technique, framework, concept, tool, dataset, metric), and extract weighted relationships with confidence scores. The resulting graph is built and analyzed using NetworkX.

**Graph capabilities:**
- PageRank centrality analysis for identifying foundational concepts
- Community detection for discovering concept clusters
- Subgraph extraction around specific topics
- Interactive Plotly visualization and static Matplotlib rendering
- GEXF and JSON export for external analysis tools

> **Guarantee:** The knowledge graph is a queryable, structured representation of the knowledge base -- not a visualization artifact.

---

## Hybrid Retrieval System

### Embedding Layer

Text is chunked with configurable size (default 1000 characters) and overlap (200 characters), using intelligent boundary detection that respects paragraph breaks, newlines, sentence endings, and word boundaries. Embeddings are generated using `multi-qa-mpnet-base-dot-v1` (768 dimensions) from sentence-transformers, with a TF-IDF hash-based fallback when the model is unavailable.

### Adaptive Search Weighting

The hybrid search layer classifies each query and applies dynamic weights:

| Query Type | Vector Weight | Keyword Weight | Trigger |
| :--- | :--- | :--- | :--- |
| Code queries | 0.50 | 0.50 | Code-like tokens detected |
| Factual queries | 0.60 | 0.40 | Specific entity or fact pattern |
| Conceptual queries | 0.80 | 0.20 | Abstract or relationship question |
| Short queries (1--2 words) | -0.10 adjustment | +0.10 adjustment | Token count <= 2 |
| Exact-match (quoted) | -0.20 adjustment | +0.20 adjustment | Quoted phrase detected |

Weights are further refined by a feedback learning loop (`search_query_log`, `search_feedback`, `weight_patterns` tables) that tracks which weight configurations produce the best user-rated results.

> **Guarantee:** Search quality improves over time without manual retuning, driven by observed user interactions.

---

## Hardest Problems Solved

### 1. Scraping Hostile Platforms at Scale

**Problem:** Instagram actively detects and blocks automated access. Rate limits, IP bans, CAPTCHA challenges, and session invalidation make reliable data collection non-trivial.

**Solution:** The scraper implements a multi-layer resilience strategy: rotating proxy pools, account credential cycling with persistent state tracking, adaptive cooldown periods that back off on detection signals, and graceful degradation that preserves partial progress. Account state (active, rate-limited, banned) is tracked per-session and persisted across runs.

### 2. Adaptive Retrieval Without Manual Tuning

**Problem:** A fixed vector-to-keyword weight ratio works well for some query types and poorly for others. Code queries need strong keyword matching; conceptual queries need strong semantic matching. Manual tuning does not scale.

**Solution:** The hybrid search system classifies each incoming query, applies a base weight configuration for the detected query type, then adjusts further based on query-specific signals (length, quoted phrases, code tokens). A feedback loop records user interactions and learns which weight patterns produce the best results for observed query distributions, progressively refining the default weights.

### 3. Structured Knowledge from Unstructured Text

**Problem:** Video transcripts and research papers contain latent concept relationships invisible to keyword and vector search. "Attention mechanism" and "transformer architecture" are deeply related, but a document about one may never mention the other by name.

**Solution:** The concept extraction pipeline uses Claude to identify concepts, classify them into a controlled taxonomy, and extract explicit relationships with confidence scores and relationship types. The resulting NetworkX graph makes latent relationships queryable -- enabling graph-based retrieval that surfaces documents connected through concept chains, not just direct textual similarity.

---

## System Domains

| Domain | Responsibility | Key Modules |
| :--- | :--- | :--- |
| **Ingestion** | Source-specific collection, rate-limit handling, credential management | `downloader.py`, `arxiv_collector.py`, `github_collector.py` |
| **Processing** | Transcription, OCR, summarization, text normalization | `transcriber.py`, `mistral_ocr.py`, `summarizer.py` |
| **Storage** | Schema management, migrations, unified content table, FTS indexes | `create_db.sql`, `db_migration.py`, `init_db.py` |
| **Knowledge** | Concept extraction, graph construction, centrality analysis | `concept_extractor.py`, `knowledge_graph.py` |
| **Embedding** | Text chunking, vector generation, batch processing | `chunking.py`, `embeddings.py`, `generate_embeddings.py` |
| **Retrieval** | Vector search, keyword search, hybrid fusion, adaptive weighting | `vector_search.py`, `hybrid_search.py`, `context_builder.py` |
| **Generation** | LLM context assembly, response generation, source citation | `llm_integration.py`, `context_builder.py` |
| **Evaluation** | Retrieval metrics, answer quality, test generation | `evaluation/*.py` |
| **Web** | Flask interface, REST API, Swagger documentation | `app.py`, `api/*.py` |

---

## Deep Dive: Technical Documentation

| Document | Focus Area |
| :--- | :--- |
| **[RAG Pipeline](src/README_RAG.md)** | End-to-end RAG usage, CLI commands, query API |
| **[Knowledge Graph](src/README_KNOWLEDGE_GRAPH.md)** | Concept extraction, graph analysis, visualization |
| **[Vector and Hybrid Search](src/README_VECTOR_SEARCH.md)** | Embedding generation, search strategies, adaptive weighting |
| **[ArXiv Collector](src/README_arxiv_collector.md)** | Paper collection, OCR pipeline, batch processing |
| **[Application Guide](src/README.md)** | Installation, configuration, CLI usage, web interface |

---

## Architectural Patterns

| Pattern | Implementation |
| :--- | :--- |
| **Source-Agnostic Schema** | Unified `ai_content` table with source-specific metadata in dedicated tables; downstream consumers are source-blind |
| **Adaptive Weighting** | Query classification, base weights, signal adjustments, feedback-refined weights via `weight_patterns` |
| **Concept Knowledge Graph** | LLM extraction into typed nodes and weighted edges, NetworkX analysis, queryable graph structure |
| **Batch LLM Processing** | Claude Message Batches API with async polling, UUID tracking, 50% cost reduction over sequential calls |
| **Graceful Degradation** | Mistral OCR with PyPDF2 fallback; sentence-transformers with TF-IDF hash fallback; partial progress preservation |
| **Evaluation-Driven Development** | Programmatic test query generation from knowledge graph; precision, recall, NDCG, MRR benchmarks |

---

## Evaluation Framework

The evaluation suite generates test queries programmatically from knowledge graph concepts, ensuring coverage evolves with the knowledge base. Metrics are computed across search strategies:

| Metric | Purpose |
| :--- | :--- |
| **Precision@k** | Fraction of retrieved results that are relevant |
| **Recall@k** | Fraction of relevant results that are retrieved |
| **F1@k** | Harmonic mean of precision and recall |
| **NDCG** | Normalized discounted cumulative gain -- measures ranking quality |
| **MRR** | Mean reciprocal rank -- measures position of first relevant result |

Results are viewable through an interactive evaluation dashboard.

---

<details>
<summary><h2>Folder Structure</h2></summary>
<br>

```
src/
├── run.py                          --- CLI entry point
├── app.py                          --- Flask web interface
├── downloader.py                   --- Instagram scraper, proxy rotation, rate limiting
├── transcriber.py                  --- Whisper audio transcription
├── summarizer.py                   --- Claude batch summarization
├── arxiv_collector.py              --- ArXiv paper collection + Mistral OCR
├── github_collector.py             --- GitHub repository collection
├── mistral_ocr.py                  --- Mistral AI OCR wrapper
│
├── embeddings.py                   --- Sentence-transformers embedding generation
├── generate_embeddings.py          --- Batch embedding orchestration
├── vector_search.py                --- Pure vector similarity search
├── hybrid_search.py                --- Hybrid vector + keyword search
├── chunking.py                     --- Text chunking with overlap
├── context_builder.py              --- RAG context selection and formatting
├── llm_integration.py              --- Claude response generation
│
├── concept_extractor.py            --- LLM-powered concept extraction
├── knowledge_graph.py              --- Graph construction, analysis, visualization
├── concept_schema.sql              --- Knowledge graph schema
│
├── create_db.sql                   --- Database schema
├── db_migration.py                 --- Schema migrations
├── init_db.py                      --- Database initialization
│
├── api/
│   ├── api.py                      --- REST API endpoints
│   ├── api_knowledge.py            --- Knowledge graph API
│   └── swagger.py                  --- OpenAPI specification
│
├── evaluation/
│   ├── retrieval_metrics.py        --- Precision, recall, NDCG, MRR
│   ├── answer_evaluator.py         --- Answer quality evaluation
│   ├── test_queries.py             --- Programmatic test generation
│   ├── test_runner.py              --- Evaluation orchestration
│   └── dashboard.py                --- Interactive results dashboard
│
├── templates/                      --- Flask HTML templates
├── data/
│   ├── audio/                      --- Transcribed audio files
│   ├── transcripts/                --- JSON transcript output
│   ├── papers/                     --- ArXiv paper text
│   ├── visualizations/             --- Knowledge graph renders
│   └── summaries_cache/            --- Cached Claude summaries
│
└── requirements.txt                --- Python dependencies
```

</details>

---

<div align="center">
  <img src="terminal-bottom-panel.svg" width="100%" alt="Terminal footer">
</div>
