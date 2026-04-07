# RAG Implementation for Instagram Knowledge Base

This document explains the Retrieval Augmented Generation (RAG) system implemented in the Instagram Knowledge Base.

## Overview

The RAG system extends the vector search capabilities with:

1. **Context Builder**: Intelligently selects and formats chunks of retrieved content
2. **LLM Integration**: Connects with Claude or other LLMs to generate high-quality responses
3. **Command-line Interface**: Easy-to-use CLI for RAG queries

## Requirements

The following packages are required for RAG functionality:
- anthropic>=0.40.0
- sentence-transformers>=2.2.2
- torch>=2.0.0
- transformers>=4.30.0

These dependencies are already included in the `requirements.txt` file.

## Getting Started

Before using the RAG system, make sure you have:

1. Completed the database migration: `python run.py --migrate`
2. Generated embeddings for your content: `python run.py --generate-embeddings`
3. Set your Anthropic API key as an environment variable: `export ANTHROPIC_API_KEY=your_api_key`

## Using the RAG System

### Basic Usage

To ask a question using RAG:

```bash
python run.py --rag-query "What techniques are used for deep learning?"
```

This will:
1. Search for relevant content using hybrid search
2. Build a context from the search results
3. Generate a response using Claude
4. Save the response to a file

### Advanced Options

The RAG system supports various options:

```bash
# Use vector search instead of hybrid search
python run.py --rag-query "What is PyTorch?" --rag-search-type vector

# Filter results by source type (github, research_paper, instagram)
python run.py --rag-query "How to use PyTorch?" --search-source github

# Adjust token limits for context and response
python run.py --rag-query "Explain LSTM networks" --rag-max-tokens-context 6000 --rag-max-tokens-answer 2000

# Adjust the temperature for more precise or creative responses
python run.py --rag-query "Compare TensorFlow and PyTorch" --rag-temperature 0.7

# Stream the response as it's generated
python run.py --rag-query "What are transformers?" --rag-stream

# Specify the Claude model to use
python run.py --rag-query "Explain GAN architecture" --rag-model claude-3-sonnet-20240229
```

### Adjusting Search Weights

You can adjust the balance between vector and keyword search:

```bash
# Emphasize vector search (semantic meaning)
python run.py --rag-query "What is the intuition behind neural networks?" --vector-weight 0.8 --keyword-weight 0.2

# Emphasize keyword search (exact matching)
python run.py --rag-query "PyTorch CUDA error" --vector-weight 0.3 --keyword-weight 0.7
```

## Module Components

### Context Builder

The `context_builder.py` module:
- Selects diverse and relevant chunks from search results
- Formats them with source citations
- Respects token limits
- Builds prompts for LLM consumption
- Provides metadata for sources

### LLM Integration

The `llm_integration.py` module:
- Supports different LLM providers (currently Claude)
- Handles API communication
- Implements retry logic
- Supports streaming responses
- Saves responses for future reference

## Custom Usage

You can use the RAG components in your own Python code:

```python
from context_builder import ContextBuilder
from llm_integration import RAGAssistant, ClaudeProvider

# Create components
llm = ClaudeProvider(model="claude-3-opus-20240229")
ctx_builder = ContextBuilder(max_tokens=4000)

# Create RAG assistant
rag = RAGAssistant(
    llm_provider=llm,
    context_builder=ctx_builder,
    max_tokens_answer=1000
)

# Answer query
response = rag.answer_query(
    query="What are the advantages of transformers over RNNs?",
    search_type="hybrid",
    top_k=10
)

# Use the result
print(response["answer"])
for source in response["sources"]:
    print(f"Source: {source['title']} ({source['source_type']})")
```

## Future Extensions

Planned enhancements for the RAG system:
- Support for additional LLMs (OpenAI, Mistral, Llama, etc.)
- Web interface for RAG queries
- Learning from user feedback
- Chat history/conversation support
- More precise token counting
- Custom prompt templates for different query types 