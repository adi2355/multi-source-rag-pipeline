# Vector Search in Instagram Knowledge Base

This document explains how to use the vector search capabilities of the Instagram Knowledge Base system. Vector search provides semantic search capabilities, allowing users to find content based on meaning rather than just keywords.

## Installation

First, make sure you have all the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

Some additional dependencies for vector search might require specific system packages. If you encounter any issues during installation, check the error messages for specific requirements.

## Database Migration

Before using vector search, make sure your database is migrated to the latest schema:

```bash
python run.py --migrate
```

This will create all necessary tables, including:
- `content_embeddings`: Stores vector embeddings for content chunks
- `ai_content`: Unified content storage across different source types
- `ai_content_fts`: Full-text search virtual table for keyword search

## Generating Embeddings

Before you can search, you need to generate embeddings for your content:

```bash
python run.py --generate-embeddings
```

This will generate vector embeddings for all content in your database. You can customize the process with these options:

- `--embeddings-source [research_paper|github|instagram]`: Only generate embeddings for a specific content type
- `--embeddings-limit N`: Process only N content items
- `--embeddings-batch N`: Process in batches of N items (default: 50)
- `--embeddings-chunk N`: Set chunk size in characters (default: 500)
- `--embeddings-overlap N`: Set chunk overlap in characters (default: 100)
- `--embeddings-force`: Regenerate embeddings for content that already has them

Example:
```bash
python run.py --generate-embeddings --embeddings-source research_paper --embeddings-limit 100 --embeddings-chunk 1000 --embeddings-overlap 200
```

## Performing Searches

### Vector Search

Vector search finds content semantically similar to the query:

```bash
python run.py --vector-search "deep learning techniques for image classification"
```

Options:
- `--search-top-k N`: Return top N results (default: 5)
- `--search-source [research_paper|github|instagram]`: Search only in a specific content type
- `--in-memory-index`: Use in-memory indexing for faster searches (uses more RAM)

### Hybrid Search

Hybrid search combines vector search with keyword search for better results:

```bash
python run.py --hybrid-search "deep learning techniques for image classification"
```

Options:
- `--search-top-k N`: Return top N results (default: 5)
- `--search-source [research_paper|github|instagram]`: Search only in a specific content type
- `--vector-weight N`: Weight for vector search component (0-1)
- `--keyword-weight N`: Weight for keyword search component (0-1)
- `--adaptive-weights`: Use adaptive weighting based on query type (default: true)

If you don't specify weights, the system will automatically determine appropriate weights based on the query characteristics.

## Query Types and Weight Adaptation

The system classifies queries into several types and applies different vector-to-keyword weights:

1. **Code Queries** (e.g., "python function for sorting arrays"): 0.5:0.5
2. **Factual Queries** (e.g., "when was transformer architecture introduced"): 0.6:0.4
3. **Concept Queries** (e.g., "explain attention mechanism in transformers"): 0.8:0.2
4. **Short Queries** (1-2 words): Reduce vector weight by 0.1
5. **Long Queries** (6+ words): Increase vector weight by 0.1
6. **Exact Match Queries** (with quotes): Reduce vector weight by 0.2

The system learns from user feedback to improve these weights over time.

## Advanced: Directly Using the Modules

For more advanced usage, you can import the modules directly in your code:

```python
from embeddings import EmbeddingGenerator
from vector_search import search_by_text
from hybrid_search import hybrid_search

# Generate embeddings for a query
embedding_generator = EmbeddingGenerator()
query_embedding = embedding_generator.generate_embedding("your query text")

# Perform vector search
vector_results = search_by_text(
    query_text="your query text",
    top_k=5,
    embedding_generator=embedding_generator
)

# Perform hybrid search
hybrid_results = hybrid_search(
    query="your query text",
    top_k=5,
    vector_weight=0.7,
    keyword_weight=0.3
)
```

## Troubleshooting

### Missing Sentence Transformers Model

If you see an error about missing models, the system will fall back to a simpler TF-IDF based embedding. For better results, make sure you have internet access during the first run so the model can be downloaded.

### Performance Issues

- For large databases, use `--in-memory-index` for faster searching
- Adjust chunk size and overlap based on your content type
- Use more specific queries for better results

### No Search Results

- Make sure you've generated embeddings first
- Check if you have content in your database
- Try using broader queries or different source types

## Further Reading

- Learn more about vector search in the code comments in `/home/adi235/MistralOCR/Instagram-Scraper/vector_search.py`
- Check hybrid search implementation in `/home/adi235/MistralOCR/Instagram-Scraper/hybrid_search.py`
- See the chunking strategy in `/home/adi235/MistralOCR/Instagram-Scraper/chunking.py` 