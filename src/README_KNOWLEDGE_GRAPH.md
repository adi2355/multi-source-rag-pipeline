# Knowledge Graph & Concept Extraction

This module provides tools for extracting AI/ML concepts from content, organizing them into a knowledge graph, and visualizing concept relationships.

## Overview

The Knowledge Graph module is built on top of the concept extraction system and provides:

1. **Concept Extraction**: Extract AI/ML concepts and their relationships from content using Claude
2. **Knowledge Graph Construction**: Build a graph representation of concepts and relationships
3. **Graph Analysis**: Analyze concept centrality, communities, and relationship patterns
4. **Visualization**: Generate interactive and static visualizations of the concept knowledge graph
5. **Command-line Interface**: Access all functionality through command-line arguments

## Requirements

The Knowledge Graph system requires the following Python packages:

- `networkx` - For graph construction and analysis
- `matplotlib` - For static visualizations
- `plotly` - For interactive HTML visualizations
- `anthropic` - For concept extraction using Claude

All required packages are included in the main `requirements.txt` file.

## Getting Started

To use the Knowledge Graph system, follow these steps:

1. **Extract Concepts**: First, you need to extract concepts from content in the database
   ```bash
   python run.py --concepts --concepts-batch
   ```

2. **Analyze the Knowledge Graph**: View statistics about the extracted concepts and relationships
   ```bash
   python run.py --kg-analyze
   ```

3. **Generate Visualizations**: Create visual representations of the concept graph
   ```bash
   python run.py --kg-visualize
   ```

## Command-line Arguments

The following command-line arguments are available for Knowledge Graph operations:

### Concept Extraction

- `--concepts`: Run AI concept extraction
- `--concepts-batch`: Run extraction in batch mode (recommended)
- `--concepts-batch-size N`: Set batch size for extraction (default: 10)
- `--concepts-limit N`: Maximum number of items to process
- `--concepts-source TYPE`: Only extract from a specific source type (`research_paper`, `github`, or `instagram`)
- `--concepts-force`: Force reprocessing of content that already has concepts

### Knowledge Graph Analysis

- `--kg-analyze`: Show knowledge graph statistics
- `--kg-concept ID`: Analyze a specific concept by ID
- `--kg-search "TERM"`: Search and visualize concepts matching a term
- `--kg-visualize`: Generate standard visualizations of the knowledge graph
- `--kg-output-dir DIR`: Set the output directory for visualizations (default: `visualizations`)

## Example Usage

### Extract Concepts from All Content

```bash
python run.py --concepts --concepts-batch
```

### Extract Concepts from Specific Source Type

```bash
python run.py --concepts --concepts-source research_paper --concepts-batch
```

### View Knowledge Graph Statistics

```bash
python run.py --kg-analyze
```

### Generate Visualizations

```bash
python run.py --kg-visualize --kg-output-dir my_visualizations
```

### Analyze a Specific Concept

```bash
# First, find a concept ID using search:
python run.py --kg-search "transformer"

# Then analyze that specific concept:
python run.py --kg-concept 123
```

### Search and Visualize Concepts

```bash
python run.py --kg-search "neural network" --kg-output-dir visualizations
```

## Available Visualizations

When you run `--kg-visualize`, the system generates:

1. **Static graph image** (`knowledge_graph.png`) - A PNG visualization using matplotlib
2. **Interactive HTML** (`knowledge_graph.html`) - An interactive visualization using Plotly
3. **Community visualization** (`knowledge_graph_communities.html`) - Communities of related concepts
4. **JSON export** (`knowledge_graph.json`) - Graph data in JSON format for custom visualization
5. **GEXF export** (`knowledge_graph.gexf`) - Graph data for use in Gephi (advanced graph visualization tool)

## Direct CLI Access (Advanced)

For advanced usage, you can directly use the `knowledge_graph.py` module, which provides a richer set of commands:

```bash
# Query the knowledge graph
python knowledge_graph.py query --concepts
python knowledge_graph.py query --category "algorithm"
python knowledge_graph.py query --search "transformer"
python knowledge_graph.py query --related 42

# Generate visualizations
python knowledge_graph.py visualize --concept-id 42 --interactive
python knowledge_graph.py visualize --all --output-dir my_visualizations

# Analyze the graph
python knowledge_graph.py analyze --stats
python knowledge_graph.py analyze --concept-id 42
python knowledge_graph.py analyze --communities
python knowledge_graph.py analyze --centrality
```

## Using the Knowledge Graph API

You can also use the Knowledge Graph API directly in your Python code:

```python
from knowledge_graph import ConceptQuery, KnowledgeGraph, GraphVisualizer, KnowledgeGraphManager

# Query concepts
query = ConceptQuery()
concepts = query.search_concepts("neural network")
related = query.get_related_concepts(concept_id=42)

# Build and analyze graph
kg = KnowledgeGraph()
graph = kg.build_graph(min_confidence=0.6)
central_concepts = kg.get_central_concepts(centrality_type="pagerank")
communities = kg.get_concept_communities()

# Generate visualizations
viz = GraphVisualizer(graph)
viz.visualize_interactive(output_file="my_graph.html")
```

## Future Extensions

Future enhancements to the Knowledge Graph system may include:

1. Web interface for knowledge graph exploration
2. Natural language querying of the knowledge graph
3. Integration with other concept sources
4. Time-based analysis of concept evolution
5. Enhanced visualization options with custom layouts

For issues or feature requests, please submit an issue to the project repository. 