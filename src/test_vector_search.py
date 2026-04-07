#!/usr/bin/env python3
"""
Test script for vector search functionality.

This script demonstrates and tests the different search capabilities of the system:
1. Vector search
2. Keyword search
3. Hybrid search with different weighting schemes
"""
import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any, Optional

# Import vector search modules
try:
    import embeddings
    import vector_search
    import hybrid_search
    import config
except ImportError as e:
    print(f"Error importing vector search modules: {str(e)}")
    print("Make sure you have installed all required dependencies from requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vector_search_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_vector_search')

# Sample test queries for different content types
TEST_QUERIES = {
    "general": [
        "How do transformer models work?",
        "What are the recent advances in natural language processing?",
        "Explain self-attention mechanism in neural networks",
        "Methods for improving computer vision models",
        "Best practices for machine learning model deployment"
    ],
    "code": [
        "Python function to implement K-means clustering",
        "How to optimize TensorFlow models for production",
        "Example of using BERT for text classification in PyTorch",
        "Code for implementing a convolutional neural network",
        "SQL query optimization techniques"
    ],
    "factual": [
        "When was GPT-3 released?",
        "Who created the transformer architecture?",
        "What is the BLEU score used for?",
        "How many parameters are in GPT-4?",
        "Which institutions are leading AI research?"
    ]
}

def test_vector_search(queries: List[str], top_k: int = 5, 
                      source_type: Optional[str] = None,
                      in_memory_index: bool = False) -> List[Dict[str, Any]]:
    """Run vector search tests for the provided queries"""
    results = []
    
    logger.info(f"Testing vector search with {len(queries)} queries")
    embedding_generator = embeddings.EmbeddingGenerator()
    
    # Create in-memory index if requested
    index = None
    if in_memory_index:
        logger.info("Creating in-memory index...")
        index = vector_search.create_memory_index()
        logger.info(f"Created in-memory index with {len(index.get('vectors', []))} vectors")
    
    # Run tests for each query
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        start_time = time.time()
        
        if in_memory_index and index and len(index.get('vectors', [])) > 0:
            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query)
            
            # Search using in-memory index
            query_results = vector_search.search_memory_index(
                query_embedding, index, top_k=top_k
            )
            
            # Fetch chunk text and enrich results
            query_results = vector_search.enrich_search_results(query_results)
        else:
            # Use standard search
            query_results = vector_search.search_by_text(
                query_text=query,
                top_k=top_k,
                source_type=source_type,
                embedding_generator=embedding_generator
            )
        
        elapsed = time.time() - start_time
        
        # Add to results
        results.append({
            'query': query,
            'results': query_results,
            'count': len(query_results),
            'elapsed': elapsed
        })
        
        logger.info(f"Found {len(query_results)} results in {elapsed:.4f} seconds")
    
    return results

def test_keyword_search(queries: List[str], top_k: int = 5,
                       source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run keyword search tests for the provided queries"""
    results = []
    
    logger.info(f"Testing keyword search with {len(queries)} queries")
    
    # Run tests for each query
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        start_time = time.time()
        
        # Perform keyword search
        query_results = hybrid_search.keyword_search(
            query=query,
            top_k=top_k,
            source_type=source_type
        )
        
        elapsed = time.time() - start_time
        
        # Add to results
        results.append({
            'query': query,
            'results': query_results,
            'count': len(query_results),
            'elapsed': elapsed
        })
        
        logger.info(f"Found {len(query_results)} results in {elapsed:.4f} seconds")
    
    return results

def test_hybrid_search(queries: List[str], top_k: int = 5,
                      source_type: Optional[str] = None,
                      vector_weight: Optional[float] = None,
                      keyword_weight: Optional[float] = None,
                      adaptive: bool = True) -> List[Dict[str, Any]]:
    """Run hybrid search tests for the provided queries"""
    results = []
    
    logger.info(f"Testing hybrid search with {len(queries)} queries")
    embedding_generator = embeddings.EmbeddingGenerator()
    
    # Load weight history if using adaptive weighting
    weight_history = None
    if adaptive and (vector_weight is None or keyword_weight is None):
        weight_history = hybrid_search.load_weights_history()
    
    # Run tests for each query
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {query}")
        
        # Determine weights if adaptive
        query_vector_weight = vector_weight
        query_keyword_weight = keyword_weight
        
        if adaptive and (query_vector_weight is None or query_keyword_weight is None):
            if weight_history:
                # Get query type
                query_type = hybrid_search.classify_query_type(query)
                
                # Use query-specific weights if available
                normalized_query = query.lower().strip()
                if normalized_query in weight_history['queries']:
                    query_vector_weight = weight_history['queries'][normalized_query]['vector_weight']
                    query_keyword_weight = weight_history['queries'][normalized_query]['keyword_weight']
                    logger.info(f"Using query-specific weights: vector={query_vector_weight:.2f}, keyword={query_keyword_weight:.2f}")
                else:
                    # Use query type defaults
                    type_key = f"default_{query_type}_vector_weight"
                    if type_key in weight_history:
                        query_vector_weight = weight_history[type_key]
                        query_keyword_weight = 1.0 - query_vector_weight
                        logger.info(f"Using {query_type} query type weights: vector={query_vector_weight:.2f}, keyword={query_keyword_weight:.2f}")
                    else:
                        # Determine weights based on query
                        query_vector_weight, query_keyword_weight = hybrid_search.determine_weights(query)
                        logger.info(f"Using determined weights: vector={query_vector_weight:.2f}, keyword={query_keyword_weight:.2f}")
        
        start_time = time.time()
        
        # Perform hybrid search
        query_results = hybrid_search.hybrid_search(
            query=query,
            top_k=top_k,
            source_type=source_type,
            vector_weight=query_vector_weight,
            keyword_weight=query_keyword_weight,
            embedding_generator=embedding_generator
        )
        
        elapsed = time.time() - start_time
        
        # Add to results
        results.append({
            'query': query,
            'results': query_results,
            'count': len(query_results),
            'elapsed': elapsed,
            'vector_weight': query_vector_weight,
            'keyword_weight': query_keyword_weight
        })
        
        logger.info(f"Found {len(query_results)} results in {elapsed:.4f} seconds with weights: vector={query_vector_weight:.2f}, keyword={query_keyword_weight:.2f}")
    
    return results

def compare_search_methods(queries: List[str], top_k: int = 5,
                          source_type: Optional[str] = None) -> Dict[str, Any]:
    """Compare vector, keyword, and hybrid search for the provided queries"""
    logger.info(f"Comparing search methods for {len(queries)} queries")
    
    # Run each search method
    vector_results = test_vector_search(queries, top_k, source_type)
    keyword_results = test_keyword_search(queries, top_k, source_type)
    hybrid_results = test_hybrid_search(queries, top_k, source_type)
    
    # Compile comparison stats
    comparison = []
    for i, query in enumerate(queries):
        vector_count = vector_results[i]['count'] if i < len(vector_results) else 0
        vector_time = vector_results[i]['elapsed'] if i < len(vector_results) else 0
        
        keyword_count = keyword_results[i]['count'] if i < len(keyword_results) else 0
        keyword_time = keyword_results[i]['elapsed'] if i < len(keyword_results) else 0
        
        hybrid_count = hybrid_results[i]['count'] if i < len(hybrid_results) else 0
        hybrid_time = hybrid_results[i]['elapsed'] if i < len(hybrid_results) else 0
        hybrid_vector_weight = hybrid_results[i].get('vector_weight', 0) if i < len(hybrid_results) else 0
        hybrid_keyword_weight = hybrid_results[i].get('keyword_weight', 0) if i < len(hybrid_results) else 0
        
        # Add to comparison
        comparison.append({
            'query': query,
            'vector': {
                'count': vector_count,
                'time': vector_time
            },
            'keyword': {
                'count': keyword_count,
                'time': keyword_time
            },
            'hybrid': {
                'count': hybrid_count,
                'time': hybrid_time,
                'vector_weight': hybrid_vector_weight,
                'keyword_weight': hybrid_keyword_weight
            }
        })
    
    # Calculate averages
    avg_vector_count = sum(c['vector']['count'] for c in comparison) / len(comparison)
    avg_vector_time = sum(c['vector']['time'] for c in comparison) / len(comparison)
    
    avg_keyword_count = sum(c['keyword']['count'] for c in comparison) / len(comparison)
    avg_keyword_time = sum(c['keyword']['time'] for c in comparison) / len(comparison)
    
    avg_hybrid_count = sum(c['hybrid']['count'] for c in comparison) / len(comparison)
    avg_hybrid_time = sum(c['hybrid']['time'] for c in comparison) / len(comparison)
    
    return {
        'comparisons': comparison,
        'summary': {
            'vector': {
                'avg_count': avg_vector_count,
                'avg_time': avg_vector_time
            },
            'keyword': {
                'avg_count': avg_keyword_count,
                'avg_time': avg_keyword_time
            },
            'hybrid': {
                'avg_count': avg_hybrid_count,
                'avg_time': avg_hybrid_time
            }
        }
    }

def print_results(results: List[Dict[str, Any]], detailed: bool = False) -> None:
    """Print search results"""
    for i, result in enumerate(results):
        query = result['query']
        query_results = result['results']
        elapsed = result['elapsed']
        
        print(f"\nQuery {i+1}: {query}")
        print(f"Found {len(query_results)} results in {elapsed:.4f} seconds")
        
        if 'vector_weight' in result and 'keyword_weight' in result:
            print(f"Weights: vector={result['vector_weight']:.2f}, keyword={result['keyword_weight']:.2f}")
        
        if detailed and query_results:
            print("-" * 80)
            
            for j, res in enumerate(query_results):
                # Print common result details
                print(f"Result {j+1}")
                print(f"Content ID: {res.get('content_id', 'Unknown')}")
                print(f"Title: {res.get('title', 'Unknown')}")
                print(f"Source: {res.get('source_type', 'Unknown')}")
                
                # Print score details
                if 'similarity' in res:
                    print(f"Similarity: {res['similarity']:.4f}")
                if 'score' in res:
                    print(f"Score: {res['score']:.4f}")
                if 'combined_score' in res:
                    print(f"Combined Score: {res['combined_score']:.4f}")
                    print(f"Vector Score: {res.get('vector_score', 0):.4f}, Keyword Score: {res.get('keyword_score', 0):.4f}")
                
                # Print snippet or chunk text
                if 'snippet' in res and res['snippet']:
                    print(f"Keyword Match: {res['snippet']}")
                
                if 'chunk_text' in res and res['chunk_text']:
                    text = res['chunk_text']
                    print(f"Text: {text[:150]}..." if len(text) > 150 else f"Text: {text}")
                
                print("-" * 40)
        
        print("-" * 80)

def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print comparison results between search methods"""
    comparisons = comparison['comparisons']
    summary = comparison['summary']
    
    print("\nSearch Method Comparison")
    print("=" * 80)
    
    # Print summary
    print("\nSummary:")
    print(f"Vector Search:  Avg Results: {summary['vector']['avg_count']:.2f}, Avg Time: {summary['vector']['avg_time']:.4f}s")
    print(f"Keyword Search: Avg Results: {summary['keyword']['avg_count']:.2f}, Avg Time: {summary['keyword']['avg_time']:.4f}s")
    print(f"Hybrid Search:  Avg Results: {summary['hybrid']['avg_count']:.2f}, Avg Time: {summary['hybrid']['avg_time']:.4f}s")
    
    # Print per-query comparison
    print("\nPer-Query Results:")
    for i, comp in enumerate(comparisons):
        query = comp['query']
        
        print(f"\nQuery {i+1}: {query}")
        print(f"Vector:  {comp['vector']['count']} results in {comp['vector']['time']:.4f}s")
        print(f"Keyword: {comp['keyword']['count']} results in {comp['keyword']['time']:.4f}s")
        print(f"Hybrid:  {comp['hybrid']['count']} results in {comp['hybrid']['time']:.4f}s (weights: {comp['hybrid']['vector_weight']:.2f}:{comp['hybrid']['keyword_weight']:.2f})")
    
    print("=" * 80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test vector search functionality")
    parser.add_argument("--vector", action="store_true", help="Test vector search")
    parser.add_argument("--keyword", action="store_true", help="Test keyword search")
    parser.add_argument("--hybrid", action="store_true", help="Test hybrid search")
    parser.add_argument("--compare", action="store_true", help="Compare all search methods")
    parser.add_argument("--query-type", choices=["general", "code", "factual", "all"], default="all", 
                        help="Type of test queries to use")
    parser.add_argument("--custom-query", help="Run tests with a custom query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--source-type", choices=["research_paper", "github", "instagram"], 
                        help="Filter by source type")
    parser.add_argument("--vector-weight", type=float, help="Weight for vector search (0-1)")
    parser.add_argument("--keyword-weight", type=float, help="Weight for keyword search (0-1)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive weighting based on query")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory indexing for vector search")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Determine test queries
    test_queries = []
    
    if args.custom_query:
        test_queries = [args.custom_query]
    elif args.query_type == "all":
        for query_type in TEST_QUERIES:
            test_queries.extend(TEST_QUERIES[query_type])
    else:
        test_queries = TEST_QUERIES.get(args.query_type, [])
        if not test_queries:
            logger.error(f"No test queries available for type: {args.query_type}")
            return
    
    # Run tests
    if args.vector:
        logger.info("Running vector search tests")
        results = test_vector_search(
            queries=test_queries,
            top_k=args.top_k,
            source_type=args.source_type,
            in_memory_index=args.in_memory
        )
        
        print("\nVector Search Results:")
        print_results(results, detailed=args.detailed)
    
    if args.keyword:
        logger.info("Running keyword search tests")
        results = test_keyword_search(
            queries=test_queries,
            top_k=args.top_k,
            source_type=args.source_type
        )
        
        print("\nKeyword Search Results:")
        print_results(results, detailed=args.detailed)
    
    if args.hybrid:
        logger.info("Running hybrid search tests")
        results = test_hybrid_search(
            queries=test_queries,
            top_k=args.top_k,
            source_type=args.source_type,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            adaptive=args.adaptive
        )
        
        print("\nHybrid Search Results:")
        print_results(results, detailed=args.detailed)
    
    if args.compare:
        logger.info("Running comparison of search methods")
        comparison = compare_search_methods(
            queries=test_queries,
            top_k=args.top_k,
            source_type=args.source_type
        )
        
        print_comparison(comparison)
    
    # Show help if no test specified
    if not (args.vector or args.keyword or args.hybrid or args.compare):
        parser.print_help()

if __name__ == "__main__":
    main() 