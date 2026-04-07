"""
Hybrid Search Module

This module combines vector search and keyword search to provide more accurate
and relevant search results. It implements adaptive weighting between the two search
techniques based on the query characteristics.
"""
import os
import logging
import sqlite3
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# Import local modules
import config
from embeddings import EmbeddingGenerator
from vector_search import search_by_text, enrich_search_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hybrid_search')

def keyword_search(query: str, top_k: int = 10, 
                  source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Perform keyword search using SQLite FTS
    
    Args:
        query: The search query
        top_k: Maximum number of results to return
        source_type: Optional filter for source type
        
    Returns:
        List of search results
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Check if FTS table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ai_content_fts'
        """)
        
        if not cursor.fetchone():
            logger.warning("FTS table does not exist, creating it...")
            # Create FTS table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS ai_content_fts USING fts5(
                    title, description, content, 
                    content='ai_content', content_rowid='id'
                )
            """)
            
            # Populate FTS table
            cursor.execute("""
                INSERT INTO ai_content_fts(rowid, title, description, content)
                SELECT id, title, description, content FROM ai_content
            """)
            
            conn.commit()
        
        # Build search query
        search_terms = " OR ".join([f'"{term}"' for term in query.split()])
        fts_query = f"{search_terms}"
        
        sql_query = """
            SELECT 
                c.id, c.title, c.description, c.date_created, 
                c.source_type_id, st.name as source_type_name,
                snippet(ai_content_fts, 2, '<b>', '</b>', '...', 30) as snippet,
                rank
            FROM ai_content_fts
            JOIN ai_content c ON c.id = ai_content_fts.rowid
            JOIN source_types st ON c.source_type_id = st.id
        """
        
        params = []
        
        # Add source type filter if provided
        if source_type:
            cursor.execute("SELECT id FROM source_types WHERE name = ?", (source_type,))
            source_type_id = cursor.fetchone()
            if source_type_id:
                sql_query += " WHERE c.source_type_id = ?"
                params.append(source_type_id[0])
            else:
                logger.warning(f"Source type '{source_type}' not found")
        
        # Add search condition and order by rank
        sql_query += f""" 
            {'WHERE' if not source_type else 'AND'} ai_content_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        
        params.extend([fts_query, top_k])
        
        # Execute query
        cursor.execute(sql_query, params)
        
        # Process results
        results = []
        for row in cursor.fetchall():
            content_id, title, description, date_created, source_type_id, source_type_name, snippet, rank = row
            
            # Rank is between 0 (best) and 1 (worst), invert for consistency with vector search
            normalized_score = 1.0 - (float(rank) / 1000.0 if float(rank) > 0 else 0)
            
            # Add to results
            results.append({
                'content_id': content_id,
                'title': title,
                'description': description,
                'date_created': date_created,
                'source_type': source_type_name,
                'snippet': snippet,
                'score': normalized_score,
                'search_type': 'keyword'
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in keyword search: {str(e)}")
        return []
        
    finally:
        if conn:
            conn.close()

def get_content_chunks(content_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get chunks for a content item
    
    Args:
        content_id: The content ID
        limit: Maximum number of chunks to return
        
    Returns:
        List of chunks with text and metadata
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get chunks for the content item
        cursor.execute("""
            SELECT 
                ce.chunk_index, ce.chunk_text,
                ac.title, ac.description, ac.source_type_id, st.name as source_type_name
            FROM content_embeddings ce
            JOIN ai_content ac ON ce.content_id = ac.id
            JOIN source_types st ON ac.source_type_id = st.id
            WHERE ce.content_id = ?
            ORDER BY ce.chunk_index
            LIMIT ?
        """, (content_id, limit))
        
        chunks = []
        for row in cursor.fetchall():
            chunk_index, chunk_text, title, description, source_type_id, source_type_name = row
            
            chunks.append({
                'content_id': content_id,
                'chunk_index': chunk_index,
                'chunk_text': chunk_text,
                'title': title,
                'description': description,
                'source_type': source_type_name
            })
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error getting content chunks: {str(e)}")
        return []
        
    finally:
        if conn:
            conn.close()

def determine_weights(query: str) -> Tuple[float, float]:
    """
    Determine weights for vector and keyword search based on query characteristics
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (vector_weight, keyword_weight)
    """
    # Default weights
    default_vector_weight = 0.7
    default_keyword_weight = 0.3
    
    # Check for code indicators
    code_patterns = [
        r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b',
        r'\bimport\b', r'\bdef\b', r'\bpython\b', r'\bjava\b', 
        r'\bjavascript\b', r'\bjs\b', r'\bc\+\+\b', r'\bsql\b'
    ]
    
    # Check for factual query indicators
    factual_patterns = [
        r'\bwho\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', 
        r'\bhow\b', r'\bwhy\b', r'\bdate\b', r'\byear\b',
        r'\bauthor\b', r'\bcreator\b', r'\bversions?\b'
    ]
    
    # Check for concept query indicators
    concept_patterns = [
        r'\bconcept\b', r'\btheory\b', r'\bframework\b', r'\barchitecture\b',
        r'\bparadigm\b', r'\bpattern\b', r'\bapproach\b', r'\bstrategy\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bcompare\b', r'\banalyze\b'
    ]
    
    # Count matches for each category
    code_matches = sum(1 for pattern in code_patterns if re.search(pattern, query.lower()))
    factual_matches = sum(1 for pattern in factual_patterns if re.search(pattern, query.lower()))
    concept_matches = sum(1 for pattern in concept_patterns if re.search(pattern, query.lower()))
    
    # Adjust weights based on query type
    if code_matches > 0:
        # Code queries benefit from more keyword search
        vector_weight = 0.5
        keyword_weight = 0.5
    elif factual_matches > concept_matches:
        # Factual queries benefit from more keyword search
        vector_weight = 0.6
        keyword_weight = 0.4
    elif concept_matches > 0:
        # Concept queries benefit from more vector search
        vector_weight = 0.8
        keyword_weight = 0.2
    else:
        # Use default weights
        vector_weight = default_vector_weight
        keyword_weight = default_keyword_weight
    
    # Also consider query length and specificity
    query_words = query.split()
    
    if len(query_words) <= 2:
        # Short queries often benefit from more keyword search
        vector_weight = max(0.4, vector_weight - 0.1)
        keyword_weight = 1.0 - vector_weight
    elif len(query_words) >= 6:
        # Longer queries often benefit from more vector search
        vector_weight = min(0.9, vector_weight + 0.1)
        keyword_weight = 1.0 - vector_weight
    
    # Check for exact quotes which indicates keyword preference
    if '"' in query:
        vector_weight = max(0.3, vector_weight - 0.2)
        keyword_weight = 1.0 - vector_weight
    
    return vector_weight, keyword_weight

def hybrid_search(query: str, top_k: int = 5, 
                 source_type: Optional[str] = None,
                 vector_weight: Optional[float] = None,
                 keyword_weight: Optional[float] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector and keyword search
    
    Args:
        query: The search query
        top_k: Maximum number of results to return
        source_type: Optional filter for source type
        vector_weight: Weight for vector search results (0-1)
        keyword_weight: Weight for keyword search results (0-1)
        embedding_generator: Optional pre-initialized embedding generator
        
    Returns:
        List of search results with combined scores
    """
    try:
        # Determine weights if not provided
        if vector_weight is None or keyword_weight is None:
            vector_weight, keyword_weight = determine_weights(query)
        
        logger.info(f"Hybrid search for '{query}' with weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        
        # Double the top_k for individual searches to have more candidates to combine
        expanded_top_k = top_k * 2
        
        # Perform vector search
        vector_results = search_by_text(
            query_text=query,
            top_k=expanded_top_k,
            source_type=source_type,
            embedding_generator=embedding_generator
        )
        
        # Convert vector results format
        formatted_vector_results = []
        for result in vector_results:
            formatted_vector_results.append({
                'content_id': result['content_id'],
                'chunk_index': result.get('chunk_index', 0),
                'score': result['similarity'],
                'chunk_text': result.get('chunk_text', ''),
                'title': result.get('title', ''),
                'source_type': result.get('source_type', ''),
                'search_type': 'vector'
            })
        
        # Perform keyword search
        keyword_results = keyword_search(
            query=query,
            top_k=expanded_top_k,
            source_type=source_type
        )
        
        # Combine results
        all_results = {}
        
        # Add vector results
        for result in formatted_vector_results:
            content_id = result['content_id']
            # Use the content_id as the key to avoid duplicates
            if content_id not in all_results:
                all_results[content_id] = {
                    'content_id': content_id,
                    'vector_score': result['score'],
                    'keyword_score': 0.0,
                    'chunk_text': result.get('chunk_text', ''),
                    'title': result.get('title', ''),
                    'source_type': result.get('source_type', ''),
                    'has_vector_match': True,
                    'has_keyword_match': False,
                    'search_type': 'hybrid'
                }
            else:
                # Update if the new vector score is higher
                if result['score'] > all_results[content_id]['vector_score']:
                    all_results[content_id]['vector_score'] = result['score']
                    all_results[content_id]['chunk_text'] = result.get('chunk_text', '')
        
        # Add keyword results
        for result in keyword_results:
            content_id = result['content_id']
            if content_id not in all_results:
                # Get chunks for this content
                chunks = get_content_chunks(content_id, limit=1)
                chunk_text = chunks[0]['chunk_text'] if chunks else ''
                
                all_results[content_id] = {
                    'content_id': content_id,
                    'vector_score': 0.0,
                    'keyword_score': result['score'],
                    'chunk_text': chunk_text,
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'source_type': result.get('source_type', ''),
                    'has_vector_match': False,
                    'has_keyword_match': True,
                    'search_type': 'hybrid'
                }
            else:
                # Update keyword score and flag
                all_results[content_id]['keyword_score'] = result['score']
                all_results[content_id]['has_keyword_match'] = True
                all_results[content_id]['snippet'] = result.get('snippet', '')
        
        # Calculate combined scores
        for content_id, result in all_results.items():
            result['combined_score'] = (
                vector_weight * result['vector_score'] + 
                keyword_weight * result['keyword_score']
            )
        
        # Convert to list and sort by combined score
        results_list = list(all_results.values())
        results_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add query to results
        for result in results_list:
            result['query'] = query
        
        # Return top k results
        top_results = results_list[:top_k]
        
        # Enrich the top results with additional metadata
        return enrich_search_results(top_results)
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []

def adjust_weights_from_feedback(query: str, result_id: int, 
                                feedback: str, 
                                weight_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Adjust search weights based on user feedback
    
    Args:
        query: The search query
        result_id: The ID of the result that received feedback
        feedback: The feedback ('relevant' or 'not_relevant')
        weight_history: Optional history of weight adjustments
        
    Returns:
        Updated weight history
    """
    # Initialize weight history if not provided
    if weight_history is None:
        weight_history = {
            'queries': {},
            'default_vector_weight': 0.7,
            'default_keyword_weight': 0.3,
        }
    
    # Normalize query for consistent keys
    normalized_query = query.lower().strip()
    query_type = classify_query_type(query)
    
    # Initialize query entry if it doesn't exist
    if normalized_query not in weight_history['queries']:
        weight_history['queries'][normalized_query] = {
            'vector_weight': weight_history.get('default_vector_weight', 0.7),
            'keyword_weight': weight_history.get('default_keyword_weight', 0.3),
            'feedback_count': 0,
            'query_type': query_type
        }
    
    # Get current weights
    current = weight_history['queries'][normalized_query]
    
    # Adjust weights based on feedback and result type
    conn = None
    try:
        # Connect to DB to get information about the result
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Check if the result exists in vector search
        cursor.execute("""
            SELECT COUNT(*) FROM content_embeddings
            WHERE content_id = ?
        """, (result_id,))
        
        has_vector_match = cursor.fetchone()[0] > 0
        
        # Check if result exists in keyword search (FTS)
        cursor.execute("""
            SELECT COUNT(*) FROM ai_content_fts
            WHERE rowid = ?
        """, (result_id,))
        
        has_keyword_match = cursor.fetchone()[0] > 0
        
        # Adjust weights
        adjustment = 0.05  # Small adjustment step
        
        if feedback == 'relevant':
            # Positive feedback - increase weight for the type that matched
            if has_vector_match and not has_keyword_match:
                # Only vector matched, increase vector weight
                current['vector_weight'] = min(0.95, current['vector_weight'] + adjustment)
                current['keyword_weight'] = 1.0 - current['vector_weight']
            elif has_keyword_match and not has_vector_match:
                # Only keyword matched, increase keyword weight
                current['keyword_weight'] = min(0.95, current['keyword_weight'] + adjustment)
                current['vector_weight'] = 1.0 - current['keyword_weight']
            # If both matched, no adjustment needed
        elif feedback == 'not_relevant':
            # Negative feedback - decrease weight for the type that matched
            if has_vector_match and not has_keyword_match:
                # Only vector matched, decrease vector weight
                current['vector_weight'] = max(0.05, current['vector_weight'] - adjustment)
                current['keyword_weight'] = 1.0 - current['vector_weight']
            elif has_keyword_match and not has_vector_match:
                # Only keyword matched, decrease keyword weight
                current['keyword_weight'] = max(0.05, current['keyword_weight'] - adjustment)
                current['vector_weight'] = 1.0 - current['keyword_weight']
            # If both matched, no adjustment needed
        
        # Increment feedback count
        current['feedback_count'] += 1
        
        # Update query type specific defaults
        if current['feedback_count'] >= 3:
            # Update default weights for this query type
            query_type_key = f"default_{query_type}_vector_weight"
            if query_type_key not in weight_history:
                weight_history[query_type_key] = weight_history['default_vector_weight']
            
            # Update with running average
            weight_history[query_type_key] = (
                weight_history[query_type_key] * 0.8 + 
                current['vector_weight'] * 0.2
            )
        
        return weight_history
        
    except Exception as e:
        logger.error(f"Error adjusting weights from feedback: {str(e)}")
        return weight_history
        
    finally:
        if conn:
            conn.close()

def classify_query_type(query: str) -> str:
    """
    Classify the query type based on its characteristics
    
    Args:
        query: The search query
        
    Returns:
        Query type classification
    """
    query_lower = query.lower()
    
    # Check for code queries
    code_patterns = [
        r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b',
        r'\bimport\b', r'\bdef\b', r'\bpython\b', r'\bjava\b', 
        r'\bjavascript\b', r'\bjs\b', r'\bc\+\+\b', r'\bsql\b'
    ]
    
    # Check for factual queries
    factual_patterns = [
        r'\bwho\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', 
        r'\bhow\b', r'\bwhy\b', r'\bdate\b', r'\byear\b',
        r'\bauthor\b', r'\bcreator\b', r'\bversions?\b'
    ]
    
    # Check for concept queries
    concept_patterns = [
        r'\bconcept\b', r'\btheory\b', r'\bframework\b', r'\barchitecture\b',
        r'\bparadigm\b', r'\bpattern\b', r'\bapproach\b', r'\bstrategy\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bcompare\b', r'\banalyze\b'
    ]
    
    # Count matches for each category
    code_matches = sum(1 for pattern in code_patterns if re.search(pattern, query_lower))
    factual_matches = sum(1 for pattern in factual_patterns if re.search(pattern, query_lower))
    concept_matches = sum(1 for pattern in concept_patterns if re.search(pattern, query_lower))
    
    # Determine query type
    if code_matches > max(factual_matches, concept_matches):
        return 'code'
    elif factual_matches > concept_matches:
        return 'factual'
    elif concept_matches > 0:
        return 'concept'
    else:
        # Check for other characteristics
        if len(query.split()) >= 6:
            return 'long'
        elif '"' in query:
            return 'exact'
        else:
            return 'general'

def save_weights_history(weight_history: Dict[str, Any], 
                        filepath: str = 'data/search_weights.json') -> None:
    """
    Save weight history to a JSON file
    
    Args:
        weight_history: Weight history dictionary
        filepath: Path to save the JSON file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(weight_history, f, indent=2)
            
        logger.info(f"Weight history saved to {filepath}")
            
    except Exception as e:
        logger.error(f"Error saving weight history: {str(e)}")

def load_weights_history(filepath: str = 'data/search_weights.json') -> Dict[str, Any]:
    """
    Load weight history from a JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Weight history dictionary
    """
    default_history = {
        'queries': {},
        'default_vector_weight': 0.7,
        'default_keyword_weight': 0.3,
        'default_code_vector_weight': 0.5,
        'default_factual_vector_weight': 0.6,
        'default_concept_vector_weight': 0.8,
        'default_long_vector_weight': 0.75,
        'default_exact_vector_weight': 0.4,
        'default_general_vector_weight': 0.7,
        'last_updated': datetime.now().isoformat()
    }
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.info(f"Weight history file not found, using defaults")
            return default_history
        
        # Load from file
        with open(filepath, 'r') as f:
            history = json.load(f)
            
        logger.info(f"Weight history loaded from {filepath}")
        
        # Update last loaded time
        history['last_updated'] = datetime.now().isoformat()
        
        return history
            
    except Exception as e:
        logger.error(f"Error loading weight history: {str(e)}")
        return default_history

def main():
    """Main function for direct script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid search for content")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    parser.add_argument("--source-type", help="Filter by source type")
    parser.add_argument("--vector-weight", type=float, help="Weight for vector search (0-1)")
    parser.add_argument("--keyword-weight", type=float, help="Weight for keyword search (0-1)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive weights based on query")
    
    args = parser.parse_args()
    
    # Set weights
    vector_weight = args.vector_weight
    keyword_weight = args.keyword_weight
    
    # If adaptive or weights not provided, determine automatically
    if args.adaptive or (vector_weight is None and keyword_weight is None):
        # Load weight history
        weight_history = load_weights_history()
        
        # Get query type
        query_type = classify_query_type(args.query)
        
        # Use query-specific weights if available
        normalized_query = args.query.lower().strip()
        if normalized_query in weight_history['queries']:
            vector_weight = weight_history['queries'][normalized_query]['vector_weight']
            keyword_weight = weight_history['queries'][normalized_query]['keyword_weight']
            print(f"Using query-specific weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        else:
            # Use query type defaults
            type_key = f"default_{query_type}_vector_weight"
            if type_key in weight_history:
                vector_weight = weight_history[type_key]
                keyword_weight = 1.0 - vector_weight
                print(f"Using {query_type} query type weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
            else:
                # Determine weights based on query
                vector_weight, keyword_weight = determine_weights(args.query)
                print(f"Using determined weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
    
    # Perform search
    results = hybrid_search(
        query=args.query,
        top_k=args.top_k,
        source_type=args.source_type,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight
    )
    
    # Display results
    print(f"\nHybrid search results for: {args.query}")
    print(f"Weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
    print("=" * 80)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} - Combined Score: {result.get('combined_score', 0):.4f}")
        print(f"Vector Score: {result.get('vector_score', 0):.4f}, Keyword Score: {result.get('keyword_score', 0):.4f}")
        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Source: {result.get('source_type', 'Unknown')}")
        print(f"Content ID: {result['content_id']}")
        print("-" * 40)
        
        # Show snippet or chunk text
        if 'snippet' in result and result['snippet']:
            print(f"Keyword Match: {result['snippet']}")
        
        if 'chunk_text' in result and result['chunk_text']:
            text = result['chunk_text']
            print(text[:300] + "..." if len(text) > 300 else text)
        
        print("-" * 40)
        
        # Display concepts if available
        if 'concepts' in result:
            print("Concepts:")
            for concept in result['concepts'][:5]:  # Show top 5 concepts
                print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
        
        print()
    
    return results

if __name__ == "__main__":
    main() 