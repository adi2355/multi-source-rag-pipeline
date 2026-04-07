"""
Vector search module for semantic retrieval of content

This module provides functionality to search content using vector similarity,
enabling semantic search capabilities for the knowledge base.
"""
import os
import logging
import sqlite3
import pickle
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

# Import local modules
import config
from embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_search')

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def vector_search(query_embedding: np.ndarray, top_k: int = 5, 
                 source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for similar content using vector embeddings
    
    Args:
        query_embedding: The query embedding vector
        top_k: Number of results to return
        source_type: Optional filter by source type
        
    Returns:
        List of search results with similarity scores and metadata
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Build query to get all embeddings (potentially filtered by source type)
        query = """
            SELECT 
                ce.content_id, ce.chunk_index, ce.embedding_vector, ce.chunk_text,
                ac.title, ac.source_type_id
            FROM content_embeddings ce
            JOIN ai_content ac ON ce.content_id = ac.id
        """
        
        params = []
        
        # Add source type filter if provided
        if source_type:
            cursor.execute("SELECT id FROM source_types WHERE name = ?", (source_type,))
            source_type_id = cursor.fetchone()
            if source_type_id:
                query += " WHERE ac.source_type_id = ?"
                params.append(source_type_id[0])
            else:
                logger.warning(f"Source type '{source_type}' not found")
        
        # Execute query
        cursor.execute(query, params)
        
        # Calculate similarities and collect results
        results = []
        for row in cursor.fetchall():
            content_id, chunk_index, embedding_binary, chunk_text, title, source_type_id = row
            
            # Get source type name
            cursor.execute("SELECT name FROM source_types WHERE id = ?", (source_type_id,))
            source_type_result = cursor.fetchone()
            source_type_name = source_type_result[0] if source_type_result else "unknown"
            
            # Skip invalid embeddings
            if not embedding_binary:
                continue
                
            try:
                # Deserialize embedding
                embedding = pickle.loads(embedding_binary)
                
                # Calculate similarity
                similarity = cosine_similarity(query_embedding, embedding)
                
                # Add to results
                results.append({
                    'content_id': content_id,
                    'chunk_index': chunk_index,
                    'similarity': similarity,
                    'chunk_text': chunk_text,
                    'title': title,
                    'source_type': source_type_name
                })
            except Exception as e:
                logger.warning(f"Error processing embedding for content {content_id}, chunk {chunk_index}: {str(e)}")
                continue
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return []
        
    finally:
        if conn:
            conn.close()

def search_by_text(query_text: str, top_k: int = 5, source_type: Optional[str] = None,
                  embedding_generator: Optional[EmbeddingGenerator] = None) -> List[Dict[str, Any]]:
    """
    Search for content similar to the query text
    
    Args:
        query_text: The query text
        top_k: Number of results to return
        source_type: Optional filter by source type
        embedding_generator: Optional pre-initialized embedding generator
        
    Returns:
        List of search results with similarity scores and metadata
    """
    # Create or use provided embedding generator
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
    
    # Generate embedding for query
    query_embedding = embedding_generator.generate_embedding(query_text)
    
    # Perform vector search
    results = vector_search(query_embedding, top_k=top_k, source_type=source_type)
    
    # Add original query to results
    for result in results:
        result['query'] = query_text
    
    return results

def enrich_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich search results with additional metadata from the database
    
    Args:
        results: List of search results
        
    Returns:
        Enriched search results
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        for result in results:
            content_id = result['content_id']
            
            # Get content metadata
            cursor.execute("""
                SELECT 
                    title, description, date_created, url, metadata
                FROM ai_content 
                WHERE id = ?
            """, (content_id,))
            
            content_data = cursor.fetchone()
            if content_data:
                title, description, date_created, url, metadata_json = content_data
                
                # Add to result
                result['title'] = title or result.get('title', '')
                result['description'] = description
                result['date_created'] = date_created
                result['url'] = url
                
                # Parse and add metadata if available
                if metadata_json:
                    try:
                        import json
                        metadata = json.loads(metadata_json)
                        result['metadata'] = metadata
                    except:
                        # Ignore metadata parsing errors
                        pass
            
            # Get concepts associated with this content
            try:
                cursor.execute("""
                    SELECT c.name, c.category, cc.importance
                    FROM concepts c
                    JOIN content_concepts cc ON c.id = cc.concept_id
                    WHERE cc.content_id = ?
                    ORDER BY cc.importance DESC
                """, (content_id,))
                
                concepts = [{
                    'name': row[0],
                    'category': row[1],
                    'importance': row[2]
                } for row in cursor.fetchall()]
                
                if concepts:
                    result['concepts'] = concepts
            except:
                # Concepts table might not exist yet
                pass
        
        return results
        
    except Exception as e:
        logger.error(f"Error enriching search results: {str(e)}")
        return results
        
    finally:
        if conn:
            conn.close()

def create_memory_index() -> Dict[str, Any]:
    """
    Create an in-memory index of all embeddings for faster search
    
    Returns:
        Dictionary containing the in-memory index
    """
    logger.info("Creating in-memory embedding index...")
    conn = None
    try:
        start_time = time.time()
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get all embeddings
        cursor.execute("""
            SELECT 
                ce.content_id, ce.chunk_index, ce.embedding_vector,
                ac.title, ac.source_type_id
            FROM content_embeddings ce
            JOIN ai_content ac ON ce.content_id = ac.id
        """)
        
        # Process embeddings
        content_ids = []
        chunk_indices = []
        vectors = []
        metadata = []
        
        for row in cursor.fetchall():
            content_id, chunk_index, embedding_binary, title, source_type_id = row
            
            # Skip invalid embeddings
            if not embedding_binary:
                continue
                
            try:
                # Deserialize embedding
                embedding = pickle.loads(embedding_binary)
                
                # Add to arrays
                content_ids.append(content_id)
                chunk_indices.append(chunk_index)
                vectors.append(embedding)
                metadata.append({
                    'content_id': content_id,
                    'chunk_index': chunk_index,
                    'title': title,
                    'source_type_id': source_type_id
                })
            except:
                # Skip problematic embeddings
                continue
        
        # Convert to numpy array for faster operations
        if vectors:
            vectors_array = np.array(vectors)
        else:
            vectors_array = np.array([])
        
        # Create index
        index = {
            'content_ids': content_ids,
            'chunk_indices': chunk_indices,
            'vectors': vectors_array,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        elapsed = time.time() - start_time
        logger.info(f"In-memory index created with {len(vectors)} embeddings in {elapsed:.2f}s")
        
        return index
        
    except Exception as e:
        logger.error(f"Error creating in-memory index: {str(e)}")
        return {
            'content_ids': [],
            'chunk_indices': [],
            'vectors': np.array([]),
            'metadata': [],
            'created_at': datetime.now().isoformat(),
            'error': str(e)
        }
        
    finally:
        if conn:
            conn.close()

def search_memory_index(query_embedding: np.ndarray, index: Dict[str, Any], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the in-memory index for similar embeddings
    
    Args:
        query_embedding: The query embedding
        index: The in-memory index
        top_k: Number of results to return
        
    Returns:
        List of search results
    """
    try:
        vectors = index.get('vectors')
        if len(vectors) == 0:
            logger.warning("Empty in-memory index")
            return []
        
        # Calculate similarities in a vectorized way
        similarities = np.dot(vectors, query_embedding) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Collect results
        results = []
        for idx in top_indices:
            results.append({
                'content_id': index['content_ids'][idx],
                'chunk_index': index['chunk_indices'][idx],
                'similarity': float(similarities[idx]),
                'metadata': index['metadata'][idx]
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching in-memory index: {str(e)}")
        return []

def debug_search(query_text: str, top_k: int = 5):
    """
    Debug function to test and display search results
    
    Args:
        query_text: Query text
        top_k: Number of results to show
    """
    # Create embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Search
    logger.info(f"Searching for: {query_text}")
    results = search_by_text(
        query_text, 
        top_k=top_k, 
        embedding_generator=embedding_generator
    )
    
    # Enrich results
    results = enrich_search_results(results)
    
    # Display results
    print(f"\nSearch results for: {query_text}")
    print("=" * 80)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} - Similarity: {result['similarity']:.4f}")
        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Source: {result.get('source_type', 'Unknown')}")
        print(f"Content ID: {result['content_id']}, Chunk: {result['chunk_index']}")
        print("-" * 40)
        print(result.get('chunk_text', '')[:300] + "..." if len(result.get('chunk_text', '')) > 300 else result.get('chunk_text', ''))
        print("-" * 40)
        
        # Display concepts if available
        if 'concepts' in result:
            print("Concepts:")
            for concept in result['concepts'][:5]:  # Show top 5 concepts
                print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
        
        print()
    
    return results

def main():
    """Main function for direct script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector search for content")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    parser.add_argument("--source-type", help="Filter by source type")
    parser.add_argument("--create-index", action="store_true", help="Create in-memory index before searching")
    
    args = parser.parse_args()
    
    if args.create_index:
        # Create and use in-memory index
        index = create_memory_index()
        
        # Create embedding generator
        embedding_generator = EmbeddingGenerator()
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(args.query)
        
        # Search using in-memory index
        results = search_memory_index(query_embedding, index, top_k=args.top_k)
        
        # Fetch chunk text and enrich results
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        for result in results:
            # Get chunk text
            cursor.execute(
                "SELECT chunk_text FROM content_embeddings WHERE content_id = ? AND chunk_index = ?",
                (result['content_id'], result['chunk_index'])
            )
            chunk_text = cursor.fetchone()
            if chunk_text:
                result['chunk_text'] = chunk_text[0]
        
        conn.close()
        
        # Enrich results
        results = enrich_search_results(results)
    else:
        # Use standard search
        results = debug_search(args.query, top_k=args.top_k)
    
if __name__ == "__main__":
    main() 