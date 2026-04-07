#!/usr/bin/env python3
"""
RAG Context Builder Module for Instagram Knowledge Base

This module handles the selection, formatting, and optimization of retrieved
context for use with Large Language Models (LLMs). It takes search results
from vector and hybrid search, performs ranking and selection, and formats
the context appropriately for LLM consumption.
"""
import os
import sys
import logging
import json
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional, Set
import numpy as np

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
import config
from embeddings import EmbeddingGenerator
import vector_search
import hybrid_search
from chunking import chunk_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/context_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('context_builder')

# Constants for LLM context
MAX_TOKENS_DEFAULT = 4000
MAX_RESULTS_DEFAULT = 20
TOKEN_ESTIMATOR_RATIO = 4  # Roughly 4 characters per token for English text
CONTEXT_HEADER = """The following information comes from various sources in the knowledge base.
Use this information to answer the user's question, and cite the source number [1], [2], etc. when using specific information.
If the information doesn't help answer the question, acknowledge this and provide your best response based on your general knowledge."""

class ContextBuilder:
    """
    Class for selecting, formatting and optimizing context for LLM consumption
    """
    
    def __init__(self, max_tokens: int = MAX_TOKENS_DEFAULT, 
                max_results: int = MAX_RESULTS_DEFAULT,
                db_path: str = config.DB_PATH):
        """
        Initialize the context builder
        
        Args:
            max_tokens: Maximum tokens to include in the context
            max_results: Maximum number of search results to consider
            db_path: Path to the SQLite database
        """
        self.max_tokens = max_tokens
        self.max_results = max_results
        self.db_path = db_path
        self.embedding_generator = EmbeddingGenerator()
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on characters
        return len(text) // TOKEN_ESTIMATOR_RATIO
    
    def get_content_metadata(self, content_id: int) -> Dict[str, Any]:
        """
        Get additional metadata for a content item
        
        Args:
            content_id: ID of the content in the ai_content table
            
        Returns:
            Dictionary with content metadata
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get content metadata
            cursor.execute("""
                SELECT ac.id, ac.title, ac.url, ac.metadata, st.name as source_type
                FROM ai_content ac
                JOIN source_types st ON ac.source_type_id = st.id
                WHERE ac.id = ?
            """, (content_id,))
            
            result = cursor.fetchone()
            if not result:
                return {}
                
            content_id, title, url, metadata_str, source_type = result
            
            # Parse JSON metadata if available
            metadata = {}
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except:
                    pass
            
            # Get additional source-specific metadata
            if source_type == 'github':
                cursor.execute("""
                    SELECT name, full_name, stars, language 
                    FROM github_repos 
                    WHERE id = (SELECT source_id FROM ai_content WHERE id = ?)
                """, (content_id,))
                
                repo_info = cursor.fetchone()
                if repo_info:
                    name, full_name, stars, language = repo_info
                    metadata.update({
                        'repo_name': name,
                        'full_name': full_name,
                        'stars': stars,
                        'language': language
                    })
            
            elif source_type == 'research_paper':
                cursor.execute("""
                    SELECT title, authors, publication, year
                    FROM research_papers
                    WHERE id = (SELECT source_id FROM ai_content WHERE id = ?)
                """, (content_id,))
                
                paper_info = cursor.fetchone()
                if paper_info:
                    paper_title, authors, publication, year = paper_info
                    metadata.update({
                        'paper_title': paper_title,
                        'authors': authors,
                        'publication': publication,
                        'year': year
                    })
            
            return {
                'content_id': content_id,
                'title': title,
                'url': url,
                'source_type': source_type,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting content metadata: {str(e)}")
            return {}
        finally:
            if conn:
                conn.close()
    
    def select_context(self, search_results: List[Dict[str, Any]], 
                      diversity_factor: float = 0.2) -> List[Dict[str, Any]]:
        """
        Select diverse context from search results to maximize information value
        while respecting token limits
        
        Args:
            search_results: List of search results
            diversity_factor: Weight given to diversity vs relevance (0-1)
            
        Returns:
            List of selected context items
        """
        if not search_results:
            logger.warning("No search results to select context from")
            return []
        
        # Ensure we don't exceed max results
        results = search_results[:self.max_results]
        
        # Calculate maximum scores for normalization
        max_score = max(r.get('score', 0) for r in results) if results else 1
        
        # If max_score is zero, set it to 1 to avoid division by zero
        if max_score == 0:
            logger.warning("All search results have a score of 0, normalizing to 1")
            max_score = 1
        
        # Get content types and ids for diversity calculation
        content_types = {}
        content_ids = {}
        
        for idx, result in enumerate(results):
            content_id = result.get('content_id')
            source_type = result.get('source_type', 'unknown')
            
            if source_type not in content_types:
                content_types[source_type] = []
            content_types[source_type].append(idx)
            
            if content_id not in content_ids:
                content_ids[content_id] = []
            content_ids[content_id].append(idx)
        
        # Calculate diversity penalties
        diversity_scores = [0] * len(results)
        
        # Selected items
        selected = []
        selected_ids = set()
        current_tokens = 0
        
        # Process until we reach token limit or run out of results
        while results and current_tokens < self.max_tokens:
            best_idx = -1
            best_score = -1
            
            for idx, result in enumerate(results):
                if idx in selected_ids:
                    continue
                
                # Calculate combined score with diversity
                base_score = result.get('score', 0) / max_score
                content_id = result.get('content_id')
                source_type = result.get('source_type', 'unknown')
                
                # Penalize if we already have content from this source/content_id
                type_penalty = sum(1 for i in selected_ids if i in content_types.get(source_type, []))
                id_penalty = sum(1 for i in selected_ids if i in content_ids.get(content_id, []))
                
                diversity_penalty = (type_penalty * 0.1 + id_penalty * 0.2) * diversity_factor
                combined_score = base_score - diversity_penalty
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            # Add best result to selection
            result = results[best_idx]
            chunk_text = result.get('chunk_text', '')
            chunk_tokens = self.estimate_tokens(chunk_text)
            
            # Check if adding this chunk would exceed token limit
            if current_tokens + chunk_tokens <= self.max_tokens:
                selected.append(result)
                selected_ids.add(best_idx)
                current_tokens += chunk_tokens
            else:
                # Try to trim the chunk to fit
                max_chars = (self.max_tokens - current_tokens) * TOKEN_ESTIMATOR_RATIO
                if max_chars > 100:  # Only use if we can include enough meaningful text
                    trimmed_text = chunk_text[:max_chars] + "..."
                    result['chunk_text'] = trimmed_text
                    selected.append(result)
                break
        
        logger.info(f"Selected {len(selected)} chunks with ~{current_tokens} tokens for context")
        return selected
    
    def format_context(self, selected_context: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format selected context for LLM consumption with source citations
        
        Args:
            selected_context: List of selected context items
            
        Returns:
            Tuple of (formatted_context, source_metadata)
        """
        if not selected_context:
            return "", []
        
        context_parts = [CONTEXT_HEADER, ""]
        source_metadata = []
        
        # Track content IDs to avoid redundant sources
        seen_content_ids = set()
        
        for idx, item in enumerate(selected_context):
            content_id = item.get('content_id')
            source_type = item.get('source_type', 'unknown')
            chunk_text = item.get('chunk_text', '')
            chunk_index = item.get('chunk_index', 0)
            
            # Get source number (either existing or new)
            source_num = None
            for i, source in enumerate(source_metadata):
                if source.get('content_id') == content_id:
                    source_num = i + 1
                    break
            
            if source_num is None:
                # New source
                metadata = self.get_content_metadata(content_id)
                metadata.update({
                    'content_id': content_id,
                    'source_type': source_type
                })
                source_metadata.append(metadata)
                source_num = len(source_metadata)
            
            # Add context with source citation
            context_parts.append(f"[Source {source_num}]")
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line between sources
        
        # Add source information at the end
        context_parts.append("\nSource Information:")
        for idx, source in enumerate(source_metadata):
            title = source.get('title', 'Untitled')
            url = source.get('url', '')
            source_type = source.get('source_type', 'Unknown')
            
            source_info = f"[{idx+1}] {title} ({source_type})"
            if url:
                source_info += f" - {url}"
                
            context_parts.append(source_info)
        
        return "\n".join(context_parts), source_metadata
    
    def build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build a prompt for the LLM that includes the query and context
        
        Args:
            query: User query
            context: Formatted context
            
        Returns:
            Complete LLM prompt
        """
        prompt = f"""Answer the following question based on the provided context information.
If the context doesn't contain relevant information, acknowledge this and provide your best response based on general knowledge.
When using information from the context, always cite the source number (e.g., [Source 1]).

Question: {query}

Context:
{context}

Answer:"""
        
        return prompt
    
    def build_context_for_query(self, query: str, 
                              search_type: str = 'hybrid',
                              vector_weight: Optional[float] = None,
                              keyword_weight: Optional[float] = None,
                              source_type: Optional[str] = None,
                              top_k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build context for a user query using the specified search method
        
        Args:
            query: User query
            search_type: Type of search ('vector', 'keyword', or 'hybrid')
            vector_weight: Weight for vector search in hybrid search (0-1)
            keyword_weight: Weight for keyword search in hybrid search (0-1)
            source_type: Filter results by source type
            top_k: Number of top results to consider
            
        Returns:
            Tuple of (LLM prompt with context, source metadata)
        """
        start_time = time.time()
        search_results = []
        
        # Perform search
        if search_type == 'vector':
            search_results = vector_search.search_by_text(
                query_text=query,
                top_k=top_k,
                source_type=source_type,
                embedding_generator=self.embedding_generator
            )
        elif search_type == 'hybrid':
            search_results = hybrid_search.hybrid_search(
                query=query,
                top_k=top_k,
                source_type=source_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                embedding_generator=self.embedding_generator
            )
        else:
            logger.error(f"Unsupported search type: {search_type}")
            return "", []
            
        # Select and format context
        selected_context = self.select_context(search_results)
        context, source_metadata = self.format_context(selected_context)
        
        # Build complete prompt
        prompt = self.build_rag_prompt(query, context)
        
        elapsed = time.time() - start_time
        logger.info(f"Built context for query in {elapsed:.2f}s with {len(selected_context)} chunks")
        
        return prompt, source_metadata

def main():
    """Main function for direct script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Context Builder")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--search-type", choices=['vector', 'hybrid'], default='hybrid', 
                       help="Search type to use")
    parser.add_argument("--source-type", help="Filter by source type")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results to consider")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT, 
                       help="Maximum tokens for context")
    parser.add_argument("--vector-weight", type=float, help="Weight for vector search (0-1)")
    parser.add_argument("--keyword-weight", type=float, help="Weight for keyword search (0-1)")
    parser.add_argument("--output", help="Output file for the generated prompt")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create context builder
    builder = ContextBuilder(max_tokens=args.max_tokens)
    
    # Build context
    prompt, source_metadata = builder.build_context_for_query(
        query=args.query,
        search_type=args.search_type,
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight,
        source_type=args.source_type,
        top_k=args.top_k
    )
    
    # Print or save prompt
    if args.output:
        with open(args.output, 'w') as f:
            f.write(prompt)
        print(f"Prompt saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("GENERATED PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)
        
    # Print source metadata
    print("\nSource Metadata:")
    for idx, source in enumerate(source_metadata):
        title = source.get('title', 'Untitled')
        source_type = source.get('source_type', 'Unknown')
        content_id = source.get('content_id', 'N/A')
        print(f"[{idx+1}] {title} ({source_type}, ID: {content_id})")
    
if __name__ == "__main__":
    main() 