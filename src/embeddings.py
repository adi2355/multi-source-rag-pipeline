"""
Embedding generation module for creating vector representations of content

This module provides functionality to generate and store vector embeddings
for content in the knowledge base using sentence-transformers.
"""
import os
import logging
import sqlite3
import pickle
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple

# Import local modules
import config
from chunking import chunk_text, prepare_content_for_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embeddings')

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Using fallback embedding method.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmbeddingGenerator:
    """Class for generating embeddings from text content"""
    
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.embedding_size = 0
        
        # Initialize the model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence-transformers model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_size = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding size: {self.embedding_size}")
            except Exception as e:
                logger.error(f"Error loading sentence-transformers model: {str(e)}")
                self.model = None
        else:
            logger.warning("Using fallback embedding model (simple TF-IDF)")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using the loaded model
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding generation")
            # Return a zero vector with correct dimensions
            if self.embedding_size > 0:
                return np.zeros(self.embedding_size)
            else:
                return np.zeros(768)  # Default fallback size
                
        if self.model is not None:
            # Use sentence-transformers
            try:
                embedding = self.model.encode(text, show_progress_bar=False)
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding with sentence-transformers: {str(e)}")
                return self._fallback_embedding(text)
        else:
            # Use fallback method
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback method for generating embeddings when sentence-transformers is not available
        This implements a simple TF-IDF based embedding
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Simple fallback: convert text to bag of words and hash to fixed size
        from collections import Counter
        import hashlib
        
        # Default embedding size
        embedding_size = 768
        
        # Tokenize text (simple approach)
        tokens = text.lower().split()
        
        # Calculate term frequencies
        term_counts = Counter(tokens)
        
        # Generate a deterministic embedding based on term frequencies
        result = np.zeros(embedding_size)
        
        for term, count in term_counts.items():
            # Hash the term to get a position in the embedding
            term_hash = int(hashlib.md5(term.encode()).hexdigest(), 16)
            position = term_hash % embedding_size
            
            # Use term count as value, normalized by total tokens
            result[position] += count / len(tokens)
        
        # Normalize the vector
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
            
        return result
    
    def process_content_item(self, content_id: int, force_update: bool = False, 
                            chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """
        Process a content item from the database, generating and storing embeddings
        
        Args:
            content_id: ID of the content item in the ai_content table
            force_update: Whether to update existing embeddings
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Boolean indicating success
        """
        conn = None
        try:
            # Connect to database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get content data
            cursor.execute("""
                SELECT title, description, content, source_type_id 
                FROM ai_content WHERE id = ?
            """, (content_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Content with ID {content_id} not found")
                return False
            
            title, description, content, source_type_id = result
            
            # Get source type name
            cursor.execute("SELECT name FROM source_types WHERE id = ?", (source_type_id,))
            source_type_result = cursor.fetchone()
            source_type = source_type_result[0] if source_type_result else "unknown"
            
            # Skip if no meaningful content
            if not content or len(content.strip()) < 50:
                logger.warning(f"Content with ID {content_id} has insufficient text for embedding")
                return False
            
            # Check if we should update
            if not force_update:
                cursor.execute(
                    "SELECT COUNT(*) FROM content_embeddings WHERE content_id = ?",
                    (content_id,)
                )
                if cursor.fetchone()[0] > 0:
                    logger.debug(f"Embeddings already exist for content ID {content_id}")
                    return True
            
            # Prepare text for embedding
            full_text = prepare_content_for_embedding(title, description, content)
            
            # Generate chunks
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                logger.warning(f"Failed to generate chunks for content ID {content_id}")
                return False
            
            logger.info(f"Generated {len(chunks)} chunks for content ID {content_id}")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Generate embedding
                start_time = time.time()
                embedding = self.generate_embedding(chunk)
                elapsed = time.time() - start_time
                
                logger.debug(f"Generated embedding for chunk {i+1}/{len(chunks)} in {elapsed:.2f}s")
                
                # Convert numpy array to binary for storage
                embedding_binary = pickle.dumps(embedding)
                
                # Store in database
                cursor.execute("""
                    INSERT OR REPLACE INTO content_embeddings 
                    (content_id, embedding_vector, embedding_model, chunk_index, chunk_text, date_created)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    content_id,
                    embedding_binary,
                    self.model_name,
                    i,
                    chunk,
                    datetime.now().isoformat()
                ))
            
            # Update ai_content to mark as processed
            cursor.execute("""
                UPDATE ai_content 
                SET metadata = json.set(COALESCE(metadata, '{}'), '$.embeddings_generated', 1),
                    date_indexed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), content_id))
            
            conn.commit()
            logger.info(f"Successfully processed content ID {content_id} with {len(chunks)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error processing content ID {content_id}: {str(e)}")
            if conn:
                conn.rollback()
            return False
            
        finally:
            if conn:
                conn.close()
    
    def process_batch(self, batch_size: int = 10, max_items: Optional[int] = None, 
                     source_type: Optional[str] = None) -> int:
        """
        Process a batch of content items from the database
        
        Args:
            batch_size: Number of items to process in a batch
            max_items: Maximum total number of items to process
            source_type: If provided, only process items of this source type
            
        Returns:
            Number of items successfully processed
        """
        conn = None
        try:
            # Connect to database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Build query to get unprocessed content
            query = """
                SELECT c.id FROM ai_content c
                WHERE c.content IS NOT NULL 
                AND LENGTH(c.content) > 100
                AND (
                    json_extract(c.metadata, '$.embeddings_generated') IS NULL 
                    OR json_extract(c.metadata, '$.embeddings_generated') = 0
                )
            """
            
            params = []
            
            # Add source type filter if provided
            if source_type:
                cursor.execute("SELECT id FROM source_types WHERE name = ?", (source_type,))
                source_type_id = cursor.fetchone()
                if source_type_id:
                    query += " AND c.source_type_id = ?"
                    params.append(source_type_id[0])
                else:
                    logger.warning(f"Source type '{source_type}' not found")
            
            # Add limit
            if max_items:
                query += " LIMIT ?"
                params.append(max_items)
            
            # Get content IDs
            cursor.execute(query, params)
            content_ids = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Found {len(content_ids)} content items to process")
            
            # Process in batches
            processed_count = 0
            for i in range(0, len(content_ids), batch_size):
                batch = content_ids[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(content_ids)-1)//batch_size + 1}")
                
                for content_id in batch:
                    success = self.process_content_item(content_id)
                    if success:
                        processed_count += 1
                    
                    # Brief pause to avoid overwhelming resources
                    time.sleep(0.1)
                
                # Longer pause between batches
                if i + batch_size < len(content_ids):
                    logger.info(f"Batch complete. Pausing before next batch...")
                    time.sleep(1)
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return 0
            
        finally:
            if conn:
                conn.close()

def main():
    """Main function for direct script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for content")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to process")
    parser.add_argument("--source-type", help="Process only this source type (instagram, github, research_paper)")
    parser.add_argument("--model", default="multi-qa-mpnet-base-dot-v1", help="Name of the sentence-transformers model to use")
    parser.add_argument("--content-id", type=int, help="Process a specific content ID")
    parser.add_argument("--force-update", action="store_true", help="Force update of existing embeddings")
    
    args = parser.parse_args()
    
    # Create embedding generator
    generator = EmbeddingGenerator(model_name=args.model)
    
    if args.content_id:
        # Process specific content ID
        logger.info(f"Processing content ID: {args.content_id}")
        success = generator.process_content_item(args.content_id, force_update=args.force_update)
        if success:
            logger.info(f"Successfully processed content ID: {args.content_id}")
        else:
            logger.error(f"Failed to process content ID: {args.content_id}")
    else:
        # Process batch
        logger.info("Processing batch of content items")
        processed = generator.process_batch(
            batch_size=args.batch_size,
            max_items=args.max_items,
            source_type=args.source_type
        )
        logger.info(f"Successfully processed {processed} content items")

if __name__ == "__main__":
    main() 