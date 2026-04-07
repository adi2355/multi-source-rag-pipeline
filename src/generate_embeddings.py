#!/usr/bin/env python3
"""
Embedding Generation Script

This script generates vector embeddings for all content items in the database.
It processes content in batches, chunks the text, and stores the embeddings
in the content_embeddings table.
"""
import os
import sys
import logging
import sqlite3
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
import config
from chunking import chunk_text, prepare_content_for_embedding
from embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/embedding_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('generate_embeddings')

def ensure_embedding_table_exists(conn: sqlite3.Connection) -> None:
    """
    Ensure content_embeddings table exists in the database
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    # Check if content_embeddings table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='content_embeddings'
    """)
    
    if not cursor.fetchone():
        logger.info("Creating content_embeddings table...")
        cursor.execute("""
            CREATE TABLE content_embeddings (
                id INTEGER PRIMARY KEY,
                content_id INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                chunk_text TEXT,
                date_created TEXT NOT NULL,
                FOREIGN KEY(content_id) REFERENCES ai_content(id),
                UNIQUE(content_id, chunk_index)
            )
        """)
        
        # Create indices for faster lookup
        cursor.execute("""
            CREATE INDEX idx_content_embeddings_content_id
            ON content_embeddings(content_id)
        """)
        
        conn.commit()
        logger.info("content_embeddings table created")

def get_content_items(conn: sqlite3.Connection, 
                     source_type: Optional[str] = None,
                     limit: Optional[int] = None,
                     offset: int = 0,
                     skip_existing: bool = True) -> List[Dict[str, Any]]:
    """
    Get content items from the database
    
    Args:
        conn: SQLite database connection
        source_type: Optional filter by source type
        limit: Maximum number of items to return
        offset: Number of items to skip
        skip_existing: Skip items that already have embeddings
        
    Returns:
        List of content items
    """
    cursor = conn.cursor()
    
    # Build query
    query = """
        SELECT 
            c.id, c.source_type_id, c.source_id, c.title, c.description,
            c.content, c.date_created, c.metadata, st.name as source_type_name
        FROM ai_content c
        JOIN source_types st ON c.source_type_id = st.id
    """
    
    params = []
    
    # Add source type filter if provided
    if source_type:
        query += " WHERE st.name = ?"
        params.append(source_type)
    
    # Skip items that already have embeddings
    if skip_existing:
        if source_type:
            query += " AND NOT EXISTS"
        else:
            query += " WHERE NOT EXISTS"
            
        query += """
            (SELECT 1 FROM content_embeddings ce 
             WHERE ce.content_id = c.id)
        """
    
    # Add order by and limit/offset
    query += " ORDER BY c.id"
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
        
    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)
    
    # Execute query
    cursor.execute(query, params)
    
    # Process results
    content_items = []
    for row in cursor.fetchall():
        (content_id, source_type_id, source_id, title, description, 
         content, date_created, metadata, source_type_name) = row
        
        content_items.append({
            'id': content_id,
            'source_type_id': source_type_id,
            'source_id': source_id,
            'title': title,
            'description': description,
            'content': content,
            'date_created': date_created,
            'metadata': metadata,
            'source_type_name': source_type_name
        })
    
    return content_items

def process_content_items(content_items: List[Dict[str, Any]], 
                         chunk_size: int = 500,
                         chunk_overlap: int = 100) -> Tuple[int, int, int]:
    """
    Process content items, chunk text, and generate embeddings
    
    Args:
        content_items: List of content items
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Tuple of (total_items, successful_items, total_chunks)
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        ensure_embedding_table_exists(conn)
        cursor = conn.cursor()
        
        # Create embedding generator
        embedding_generator = EmbeddingGenerator()
        model_name = embedding_generator.model_name
        
        # Track statistics
        total_items = len(content_items)
        successful_items = 0
        total_chunks = 0
        start_time = time.time()
        
        # Process each content item
        for i, item in enumerate(content_items):
            try:
                # Prepare text for chunking (combine title, description, content)
                text = prepare_content_for_embedding(
                    title=item.get('title', ''),
                    description=item.get('description', ''),
                    content=item.get('content', '')
                )
                
                # Skip if no text
                if not text.strip():
                    logger.warning(f"Skipping content {item['id']} - no text content")
                    continue
                
                # Chunk text
                text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                
                # Process each chunk
                for chunk_idx, chunk_content in enumerate(text_chunks):
                    # Generate embedding
                    embedding = embedding_generator.generate_embedding(chunk_content)
                    
                    if embedding is not None:
                        # Serialize embedding (binary pickle)
                        import pickle
                        embedding_binary = pickle.dumps(embedding)
                        
                        # Store in database
                        cursor.execute("""
                            INSERT OR REPLACE INTO content_embeddings
                            (content_id, chunk_index, chunk_text, embedding_vector, embedding_model, date_created)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            item['id'],
                            chunk_idx,
                            chunk_content,
                            embedding_binary,
                            model_name,
                            datetime.now().isoformat()
                        ))
                        
                        total_chunks += 1
                
                # Commit after each item
                conn.commit()
                
                successful_items += 1
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == total_items:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    logger.info(f"Processed {i+1}/{total_items} items, {total_chunks} chunks "
                                f"({successful_items} successful), "
                                f"avg {avg_time:.2f}s per item")
            
            except Exception as e:
                logger.error(f"Error processing content {item['id']}: {str(e)}")
                # Continue with next item
                continue
        
        return total_items, successful_items, total_chunks
        
    except Exception as e:
        logger.error(f"Error in process_content_items: {str(e)}")
        if conn:
            conn.rollback()
        return len(content_items), 0, 0
        
    finally:
        if conn:
            conn.close()

def main():
    """Main function for script execution"""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for content in the database"
    )
    
    parser.add_argument(
        "--source-type", 
        help="Process only content of this source type (e.g., 'instagram', 'github', 'arxiv')"
    )
    
    parser.add_argument(
        "--limit", type=int, 
        help="Maximum number of items to process"
    )
    
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of items to process in each batch"
    )
    
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Size of text chunks in characters"
    )
    
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="Overlap between text chunks in characters"
    )
    
    parser.add_argument(
        "--force", action="store_true",
        help="Process items even if they already have embeddings"
    )
    
    parser.add_argument(
        "--stats", action="store_true",
        help="Show database statistics and exit"
    )
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(config.DB_PATH)
    
    if args.stats:
        # Show database statistics
        cursor = conn.cursor()
        
        # Get total content count
        cursor.execute("SELECT COUNT(*) FROM ai_content")
        total_content = cursor.fetchone()[0]
        
        # Get count by source type
        cursor.execute("""
            SELECT st.name, COUNT(*)
            FROM ai_content c
            JOIN source_types st ON c.source_type_id = st.id
            GROUP BY st.name
        """)
        
        counts_by_source = cursor.fetchall()
        
        # Get embedding statistics
        cursor.execute("""
            SELECT COUNT(*) FROM content_embeddings
        """)
        
        total_embeddings = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT content_id) FROM content_embeddings
        """)
        
        content_with_embeddings = cursor.fetchone()[0]
        
        # Print statistics
        print("\nDatabase Statistics:")
        print("=" * 40)
        print(f"Total content items: {total_content}")
        print(f"Content items with embeddings: {content_with_embeddings}")
        print(f"Total embedding chunks: {total_embeddings}")
        print("\nContent by source type:")
        
        for source_name, count in counts_by_source:
            print(f"- {source_name}: {count}")
        
        # Calculate items without embeddings
        items_without_embeddings = total_content - content_with_embeddings
        print(f"\nContent items without embeddings: {items_without_embeddings}")
        
        conn.close()
        return
    
    # Start processing
    logger.info("Starting embedding generation...")
    logger.info(f"Source type: {args.source_type if args.source_type else 'all'}")
    logger.info(f"Limit: {args.limit if args.limit else 'none'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    logger.info(f"Force reprocess: {args.force}")
    
    # Initialize counters
    total_processed = 0
    total_successful = 0
    total_chunks = 0
    
    # Process in batches
    offset = 0
    while True:
        # Get batch of content items
        conn = sqlite3.connect(config.DB_PATH)
        content_items = get_content_items(
            conn,
            source_type=args.source_type,
            limit=args.batch_size,
            offset=offset,
            skip_existing=not args.force
        )
        conn.close()
        
        # Exit if no more items
        if not content_items:
            break
        
        # Process batch
        logger.info(f"Processing batch of {len(content_items)} items (offset {offset})...")
        
        batch_total, batch_successful, batch_chunks = process_content_items(
            content_items,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Update counters
        total_processed += batch_total
        total_successful += batch_successful
        total_chunks += batch_chunks
        
        # Update offset for next batch
        offset += len(content_items)
        
        # Exit if limit reached
        if args.limit and total_processed >= args.limit:
            break
    
    # Log final statistics
    logger.info("Embedding generation complete!")
    logger.info(f"Total items processed: {total_processed}")
    logger.info(f"Successfully processed: {total_successful}")
    logger.info(f"Total chunks generated: {total_chunks}")
    
if __name__ == "__main__":
    main() 