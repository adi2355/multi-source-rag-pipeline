"""
Text chunking module for breaking content into appropriate-sized pieces for embedding

This module provides functions to split content into overlapping chunks for better
context preservation and more effective embeddings for vector search.
"""
import logging
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chunking')

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum size in characters for each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position with potential overlap
        end = min(start + chunk_size, text_length)
        
        # If we're not at the very end, try to find a good breaking point
        if end < text_length:
            # Look for natural text boundaries in order of preference
            # Check for double newlines (paragraph breaks)
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # Check for single newlines
                newline = text.rfind("\n", start, end)
                if newline != -1 and newline > start + chunk_size // 2:
                    end = newline + 1
                else:
                    # Check for sentence boundaries (period followed by space)
                    sentence = text.rfind(". ", start, end)
                    if sentence != -1 and sentence > start + chunk_size // 2:
                        end = sentence + 2
                    else:
                        # Last resort: check for any space
                        space = text.rfind(" ", start, end)
                        if space != -1 and space > start + chunk_size // 2:
                            end = space + 1
        
        # Extract the chunk and add to results
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position for next chunk, ensuring overlap
        start = max(start + 1, end - overlap)
    
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks

def chunk_with_metadata(content: Dict[str, Any], 
                        chunk_size: int = 1000, 
                        overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split content with metadata into chunks, preserving metadata in each chunk
    
    Args:
        content: Dictionary with text and metadata
        chunk_size: Maximum size in characters for each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries, each with a text chunk and the original metadata
    """
    if not content or 'text' not in content:
        return []
    
    # Extract text and metadata
    text = content['text']
    metadata = {k: v for k, v in content.items() if k != 'text'}
    
    # Chunk the text
    text_chunks = chunk_text(text, chunk_size, overlap)
    
    # Create result with metadata preserved
    result = []
    for i, chunk in enumerate(text_chunks):
        chunk_data = metadata.copy()
        chunk_data['text'] = chunk
        chunk_data['chunk_index'] = i
        chunk_data['total_chunks'] = len(text_chunks)
        result.append(chunk_data)
    
    return result

def prepare_content_for_embedding(title: str, description: str, content: str) -> str:
    """
    Prepare content by combining title, description, and content with appropriate formatting
    
    Args:
        title: Content title
        description: Content description or summary
        content: Main content text
        
    Returns:
        Formatted text ready for chunking and embedding
    """
    parts = []
    
    if title:
        parts.append(f"Title: {title.strip()}")
    
    if description:
        parts.append(f"Description: {description.strip()}")
    
    if content:
        parts.append(f"Content: {content.strip()}")
    
    return "\n\n".join(parts)

if __name__ == "__main__":
    # Test the chunking functionality
    test_text = """
    Title: Understanding Vector Embeddings
    
    This is a test document that will be split into chunks. Vector embeddings are numerical representations 
    of concepts converted into a series of numbers so that computers can understand how semantically close 
    two concepts are to each other.
    
    The main idea behind vector embeddings is to represent data in a way that captures semantic similarity
    through spatial relationships in a high-dimensional vector space.
    
    When we perform operations like measuring the cosine similarity between two vectors, we're essentially
    quantifying how similar the underlying concepts are.
    
    This has applications in search, recommendations, classification, and many other machine learning tasks.
    """
    
    chunks = chunk_text(test_text, chunk_size=200, overlap=50)
    
    print(f"Split text into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print("-" * 40)
        print(chunk)
        print("-" * 40) 