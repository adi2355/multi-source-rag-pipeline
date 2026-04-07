#!/usr/bin/env python3
"""
LLM Integration Module for Instagram Knowledge Base

This module handles integration with Language Models like Claude for generating
responses to user queries based on retrieved context from the knowledge base.
It supports different LLM providers and handles rate limiting, retry logic, and
streaming responses.
"""
import os
import sys
import logging
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Union, Generator, Callable
from datetime import datetime
import requests

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
import config
from context_builder import ContextBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/llm_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('llm_integration')

# Check if Anthropic SDK is available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not available. Install with 'pip install anthropic'")

# Default model configuration
DEFAULT_MODEL = "claude-3-sonnet-20240229"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.5

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self):
        self.name = "base"
        
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> str:
        """Generate text from prompt"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def generate_streaming(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                           temperature: float = DEFAULT_TEMPERATURE, 
                           **kwargs) -> Generator[str, None, None]:
        """Generate text from prompt with streaming response"""
        raise NotImplementedError("Subclasses must implement this method")

class ClaudeProvider(LLMProvider):
    """Provider for Anthropic's Claude models"""
    
    def __init__(self, api_key: Optional[str] = None, 
                model: str = DEFAULT_MODEL,
                max_retries: int = 3,
                retry_delay: int = 5):
        """
        Initialize Claude provider
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        super().__init__()
        self.name = "claude"
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize client
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK not available. Install with 'pip install anthropic'")
            
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> str:
        """
        Generate text from prompt using Claude
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0-1)
            
        Returns:
            Generated text
        """
        # Prepare message structure
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Add retry logic
        for attempt in range(self.max_retries):
            try:
                # Debug the model name
                logger.info(f"Using model: {self.model}")
                
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                # Return the content
                return response.content[0].text
                
            except Exception as e:
                logger.error(f"Error generating response from Claude (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        # This should never be reached due to the raise in the loop
        return "Error generating response."
        
    def generate_streaming(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                          temperature: float = DEFAULT_TEMPERATURE, 
                          **kwargs) -> Generator[str, None, None]:
        """
        Generate text from prompt with streaming response
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0-1)
            
        Yields:
            Chunks of generated text
        """
        # Prepare message structure
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Add retry logic for streaming
        for attempt in range(self.max_retries):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                    
                # If we reach here, streaming finished successfully
                break
                
            except Exception as e:
                logger.error(f"Error streaming response from Claude (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    # Yield error message
                    yield f"\n[Error during streaming. Retrying ({attempt+1}/{self.max_retries})...]\n"
                else:
                    yield f"\n[Error: Failed to complete response after {self.max_retries} attempts.]\n"
                    raise

class RAGAssistant:
    """
    RAG-powered assistant that combines context building with LLM generation
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None,
                context_builder: Optional[ContextBuilder] = None,
                max_tokens_answer: int = DEFAULT_MAX_TOKENS,
                max_tokens_context: int = 4000,
                temperature: float = 0.5):
        """
        Initialize RAG assistant
        
        Args:
            llm_provider: LLM provider instance (defaults to Claude)
            context_builder: Context builder instance (defaults to new ContextBuilder)
            max_tokens_answer: Maximum tokens for LLM answer
            max_tokens_context: Maximum tokens for context
            temperature: Temperature for LLM generation
        """
        # Create default LLM provider if not provided
        if llm_provider is None:
            try:
                self.llm = ClaudeProvider()
            except (ImportError, ValueError) as e:
                logger.error(f"Failed to initialize Claude provider: {str(e)}")
                raise ValueError("No LLM provider available. Please install required packages or provide an API key.")
        else:
            self.llm = llm_provider
            
        # Create context builder if not provided
        self.context_builder = context_builder or ContextBuilder(max_tokens=max_tokens_context)
        
        # Configuration
        self.max_tokens_answer = max_tokens_answer
        self.max_tokens_context = max_tokens_context
        self.temperature = temperature
        
        logger.info(f"Initialized RAG Assistant with {self.llm.name} provider")
        
    def answer_query(self, query: str, search_type: str = 'hybrid',
                   vector_weight: Optional[float] = None,
                   keyword_weight: Optional[float] = None,
                   source_type: Optional[str] = None,
                   top_k: int = 10,
                   temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Answer a user query using RAG
        
        Args:
            query: User query
            search_type: Type of search ('vector', 'keyword', or 'hybrid')
            vector_weight: Weight for vector search in hybrid search (0-1)
            keyword_weight: Weight for keyword search in hybrid search (0-1)
            source_type: Filter results by source type
            top_k: Number of top results to consider
            temperature: Override default temperature
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Build context
        prompt, source_metadata = self.context_builder.build_context_for_query(
            query=query,
            search_type=search_type,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            source_type=source_type,
            top_k=top_k
        )
        
        # Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_tokens_answer,
            temperature=temperature or self.temperature
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Generated answer for query in {elapsed:.2f}s")
        
        # Return results
        return {
            "query": query,
            "answer": answer,
            "sources": source_metadata,
            "metadata": {
                "time_taken": elapsed,
                "llm_provider": self.llm.name,
                "llm_model": getattr(self.llm, "model", "unknown"),
                "search_type": search_type,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    def answer_query_streaming(self, query: str, 
                             callback: Optional[Callable[[str], None]] = None,
                             search_type: str = 'hybrid',
                             vector_weight: Optional[float] = None,
                             keyword_weight: Optional[float] = None,
                             source_type: Optional[str] = None,
                             top_k: int = 10,
                             temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Answer a user query using RAG with streaming response
        
        Args:
            query: User query
            callback: Optional callback function for streaming chunks
            search_type: Type of search ('vector', 'keyword', or 'hybrid')
            vector_weight: Weight for vector search in hybrid search (0-1)
            keyword_weight: Weight for keyword search in hybrid search (0-1)
            source_type: Filter results by source type
            top_k: Number of top results to consider
            temperature: Override default temperature
            
        Returns:
            Dictionary with complete answer, sources, and metadata
        """
        start_time = time.time()
        
        # Build context
        prompt, source_metadata = self.context_builder.build_context_for_query(
            query=query,
            search_type=search_type,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            source_type=source_type,
            top_k=top_k
        )
        
        # Generate answer with streaming
        answer_chunks = []
        for chunk in self.llm.generate_streaming(
            prompt=prompt,
            max_tokens=self.max_tokens_answer,
            temperature=temperature or self.temperature
        ):
            answer_chunks.append(chunk)
            if callback:
                callback(chunk)
        
        # Combine chunks into complete answer
        answer = "".join(answer_chunks)
        
        elapsed = time.time() - start_time
        logger.info(f"Generated streaming answer for query in {elapsed:.2f}s")
        
        # Return results
        return {
            "query": query,
            "answer": answer,
            "sources": source_metadata,
            "metadata": {
                "time_taken": elapsed,
                "llm_provider": self.llm.name,
                "llm_model": getattr(self.llm, "model", "unknown"),
                "search_type": search_type,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def save_response(self, response: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save response to file
        
        Args:
            response: Response dictionary
            filename: Optional filename (defaults to timestamp-based name)
            
        Returns:
            Path to saved file
        """
        # Create responses directory if it doesn't exist
        os.makedirs("responses", exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = response.get("query", "query")[:30].replace(" ", "_").lower()
            filename = f"response_{timestamp}_{query_slug}.json"
            
        filepath = os.path.join("responses", filename)
        
        # Save response
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2, default=str)
            
        return filepath

def main():
    """Main function for direct script execution"""
    parser = argparse.ArgumentParser(description="RAG LLM Integration")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--search-type", choices=['vector', 'hybrid'], default='hybrid', 
                       help="Search type to use")
    parser.add_argument("--source-type", help="Filter by source type")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results to consider")
    parser.add_argument("--max-tokens-answer", type=int, default=DEFAULT_MAX_TOKENS, 
                       help="Maximum tokens for answer")
    parser.add_argument("--max-tokens-context", type=int, default=4000, 
                       help="Maximum tokens for context")
    parser.add_argument("--vector-weight", type=float, help="Weight for vector search (0-1)")
    parser.add_argument("--keyword-weight", type=float, help="Weight for keyword search (0-1)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, 
                       help="Temperature for generation")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--save", action="store_true", help="Save the response to file")
    parser.add_argument("--output", help="Output file for the response")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Create LLM provider
        llm_provider = ClaudeProvider(model=args.model)
        
        # Create context builder
        context_builder = ContextBuilder(max_tokens=args.max_tokens_context)
        
        # Create RAG assistant
        rag = RAGAssistant(
            llm_provider=llm_provider,
            context_builder=context_builder,
            max_tokens_answer=args.max_tokens_answer,
            max_tokens_context=args.max_tokens_context,
            temperature=args.temperature
        )
        
        # Answer query
        if args.stream:
            # Define callback for streaming
            def print_chunk(chunk):
                print(chunk, end="", flush=True)
                
            print("\nGenerating response...\n")
            response = rag.answer_query_streaming(
                query=args.query,
                callback=print_chunk,
                search_type=args.search_type,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight,
                source_type=args.source_type,
                top_k=args.top_k
            )
            print("\n")  # Add newline after streaming
        else:
            print("\nGenerating response...\n")
            response = rag.answer_query(
                query=args.query,
                search_type=args.search_type,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight,
                source_type=args.source_type,
                top_k=args.top_k
            )
            print(response["answer"])
            print("")
            
        # Print source information
        print("\nSources:")
        for idx, source in enumerate(response["sources"]):
            title = source.get("title", "Untitled")
            source_type = source.get("source_type", "Unknown")
            print(f"[{idx+1}] {title} ({source_type})")
            
        # Save response if requested
        if args.save or args.output:
            filepath = rag.save_response(response, args.output)
            print(f"\nResponse saved to: {filepath}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 