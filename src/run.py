"""
Main script to run the complete Instagram Knowledge Base system
"""
import os
import argparse
import logging
from time import time
import sqlite3

# Import our modules
from config import DATA_DIR
import config
import downloader
import transcriber
import indexer
import summarizer
from app import app

# Import the new modules
try:
    import db_migration
    has_db_migration = True
except ImportError:
    has_db_migration = False

try:
    import github_collector
except ImportError as e:
    github_collector_error = str(e)
    github_collector = None

try:
    import arxiv_collector
except ImportError as e:
    arxiv_collector_error = str(e)
    arxiv_collector = None

try:
    import concept_extractor
    import chunking
    import embeddings
    import generate_embeddings
    import vector_search
    import hybrid_search
    import context_builder
    import llm_integration
    has_vector_search = True
    has_rag = True
except ImportError as e:
    has_vector_search = False
    has_rag = False
    import_error = str(e)
    missing_module = str(e).split("No module named ")[-1].strip("'")
    import sys
    print(f"Current sys.path: {sys.path}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")

# Try to import knowledge graph module
try:
    import knowledge_graph
    has_knowledge_graph = True
except ImportError:
    has_knowledge_graph = False

# Try to import evaluation modules
try:
    from evaluation import dashboard
    from evaluation.test_runner import RAGTestRunner
    has_evaluation = True
except ImportError:
    has_evaluation = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_kb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

def setup():
    """Setup necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def run_downloader(force_refresh=False, use_auth=True):
    """Run the Instagram downloader"""
    logger.info("Starting Instagram content download")
    start_time = time()
    downloader.download_from_instagram(force_refresh=force_refresh, use_auth=use_auth)
    logger.info(f"Download completed in {time() - start_time:.2f} seconds")

def run_transcriber(batch_size=16, extraction_workers=4, auto_batch_size=True):
    """Run the audio extraction and transcription"""
    logger.info("Starting audio extraction and transcription")
    start_time = time()
    transcriber.process_videos(
        batch_size=batch_size,
        extraction_workers=extraction_workers,
        auto_batch_size=auto_batch_size
    )
    logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")

def run_summarizer():
    """Run the transcript summarization using Claude"""
    logger.info("Starting transcript summarization using Claude")
    start_time = time()
    if args.no_batch:
        logger.info("Batch processing disabled, using sequential processing")
        summarizer.summarize_transcripts(use_batch_api=False)
    else:
        logger.info("Using batch processing with Claude API for cost savings (50% cheaper)")
        summarizer.summarize_transcripts(use_batch_api=True)
    logger.info(f"Summarization completed in {time() - start_time:.2f} seconds")

def run_indexer():
    """Run the knowledge base indexer"""
    logger.info("Starting indexing of transcripts")
    start_time = time()
    indexer.index_transcripts()
    logger.info(f"Indexing completed in {time() - start_time:.2f} seconds")

def run_web_interface(port=5000, debug=False):
    """Run the web interface with API endpoints"""
    from app import app
    
    # The app module already registers all available blueprints when imported
    # So we don't need to register them again, just run the app
    logger.info(f"Starting web interface on port {port}, debug={debug}")
    logger.info(f"Registered blueprints: {list(app.blueprints.keys())}")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)

def run_db_migration():
    """Run database migration to support multiple content sources"""
    if not has_vector_search:
        logger.error("Database migration module not available")
        return
    
    logger.info("Starting database migration")
    start_time = time()
    success = db_migration.migrate_database()
    
    if success:
        logger.info(f"Database migration completed in {time() - start_time:.2f} seconds")
    else:
        logger.error(f"Database migration failed after {time() - start_time:.2f} seconds")

def run_github_collector(max_repos=None):
    """Run GitHub repository collection"""
    if not has_vector_search:
        logger.error("GitHub collector module not available")
        return
    
    logger.info("Starting GitHub repository collection")
    start_time = time()
    success_count = github_collector.collect_github_repos(max_repos=max_repos)
    logger.info(f"GitHub collection completed in {time() - start_time:.2f} seconds, processed {success_count} repositories")

def run_papers_collector(args):
    try:
        import arxiv_collector
        
        if arxiv_collector is None:
            logger.error("ArXiv collector module is not available")
            return
            
        if args.paper_url:
            logger.info(f"Downloading paper from URL: {args.paper_url}")
            arxiv_collector.download_paper_from_url(args.paper_url)
        elif args.download_papers:
            logger.info(f"Downloading up to {args.max_papers} papers without processing")
            arxiv_collector.download_papers_only(max_papers=args.max_papers)
        elif args.process_papers:
            logger.info(f"Processing previously downloaded papers (max: {args.max_papers})")
            arxiv_collector.batch_process_pdfs(max_papers=args.max_papers)
        else:
            logger.info(f"Collecting up to {args.max_papers} papers")
            arxiv_collector.collect_papers(max_papers=args.max_papers)
    except Exception as e:
        logger.error(f"Failed to run ArXiv collector: {e}")
        import traceback
        traceback.print_exc()

def run_concept_extractor(limit=None, source_type=None, batch=False, batch_size=5, force=False):
    """Run concept extraction on content"""
    if not has_vector_search:
        logger.error("Concept extractor module not available")
        return
    
    logger.info("Starting concept extraction")
    start_time = time()
    
    if batch:
        logger.info(f"Processing content in batch mode with batch size {batch_size}")
        processed = concept_extractor.process_in_batches(batch_size=batch_size, force=force)
        logger.info(f"Batch processing completed. Processed {processed} items.")
        return processed
    
    if source_type:
        logger.info(f"Processing {limit or 'all'} items from source type: {source_type}")
        processed = concept_extractor.process_unprocessed_content(limit=limit or 5, source_type=source_type, force=force)
        logger.info(f"Processed {processed} items from {source_type}")
    else:
        # Process some content from each source type
        total_processed = 0
        for src_type in ["research_paper", "github", "instagram"]:
            logger.info(f"Processing source type: {src_type}")
            processed = concept_extractor.process_unprocessed_content(limit=limit or 3, source_type=src_type, force=force)
            logger.info(f"Processed {processed} items from {src_type}")
            total_processed += processed
        
        logger.info(f"Concept extraction completed in {time() - start_time:.2f} seconds, processed {total_processed} items")

def run_embedding_generation(source_type=None, limit=None, batch_size=50, 
                           chunk_size=500, chunk_overlap=100, force=False):
    """Generate embeddings for content"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return
    
    logger.info("Starting embedding generation")
    start_time = time()
    
    args = []
    if source_type:
        args.extend(["--source-type", source_type])
    
    if limit:
        args.extend(["--limit", str(limit)])
    
    if batch_size:
        args.extend(["--batch-size", str(batch_size)])
    
    if chunk_size:
        args.extend(["--chunk-size", str(chunk_size)])
    
    if chunk_overlap:
        args.extend(["--chunk-overlap", str(chunk_overlap)])
    
    if force:
        args.append("--force")
    
    # Run embedding generator script
    import sys
    old_args = sys.argv
    sys.argv = ["generate_embeddings.py"] + args
    try:
        generate_embeddings.main()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
    finally:
        sys.argv = old_args
    
    logger.info(f"Embedding generation completed in {time() - start_time:.2f} seconds")

def run_vector_search(query, top_k=5, source_type=None, in_memory_index=False):
    """Run vector search for a query"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return []
    
    logger.info(f"Running vector search for: {query}")
    start_time = time()
    
    try:
        if in_memory_index:
            # Create in-memory index for faster search
            index = vector_search.create_memory_index()
            
            # Create embedding generator
            embedding_generator = embeddings.EmbeddingGenerator()
            
            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query)
            
            # Search using in-memory index
            results = vector_search.search_memory_index(query_embedding, index, top_k=top_k)
            
            # Fetch chunk text and enrich results
            results = vector_search.enrich_search_results(results)
        else:
            # Use standard search
            results = vector_search.debug_search(query, top_k=top_k)
        
        logger.info(f"Vector search completed in {time() - start_time:.2f} seconds with {len(results)} results")
        
        # Print results
        print(f"\nVector search results for: {query}")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} - Similarity: {result.get('similarity', 0):.4f}")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Source: {result.get('source_type', 'Unknown')}")
            print(f"Content ID: {result.get('content_id', 'Unknown')}")
            print("-" * 40)
            if 'chunk_text' in result:
                text = result['chunk_text']
                print(text[:300] + "..." if len(text) > 300 else text)
            print("-" * 40)
            
            if 'concepts' in result:
                print("Concepts:")
                for concept in result['concepts'][:5]:
                    print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
            print()
        
        return results
    
    except Exception as e:
        logger.error(f"Error performing vector search: {str(e)}")
        return []

def run_hybrid_search(query, top_k=5, source_type=None, vector_weight=None, 
                     keyword_weight=None, adaptive=True):
    """Run hybrid search for a query"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return []
    
    logger.info(f"Running hybrid search for: {query}")
    start_time = time()
    
    try:
        # Load weight history if using adaptive weighting
        weight_history = None
        if adaptive and (vector_weight is None or keyword_weight is None):
            weight_history = hybrid_search.load_weights_history()
            
            # Get query type
            query_type = hybrid_search.classify_query_type(query)
            
            # Use query-specific weights if available
            normalized_query = query.lower().strip()
            if normalized_query in weight_history['queries']:
                vector_weight = weight_history['queries'][normalized_query]['vector_weight']
                keyword_weight = weight_history['queries'][normalized_query]['keyword_weight']
                logger.info(f"Using query-specific weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
            else:
                # Use query type defaults
                type_key = f"default_{query_type}_vector_weight"
                if type_key in weight_history:
                    vector_weight = weight_history[type_key]
                    keyword_weight = 1.0 - vector_weight
                    logger.info(f"Using {query_type} query type weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        
        # Perform hybrid search
        results = hybrid_search.hybrid_search(
            query=query,
            top_k=top_k,
            source_type=source_type,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
        
        logger.info(f"Hybrid search completed in {time() - start_time:.2f} seconds with {len(results)} results")
        
        # Get actual weights used for display
        if vector_weight is None or keyword_weight is None:
            vector_weight, keyword_weight = hybrid_search.determine_weights(query)
        
        # Print results
        print(f"\nHybrid search results for: {query}")
        print(f"Weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} - Combined Score: {result.get('combined_score', 0):.4f}")
            print(f"Vector Score: {result.get('vector_score', 0):.4f}, Keyword Score: {result.get('keyword_score', 0):.4f}")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Source: {result.get('source_type', 'Unknown')}")
            print(f"Content ID: {result['content_id']}")
            print("-" * 40)
            
            if 'snippet' in result and result['snippet']:
                print(f"Keyword Match: {result['snippet']}")
            
            if 'chunk_text' in result and result['chunk_text']:
                text = result['chunk_text']
                print(text[:300] + "..." if len(text) > 300 else text)
            
            print("-" * 40)
            
            if 'concepts' in result:
                print("Concepts:")
                for concept in result['concepts'][:5]:
                    print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
            
            print()
        
        return results
    
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        return []

def run_rag_query(query, search_type='hybrid', source_type=None, top_k=5, 
                 vector_weight=None, keyword_weight=None, 
                 max_tokens_context=4000, max_tokens_answer=1000,
                 temperature=0.5, model=None, stream=False):
    """Run a RAG query and get a response from the LLM"""
    if not has_rag:
        logger.error(f"RAG modules not available: {import_error}")
        return
    
    logger.info(f"Running RAG query: {query}")
    start_time = time()
    
    try:
        # First check if we have any content in the database with embeddings
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Check if content_embeddings table exists and has data
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='table' AND name='content_embeddings'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM content_embeddings")
            embedding_count = cursor.fetchone()[0]
            
            if embedding_count == 0:
                logger.warning("No embeddings found in the database. Run --generate-embeddings first.")
                print("No content embeddings found in the database.")
                print("Please run the following command to generate embeddings first:")
                print("  python run.py --generate-embeddings")
                return
        else:
            logger.warning("Content embeddings table does not exist. Run --migrate and --generate-embeddings first.")
            print("Content embeddings table does not exist.")
            print("Please run the following commands to set up the database and generate embeddings:")
            print("  1. python run.py --migrate")
            print("  2. python run.py --generate-embeddings")
            return
        
        conn.close()
        
        # Create LLM provider
        llm_provider = llm_integration.ClaudeProvider(model=model or "claude-3-sonnet-20240229")
        
        # Create context builder
        ctx_builder = context_builder.ContextBuilder(max_tokens=max_tokens_context)
        
        # Create RAG assistant
        rag = llm_integration.RAGAssistant(
            llm_provider=llm_provider,
            context_builder=ctx_builder,
            max_tokens_answer=max_tokens_answer,
            max_tokens_context=max_tokens_context,
            temperature=temperature
        )
        
        # Answer query
        if stream:
            # Define callback for streaming
            def print_chunk(chunk):
                print(chunk, end="", flush=True)
                
            response = rag.answer_query_streaming(
                query=query,
                callback=print_chunk,
                search_type=search_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                source_type=source_type,
                top_k=top_k
            )
            print("\n")  # Add newline after streaming
        else:
            response = rag.answer_query(
                query=query,
                search_type=search_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                source_type=source_type,
                top_k=top_k
            )
            print(response["answer"])
            print("")
            
        # Print source information
        print("\nSources:")
        for idx, source in enumerate(response["sources"]):
            title = source.get("title", "Untitled")
            source_type = source.get("source_type", "Unknown")
            print(f"[{idx+1}] {title} ({source_type})")
            
        # Save response
        timestamp = response["metadata"]["timestamp"].split("T")[0]
        filepath = rag.save_response(response)
        print(f"\nResponse saved to: {filepath}")
        
        logger.info(f"RAG query completed in {time() - start_time:.2f} seconds")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error running RAG query: {str(e)}\n{error_traceback}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{error_traceback}")

def run_evaluation_dashboard(port=5050, debug=False):
    """Run the evaluation dashboard"""
    try:
        from evaluation import evaluation_bp
        from app import app
        
        # Register the evaluation blueprint
        app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
        
        # Add a redirect from root to evaluation dashboard
        @app.route('/')
        def redirect_to_evaluation():
            return redirect(url_for('evaluation_bp.evaluation_dashboard'))
            
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=debug)
    except ImportError:
        logger.error("Evaluation module not available")
        
def run_create_test_dataset(concept_queries=15, content_queries=10):
    """Create a test dataset for RAG evaluation"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info("Creating test dataset for RAG evaluation")
    start_time = time()
    
    from evaluation.test_queries import TestQueryGenerator
    generator = TestQueryGenerator()
    dataset_id = generator.create_test_dataset(
        concept_queries=concept_queries,
        content_queries=content_queries
    )
    
    logger.info(f"Test dataset creation completed in {time() - start_time:.2f} seconds")
    logger.info(f"Dataset ID: {dataset_id}")
    return True

def run_evaluation_tests(dataset_id, search_type='hybrid', top_k=10, vector_weight=0.7, keyword_weight=0.3):
    """Run retrieval tests on a test dataset"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info(f"Running retrieval tests on dataset {dataset_id}")
    start_time = time()
    
    test_runner = RAGTestRunner()
    results = test_runner.run_retrieval_tests(
        dataset_id=dataset_id,
        search_types=[search_type],
        top_k=[top_k],
        vector_weights=[vector_weight],
        keyword_weights=[keyword_weight]
    )
    
    logger.info(f"Retrieval tests completed in {time() - start_time:.2f} seconds")
    logger.info(f"Results: {results}")
    return True

def run_answer_tests(dataset_id, search_type='hybrid', top_k=5, vector_weight=0.7, keyword_weight=0.3, max_queries=10):
    """Run answer quality tests on a test dataset"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info(f"Running answer quality tests on dataset {dataset_id}")
    start_time = time()
    
    test_runner = RAGTestRunner()
    results = test_runner.run_answer_tests(
        dataset_id=dataset_id,
        search_type=search_type,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        max_queries=max_queries
    )
    
    logger.info(f"Answer quality tests completed in {time() - start_time:.2f} seconds")
    logger.info(f"Results: {results}")
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Instagram Knowledge Base system')
    
    # Main operation groups
    parser.add_argument('--knowledge', action='store_true', help='Build or update the knowledge base')
    parser.add_argument('--embeddings', action='store_true', help='Generate and index embeddings')
    parser.add_argument('--papers', action='store_true', help='Collect and process research papers')
    parser.add_argument('--visualize', action='store_true', help='Run visualization tools')
    parser.add_argument('--all', action='store_true', help='Run all main operations')
    
    # Add transcriber options
    parser.add_argument('--transcribe', action='store_true', help='Run audio extraction and transcription')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for transcription')
    parser.add_argument('--extraction-workers', type=int, default=4, help='Number of parallel audio extraction workers')
    parser.add_argument('--auto-batch-size', action='store_true', help='Automatically determine optimal batch size')
    
    # Other arguments
    parser.add_argument('--download-papers', action='store_true', help='Download papers without processing')
    parser.add_argument('--process-papers', action='store_true', help='Process previously downloaded papers')
    parser.add_argument('--max-papers', type=int, default=50, help='Maximum number of papers to process')
    parser.add_argument('--paper-url', type=str, help='Download a single paper from the provided URL')
    return parser.parse_args()

def main():
    """Main function to parse arguments and run the system"""
    args = parse_args()

    if args.all:
        args.knowledge = True
        args.embeddings = True
        args.papers = True

    if args.knowledge:
        run_knowledge_builder(args)

    if args.embeddings:
        run_embeddings_generator(args)

    if args.papers or args.download_papers or args.process_papers or args.paper_url:
        run_papers_collector(args)

    if args.visualize:
        run_visualizer(args)
        
    # Add support for transcriber
    if args.transcribe:
        run_transcriber(
            batch_size=args.batch_size,
            extraction_workers=args.extraction_workers,
            auto_batch_size=args.auto_batch_size
        )

    # The following if statements refer to arguments that no longer exist
    # Comment them out for now
    """
    if args.download:
        run_downloader(force_refresh=args.refresh_force, use_auth=not args.no_auth)
    
    if args.github:
        run_github_collector(args.github_max)
        
    if args.transcribe:
        run_transcriber()
    
    if args.summarize:
        run_summarizer()
        
    if args.concepts:
        run_concept_extractor(
            limit=args.concepts_limit, 
            source_type=args.concepts_source,
            batch=args.concepts_batch,
            batch_size=args.concepts_batch_size,
            force=args.concepts_force
        )
        
    if args.kg_analyze:
        if has_knowledge_graph:
            run_knowledge_graph(analyze=True)
        else:
            logger.error("Knowledge graph module not available")
    
    if args.index:
        run_indexer()
    
    # Handle search-related arguments
    if args.search:
        if args.search_type == 'vector':
            results = run_vector_search(
                query=args.search, 
                top_k=args.top_k, 
                source_type=args.search_source,
                in_memory_index=args.in_memory
            )
            print(f"\nTop {args.top_k} results for vector search: '{args.search}'")
        else:
            results = run_hybrid_search(
                query=args.search, 
                top_k=args.top_k, 
                source_type=args.search_source,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight
            )
            print(f"\nTop {args.top_k} results for {args.search_type} search: '{args.search}'")
        
        print("\n" + "-"*80)
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.get('title', 'No title')} (Score: {result.get('similarity', 0):.4f})")
            print(f"   Source: {result.get('source_type', 'Unknown')}, ID: {result.get('content_id', 'Unknown')}")
            print(f"   {result.get('snippet', '')[:200]}...")
        print("\n" + "-"*80)
    
    # Handle RAG query
    if args.rag_query:
        run_rag_query(
            query=args.rag_query,
            search_type=args.search_type,
            source_type=args.search_source,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            max_tokens_context=args.rag_tokens_context,
            max_tokens_answer=args.rag_tokens_answer,
            temperature=args.rag_temperature,
            model=args.rag_model,
            stream=args.rag_stream
        )
    
    # Handle evaluation arguments
    if args.evaluation_dashboard:
        run_evaluation_dashboard(port=args.evaluation_port, debug=args.debug)
        
    if args.create_test_dataset:
        run_create_test_dataset(
            concept_queries=args.concept_queries,
            content_queries=args.content_queries
        )
        
    if args.evaluation_tests:
        run_evaluation_tests(
            dataset_id=args.evaluation_tests,
            search_type=args.search_type,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight
        )
        
    if args.answer_tests:
        run_answer_tests(
            dataset_id=args.answer_tests,
            search_type=args.search_type,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            max_queries=args.max_queries
        )
        
    # Run the web interface if requested
    if args.api:
        run_web_interface(port=args.api_port, debug=args.debug)
        
    if args.web or args.all:
        run_web_interface(port=args.port, debug=args.debug)
    """

if __name__ == "__main__":
    main() 