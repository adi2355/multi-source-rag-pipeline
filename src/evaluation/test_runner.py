import os
import logging
import json
import time
import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.retrieval_metrics import RetrievalEvaluator
from evaluation.answer_evaluator import AnswerEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_runner')

class RAGTestRunner:
    """Automated test runner for RAG system"""
    
    def __init__(self, db_path=None, rag_system=None):
        self.db_path = db_path or config.DB_PATH
        self.rag_system = rag_system
        self.retrieval_evaluator = RetrievalEvaluator(db_path=self.db_path)
        self.answer_evaluator = AnswerEvaluator()
    
    def load_test_dataset(self, dataset_id=None, dataset_name=None, file_path=None):
        """Load test dataset from database or file"""
        if file_path:
            # Load from file
            try:
                with open(file_path, 'r') as f:
                    dataset = json.load(f)
                return dataset
            except Exception as e:
                logger.error(f"Error loading test dataset from file: {str(e)}")
                return None
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if dataset_id:
                cursor.execute(
                    "SELECT queries_json FROM test_datasets WHERE id = ?",
                    (dataset_id,)
                )
            elif dataset_name:
                cursor.execute(
                    "SELECT queries_json FROM test_datasets WHERE name = ?",
                    (dataset_name,)
                )
            else:
                # Get latest dataset
                cursor.execute(
                    "SELECT queries_json FROM test_datasets ORDER BY date_created DESC LIMIT 1"
                )
                
            row = cursor.fetchone()
            
            if not row:
                logger.error("Test dataset not found")
                return None
                
            return json.loads(row[0])
            
        except Exception as e:
            logger.error(f"Error loading test dataset: {str(e)}")
            return None
            
        finally:
            conn.close()
    
    def run_retrieval_tests(self, dataset, search_types=None, top_k=10, 
                          vector_weights=None, keyword_weights=None):
        """
        Run retrieval tests on dataset
        
        Args:
            dataset: Test dataset
            search_types: List of search types to test ('vector', 'hybrid', 'keyword')
            top_k: Number of results to retrieve
            vector_weights: List of vector weights to test with hybrid search
            keyword_weights: List of keyword weights to test with hybrid search
            
        Returns:
            Dict with test results
        """
        if not dataset or 'queries' not in dataset:
            logger.error("Invalid test dataset")
            return None
            
        if not self.rag_system:
            logger.error("RAG system not provided")
            return None
            
        # Default search types if not specified
        if not search_types:
            search_types = ['vector', 'hybrid']
            
        # Default weights if not specified
        if not vector_weights and 'hybrid' in search_types:
            vector_weights = [0.7]
            keyword_weights = [0.3]
            
        # Ensure weights are paired correctly
        weight_pairs = []
        if 'hybrid' in search_types:
            for i, v_weight in enumerate(vector_weights):
                k_weight = keyword_weights[i] if i < len(keyword_weights) else 1.0 - v_weight
                weight_pairs.append((v_weight, k_weight))
                
        # Prepare result structure
        test_results = {
            'dataset_name': dataset.get('name', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'search_results': {},
            'metrics': {}
        }
        
        # Run tests for each search type
        for search_type in search_types:
            if search_type == 'hybrid':
                # Test each weight combination
                for v_weight, k_weight in weight_pairs:
                    config_name = f"{search_type}_v{v_weight}_k{k_weight}"
                    
                    logger.info(f"Running {search_type} search tests with weights v={v_weight}, k={k_weight}")
                    
                    search_results = {}
                    ground_truth = {}
                    
                    # Process each query
                    for query in dataset['queries']:
                        query_id = query['query_id']
                        query_text = query['query_text']
                        
                        logger.info(f"Testing query: {query_text}")
                        
                        # Record ground truth
                        ground_truth[query_id] = query['relevant_content']
                        
                        # Run search
                        try:
                            start_time = time.time()
                            results = self.rag_system.search(
                                query=query_text,
                                search_type=search_type,
                                top_k=top_k,
                                vector_weight=v_weight,
                                keyword_weight=k_weight
                            )
                            search_time = time.time() - start_time
                            
                            # Extract content IDs
                            content_ids = [r.get('content_id') for r in results]
                            
                            # Save search results
                            search_results[query_id] = {
                                'content_ids': content_ids,
                                'search_time': search_time
                            }
                            
                        except Exception as e:
                            logger.error(f"Error searching for query {query_id}: {str(e)}")
                            search_results[query_id] = {
                                'content_ids': [],
                                'search_time': 0,
                                'error': str(e)
                            }
                    
                    # Save search results
                    test_results['search_results'][config_name] = search_results
                    
                    # Calculate metrics
                    metrics = self.retrieval_evaluator.evaluate_search_results(
                        {qid: res['content_ids'] for qid, res in search_results.items()},
                        ground_truth
                    )
                    
                    # Save metrics
                    test_results['metrics'][config_name] = metrics
                    
                    # Log summary metrics
                    if 'aggregate' in metrics:
                        logger.info(f"Search type {config_name} metrics:")
                        for metric, value in metrics['aggregate'].items():
                            logger.info(f"  {metric}: {value:.4f}")
            else:
                # Run standard search type
                logger.info(f"Running {search_type} search tests")
                
                search_results = {}
                ground_truth = {}
                
                # Process each query
                for query in dataset['queries']:
                    query_id = query['query_id']
                    query_text = query['query_text']
                    
                    logger.info(f"Testing query: {query_text}")
                    
                    # Record ground truth
                    ground_truth[query_id] = query['relevant_content']
                    
                    # Run search
                    try:
                        start_time = time.time()
                        results = self.rag_system.search(
                            query=query_text,
                            search_type=search_type,
                            top_k=top_k
                        )
                        search_time = time.time() - start_time
                        
                        # Extract content IDs
                        content_ids = [r.get('content_id') for r in results]
                        
                        # Save search results
                        search_results[query_id] = {
                            'content_ids': content_ids,
                            'search_time': search_time
                        }
                        
                    except Exception as e:
                        logger.error(f"Error searching for query {query_id}: {str(e)}")
                        search_results[query_id] = {
                            'content_ids': [],
                            'search_time': 0,
                            'error': str(e)
                        }
                
                # Save search results
                test_results['search_results'][search_type] = search_results
                
                # Calculate metrics
                metrics = self.retrieval_evaluator.evaluate_search_results(
                    {qid: res['content_ids'] for qid, res in search_results.items()},
                    ground_truth
                )
                
                # Save metrics
                test_results['metrics'][search_type] = metrics
                
                # Log summary metrics
                if 'aggregate' in metrics:
                    logger.info(f"Search type {search_type} metrics:")
                    for metric, value in metrics['aggregate'].items():
                        logger.info(f"  {metric}: {value:.4f}")
        
        # Save test results
        self._save_test_results(test_results, 'retrieval')
        
        return test_results
    
    def run_answer_tests(self, dataset, search_type='hybrid', top_k=5,
                       vector_weight=None, keyword_weight=None, max_queries=10):
        """
        Run answer quality tests on dataset
        
        Args:
            dataset: Test dataset
            search_type: Search type to use ('vector', 'hybrid', 'keyword')
            top_k: Number of results to retrieve
            vector_weight: Vector weight for hybrid search
            keyword_weight: Keyword weight for hybrid search
            max_queries: Maximum number of queries to test
            
        Returns:
            Dict with test results
        """
        if not dataset or 'queries' not in dataset:
            logger.error("Invalid test dataset")
            return None
            
        if not self.rag_system:
            logger.error("RAG system not provided")
            return None
            
        # Limit number of queries to test
        queries = dataset['queries'][:max_queries]
        
        # Prepare result structure
        test_results = {
            'dataset_name': dataset.get('name', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'search_type': search_type,
                'top_k': top_k,
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight
            },
            'answer_results': {}
        }
        
        # Process each query
        for query in queries:
            query_id = query['query_id']
            query_text = query['query_text']
            
            logger.info(f"Testing query: {query_text}")
            
            try:
                # Generate answer
                start_time = time.time()
                response = self.rag_system.answer_query(
                    query=query_text,
                    search_type=search_type,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight
                )
                answer_time = time.time() - start_time
                
                answer = response.get('answer', '')
                context = response.get('context', '')
                
                # Evaluate answer
                faithfulness = self.answer_evaluator.evaluate_faithfulness(
                    answer=answer,
                    context=context,
                    query=query_text
                )
                
                relevance = self.answer_evaluator.evaluate_relevance(
                    answer=answer,
                    query=query_text
                )
                
                # Save results
                test_results['answer_results'][query_id] = {
                    'query_text': query_text,
                    'answer_text': answer,
                    'context_text': context,
                    'answer_time': answer_time,
                    'faithfulness_score': faithfulness.get('score', 0),
                    'faithfulness_explanation': faithfulness.get('explanation', ''),
                    'relevance_score': relevance.get('score', 0),
                    'relevance_explanation': relevance.get('explanation', '')
                }
                
                # Save to answer evaluations table
                self.answer_evaluator.save_evaluation_results({
                    'query_id': query_id,
                    'query_text': query_text,
                    'answer_text': answer,
                    'context_text': context,
                    'faithfulness_score': faithfulness.get('score', 0),
                    'faithfulness_explanation': faithfulness.get('explanation', ''),
                    'relevance_score': relevance.get('score', 0),
                    'relevance_explanation': relevance.get('explanation', '')
                })
                
            except Exception as e:
                logger.error(f"Error generating answer for query {query_id}: {str(e)}")
                test_results['answer_results'][query_id] = {
                    'query_text': query_text,
                    'error': str(e)
                }
        
        # Calculate aggregate metrics
        faithfulness_scores = [r['faithfulness_score'] for r in test_results['answer_results'].values() 
                            if 'faithfulness_score' in r]
        relevance_scores = [r['relevance_score'] for r in test_results['answer_results'].values() 
                           if 'relevance_score' in r]
        
        test_results['metrics'] = {
            'avg_faithfulness': sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'combined_score': (sum(faithfulness_scores) + sum(relevance_scores)) / (len(faithfulness_scores) + len(relevance_scores)) 
                            if faithfulness_scores and relevance_scores else 0
        }
        
        # Log metrics
        logger.info(f"Answer quality metrics:")
        for metric, value in test_results['metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save test results
        self._save_test_results(test_results, 'answer_quality')
        
        return test_results
    
    def _save_test_results(self, results, test_type):
        """Save test results to database and file"""
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create test_results table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY,
                    test_type TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    date_created TEXT NOT NULL,
                    results_json TEXT NOT NULL
                )
            """)
            
            # Create test name
            test_name = f"{test_type}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Insert test results
            cursor.execute("""
                INSERT INTO test_results (test_type, test_name, date_created, results_json)
                VALUES (?, ?, ?, ?)
            """, (
                test_type,
                test_name,
                datetime.now().isoformat(),
                json.dumps(results)
            ))
            
            conn.commit()
            
            # Save to file as well
            results_dir = os.path.join(config.DATA_DIR, "evaluation", "results")
            os.makedirs(results_dir, exist_ok=True)
            
            with open(os.path.join(results_dir, f"{test_name}.json"), 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Test results saved to {os.path.join(results_dir, f'{test_name}.json')}")
            
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
            conn.rollback()
            
        finally:
            conn.close() 