import numpy as np
import logging
import sqlite3
import json
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('retrieval_metrics')

class RetrievalEvaluator:
    """Evaluator for retrieval quality metrics"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        
    def calculate_precision_recall(self, query_id, retrieved_ids, relevant_ids):
        """
        Calculate precision, recall, and F1 score
        
        Args:
            query_id: ID of the query
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)
            
        Returns:
            Dict with precision, recall, and F1 score
        """
        if not retrieved_ids or not relevant_ids:
            return {
                'query_id': query_id,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'retrieved_count': len(retrieved_ids),
                'relevant_count': len(relevant_ids),
                'true_positives': 0
            }
            
        # Convert to sets for intersection
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # Calculate intersection (true positives)
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        # Calculate metrics
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        
        return {
            'query_id': query_id,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(relevant_set),
            'true_positives': true_positives
        }
    
    def calculate_ndcg(self, query_id, retrieved_ids, relevant_ids_with_scores, k=None):
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Args:
            query_id: ID of the query
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids_with_scores: Dict mapping relevant doc IDs to relevance scores
            k: Cutoff for NDCG calculation (None for all)
            
        Returns:
            Dict with NDCG score
        """
        if not retrieved_ids or not relevant_ids_with_scores:
            return {
                'query_id': query_id,
                'ndcg': 0.0,
                'dcg': 0.0,
                'idcg': 0.0,
                'k': k or len(retrieved_ids)
            }
            
        # Use k if specified, otherwise use all retrieved docs
        k = k or len(retrieved_ids)
        retrieved_ids = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            # Get relevance score, default to 0 if not relevant
            rel = relevant_ids_with_scores.get(str(doc_id), relevant_ids_with_scores.get(doc_id, 0.0))
            # Position is 1-indexed in the formula
            position = i + 1
            dcg += rel / np.log2(position + 1)
        
        # Calculate ideal DCG
        # Sort relevance scores in descending order
        ideal_scores = sorted(relevant_ids_with_scores.values(), reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            'query_id': query_id,
            'ndcg': ndcg,
            'dcg': dcg,
            'idcg': idcg,
            'k': k
        }
        
    def evaluate_search_results(self, search_results, ground_truth, metrics=None):
        """
        Evaluate search results against ground truth
        
        Args:
            search_results: Dict with query_id -> list of retrieved doc IDs
            ground_truth: Dict with query_id -> list of relevant doc IDs
            metrics: List of metrics to calculate, or None for all
            
        Returns:
            Dict with evaluation results
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'ndcg']
            
        results = {}
        
        for query_id, retrieved_ids in search_results.items():
            if query_id not in ground_truth:
                continue
                
            relevant_data = ground_truth[query_id]
            
            # Handle different ground truth formats
            if isinstance(relevant_data, list):
                relevant_ids = relevant_data
                relevant_with_scores = {doc_id: 1.0 for doc_id in relevant_ids}
            elif isinstance(relevant_data, dict):
                relevant_ids = list(relevant_data.keys())
                relevant_with_scores = relevant_data
            else:
                logger.warning(f"Unknown ground truth format for query {query_id}")
                continue
                
            query_results = {}
            
            # Calculate metrics
            if any(m in metrics for m in ['precision', 'recall', 'f1']):
                pr_results = self.calculate_precision_recall(
                    query_id, retrieved_ids, relevant_ids
                )
                query_results.update({
                    'precision': pr_results['precision'],
                    'recall': pr_results['recall'],
                    'f1': pr_results['f1']
                })
                
            if 'ndcg' in metrics:
                ndcg_results = self.calculate_ndcg(
                    query_id, retrieved_ids, relevant_with_scores
                )
                query_results['ndcg'] = ndcg_results['ndcg']
                
            results[query_id] = query_results
            
        # Calculate aggregate metrics
        agg_results = {}
        
        for metric in metrics:
            metric_values = [results[q][metric] for q in results if metric in results[q]]
            if metric_values:
                agg_results[f'avg_{metric}'] = sum(metric_values) / len(metric_values)
                
        return {
            'query_results': results,
            'aggregate': agg_results
        }
    
    def save_evaluation_results(self, eval_results, evaluation_name, notes=None):
        """Save evaluation results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create evaluation_results table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    date_created TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    notes TEXT
                )
            """)
            
            # Insert evaluation results
            cursor.execute("""
                INSERT INTO evaluation_results (name, date_created, metrics_json, notes)
                VALUES (?, ?, ?, ?)
            """, (
                evaluation_name,
                datetime.now().isoformat(),
                json.dumps(eval_results),
                notes
            ))
            
            eval_id = cursor.lastrowid
            conn.commit()
            
            return eval_id
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            conn.rollback()
            return None
            
        finally:
            conn.close() 