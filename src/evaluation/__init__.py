"""
RAG Evaluation & Metrics System

This module provides comprehensive functionality for evaluating and analyzing the performance
of the RAG (Retrieval-Augmented Generation) system.

Components:
- Retrieval Quality Metrics: Measures precision, recall, F1, and NDCG at various k values
- Test Query Generator: Creates test datasets with ground truth relevance scores
- Answer Quality Evaluator: Assesses answer faithfulness and relevance using Claude API
- Test Runner: Automates evaluation across multiple parameters and settings
- Dashboard: Web interface for visualizing metrics and analyzing results
"""

import os
import logging

# Configure logging
logger = logging.getLogger('evaluation')
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

# Create necessary directories
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates', 'evaluation'), exist_ok=True)

# Import main components to make them available at the module level
try:
    from .retrieval_metrics import RetrievalEvaluator
    from .test_queries import TestQueryGenerator
    from .answer_evaluator import AnswerEvaluator
    from .test_runner import RAGTestRunner
    from .dashboard import evaluation_bp
    
    __all__ = [
        'RetrievalEvaluator',
        'TestQueryGenerator',
        'AnswerEvaluator',
        'RAGTestRunner',
        'evaluation_bp'
    ]
except ImportError as e:
    logger.warning(f"Error importing evaluation components: {e}")
    __all__ = [] 