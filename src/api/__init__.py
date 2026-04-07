"""
API Module for Instagram Knowledge Base

This module provides REST API endpoints for accessing the knowledge base,
searching content, and interacting with the RAG system.
"""

# Import necessary components
from .api import api_bp, setup_api_routes
from .api_knowledge import setup_knowledge_routes

__all__ = ['api_bp', 'setup_api_routes', 'setup_knowledge_routes'] 