"""
Core API endpoints for the Instagram Knowledge Base
"""
from flask import Blueprint, request, jsonify, Response, stream_with_context
import logging
from datetime import datetime
import sys
import os
import json
import sqlite3

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api')

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

def setup_api_routes():
    """Setup additional routes for the API blueprint"""
    pass

@api_bp.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api_bp.route('/search', methods=['POST'])
def search():
    """API endpoint for search"""
    # Parse request
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # Extract parameters
    query = data['query']
    search_type = data.get('search_type', 'hybrid')
    top_k = data.get('top_k', 10)
    vector_weight = data.get('vector_weight')
    keyword_weight = data.get('keyword_weight')
    source_type = data.get('source_type')
    page = data.get('page', 1)
    
    logger.info(f"API search request: {query}")
    
    try:
        # Import locally to avoid circular imports
        from run import run_hybrid_search, run_vector_search
        
        # Perform search
        if search_type == 'hybrid':
            search_results = run_hybrid_search(
                query=query,
                top_k=top_k,
                source_type=source_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )
        elif search_type == 'vector':
            search_results = run_vector_search(
                query=query,
                top_k=top_k,
                source_type=source_type
            )
        else:  # keyword search
            # Implementation requires direct database access
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Construct query
            base_query = """
                SELECT c.id, c.title, c.description, c.url, s.name AS source_type
                FROM ai_content c
                JOIN source_types s ON c.source_type_id = s.id
                WHERE (c.title LIKE ? OR c.description LIKE ? OR c.content LIKE ?)
            """
            
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if source_type:
                base_query += " AND s.name = ?"
                params.append(source_type)
                
            base_query += " LIMIT ?"
            params.append(top_k)
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            
            search_results = []
            for row in rows:
                search_results.append({
                    'content_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'url': row[3],
                    'source_type': row[4],
                    'similarity': 0.5,  # Placeholder for keyword search
                    'snippet': row[2] if row[2] else "No description available"
                })
            
            conn.close()
        
        # Log search query
        log_search_query(query, search_type, len(search_results))
        
        # Calculate total pages (for pagination)
        total_pages = 1  # Simple implementation
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'content_id': result.get('content_id'),
                'title': result.get('title', ''),
                'source_type': result.get('source_type', ''),
                'similarity': result.get('similarity', 0),
                'snippet': result.get('snippet', '')[:300] + '...' if result.get('snippet') and len(result.get('snippet', '')) > 300 else result.get('snippet', ''),
                'url': result.get('url', '')
            })
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'top_k': top_k,
            'vector_weight': vector_weight,
            'keyword_weight': keyword_weight,
            'source_type': source_type,
            'results': formatted_results,
            'page': page,
            'total_pages': total_pages,
            'search_time': 0.5,  # Placeholder
            'query_log_id': get_last_query_id()
        })
        
    except Exception as e:
        logger.error(f"Error in search API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/answer', methods=['POST'])
def answer():
    """API endpoint for generating answers"""
    # Parse request
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # Extract parameters
    query = data['query']
    search_type = data.get('search_type', 'hybrid')
    top_k = data.get('top_k', 5)
    vector_weight = data.get('vector_weight')
    keyword_weight = data.get('keyword_weight')
    source_type = data.get('source_type')
    stream = data.get('stream', False)
    model = data.get('model')
    
    logger.info(f"API answer request: {query}")
    
    try:
        # Import locally to avoid circular imports
        from run import run_rag_query
        
        if stream:
            # Not implemented in this basic version
            return jsonify({'error': 'Streaming not implemented in API yet'}), 501
        
        # Generate answer
        response = run_rag_query(
            query=query,
            search_type=search_type,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            source_type=source_type,
            model=model
        )
        
        return jsonify({
            'query': query,
            'answer': response.get('answer', ''),
            'sources': response.get('sources', []),
            'metadata': {
                'search_type': search_type,
                'top_k': top_k,
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'source_type': source_type,
                'timestamp': datetime.now().isoformat()
            },
            'query_log_id': get_last_query_id()
        })
        
    except Exception as e:
        logger.error(f"Error in answer API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/answer/stream', methods=['GET'])
def answer_stream():
    """API endpoint for streaming answer generation"""
    # Parse request parameters
    query = request.args.get('query')
    search_type = request.args.get('search_type', 'hybrid')
    top_k = int(request.args.get('top_k', 5))
    vector_weight = float(request.args.get('vector_weight', 0.7))
    keyword_weight = float(request.args.get('keyword_weight', 0.3))
    source_type = request.args.get('source_type')
    model = request.args.get('model')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    try:
        # Import locally to avoid circular imports
        from run import run_rag_query
        
        def generate():
            # First yield an event to start the stream
            yield f"data: {json.dumps({'status': 'started'})}\n\n"
            
            # Generate the answer with streaming
            for chunk in run_rag_query(
                query=query,
                search_type=search_type,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                source_type=source_type,
                model=model,
                stream=True
            ):
                if isinstance(chunk, dict) and 'sources' in chunk:
                    # Sources come at the end
                    yield f"data: {json.dumps({'sources': chunk['sources']})}\n\n"
                elif isinstance(chunk, str):
                    # Text chunks
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # End of stream
            yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate()), content_type='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in streaming answer API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/feedback', methods=['POST'])
def save_feedback():
    """API endpoint for saving user feedback"""
    # Parse request
    data = request.json
    
    if not data or 'content_id' not in data or 'feedback_score' not in data:
        return jsonify({'error': 'content_id and feedback_score parameters are required'}), 400
    
    try:
        # Extract parameters
        content_id = data['content_id']
        query_log_id = data.get('query_log_id')
        feedback_score = data['feedback_score']
        feedback_type = data.get('feedback_type', 'relevance')
        feedback_text = data.get('feedback_text', '')
        
        # Save to database
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_log_id INTEGER,
                content_id INTEGER,
                feedback_score INTEGER,
                feedback_type TEXT,
                feedback_text TEXT,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES ai_content(id)
            )
        """)
        
        # Insert feedback
        cursor.execute("""
            INSERT INTO user_feedback 
            (query_log_id, content_id, feedback_score, feedback_type, feedback_text)
            VALUES (?, ?, ?, ?, ?)
        """, (query_log_id, content_id, feedback_score, feedback_type, feedback_text))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Feedback saved successfully'})
        
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

def log_search_query(query, search_type, results_count):
    """Log search query to database"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                search_type TEXT,
                results_count INTEGER,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert query log
        cursor.execute("""
            INSERT INTO query_logs (query, search_type, results_count)
            VALUES (?, ?, ?)
        """, (query, search_type, results_count))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")

def get_last_query_id():
    """Get the ID of the last logged query"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT last_insert_rowid()")
        query_id = cursor.fetchone()[0]
        
        conn.close()
        return query_id
    except Exception as e:
        logger.error(f"Error getting last query ID: {str(e)}")
        return None 