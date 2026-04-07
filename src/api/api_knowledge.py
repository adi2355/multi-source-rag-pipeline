"""
Knowledge Graph and Content API endpoints
"""
from flask import request, jsonify
import logging
import sqlite3
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from api.api import api_bp

logger = logging.getLogger('api.knowledge')

def setup_knowledge_routes():
    """Setup knowledge graph and content routes for the API blueprint"""
    pass

@api_bp.route('/concepts/search', methods=['GET'])
def search_concepts():
    """API endpoint to search concepts"""
    query = request.args.get('q', '')
    category = request.args.get('category')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
        
    try:
        # Import knowledge graph module (using try/except to handle possible import errors)
        try:
            from knowledge_graph import KnowledgeGraph
            graph = KnowledgeGraph()
            results = graph.search_concepts(query, limit=limit, category=category)
            
            return jsonify({
                "query": query,
                "results": results
            })
        except ImportError:
            # If knowledge_graph module is not available, fallback to direct DB query
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Build query
            base_query = """
                SELECT id, name, category, importance, confidence
                FROM concepts
                WHERE name LIKE ?
            """
            
            params = [f"%{query}%"]
            
            if category:
                base_query += " AND category = ?"
                params.append(category)
                
            base_query += " ORDER BY importance DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'importance': row[3],
                    'confidence': row[4]
                })
            
            conn.close()
            return jsonify({
                "query": query,
                "results": results
            })
            
    except Exception as e:
        logger.error(f"Error in concept search API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/concepts/<int:concept_id>', methods=['GET'])
def get_concept(concept_id):
    """API endpoint to get concept details"""
    try:
        try:
            from knowledge_graph import KnowledgeGraph
            graph = KnowledgeGraph()
            concept = graph.get_concept(concept_id=concept_id)
            
            if not concept:
                return jsonify({"error": "Concept not found"}), 404
                
            return jsonify(concept)
        except ImportError:
            # Fallback to direct DB query
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get concept details
            cursor.execute("""
                SELECT id, name, category, description, importance, confidence
                FROM concepts
                WHERE id = ?
            """, (concept_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return jsonify({"error": "Concept not found"}), 404
                
            concept = {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3],
                'importance': row[4],
                'confidence': row[5]
            }
            
            # Get related content
            cursor.execute("""
                SELECT c.id, c.title, cc.importance
                FROM ai_content c
                JOIN content_concepts cc ON c.id = cc.content_id
                WHERE cc.concept_id = ?
                ORDER BY cc.importance DESC
                LIMIT 10
            """, (concept_id,))
            
            related_content = []
            for content_row in cursor.fetchall():
                related_content.append({
                    'id': content_row[0],
                    'title': content_row[1],
                    'importance': content_row[2]
                })
            
            # Get related concepts
            cursor.execute("""
                SELECT c2.id, c2.name, c2.category, COUNT(*) as frequency
                FROM concepts c1
                JOIN content_concepts cc1 ON c1.id = cc1.concept_id
                JOIN content_concepts cc2 ON cc1.content_id = cc2.content_id
                JOIN concepts c2 ON cc2.concept_id = c2.id
                WHERE c1.id = ? AND c2.id != ?
                GROUP BY c2.id
                ORDER BY frequency DESC
                LIMIT 10
            """, (concept_id, concept_id))
            
            related_concepts = []
            for rel_row in cursor.fetchall():
                related_concepts.append({
                    'id': rel_row[0],
                    'name': rel_row[1],
                    'category': rel_row[2],
                    'frequency': rel_row[3]
                })
            
            concept['related_content'] = related_content
            concept['related_concepts'] = related_concepts
            
            conn.close()
            return jsonify(concept)
            
    except Exception as e:
        logger.error(f"Error in get concept API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/content/<int:content_id>', methods=['GET'])
def get_content(content_id):
    """API endpoint to get content details"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get content details
        cursor.execute("""
            SELECT c.id, c.title, c.description, c.content, c.url, 
                   c.date_created, c.date_collected, c.metadata,
                   s.name as source_type
            FROM ai_content c
            JOIN source_types s ON c.source_type_id = s.id
            WHERE c.id = ?
        """, (content_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "Content not found"}), 404
            
        # Get concepts for this content
        cursor.execute("""
            SELECT c.id, c.name, c.category, cc.importance
            FROM concepts c
            JOIN content_concepts cc ON c.id = cc.concept_id
            WHERE cc.content_id = ?
            ORDER BY cc.importance DESC
        """, (content_id,))
        
        concepts = []
        for concept_row in cursor.fetchall():
            concepts.append({
                'id': concept_row[0],
                'name': concept_row[1],
                'category': concept_row[2],
                'importance': concept_row[3]
            })
        
        # Get related content based on common concepts
        cursor.execute("""
            SELECT c2.id, c2.title, s.name as source_type, COUNT(*) as common_concepts
            FROM ai_content c1
            JOIN content_concepts cc1 ON c1.id = cc1.content_id
            JOIN content_concepts cc2 ON cc1.concept_id = cc2.concept_id
            JOIN ai_content c2 ON cc2.content_id = c2.id
            JOIN source_types s ON c2.source_type_id = s.id
            WHERE c1.id = ? AND c2.id != ?
            GROUP BY c2.id
            ORDER BY common_concepts DESC
            LIMIT 5
        """, (content_id, content_id))
        
        related_content = []
        for rel_row in cursor.fetchall():
            related_content.append({
                'id': rel_row[0],
                'title': rel_row[1],
                'source_type': rel_row[2],
                'common_concepts': rel_row[3]
            })
        
        content = {
            'id': row[0],
            'title': row[1],
            'description': row[2],
            'content': row[3],
            'url': row[4],
            'date_created': row[5],
            'date_collected': row[6],
            'metadata': row[7],
            'source_type': row[8],
            'concepts': concepts,
            'related_content': related_content
        }
        
        conn.close()
        return jsonify(content)
        
    except Exception as e:
        logger.error(f"Error in get content API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/kg/stats', methods=['GET'])
def knowledge_graph_stats():
    """API endpoint to get knowledge graph statistics"""
    try:
        try:
            from knowledge_graph import KnowledgeGraph
            graph = KnowledgeGraph()
            stats = graph.get_statistics()
            return jsonify(stats)
        except ImportError:
            # Fallback to direct DB query
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get concept count
            cursor.execute("SELECT COUNT(*) FROM concepts")
            concepts_count = cursor.fetchone()[0]
            
            # Get content count
            cursor.execute("SELECT COUNT(*) FROM ai_content")
            content_count = cursor.fetchone()[0]
            
            # Get relationship count
            cursor.execute("SELECT COUNT(*) FROM content_concepts")
            relationships_count = cursor.fetchone()[0]
            
            # Get concept categories
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM concepts
                GROUP BY category
                ORDER BY count DESC
            """)
            
            categories = {}
            for row in cursor.fetchall():
                categories[row[0]] = row[1]
            
            # Get top concepts
            cursor.execute("""
                SELECT c.id, c.name, c.category, COUNT(*) as reference_count
                FROM concepts c
                JOIN content_concepts cc ON c.id = cc.concept_id
                GROUP BY c.id
                ORDER BY reference_count DESC
                LIMIT 10
            """)
            
            top_concepts = []
            for row in cursor.fetchall():
                top_concepts.append({
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'reference_count': row[3]
                })
            
            conn.close()
            
            return jsonify({
                'concepts_count': concepts_count,
                'content_count': content_count,
                'relationships_count': relationships_count,
                'categories': categories,
                'top_concepts': top_concepts
            })
            
    except Exception as e:
        logger.error(f"Error in knowledge graph stats API: {str(e)}")
        return jsonify({'error': str(e)}), 500 