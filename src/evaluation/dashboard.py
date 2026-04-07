import os
import sqlite3
import logging
import json
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('evaluation_dashboard')

# Create Flask blueprint
evaluation_bp = Blueprint('evaluation', __name__, 
                         url_prefix='/evaluation',
                         template_folder='templates',
                         static_folder='static')

@evaluation_bp.route('/')
def evaluation_dashboard():
    """Main evaluation dashboard page"""
    return render_template('evaluation/dashboard.html')

@evaluation_bp.route('/retrieval')
def retrieval_dashboard():
    """Retrieval metrics dashboard page"""
    return render_template('evaluation/retrieval.html')

@evaluation_bp.route('/answer-quality')
def answer_quality_dashboard():
    """Answer quality dashboard page"""
    return render_template('evaluation/answer_quality.html')

@evaluation_bp.route('/datasets')
def datasets_dashboard():
    """Test datasets dashboard page"""
    return render_template('evaluation/datasets.html')

#
# API Endpoints for Dashboard Data
#

@evaluation_bp.route('/api/test-results')
def test_results():
    """Get test results for dashboard"""
    test_type = request.args.get('type', 'all')
    limit = int(request.args.get('limit', 10))
    
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Build query
        query = """
            SELECT id, test_type, test_name, date_created
            FROM test_results
        """
        
        params = []
        if test_type != 'all':
            query += " WHERE test_type = ?"
            params.append(test_type)
            
        query += " ORDER BY date_created DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            # Try to extract summary metrics
            cursor.execute(
                "SELECT results_json FROM test_results WHERE id = ?",
                (row[0],)
            )
            
            metrics_row = cursor.fetchone()
            metrics = None
            
            if metrics_row:
                try:
                    results_json = json.loads(metrics_row[0])
                    if 'metrics' in results_json:
                        metrics = results_json['metrics']
                except:
                    pass
            
            results.append({
                'id': row[0],
                'test_type': row[1],
                'test_name': row[2],
                'date_created': row[3],
                'metrics': metrics
            })
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting test results: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/test-result/<int:result_id>')
def test_result_detail(result_id):
    """Get detailed test result"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT test_type, test_name, date_created, results_json FROM test_results WHERE id = ?",
            (result_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'error': 'Test result not found'}), 404
            
        test_type, test_name, date_created, results_json = row
        
        results = json.loads(results_json)
        
        return jsonify({
            'id': result_id,
            'test_type': test_type,
            'test_name': test_name,
            'date_created': date_created,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error getting test result detail: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/metrics/summary')
def metrics_summary():
    """Get summary metrics for dashboard"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get latest retrieval test metrics
        cursor.execute("""
            SELECT results_json FROM test_results 
            WHERE test_type = 'retrieval'
            ORDER BY date_created DESC
            LIMIT 1
        """)
        
        retrieval_row = cursor.fetchone()
        retrieval_metrics = None
        
        if retrieval_row:
            try:
                results = json.loads(retrieval_row[0])
                retrieval_metrics = {}
                
                for search_type, metrics in results.get('metrics', {}).items():
                    if 'aggregate' in metrics:
                        retrieval_metrics[search_type] = metrics['aggregate']
            except:
                pass
        
        # Get latest answer quality metrics
        cursor.execute("""
            SELECT results_json FROM test_results 
            WHERE test_type = 'answer_quality'
            ORDER BY date_created DESC
            LIMIT 1
        """)
        
        answer_row = cursor.fetchone()
        answer_metrics = None
        
        if answer_row:
            try:
                results = json.loads(answer_row[0])
                answer_metrics = results.get('metrics', {})
            except:
                pass
        
        # Get historical metrics for trends
        cursor.execute("""
            SELECT test_type, date_created, results_json 
            FROM test_results
            WHERE test_type IN ('retrieval', 'answer_quality')
            ORDER BY date_created ASC
            LIMIT 20
        """)
        
        trends = {'retrieval': [], 'answer_quality': []}
        
        for row in cursor.fetchall():
            test_type, date_created, results_json = row
            
            try:
                results = json.loads(results_json)
                
                if test_type == 'retrieval':
                    for search_type, metrics in results.get('metrics', {}).items():
                        if 'aggregate' in metrics and 'avg_f1' in metrics['aggregate']:
                            trends['retrieval'].append({
                                'date': date_created,
                                'search_type': search_type,
                                'f1': metrics['aggregate']['avg_f1']
                            })
                elif test_type == 'answer_quality':
                    if 'metrics' in results and 'combined_score' in results['metrics']:
                        trends['answer_quality'].append({
                            'date': date_created,
                            'combined_score': results['metrics']['combined_score']
                        })
            except:
                pass
        
        return jsonify({
            'retrieval': retrieval_metrics,
            'answer_quality': answer_metrics,
            'trends': trends
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/datasets')
def test_datasets():
    """Get test datasets"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, name, date_created, 
                  (SELECT COUNT(*) FROM json_each(queries_json)) as query_count
            FROM test_datasets
            ORDER BY date_created DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'date_created': row[2],
                'query_count': row[3]
            })
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting test datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/dataset/<int:dataset_id>')
def dataset_detail(dataset_id):
    """Get detailed dataset information"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT name, date_created, queries_json FROM test_datasets WHERE id = ?",
            (dataset_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'error': 'Dataset not found'}), 404
            
        name, date_created, queries_json = row
        
        queries = json.loads(queries_json)
        
        return jsonify({
            'id': dataset_id,
            'name': name,
            'date_created': date_created,
            'queries': queries
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset detail: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/answer-evaluations')
def answer_evaluations():
    """Get answer evaluations"""
    limit = int(request.args.get('limit', 20))
    
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, query_id, query_text, faithfulness_score, relevance_score, date_evaluated
            FROM answer_evaluations
            ORDER BY date_evaluated DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'query_id': row[1],
                'query_text': row[2],
                'faithfulness_score': row[3],
                'relevance_score': row[4],
                'date_evaluated': row[5]
            })
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting answer evaluations: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

@evaluation_bp.route('/api/answer-evaluation/<int:eval_id>')
def answer_evaluation_detail(eval_id):
    """Get detailed answer evaluation"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT query_id, query_text, answer_text, context_text,
                  faithfulness_score, faithfulness_explanation,
                  relevance_score, relevance_explanation,
                  date_evaluated
            FROM answer_evaluations
            WHERE id = ?
        """, (eval_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'error': 'Answer evaluation not found'}), 404
            
        return jsonify({
            'id': eval_id,
            'query_id': row[0],
            'query_text': row[1],
            'answer_text': row[2],
            'context_text': row[3],
            'faithfulness_score': row[4],
            'faithfulness_explanation': row[5],
            'relevance_score': row[6],
            'relevance_explanation': row[7],
            'date_evaluated': row[8]
        })
        
    except Exception as e:
        logger.error(f"Error getting answer evaluation detail: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        conn.close()

# Add this blueprint to app.py
# from evaluation.dashboard import evaluation_bp
# app.register_blueprint(evaluation_bp) 