"""
Web interface for the AI Knowledge Base
"""
import os
import re
import sqlite3
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, g, send_from_directory, redirect, url_for

from config import (
    DB_PATH,
    WEB_PORT,
    DEBUG_MODE,
    DOWNLOAD_DIR,
    DATA_DIR
)

app = Flask(__name__)

# Import and register API blueprints
try:
    from api import api_bp, setup_api_routes, setup_knowledge_routes
    from api.swagger import swagger_ui_blueprint, swagger_json_bp
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(swagger_ui_blueprint)
    app.register_blueprint(swagger_json_bp)
    
    # Setup additional routes
    setup_api_routes()
    setup_knowledge_routes()
    
    has_api = True
except ImportError as e:
    print(f"API module not available: {e}")
    has_api = False

# Try to import evaluation dashboard
try:
    from evaluation.dashboard import evaluation_bp
    app.register_blueprint(evaluation_bp)
    has_evaluation = True
except ImportError as e:
    print(f"Evaluation module not available: {e}")
    has_evaluation = False

def get_db():
    """Get database connection with row factory for easy access"""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection when app context ends"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    """Home page with search interface"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Search interface page"""
    # Simple search interface that uses the API for actual searching
    return render_template('search.html')

@app.route('/chat')
def chat():
    """Chat interface with RAG assistant"""
    return render_template('chat.html')

@app.route('/concepts')
def concepts():
    """Concepts explorer page"""
    return render_template('concepts.html')

@app.route('/content/<int:content_id>')
def content_details(content_id):
    """Content details page"""
    return render_template('content_details.html', content_id=content_id)

# Legacy routes still supported for backward compatibility
@app.route('/video/<account>/<shortcode>')
def video(account, shortcode):
    """Legacy video detail page - redirects to content page if possible"""
    db = get_db()
    cursor = db.cursor()
    
    # Try to find the content ID for this video
    cursor.execute('''
    SELECT id FROM ai_content 
    WHERE metadata LIKE ?
    ''', (f'%"shortcode": "{shortcode}"%',))
    
    result = cursor.fetchone()
    if result:
        # Redirect to the new content page
        return redirect(url_for('content_details', content_id=result['id']))
    
    # Get video details from old table if still exists
    try:
        cursor.execute('''
        SELECT v.* FROM videos v
        WHERE v.account = ? AND v.shortcode = ?
        ''', (account, shortcode))
        video = cursor.fetchone()
        
        if not video:
            return "Video not found", 404
        
        # Get tags
        cursor.execute("SELECT tag FROM tags WHERE video_id = ?", (video['id'],))
        tags = [tag['tag'] for tag in cursor.fetchall()]
        
        return render_template(
            'legacy_video.html',
            video=video,
            tags=tags
        )
    except sqlite3.OperationalError:
        # Table doesn't exist anymore
        return "Video not found - database has been migrated", 404

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/stats')
def stats():
    """Legacy statistics page - redirects to admin dashboard"""
    return redirect(url_for('index'))

@app.route('/media/<path:path>')
def media(path):
    """Serve media files"""
    return send_from_directory(DOWNLOAD_DIR, path)

@lru_cache(maxsize=100)
def get_video_by_shortcode(shortcode):
    """Legacy cache for videos by shortcode"""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    
    try:
        cursor.execute('SELECT * FROM videos WHERE shortcode = ?', (shortcode,))
        video = cursor.fetchone()
        db.close()
        return dict(video) if video else None
    except sqlite3.OperationalError:
        db.close()
        return None

@lru_cache(maxsize=30)
def get_recent_videos(limit=10, account=None):
    """Legacy cache for recent videos"""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    
    try:
        if account:
            cursor.execute('SELECT * FROM videos WHERE account = ? ORDER BY timestamp DESC LIMIT ?', (account, limit))
        else:
            cursor.execute('SELECT * FROM videos ORDER BY timestamp DESC LIMIT ?', (limit,))
        
        videos = [dict(row) for row in cursor.fetchall()]
        db.close()
        return videos
    except sqlite3.OperationalError:
        db.close()
        return []

def clear_caches():
    """Clear all cached data"""
    get_video_by_shortcode.cache_clear()
    get_recent_videos.cache_clear()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=WEB_PORT, debug=DEBUG_MODE) 