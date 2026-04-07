"""
Initialize the SQLite database with proper schema
"""
import os
import sqlite3
import logging

from config import DB_PATH, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('init_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('init_db')

def init_database():
    """Initialize the SQLite database with proper schema"""
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info(f"Initializing database at {DB_PATH}")
    
    # Create or connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Read SQL schema from file
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_db.sql")
    
    try:
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema script
        cursor.executescript(schema_sql)
        logger.info("Successfully executed schema script")
        
    except FileNotFoundError:
        logger.warning(f"Schema file not found at {schema_path}. Using embedded schema.")
        
        # Execute embedded schema if file not found
        cursor.executescript('''
        -- Content table stores all video data
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
            shortcode TEXT UNIQUE,
            account TEXT,
            filename TEXT,
            caption TEXT,
            transcript TEXT,
            summary TEXT,
            timestamp TEXT,
            download_date TEXT,
            url TEXT,
            likes INTEGER,
            comments INTEGER,
            word_count INTEGER,
            duration_seconds INTEGER,
            key_phrases TEXT
        );

        -- Tags table for improved filtering
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY,
            video_id INTEGER,
            tag TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        );

        -- Improved indexing for better performance
        CREATE INDEX IF NOT EXISTS idx_videos_account ON videos(account);
        CREATE INDEX IF NOT EXISTS idx_videos_timestamp ON videos(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

        -- Virtual FTS4 table for full-text search (using FTS4 instead of FTS5 for better compatibility)
        CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts4(
            shortcode,
            account,
            caption,
            transcript,
            summary,
            timestamp,
            content=videos,
            tokenize=porter
        );

        -- Create triggers to keep FTS table synchronized
        CREATE TRIGGER IF NOT EXISTS videos_ai AFTER INSERT ON videos BEGIN
            INSERT INTO videos_fts(docid, shortcode, account, caption, transcript, summary, timestamp)
            VALUES (new.id, new.shortcode, new.account, new.caption, new.transcript, new.summary, new.timestamp);
        END;

        CREATE TRIGGER IF NOT EXISTS videos_ad AFTER DELETE ON videos BEGIN
            DELETE FROM videos_fts WHERE docid = old.id;
        END;

        CREATE TRIGGER IF NOT EXISTS videos_au AFTER UPDATE ON videos BEGIN
            DELETE FROM videos_fts WHERE docid = old.id;
            INSERT INTO videos_fts(docid, shortcode, account, caption, transcript, summary, timestamp)
            VALUES (new.id, new.shortcode, new.account, new.caption, new.transcript, new.summary, new.timestamp);
        END;
        ''')
        logger.info("Successfully executed embedded schema")
    
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {DB_PATH}")
    logger.info("You can now run the pipeline to populate the database with your video data.")

if __name__ == "__main__":
    init_database() 