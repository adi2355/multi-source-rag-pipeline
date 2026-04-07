"""
Module for indexing transcribed content into the knowledge base
"""
import os
import json
import glob
import sqlite3
import logging
from tqdm import tqdm

from config import (
    DB_PATH,
    TRANSCRIPT_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('indexer')

def setup_database():
    """Ensure the database is set up correctly"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use the updated schema with summary field and FTS4
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
    
    -- Virtual FTS4 table for full-text search
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
    
    conn.commit()
    conn.close()
    
    logger.info("Database setup complete")

def extract_tags_from_caption(caption):
    """Extract hashtags from captions"""
    if not caption:
        return []
    
    words = caption.split()
    tags = [word[1:] for word in words if word.startswith('#')]
    return tags

def calculate_word_count(text):
    """Calculate word count from text"""
    if not text:
        return 0
    return len(text.split())

def index_transcripts():
    """Index all transcripts into the database"""
    setup_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all transcript files
    all_transcripts = []
    for root, _, _ in os.walk(TRANSCRIPT_DIR):
        transcripts = glob.glob(os.path.join(root, "*.json"))
        all_transcripts.extend(transcripts)
    
    logger.info(f"Found {len(all_transcripts)} transcripts to index")
    
    # Process each transcript
    new_count = 0
    updated_count = 0
    
    for transcript_path in tqdm(all_transcripts, desc="Indexing transcripts"):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract relevant fields, using empty strings for missing data
            shortcode = data.get("shortcode", "")
            account = data.get("account", "")
            filename = data.get("filename", "")
            caption = data.get("caption", "")
            transcript_text = data.get("text", "")
            timestamp = data.get("timestamp", "")
            download_date = data.get("download_date", "")
            url = data.get("url", "")
            likes = data.get("likes", 0)
            comments = data.get("comments", 0)
            
            # Calculate word count
            word_count = calculate_word_count(transcript_text)
            
            # Duration (if available)
            duration_seconds = data.get("duration_seconds", None)
            
            # Initially no summary or key phrases
            summary = ""
            key_phrases = ""
            
            # Check if this video is already in the database
            cursor.execute("SELECT id, summary, word_count FROM videos WHERE shortcode = ?", (shortcode,))
            result = cursor.fetchone()
            
            if result:
                # Update existing record (preserve summary if it exists)
                video_id = result[0]
                existing_summary = result[1] or ""
                existing_word_count = result[2] or 0
                
                cursor.execute('''
                UPDATE videos SET
                    account = ?,
                    filename = ?,
                    caption = ?,
                    transcript = ?,
                    timestamp = ?,
                    download_date = ?,
                    url = ?,
                    likes = ?,
                    comments = ?,
                    word_count = ?,
                    duration_seconds = ?,
                    key_phrases = ?
                WHERE id = ?
                ''', (account, filename, caption, transcript_text, timestamp, 
                      download_date, url, likes, comments, word_count,
                      duration_seconds, key_phrases, video_id))
                updated_count += 1
                
                # Remove old tags
                cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
                
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO videos (
                    shortcode, account, filename, caption, transcript, summary,
                    timestamp, download_date, url, likes, comments, 
                    word_count, duration_seconds, key_phrases
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (shortcode, account, filename, caption, transcript_text, summary,
                      timestamp, download_date, url, likes, comments, 
                      word_count, duration_seconds, key_phrases))
                video_id = cursor.lastrowid
                new_count += 1
            
            # Extract and save tags
            tags = extract_tags_from_caption(caption)
            for tag in tags:
                cursor.execute('''
                INSERT INTO tags (video_id, tag) VALUES (?, ?)
                ''', (video_id, tag))
            
            # Commit every 100 records
            if (new_count + updated_count) % 100 == 0:
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error indexing {transcript_path}: {str(e)}")
    
    # Final commit
    conn.commit()
    conn.close()
    
    logger.info(f"Indexing complete. Added {new_count} new records, updated {updated_count} existing records.")

if __name__ == "__main__":
    index_transcripts() 