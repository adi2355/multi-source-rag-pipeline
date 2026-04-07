"""
Database migration module for Instagram Knowledge Base
"""
import os
import logging
import sqlite3
from datetime import datetime
import json
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_migration')

def migrate_database():
    """
    Perform database migration to support multiple content sources and vector embeddings
    
    Returns:
        bool: Success or failure
    """
    logger.info("Starting database migration")
    start_time = datetime.now()
    
    try:
        # Connect to the database
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Check for existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videos'")
        has_videos_table = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='source_types'")
        has_source_types_table = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='github_repos'")
        has_github_repos_table = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_content'")
        has_ai_content_table = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_papers'")
        has_research_papers_table = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_embeddings'")
        has_content_embeddings_table = cursor.fetchone() is not None
        
        # Create source_types table if it doesn't exist
        if not has_source_types_table:
            cursor.execute("""
            CREATE TABLE source_types (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                UNIQUE(name)
            )
            """)
            
            # Insert initial source types
            cursor.execute("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                         ('instagram', 'Instagram videos and posts'))
            cursor.execute("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                         ('github', 'GitHub repositories'))
            cursor.execute("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                         ('research_paper', 'Scientific research papers'))
            
            logger.info("Created source_types table")
        
        # Create GitHub repositories table if it doesn't exist
        if not has_github_repos_table:
            cursor.execute("""
            CREATE TABLE github_repos (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                full_name TEXT NOT NULL,
                description TEXT,
                url TEXT,
                stars INTEGER,
                watchers INTEGER,
                forks INTEGER,
                language TEXT,
                last_push TEXT,
                created_at TEXT,
                updated_at TEXT,
                topics TEXT,
                readme TEXT,
                last_crawled TEXT,
                UNIQUE(full_name)
            )
            """)
            
            # Create index for searching
            cursor.execute("CREATE INDEX idx_github_repos_name ON github_repos(name)")
            cursor.execute("CREATE INDEX idx_github_repos_language ON github_repos(language)")
            
            logger.info("Created github_repos table")
        
        # Create research papers table if it doesn't exist
        if not has_research_papers_table:
            cursor.execute("""
            CREATE TABLE research_papers (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                publication TEXT,
                year INTEGER,
                url TEXT,
                doi TEXT,
                pdf_path TEXT,
                content TEXT,
                last_crawled TEXT,
                UNIQUE(doi)
            )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX idx_research_papers_title ON research_papers(title)")
            cursor.execute("CREATE INDEX idx_research_papers_year ON research_papers(year)")
            
            logger.info("Created research_papers table")
        
        # Check if ai_content table exists but needs migration
        if has_ai_content_table:
            # Check for the required columns
            cursor.execute("PRAGMA table_info(ai_content)")
            columns = {column[1] for column in cursor.fetchall()}
            
            # If the table structure is significantly different, we might need to recreate it
            # But let's check if we just need to add a date_indexed column
            if "date_indexed" not in columns:
                try:
                    logger.info("Adding date_indexed column to ai_content table")
                    cursor.execute("ALTER TABLE ai_content ADD COLUMN date_indexed TEXT")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add date_indexed column: {str(e)}")
            
            # Verify that required columns exist
            required_columns = {"source_id", "source_type_id"}
            missing_columns = required_columns - columns
            
            if missing_columns:
                logger.error(f"ai_content table is missing required columns: {missing_columns}, manual intervention needed")
                # We won't proceed with migration that could lose data
                return False
        else:
            # Create unified content table if it doesn't exist
            cursor.execute("""
            CREATE TABLE ai_content (
                id INTEGER PRIMARY KEY,
                source_type_id INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT,
                description TEXT,
                content TEXT,
                url TEXT,
                date_created TEXT,
                date_collected TEXT NOT NULL,
                date_indexed TEXT,
                metadata TEXT,
                is_indexed INTEGER DEFAULT 0,
                embedding_file TEXT,
                FOREIGN KEY(source_type_id) REFERENCES source_types(id),
                UNIQUE(source_type_id, source_id)
            )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX idx_ai_content_source ON ai_content(source_type_id, source_id)")
            cursor.execute("CREATE INDEX idx_ai_content_title ON ai_content(title)")
            cursor.execute("CREATE INDEX idx_ai_content_date ON ai_content(date_created)")
            
            logger.info("Created ai_content table")
            
            # Commit to ensure table is created before migration
            conn.commit()
            
            # Migrate existing Instagram videos if available
            if has_videos_table:
                try:
                    # Get source_type_id for Instagram
                    cursor.execute("SELECT id FROM source_types WHERE name = 'instagram'")
                    instagram_source_type_id = cursor.fetchone()
                    
                    if instagram_source_type_id is None:
                        # Insert instagram source type if not already present
                        cursor.execute("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                                     ('instagram', 'Instagram videos and posts'))
                        instagram_source_type_id = cursor.lastrowid
                    else:
                        instagram_source_type_id = instagram_source_type_id[0]
                    
                    # Get columns in videos table to handle different schema versions
                    cursor.execute("PRAGMA table_info(videos)")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # Build dynamic query based on available columns
                    select_columns = ["id", "shortcode"]
                    
                    # Add caption if it exists
                    if "caption" in columns:
                        select_columns.append("caption")
                    else:
                        select_columns.append("''")  # Empty string as placeholder
                    
                    # Add timestamp if it exists
                    if "timestamp" in columns:
                        select_columns.append("timestamp")
                    
                    # Add username if it exists, or try account
                    if "username" in columns:
                        select_columns.append("username")
                    elif "account" in columns:
                        select_columns.append("account")
                    
                    if "filepath" in columns:
                        select_columns.append("filepath")
                    
                    # Add where clause if downloaded column exists
                    where_clause = ""
                    if "downloaded" in columns:
                        where_clause = "WHERE downloaded = 1"
                    
                    # Create the query
                    query = f"SELECT {', '.join(select_columns)} FROM videos {where_clause}"
                    
                    # Execute the query
                    cursor.execute(query)
                    videos = cursor.fetchall()
                    
                    migrated_count = 0
                    for video in videos:
                        try:
                            # Extract data with flexible column positions
                            video_id = video[0]
                            shortcode = video[1]
                            caption = video[2] if len(video) > 2 else ""
                            
                            # Handle timestamp if available (index will depend on whether caption exists)
                            timestamp_idx = select_columns.index("timestamp") if "timestamp" in select_columns else -1
                            timestamp = video[timestamp_idx] if timestamp_idx >= 0 and timestamp_idx < len(video) else None
                            
                            # Handle username/account
                            username_idx = -1
                            if "username" in select_columns:
                                username_idx = select_columns.index("username")
                            elif "account" in select_columns:
                                username_idx = select_columns.index("account")
                                
                            username = video[username_idx] if username_idx >= 0 and username_idx < len(video) else "unknown"
                            
                            # Format date
                            date_created = datetime.now().isoformat()
                            if timestamp:
                                try:
                                    # Just use the timestamp as is, don't try to convert it
                                    date_created = timestamp
                                except:
                                    # Use current time if timestamp causes issues
                                    logger.warning(f"Could not use timestamp for video {video_id}, using current time")
                            
                            # Insert into ai_content table
                            cursor.execute("""
                            INSERT OR IGNORE INTO ai_content 
                            (source_type_id, source_id, title, description, url, date_created, date_collected, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                instagram_source_type_id,
                                str(video_id),
                                f"Instagram video from {username}",
                                caption,
                                f"https://www.instagram.com/p/{shortcode}/",
                                date_created,
                                datetime.now().isoformat(),
                                json.dumps({"shortcode": shortcode, "username": username})
                            ))
                            migrated_count += 1
                            
                            # Commit every 10 items to avoid holding locks too long
                            if migrated_count % 10 == 0:
                                conn.commit()
                                
                        except Exception as e:
                            logger.error(f"Error migrating video {video[0]}: {str(e)}")
                    
                    # Final commit
                    conn.commit()
                    logger.info(f"Migrated {migrated_count} Instagram videos to ai_content table")
                except Exception as e:
                    logger.error(f"Error migrating Instagram videos: {str(e)}")
        
        # Create content embeddings table if it doesn't exist
        if not has_content_embeddings_table:
            cursor.execute("""
            CREATE TABLE content_embeddings (
                id INTEGER PRIMARY KEY,
                content_id INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                chunk_text TEXT,
                date_created TEXT NOT NULL,
                FOREIGN KEY(content_id) REFERENCES ai_content(id),
                UNIQUE(content_id, chunk_index)
            )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX idx_content_embeddings_content ON content_embeddings(content_id)")
            
            logger.info("Created content_embeddings table")
            
        # Create FTS virtual table for full-text search if it doesn't exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_content_fts'")
        has_fts_table = cursor.fetchone() is not None
        
        if not has_fts_table and has_ai_content_table:
            try:
                # Get column names from ai_content
                cursor.execute("PRAGMA table_info(ai_content)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Check required columns for FTS
                fts_columns = ["title", "description", "content"]
                available_fts_columns = [col for col in fts_columns if col in columns]
                
                if len(available_fts_columns) > 0:
                    # Create column definitions for FTS
                    content_link = 'content="ai_content"'
                    indexed_columns = ", ".join(available_fts_columns)
                    notindexed_columns = "notindexed=id, notindexed=source_type_id"
                    
                    fts_query = f"""
                    CREATE VIRTUAL TABLE ai_content_fts USING fts4(
                        {content_link},
                        {indexed_columns},
                        {notindexed_columns}
                    )
                    """
                    
                    cursor.execute(fts_query)
                    
                    # Populate FTS table with existing content
                    select_columns = ", ".join(available_fts_columns)
                    cursor.execute(f"""
                    INSERT INTO ai_content_fts(docid, {select_columns})
                    SELECT id, {select_columns} FROM ai_content
                    WHERE content IS NOT NULL
                    """)
                    
                    logger.info("Created ai_content_fts virtual table for full-text search")
                else:
                    logger.warning("Could not create FTS table: required columns not found in ai_content")
            except Exception as e:
                logger.warning(f"Could not create FTS table: {str(e)}")
        
        # Commit changes
        conn.commit()
        
        # Done
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Database migration completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Database migration failed: {str(e)}")
        return False

if __name__ == "__main__":
    migrate_database() 