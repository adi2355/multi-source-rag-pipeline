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