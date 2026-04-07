-- Schema for Knowledge Graph & Concept Extraction

-- Concepts table
CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    first_seen_date TEXT,
    last_updated TEXT,
    reference_count INTEGER DEFAULT 1,
    UNIQUE(name)
);

-- Concept relationships table
CREATE TABLE IF NOT EXISTS concept_relationships (
    id INTEGER PRIMARY KEY,
    source_concept_id INTEGER NOT NULL,
    target_concept_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    first_seen_date TEXT,
    last_updated TEXT,
    reference_count INTEGER DEFAULT 1,
    confidence_score REAL DEFAULT 1.0,
    UNIQUE(source_concept_id, target_concept_id, relationship_type),
    FOREIGN KEY(source_concept_id) REFERENCES concepts(id),
    FOREIGN KEY(target_concept_id) REFERENCES concepts(id)
);

-- Link between content items and concepts
CREATE TABLE IF NOT EXISTS content_concepts (
    id INTEGER PRIMARY KEY,
    content_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    importance TEXT NOT NULL,
    date_extracted TEXT NOT NULL,
    UNIQUE(content_id, concept_id),
    FOREIGN KEY(content_id) REFERENCES ai_content(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

-- Tables for feedback and weight optimization
CREATE TABLE IF NOT EXISTS search_query_log (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    vector_weight REAL,
    keyword_weight REAL,
    search_type TEXT,
    source_type TEXT,
    result_count INTEGER,
    top_result_ids TEXT, -- JSON array of content IDs
    timestamp TEXT NOT NULL,
    query_features TEXT -- JSON of query features
);

CREATE TABLE IF NOT EXISTS search_feedback (
    id INTEGER PRIMARY KEY,
    query_log_id INTEGER,
    content_id INTEGER,
    feedback_score INTEGER, -- 1-5 rating
    feedback_type TEXT,
    feedback_text TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(query_log_id) REFERENCES search_query_log(id)
);

CREATE TABLE IF NOT EXISTS weight_patterns (
    id INTEGER PRIMARY KEY,
    pattern_name TEXT NOT NULL,
    query_pattern TEXT, -- JSON of query features
    vector_weight REAL,
    keyword_weight REAL,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    last_updated TEXT,
    confidence_score REAL DEFAULT 0.5,
    UNIQUE(pattern_name)
);

-- Create indices for efficient queries
CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category);
CREATE INDEX IF NOT EXISTS idx_concept_relationships_source ON concept_relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_relationships_target ON concept_relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_content_concepts_content ON content_concepts(content_id);
CREATE INDEX IF NOT EXISTS idx_content_concepts_concept ON content_concepts(concept_id); 