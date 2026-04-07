"""
Concept extractor module for identifying and extracting AI concepts from content

This module uses anthropic.claude to analyze content from various sources
(research papers, GitHub repositories, Instagram videos) and extract
AI/ML concepts, creating a structured knowledge graph of concepts.
"""
import os
import json
import time
import logging
import sqlite3
from datetime import datetime
import anthropic
import config

# Configure logging
log_dir = os.path.join(config.DATA_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'concept_extractor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('concept_extractor')

# Set up the Anthropic Claude client
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

def extract_concepts_from_text(text, content_type, title="", context=""):
    """
    Extract AI concepts from text content using Claude
    
    Args:
        text (str): The text content to analyze
        content_type (str): The type of content (research_paper, github, instagram)
        title (str): Title of the content
        context (str): Additional context or metadata
        
    Returns:
        dict: Dictionary of extracted concepts
    """
    if not text or len(text.strip()) < 50:
        logger.warning(f"Text too short for concept extraction: {text[:50]}...")
        return {"concepts": [], "relationships": []}
    
    # Truncate text if it's too long
    max_length = 20000  # Claude can handle larger texts, but we'll keep it reasonable
    if len(text) > max_length:
        logger.info(f"Truncating text from {len(text)} to {max_length} characters")
        text = text[:max_length] + "..."
    
    # Prepare prompt based on content type
    type_context = {
        "research_paper": "This is content from a research paper on AI/ML.",
        "github": "This is content from a GitHub repository related to AI/ML.",
        "instagram": "This is a transcript or summary from an Instagram video about AI/ML."
    }
    
    content_context = type_context.get(content_type, "This is AI/ML related content.")
    
    prompt = f"""
    {content_context}
    Title: {title}
    Context: {context}
    
    I need you to analyze the following content and extract key AI/ML concepts. For each concept:
    1. Provide a short description
    2. Identify related concepts
    3. Categorize it (e.g., model architecture, training technique, dataset, evaluation metric)
    4. Assess its importance in the content (high/medium/low)
    
    Your response should be a well-structured JSON with the following format:
    {{
        "concepts": [
            {{
                "name": "concept name",
                "description": "brief description",
                "category": "category",
                "importance": "high/medium/low",
                "related_concepts": ["related concept 1", "related concept 2"]
            }}
        ],
        "relationships": [
            {{
                "source": "concept1",
                "target": "concept2",
                "relationship_type": "uses/is_part_of/improves/etc"
            }}
        ]
    }}
    
    Focus on technical AI/ML concepts only. Identify between 5-15 concepts depending on content length and density.
    
    Content to analyze:
    {text}
    """
    
    try:
        # Call Claude API
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0,
            system="You are an AI expert who specializes in extracting and organizing AI/ML concepts from technical content.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        response_text = message.content[0].text
        
        # Try to parse the JSON
        try:
            # Find JSON in the response (might be wrapped in markdown code blocks)
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            import re
            json_match = re.search(json_pattern, response_text)
            
            if json_match:
                json_str = json_match.group(1)
                concepts_data = json.loads(json_str)
            else:
                # Try parsing the whole response as JSON
                concepts_data = json.loads(response_text)
                
            # Validate the structure
            if not isinstance(concepts_data, dict) or "concepts" not in concepts_data:
                raise ValueError("Invalid JSON structure")
                
            return concepts_data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from Claude: {str(e)}")
            logger.debug(f"Claude response: {response_text}")
            return {"concepts": [], "relationships": []}
    
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return {"concepts": [], "relationships": []}

def store_concepts(content_id, source_type_id, concepts_data):
    """
    Store extracted concepts in the database
    
    Args:
        content_id (int): ID of the content in ai_content table
        source_type_id (int): Source type ID
        concepts_data (dict): Dictionary containing concepts and relationships
        
    Returns:
        bool: Success or failure
    """
    # Connect to database
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if concepts table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
        has_concepts_table = cursor.fetchone() is not None
        
        # Create tables if they don't exist
        if not has_concepts_table:
            # Create concepts table
            cursor.execute("""
            CREATE TABLE concepts (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                UNIQUE(name)
            )
            """)
            
            # Create content_concepts table for many-to-many relationships
            cursor.execute("""
            CREATE TABLE content_concepts (
                id INTEGER PRIMARY KEY,
                content_id INTEGER,
                concept_id INTEGER,
                importance TEXT,
                metadata TEXT,
                UNIQUE(content_id, concept_id),
                FOREIGN KEY(content_id) REFERENCES ai_content(id),
                FOREIGN KEY(concept_id) REFERENCES concepts(id)
            )
            """)
            
            # Create concept relationships table
            cursor.execute("""
            CREATE TABLE concept_relationships (
                id INTEGER PRIMARY KEY,
                source_concept_id INTEGER,
                target_concept_id INTEGER,
                relationship_type TEXT,
                UNIQUE(source_concept_id, target_concept_id, relationship_type),
                FOREIGN KEY(source_concept_id) REFERENCES concepts(id),
                FOREIGN KEY(target_concept_id) REFERENCES concepts(id)
            )
            """)
            
            logger.info("Created concepts tables")
        
        # Store each concept
        concept_ids = {}
        for concept in concepts_data.get("concepts", []):
            name = concept.get("name", "").strip()
            if not name:
                continue
                
            # Check if concept already exists
            cursor.execute("SELECT id FROM concepts WHERE name = ?", (name,))
            result = cursor.fetchone()
            
            if result:
                concept_id = result[0]
            else:
                # Insert new concept
                cursor.execute("""
                INSERT INTO concepts (name, description, category) 
                VALUES (?, ?, ?)
                """, (
                    name,
                    concept.get("description", ""),
                    concept.get("category", "")
                ))
                concept_id = cursor.lastrowid
            
            concept_ids[name] = concept_id
            
            # Link concept to content
            try:
                cursor.execute("""
                INSERT INTO content_concepts (content_id, concept_id, importance, metadata)
                VALUES (?, ?, ?, ?)
                """, (
                    content_id,
                    concept_id,
                    concept.get("importance", "medium"),
                    json.dumps({
                        "related_concepts": concept.get("related_concepts", [])
                    })
                ))
            except sqlite3.IntegrityError:
                # Update existing link
                cursor.execute("""
                UPDATE content_concepts 
                SET importance = ?, metadata = ?
                WHERE content_id = ? AND concept_id = ?
                """, (
                    concept.get("importance", "medium"),
                    json.dumps({
                        "related_concepts": concept.get("related_concepts", [])
                    }),
                    content_id,
                    concept_id
                ))
        
        # Store relationships
        for relationship in concepts_data.get("relationships", []):
            source = relationship.get("source", "").strip()
            target = relationship.get("target", "").strip()
            rel_type = relationship.get("relationship_type", "related_to").strip()
            
            if not source or not target or source not in concept_ids or target not in concept_ids:
                continue
                
            source_id = concept_ids[source]
            target_id = concept_ids[target]
            
            try:
                cursor.execute("""
                INSERT INTO concept_relationships (source_concept_id, target_concept_id, relationship_type)
                VALUES (?, ?, ?)
                """, (source_id, target_id, rel_type))
            except sqlite3.IntegrityError:
                # Relationship already exists
                pass
        
        # Update ai_content to mark as processed for concepts
        cursor.execute("""
        UPDATE ai_content 
        SET metadata = json.set(metadata, '$.concepts_extracted', 1),
            date_indexed = ?
        WHERE id = ?
        """, (datetime.now().isoformat(), content_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error storing concepts: {str(e)}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def process_unprocessed_content(limit=5, source_type=None):
    """
    Process content that hasn't had concepts extracted yet
    
    Args:
        limit (int): Maximum number of items to process
        source_type (str, optional): If provided, only process this type of content
        
    Returns:
        int: Number of items processed
    """
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get source type IDs
        source_type_map = {}
        cursor.execute("SELECT id, name FROM source_types")
        for row in cursor.fetchall():
            source_type_map[row[1]] = row[0]
        
        # Build query
        query = """
        SELECT c.id, c.source_type_id, c.title, c.content, c.description, c.metadata
        FROM ai_content c
        WHERE 
            (json_extract(c.metadata, '$.concepts_extracted') IS NULL OR 
             json_extract(c.metadata, '$.concepts_extracted') = 0)
            AND c.content IS NOT NULL AND length(c.content) > 100
        """
        
        params = []
        if source_type and source_type in source_type_map:
            query += " AND c.source_type_id = ?"
            params.append(source_type_map[source_type])
        
        query += " ORDER BY c.date_collected DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        items = cursor.fetchall()
        
        processed_count = 0
        for item in items:
            content_id, source_type_id, title, content, description, metadata_json = item
            
            # Determine source type name
            source_type_name = next((name for name, id_val in source_type_map.items() if id_val == source_type_id), "unknown")
            
            # Parse metadata
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except:
                metadata = {}
            
            # Different handling based on content type
            context = ""
            if source_type_name == "github":
                context = f"GitHub repository: {metadata.get('full_name', '')}"
                # Use description for context if content is README
                if description:
                    context += f"\nDescription: {description}"
                
            elif source_type_name == "research_paper":
                authors = metadata.get("authors", "")
                year = metadata.get("year", "")
                context = f"Authors: {authors}\nYear: {year}"
                
                # Use abstract for context if we're processing full text
                if description and len(content) > len(description)*3:
                    context += f"\nAbstract: {description[:500]}..."
            
            # Extract concepts
            logger.info(f"Extracting concepts from {source_type_name} content: {title}")
            concepts_data = extract_concepts_from_text(
                content,
                source_type_name,
                title=title,
                context=context
            )
            
            # Store concepts
            if concepts_data and concepts_data.get("concepts"):
                logger.info(f"Found {len(concepts_data['concepts'])} concepts in {title}")
                if store_concepts(content_id, source_type_id, concepts_data):
                    processed_count += 1
            else:
                logger.warning(f"No concepts found in {title}")
                # Mark as processed even if no concepts found
                cursor.execute("""
                UPDATE ai_content 
                SET metadata = json.set(metadata, '$.concepts_extracted', 1),
                    date_indexed = ?
                WHERE id = ?
                """, (datetime.now().isoformat(), content_id))
                conn.commit()
                processed_count += 1
                
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        return processed_count
    
    except Exception as e:
        logger.error(f"Error processing content for concept extraction: {str(e)}")
        return 0
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Process some content from each source type
    for source in ["research_paper", "github", "instagram"]:
        count = process_unprocessed_content(limit=3, source_type=source)
        logger.info(f"Processed {count} items from {source}") 