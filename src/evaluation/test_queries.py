import os
import json
import random
import logging
import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_queries')

class TestQueryGenerator:
    """Generator for test queries with ground truth"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
    
    def generate_concept_queries(self, num_queries=15):
        """Generate test queries based on concepts in the knowledge graph"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get top concepts
            cursor.execute("""
                SELECT id, name, category 
                FROM concepts 
                ORDER BY reference_count DESC 
                LIMIT 50
            """)
            
            concepts = cursor.fetchall()
            
            # Shuffle concepts to get variety
            random.shuffle(concepts)
            concepts = concepts[:num_queries]
            
            test_queries = []
            
            for concept_id, name, category in concepts:
                # Create query templates based on concept category
                templates = []
                
                if category == 'algorithm':
                    templates = [
                        f"How does the {name} algorithm work?",
                        f"What is {name} used for?",
                        f"Explain the key steps in {name}"
                    ]
                elif category == 'model':
                    templates = [
                        f"What is the architecture of {name}?",
                        f"How is {name} trained?",
                        f"What are the key components of {name}?"
                    ]
                elif category == 'technique':
                    templates = [
                        f"What is {name} technique?",
                        f"How is {name} used in machine learning?",
                        f"What are the advantages of using {name}?"
                    ]
                elif category == 'framework':
                    templates = [
                        f"What are the key features of {name}?",
                        f"How does {name} compare to other frameworks?",
                        f"What are common applications of {name}?"
                    ]
                else:
                    templates = [
                        f"What is {name}?",
                        f"How is {name} used in AI?",
                        f"Explain {name} in simple terms"
                    ]
                
                # Choose a template
                query_text = random.choice(templates)
                
                # Get relevant content IDs for this concept (ground truth)
                cursor.execute("""
                    SELECT content_id, importance
                    FROM content_concepts
                    WHERE concept_id = ?
                """, (concept_id,))
                
                relevance_scores = {}
                
                for content_id, importance in cursor.fetchall():
                    # Assign relevance scores based on importance
                    if importance == 'primary':
                        relevance_scores[content_id] = 3.0
                    elif importance == 'secondary':
                        relevance_scores[content_id] = 2.0
                    else:
                        relevance_scores[content_id] = 1.0
                
                # Only add if we have ground truth
                if relevance_scores:
                    test_queries.append({
                        'query_id': f"concept_{concept_id}",
                        'query_text': query_text,
                        'concept_id': concept_id,
                        'concept_name': name,
                        'category': category,
                        'relevant_content': relevance_scores
                    })
            
            return test_queries
            
        except Exception as e:
            logger.error(f"Error generating concept queries: {str(e)}")
            return []
            
        finally:
            conn.close()
    
    def generate_content_queries(self, num_queries=10):
        """Generate test queries based on content titles and descriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get random content items with good titles
            cursor.execute("""
                SELECT id, title, description, content, source_type
                FROM ai_content
                WHERE title IS NOT NULL AND length(title) > 10
                ORDER BY RANDOM()
                LIMIT 30
            """)
            
            content_items = cursor.fetchall()
            selected_items = []
            
            # Filter for diverse content
            source_counts = {}
            
            for item in content_items:
                content_id, title, description, content, source_type = item
                
                # Track source type
                if source_type not in source_counts:
                    source_counts[source_type] = 0
                
                # Limit number per source type for diversity
                if source_counts[source_type] < 5:
                    selected_items.append(item)
                    source_counts[source_type] += 1
                
                if len(selected_items) >= num_queries:
                    break
            
            test_queries = []
            
            for content_id, title, description, content, source_type in selected_items:
                # Create query by transforming title into question
                query_text = self._title_to_question(title, source_type)
                
                # Ground truth for this query is the content itself
                relevant_content = {content_id: 3.0}  # Perfect match
                
                # Try to find related content
                cursor.execute("""
                    SELECT ac.id, cc1.importance
                    FROM content_concepts cc1
                    JOIN content_concepts cc2 ON cc1.concept_id = cc2.concept_id AND cc1.content_id != cc2.content_id
                    JOIN ai_content ac ON cc1.content_id = ac.id
                    WHERE cc2.content_id = ?
                    GROUP BY cc1.content_id
                    ORDER BY COUNT(DISTINCT cc1.concept_id) DESC
                    LIMIT 5
                """, (content_id,))
                
                for related_id, importance in cursor.fetchall():
                    # Lower relevance for related content
                    relevant_content[related_id] = 1.5
                
                test_queries.append({
                    'query_id': f"content_{content_id}",
                    'query_text': query_text,
                    'content_id': content_id,
                    'title': title,
                    'relevant_content': relevant_content
                })
            
            return test_queries
            
        except Exception as e:
            logger.error(f"Error generating content queries: {str(e)}")
            return []
            
        finally:
            conn.close()
    
    def _title_to_question(self, title, source_type):
        """Convert content title to question"""
        # Clean title
        title = title.strip()
        
        # Remove ending punctuation if present
        if title.endswith(('.', '?', '!')):
            title = title[:-1]
        
        # Generate question based on source type
        if source_type == 'instagram':  # Instagram
            templates = [
                f"Can you explain {title}?",
                f"What does the video about '{title}' discuss?",
                f"Tell me about {title}"
            ]
        elif source_type == 'github':  # GitHub
            templates = [
                f"How does {title} work?",
                f"What are the key features of {title}?",
                f"How can I use {title}?"
            ]
        elif source_type == 'research_paper':  # Research paper
            templates = [
                f"What does the paper '{title}' discuss?",
                f"Can you summarize the findings from '{title}'?",
                f"What are the key contributions of the paper '{title}'?"
            ]
        else:
            templates = [
                f"What is {title}?",
                f"Can you explain {title}?",
                f"Tell me about {title}"
            ]
        
        return random.choice(templates)
    
    def create_test_dataset(self, concept_queries=15, content_queries=10):
        """Create complete test dataset with different query types"""
        # Generate different types of queries
        concept_qs = self.generate_concept_queries(concept_queries)
        content_qs = self.generate_content_queries(content_queries)
        
        # Combine all queries
        all_queries = concept_qs + content_qs
        
        # Add some compound queries
        if len(concept_qs) >= 2:
            for i in range(min(5, len(concept_qs) // 2)):
                # Create compound query from two concepts
                concept1 = concept_qs[i * 2]
                concept2 = concept_qs[i * 2 + 1]
                
                query_text = f"Compare {concept1['concept_name']} and {concept2['concept_name']}"
                
                # Combine relevant content from both
                relevant_content = {}
                for content_id, score in concept1['relevant_content'].items():
                    relevant_content[content_id] = score
                
                for content_id, score in concept2['relevant_content'].items():
                    if content_id in relevant_content:
                        # Boost score if relevant to both concepts
                        relevant_content[content_id] = max(relevant_content[content_id], score) + 0.5
                    else:
                        relevant_content[content_id] = score
                
                all_queries.append({
                    'query_id': f"compound_{concept1['concept_id']}_{concept2['concept_id']}",
                    'query_text': query_text,
                    'concepts': [concept1['concept_name'], concept2['concept_name']],
                    'relevant_content': relevant_content
                })
        
        # Save test dataset
        dataset = {
            'name': f"test_dataset_{datetime.now().strftime('%Y%m%d')}",
            'created': datetime.now().isoformat(),
            'queries': all_queries
        }
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create test_datasets table if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_datasets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    date_created TEXT NOT NULL,
                    queries_json TEXT NOT NULL
                )
            """)
            
            # Insert dataset
            cursor.execute("""
                INSERT INTO test_datasets (name, date_created, queries_json)
                VALUES (?, ?, ?)
            """, (
                dataset['name'],
                dataset['created'],
                json.dumps(dataset['queries'])
            ))
            
            dataset_id = cursor.lastrowid
            conn.commit()
            
            # Save to file as well
            os.makedirs(os.path.join(config.DATA_DIR, "evaluation"), exist_ok=True)
            with open(os.path.join(config.DATA_DIR, "evaluation", f"{dataset['name']}.json"), 'w') as f:
                json.dump(dataset, f, indent=2)
            
            return {
                'dataset_id': dataset_id,
                'name': dataset['name'],
                'queries_count': len(all_queries),
                'file_path': os.path.join(config.DATA_DIR, "evaluation", f"{dataset['name']}.json")
            }
            
        except Exception as e:
            logger.error(f"Error saving test dataset: {str(e)}")
            conn.rollback()
            return None
            
        finally:
            conn.close() 