import logging
import json
import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import anthropic
except ImportError:
    anthropic = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('answer_evaluator')

class AnswerEvaluator:
    """Evaluator for LLM answer quality using Claude"""
    
    def __init__(self, model=None):
        """Initialize evaluator with Claude API client"""
        if anthropic is None:
            logger.warning("anthropic package not found. Please install it with pip install anthropic")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model or "claude-3-sonnet-20240229"
    
    def evaluate_faithfulness(self, answer, context, query=None):
        """
        Evaluate if the answer is faithful to the provided context
        
        Args:
            answer: The generated answer
            context: The retrieval context provided to the LLM
            query: Optional query for context
            
        Returns:
            Dict with faithfulness score and explanation
        """
        if not answer or not context:
            return {
                'score': 0,
                'explanation': "Missing answer or context for evaluation"
            }
            
        if self.client is None:
            logger.error("Claude API client not initialized")
            return {
                'score': 0,
                'explanation': "Claude API client not initialized"
            }
            
        # Create evaluation prompt
        prompt = f"""
        You are an expert evaluator assessing the quality of AI assistant responses.
        Rate how faithful this answer is to the provided context on a scale of 1-5.
        
        A faithful answer:
        - Only contains information present in the context
        - Doesn't introduce hallucinations or made-up facts
        - May say "I don't know" when information isn't in the context
        - Correctly cites information sources when available
        
        Rating scale:
        1 = Completely unfaithful (major hallucinations, contradicts context)
        2 = Mostly unfaithful (contains significant unsupported claims)
        3 = Partially faithful (minor unsupported claims, but mostly accurate)
        4 = Mostly faithful (very minor extrapolations but no contradictions)
        5 = Completely faithful (only uses information from context)
        
        {"Question: " + query if query else ""}
        
        Context provided to the assistant:
        {context}
        
        Assistant's answer:
        {answer}
        
        First, analyze what information in the answer is supported or unsupported by the context.
        Then provide your faithfulness rating (1-5) with an explanation.
        
        Format your response as a JSON object:
        {{
            "score": 1-5,
            "explanation": "Your detailed explanation"
        }}
        """
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                system="You are an expert evaluator assessing AI-generated answers. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            
            try:
                # Try to find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Validate structure
                    if not isinstance(result, dict) or "score" not in result:
                        return {
                            'score': 0,
                            'explanation': "Invalid evaluation format"
                        }
                    
                    return result
                else:
                    # Fallback if no JSON found
                    return {
                        'score': 0,
                        'explanation': "Could not parse evaluation result"
                    }
            except json.JSONDecodeError:
                return {
                    'score': 0,
                    'explanation': "Error parsing evaluation result"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating answer faithfulness: {str(e)}")
            return {
                'score': 0,
                'explanation': f"Evaluation error: {str(e)}"
            }
    
    def evaluate_relevance(self, answer, query):
        """
        Evaluate if the answer is relevant to the query
        
        Args:
            answer: The generated answer
            query: The original query
            
        Returns:
            Dict with relevance score and explanation
        """
        if not answer or not query:
            return {
                'score': 0,
                'explanation': "Missing answer or query for evaluation"
            }
            
        if self.client is None:
            logger.error("Claude API client not initialized")
            return {
                'score': 0,
                'explanation': "Claude API client not initialized"
            }
            
        # Create evaluation prompt
        prompt = f"""
        You are an expert evaluator assessing the quality of AI assistant responses.
        Rate how relevant this answer is to the user's query on a scale of 1-5.
        
        A relevant answer:
        - Directly addresses the user's question
        - Provides the specific information requested
        - Stays on topic without unnecessary tangents
        - Matches the level of detail appropriate to the question
        
        Rating scale:
        1 = Not relevant (doesn't answer the question at all)
        2 = Slightly relevant (touches on the topic but misses the main point)
        3 = Moderately relevant (addresses part of the question but has gaps)
        4 = Very relevant (addresses the question well with minor omissions)
        5 = Perfectly relevant (addresses exactly what was asked)
        
        User query:
        {query}
        
        Assistant's answer:
        {answer}
        
        First, analyze how well the answer addresses the specific query.
        Then provide your relevance rating (1-5) with an explanation.
        
        Format your response as a JSON object:
        {{
            "score": 1-5,
            "explanation": "Your detailed explanation"
        }}
        """
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                system="You are an expert evaluator assessing AI-generated answers. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            
            try:
                # Try to find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Validate structure
                    if not isinstance(result, dict) or "score" not in result:
                        return {
                            'score': 0,
                            'explanation': "Invalid evaluation format"
                        }
                    
                    return result
                else:
                    # Fallback if no JSON found
                    return {
                        'score': 0,
                        'explanation': "Could not parse evaluation result"
                    }
            except json.JSONDecodeError:
                return {
                    'score': 0,
                    'explanation': "Error parsing evaluation result"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return {
                'score': 0,
                'explanation': f"Evaluation error: {str(e)}"
            }
    
    def save_evaluation_results(self, eval_results, db_path=None):
        """Save answer evaluation results to database"""
        conn = sqlite3.connect(db_path or config.DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Create answer_evaluations table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS answer_evaluations (
                    id INTEGER PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    context_text TEXT,
                    faithfulness_score REAL,
                    faithfulness_explanation TEXT,
                    relevance_score REAL,
                    relevance_explanation TEXT,
                    date_evaluated TEXT NOT NULL
                )
            """)
            
            # Insert evaluation results
            cursor.execute("""
                INSERT INTO answer_evaluations (
                    query_id, query_text, answer_text, context_text,
                    faithfulness_score, faithfulness_explanation,
                    relevance_score, relevance_explanation,
                    date_evaluated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_results['query_id'],
                eval_results['query_text'],
                eval_results['answer_text'],
                eval_results.get('context_text'),
                eval_results.get('faithfulness_score'),
                eval_results.get('faithfulness_explanation'),
                eval_results.get('relevance_score'),
                eval_results.get('relevance_explanation'),
                datetime.now().isoformat()
            ))
            
            eval_id = cursor.lastrowid
            conn.commit()
            
            return eval_id
            
        except Exception as e:
            logger.error(f"Error saving answer evaluation: {str(e)}")
            conn.rollback()
            return None
            
        finally:
            conn.close() 