"""
Swagger documentation for the API
"""
from flask import Blueprint, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

# Create Swagger UI blueprint
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
API_URL = '/api/swagger.json'  # URL for Swagger JSON endpoint

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "AI Knowledge Base API"
    }
)

# Blueprint for Swagger JSON
swagger_json_bp = Blueprint('swagger_json', __name__)

@swagger_json_bp.route('/api/swagger.json')
def swagger_json():
    """Return Swagger specification"""
    swagger_spec = {
        "swagger": "2.0",
        "info": {
            "title": "AI Knowledge Base API",
            "description": "API for accessing and querying the AI Knowledge Base",
            "version": "1.0.0"
        },
        "basePath": "/api/v1",
        "schemes": ["http", "https"],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Check if the API is running",
                    "produces": ["application/json"],
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "timestamp": {"type": "string", "format": "date-time"},
                                    "version": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            "/search": {
                "post": {
                    "summary": "Search the knowledge base",
                    "description": "Search for content in the knowledge base",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query"
                                    },
                                    "search_type": {
                                        "type": "string",
                                        "description": "Search type (hybrid, vector, keyword)",
                                        "enum": ["hybrid", "vector", "keyword"],
                                        "default": "hybrid"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Number of results to return",
                                        "default": 10
                                    },
                                    "vector_weight": {
                                        "type": "number",
                                        "description": "Weight for vector search (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "keyword_weight": {
                                        "type": "number",
                                        "description": "Weight for keyword search (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "source_type": {
                                        "type": "string",
                                        "description": "Filter by source type",
                                        "enum": ["instagram", "github", "research_paper"]
                                    },
                                    "page": {
                                        "type": "integer",
                                        "description": "Page number for pagination",
                                        "default": 1
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "search_type": {"type": "string"},
                                    "results": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "content_id": {"type": "integer"},
                                                "title": {"type": "string"},
                                                "source_type": {"type": "string"},
                                                "similarity": {"type": "number"},
                                                "snippet": {"type": "string"},
                                                "url": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/answer": {
                "post": {
                    "summary": "Generate answer for a query",
                    "description": "Use RAG to answer a question",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Question to answer"
                                    },
                                    "search_type": {
                                        "type": "string",
                                        "description": "Search type for retrieving context",
                                        "enum": ["hybrid", "vector", "keyword"],
                                        "default": "hybrid"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Number of results to use as context",
                                        "default": 5
                                    },
                                    "vector_weight": {
                                        "type": "number",
                                        "description": "Weight for vector search (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "keyword_weight": {
                                        "type": "number",
                                        "description": "Weight for keyword search (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "source_type": {
                                        "type": "string",
                                        "description": "Filter by source type",
                                        "enum": ["instagram", "github", "research_paper"]
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "LLM model to use",
                                        "default": "claude-3-sonnet-20240229"
                                    },
                                    "stream": {
                                        "type": "boolean",
                                        "description": "Whether to stream the response",
                                        "default": False
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Generated answer",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "sources": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "metadata": {"type": "object"}
                                }
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/answer/stream": {
                "get": {
                    "summary": "Stream answer generation",
                    "description": "Stream the answer as it's being generated",
                    "produces": ["text/event-stream"],
                    "parameters": [
                        {
                            "name": "query",
                            "in": "query",
                            "required": True,
                            "type": "string",
                            "description": "Question to answer"
                        },
                        {
                            "name": "search_type",
                            "in": "query",
                            "required": False,
                            "type": "string",
                            "enum": ["hybrid", "vector", "keyword"],
                            "default": "hybrid",
                            "description": "Search type for retrieving context"
                        },
                        {
                            "name": "top_k",
                            "in": "query",
                            "required": False,
                            "type": "integer",
                            "default": 5,
                            "description": "Number of results to use as context"
                        },
                        {
                            "name": "vector_weight",
                            "in": "query",
                            "required": False,
                            "type": "number",
                            "default": 0.7,
                            "description": "Weight for vector search (0-1)"
                        },
                        {
                            "name": "keyword_weight",
                            "in": "query",
                            "required": False,
                            "type": "number",
                            "default": 0.3,
                            "description": "Weight for keyword search (0-1)"
                        },
                        {
                            "name": "model",
                            "in": "query",
                            "required": False,
                            "type": "string",
                            "description": "LLM model to use"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Event stream of answer chunks",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/feedback": {
                "post": {
                    "summary": "Submit feedback on search results",
                    "description": "Submit relevance feedback for a search result",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query_log_id": {
                                        "type": "integer",
                                        "description": "ID of the query log"
                                    },
                                    "content_id": {
                                        "type": "integer",
                                        "description": "ID of the content for which feedback is given"
                                    },
                                    "feedback_score": {
                                        "type": "integer",
                                        "description": "Feedback score (1-5)",
                                        "minimum": 1,
                                        "maximum": 5
                                    },
                                    "feedback_type": {
                                        "type": "string",
                                        "description": "Type of feedback",
                                        "default": "relevance"
                                    },
                                    "feedback_text": {
                                        "type": "string",
                                        "description": "Additional feedback comments"
                                    }
                                },
                                "required": ["content_id", "feedback_score"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Feedback submitted successfully",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "message": {"type": "string"}
                                }
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/concepts/search": {
                "get": {
                    "summary": "Search for concepts",
                    "description": "Search for concepts in the knowledge graph",
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "type": "string",
                            "description": "Search query"
                        },
                        {
                            "name": "category",
                            "in": "query",
                            "required": False,
                            "type": "string",
                            "description": "Filter by concept category"
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of results to return"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "results": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                                "category": {"type": "string"},
                                                "importance": {"type": "number"},
                                                "confidence": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/concepts/{concept_id}": {
                "get": {
                    "summary": "Get concept details",
                    "description": "Get detailed information about a concept",
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "concept_id",
                            "in": "path",
                            "required": True,
                            "type": "integer",
                            "description": "ID of the concept"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Concept details",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "name": {"type": "string"},
                                    "category": {"type": "string"},
                                    "description": {"type": "string"},
                                    "importance": {"type": "number"},
                                    "confidence": {"type": "number"},
                                    "related_content": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "related_concepts": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "404": {"description": "Concept not found"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/content/{content_id}": {
                "get": {
                    "summary": "Get content details",
                    "description": "Get detailed information about a content item",
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "content_id",
                            "in": "path",
                            "required": True,
                            "type": "integer",
                            "description": "ID of the content"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Content details",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                    "content": {"type": "string"},
                                    "url": {"type": "string"},
                                    "date_created": {"type": "string"},
                                    "date_collected": {"type": "string"},
                                    "metadata": {"type": "string"},
                                    "source_type": {"type": "string"},
                                    "concepts": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "related_content": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "404": {"description": "Content not found"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/kg/stats": {
                "get": {
                    "summary": "Get knowledge graph statistics",
                    "description": "Get statistics about the knowledge graph",
                    "produces": ["application/json"],
                    "responses": {
                        "200": {
                            "description": "Knowledge graph statistics",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "concepts_count": {"type": "integer"},
                                    "content_count": {"type": "integer"},
                                    "relationships_count": {"type": "integer"},
                                    "categories": {"type": "object"},
                                    "top_concepts": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "500": {"description": "Server error"}
                    }
                }
            }
        }
    }
    
    return jsonify(swagger_spec) 