#!/usr/bin/env python3
"""
Knowledge Graph Module for Instagram Knowledge Base

This module provides tools for querying, analyzing, and visualizing
the AI concept knowledge graph constructed from extracted concepts.
"""
import os
import sys
import logging
import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Set

# Import local modules
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/knowledge_graph.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('knowledge_graph')

# Optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Install with 'pip install networkx'")

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with 'pip install matplotlib'")

# Check for Plotly
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with 'pip install plotly'")


class ConceptQuery:
    """Query and retrieve concepts from the knowledge graph"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize with database connection
        
        Args:
            db_path: Path to SQLite database (defaults to config.DB_PATH)
        """
        self.db_path = db_path or config.DB_PATH
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection
        
        Returns:
            SQLite connection object
        """
        return sqlite3.connect(self.db_path)
    
    def get_concept_by_id(self, concept_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a concept by ID
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            Concept data or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, category, first_seen_date, 
                       last_updated, reference_count
                FROM concepts
                WHERE id = ?
            """, (concept_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "category": row[3],
                "first_seen_date": row[4],
                "last_updated": row[5],
                "reference_count": row[6]
            }
        finally:
            conn.close()
    
    def get_concept_by_name(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by name
        
        Args:
            concept_name: The name of the concept
            
        Returns:
            Concept data or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, category, first_seen_date, 
                       last_updated, reference_count
                FROM concepts
                WHERE name = ?
            """, (concept_name,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "category": row[3],
                "first_seen_date": row[4],
                "last_updated": row[5],
                "reference_count": row[6]
            }
        finally:
            conn.close()
    
    def search_concepts(self, search_term: str, 
                       category: Optional[str] = None,
                       min_references: int = 0,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for concepts matching a search term
        
        Args:
            search_term: Text to search for in concept names and descriptions
            category: Optional category to filter by
            min_references: Minimum number of references
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT id, name, description, category, first_seen_date, 
                       last_updated, reference_count
                FROM concepts
                WHERE (name LIKE ? OR description LIKE ?)
                  AND reference_count >= ?
            """
            params = [f"%{search_term}%", f"%{search_term}%", min_references]
            
            if category:
                query += " AND category = ?"
                params.append(category)
                
            query += " ORDER BY reference_count DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "first_seen_date": row[4],
                    "last_updated": row[5],
                    "reference_count": row[6]
                })
                
            return results
        finally:
            conn.close()
    
    def get_concept_categories(self) -> List[str]:
        """
        Get all distinct concept categories
        
        Returns:
            List of category names
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT DISTINCT category FROM concepts
                WHERE category IS NOT NULL AND category != ''
                ORDER BY category
            """)
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_concepts_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get concepts in a specific category
        
        Args:
            category: Category name
            limit: Maximum number of results
            
        Returns:
            List of concepts in that category
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, category, first_seen_date, 
                       last_updated, reference_count
                FROM concepts
                WHERE category = ?
                ORDER BY reference_count DESC, name
                LIMIT ?
            """, (category, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "first_seen_date": row[4],
                    "last_updated": row[5],
                    "reference_count": row[6]
                })
                
            return results
        finally:
            conn.close()
    
    def get_top_concepts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the most referenced concepts
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of top concepts
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, category, first_seen_date, 
                       last_updated, reference_count
                FROM concepts
                ORDER BY reference_count DESC, name
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "first_seen_date": row[4],
                    "last_updated": row[5],
                    "reference_count": row[6]
                })
                
            return results
        finally:
            conn.close()
    
    def get_related_concepts(self, concept_id: int) -> List[Dict[str, Any]]:
        """
        Get concepts related to a specific concept
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            List of related concepts with relationship info
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get outgoing relationships (source -> target)
            cursor.execute("""
                SELECT 
                    cr.target_concept_id, 
                    c.name,
                    c.description,
                    c.category,
                    cr.relationship_type,
                    cr.confidence_score,
                    cr.reference_count
                FROM concept_relationships cr
                JOIN concepts c ON cr.target_concept_id = c.id
                WHERE cr.source_concept_id = ?
                ORDER BY cr.confidence_score DESC
            """, (concept_id,))
            
            outgoing = []
            for row in cursor.fetchall():
                outgoing.append({
                    "concept_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "relationship": row[4],
                    "confidence": row[5],
                    "reference_count": row[6],
                    "direction": "outgoing"
                })
                
            # Get incoming relationships (target <- source)
            cursor.execute("""
                SELECT 
                    cr.source_concept_id, 
                    c.name,
                    c.description,
                    c.category,
                    cr.relationship_type,
                    cr.confidence_score,
                    cr.reference_count
                FROM concept_relationships cr
                JOIN concepts c ON cr.source_concept_id = c.id
                WHERE cr.target_concept_id = ?
                ORDER BY cr.confidence_score DESC
            """, (concept_id,))
            
            incoming = []
            for row in cursor.fetchall():
                incoming.append({
                    "concept_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "relationship": row[4],
                    "confidence": row[5],
                    "reference_count": row[6],
                    "direction": "incoming"
                })
                
            return outgoing + incoming
        finally:
            conn.close()
    
    def get_content_with_concept(self, concept_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get content items that contain a specific concept
        
        Args:
            concept_id: ID of the concept
            limit: Maximum number of results
            
        Returns:
            List of content items with concept importance
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    ac.id,
                    ac.title,
                    ac.description,
                    st.name as source_type,
                    cc.importance,
                    cc.date_extracted
                FROM content_concepts cc
                JOIN ai_content ac ON cc.content_id = ac.id
                JOIN source_types st ON ac.source_type_id = st.id
                WHERE cc.concept_id = ?
                ORDER BY 
                    CASE 
                        WHEN cc.importance = 'primary' THEN 1
                        WHEN cc.importance = 'secondary' THEN 2
                        ELSE 3
                    END,
                    ac.date_collected DESC
                LIMIT ?
            """, (concept_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "content_id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "source_type": row[3],
                    "importance": row[4],
                    "date_extracted": row[5]
                })
                
            return results
        finally:
            conn.close()


class RelationshipQuery:
    """Query and analyze concept relationships in the knowledge graph"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize with database connection
        
        Args:
            db_path: Path to SQLite database (defaults to config.DB_PATH)
        """
        self.db_path = db_path or config.DB_PATH
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection
        
        Returns:
            SQLite connection object
        """
        return sqlite3.connect(self.db_path)
    
    def get_relationship_types(self) -> List[str]:
        """
        Get all distinct relationship types
        
        Returns:
            List of relationship types
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT DISTINCT relationship_type 
                FROM concept_relationships
                ORDER BY relationship_type
            """)
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_relationships_by_type(self, rel_type: str, 
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get relationships of a specific type
        
        Args:
            rel_type: Type of relationship
            limit: Maximum number of results
            
        Returns:
            List of relationships
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    cr.id,
                    s.id as source_id,
                    s.name as source_name,
                    t.id as target_id,
                    t.name as target_name,
                    cr.relationship_type,
                    cr.confidence_score,
                    cr.reference_count
                FROM concept_relationships cr
                JOIN concepts s ON cr.source_concept_id = s.id
                JOIN concepts t ON cr.target_concept_id = t.id
                WHERE cr.relationship_type = ?
                ORDER BY cr.confidence_score DESC, cr.reference_count DESC
                LIMIT ?
            """, (rel_type, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "source_id": row[1],
                    "source_name": row[2],
                    "target_id": row[3],
                    "target_name": row[4],
                    "relationship_type": row[5],
                    "confidence_score": row[6],
                    "reference_count": row[7]
                })
                
            return results
        finally:
            conn.close()
    
    def get_relationship(self, source_id: int, target_id: int, 
                       rel_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific relationship between two concepts
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            rel_type: Optional relationship type (if None, returns any relationship)
            
        Returns:
            Relationship data or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT 
                    cr.id,
                    s.name as source_name,
                    t.name as target_name,
                    cr.relationship_type,
                    cr.confidence_score,
                    cr.reference_count,
                    cr.first_seen_date,
                    cr.last_updated
                FROM concept_relationships cr
                JOIN concepts s ON cr.source_concept_id = s.id
                JOIN concepts t ON cr.target_concept_id = t.id
                WHERE cr.source_concept_id = ? AND cr.target_concept_id = ?
            """
            params = [source_id, target_id]
            
            if rel_type:
                query += " AND cr.relationship_type = ?"
                params.append(rel_type)
                
            cursor.execute(query, params)
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return {
                "id": row[0],
                "source_name": row[1],
                "target_name": row[2],
                "relationship_type": row[3],
                "confidence_score": row[4],
                "reference_count": row[5],
                "first_seen_date": row[6],
                "last_updated": row[7]
            }
        finally:
            conn.close()
    
    def get_all_relationships(self, min_confidence: float = 0.5,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all relationships with a minimum confidence score
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0)
            limit: Maximum number of results
            
        Returns:
            List of relationships
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    cr.id,
                    cr.source_concept_id,
                    s.name as source_name,
                    cr.target_concept_id,
                    t.name as target_name,
                    cr.relationship_type,
                    cr.confidence_score,
                    cr.reference_count
                FROM concept_relationships cr
                JOIN concepts s ON cr.source_concept_id = s.id
                JOIN concepts t ON cr.target_concept_id = t.id
                WHERE cr.confidence_score >= ?
                ORDER BY cr.confidence_score DESC, cr.reference_count DESC
                LIMIT ?
            """, (min_confidence, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "source_id": row[1],
                    "source_name": row[2],
                    "target_id": row[3],
                    "target_name": row[4],
                    "relationship_type": row[5],
                    "confidence_score": row[6],
                    "reference_count": row[7]
                })
                
            return results
        finally:
            conn.close()


class KnowledgeGraph:
    """Build and analyze a NetworkX graph from concept data"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize knowledge graph builder
        
        Args:
            db_path: Path to SQLite database (defaults to config.DB_PATH)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available. Install with 'pip install networkx'")
            
        self.db_path = db_path or config.DB_PATH
        self.concept_query = ConceptQuery(db_path)
        self.rel_query = RelationshipQuery(db_path)
        self.graph = None
        
    def build_graph(self, min_confidence: float = 0.5, 
                  min_references: int = 1,
                  include_categories: List[str] = None,
                  exclude_categories: List[str] = None,
                  relationship_types: List[str] = None) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from concept relationships
        
        Args:
            min_confidence: Minimum confidence score for relationships
            min_references: Minimum reference count for concepts
            include_categories: Only include these categories (if provided)
            exclude_categories: Exclude these categories (if provided)
            relationship_types: Only include these relationship types (if provided)
            
        Returns:
            NetworkX DiGraph object
        """
        logger.info("Building knowledge graph from concept database")
        
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Get all concepts that meet criteria
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build query to get concepts
            query = """
                SELECT id, name, description, category, reference_count
                FROM concepts
                WHERE reference_count >= ?
            """
            params = [min_references]
            
            if include_categories:
                placeholders = ", ".join(["?"] * len(include_categories))
                query += f" AND category IN ({placeholders})"
                params.extend(include_categories)
                
            if exclude_categories:
                placeholders = ", ".join(["?"] * len(exclude_categories))
                query += f" AND category NOT IN ({placeholders})"
                params.extend(exclude_categories)
                
            cursor.execute(query, params)
            
            # Add nodes for concepts
            for row in cursor.fetchall():
                concept_id, name, description, category, ref_count = row
                G.add_node(concept_id, 
                           name=name, 
                           description=description, 
                           category=category,
                           reference_count=ref_count)
            
            # Get relationships
            rel_query = """
                SELECT source_concept_id, target_concept_id, relationship_type, 
                       confidence_score, reference_count
                FROM concept_relationships
                WHERE confidence_score >= ?
            """
            rel_params = [min_confidence]
            
            if relationship_types:
                placeholders = ", ".join(["?"] * len(relationship_types))
                rel_query += f" AND relationship_type IN ({placeholders})"
                rel_params.extend(relationship_types)
                
            cursor.execute(rel_query, rel_params)
            
            # Add edges for relationships
            for row in cursor.fetchall():
                source_id, target_id, rel_type, confidence, ref_count = row
                
                # Only add edges between nodes that exist in the graph
                if source_id in G and target_id in G:
                    G.add_edge(source_id, target_id, 
                              relationship=rel_type,
                              confidence=confidence,
                              reference_count=ref_count,
                              weight=confidence)  # Use confidence as edge weight
            
            logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            self.graph = G
            return G
            
        finally:
            conn.close()
    
    def get_central_concepts(self, limit: int = 10, 
                           centrality_type: str = "degree") -> List[Dict[str, Any]]:
        """
        Get the most central concepts in the graph
        
        Args:
            limit: Maximum number of concepts to return
            centrality_type: Type of centrality ('degree', 'betweenness', 'eigenvector', 'pagerank')
            
        Returns:
            List of concepts with centrality scores
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build_graph() first.")
            
        if centrality_type == "degree":
            centrality = nx.degree_centrality(self.graph)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(self.graph)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(self.graph, max_iter=300)
        elif centrality_type == "pagerank":
            centrality = nx.pagerank(self.graph)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
            
        # Get top concepts by centrality
        top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Get full concept info
        results = []
        for concept_id, score in top_concepts:
            concept_data = self.concept_query.get_concept_by_id(concept_id)
            if concept_data:
                concept_data["centrality_score"] = score
                results.append(concept_data)
                
        return results
    
    def get_concept_communities(self, algorithm: str = "louvain") -> Dict[int, int]:
        """
        Detect communities/clusters of concepts
        
        Args:
            algorithm: Community detection algorithm ('louvain', 'label_propagation', 'clique')
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build_graph() first.")
            
        # Convert to undirected graph for community detection
        undirected_graph = self.graph.to_undirected()
        
        if algorithm == "louvain":
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(undirected_graph)
            except ImportError:
                logger.warning("python-louvain package not installed. Using label propagation instead.")
                communities = self._label_propagation_communities()
        elif algorithm == "label_propagation":
            communities = self._label_propagation_communities()
        elif algorithm == "clique":
            communities = self._clique_communities()
        else:
            raise ValueError(f"Unknown community detection algorithm: {algorithm}")
            
        return communities
    
    def _label_propagation_communities(self) -> Dict[int, int]:
        """Use label propagation for community detection"""
        result = {}
        undirected_graph = self.graph.to_undirected()
        
        for i, community in enumerate(nx.algorithms.community.label_propagation_communities(undirected_graph)):
            for node in community:
                result[node] = i
                
        return result
    
    def _clique_communities(self) -> Dict[int, int]:
        """Use clique percolation for community detection"""
        result = {}
        undirected_graph = self.graph.to_undirected()
        
        for i, community in enumerate(nx.algorithms.community.k_clique_communities(undirected_graph, 3)):
            for node in community:
                result[node] = i
                
        # Assign remaining nodes to their own communities
        next_id = max(result.values()) + 1 if result else 0
        for node in undirected_graph.nodes():
            if node not in result:
                result[node] = next_id
                next_id += 1
                
        return result
    
    def get_community_summary(self, communities: Dict[int, int], 
                            min_community_size: int = 3) -> List[Dict[str, Any]]:
        """
        Generate summary of community contents
        
        Args:
            communities: Output from get_concept_communities()
            min_community_size: Minimum community size to include
            
        Returns:
            List of community summaries
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build_graph() first.")
            
        # Group nodes by community
        community_nodes = {}
        for node, community_id in communities.items():
            if community_id not in community_nodes:
                community_nodes[community_id] = []
            community_nodes[community_id].append(node)
            
        # Generate summaries for communities that meet size threshold
        summaries = []
        for community_id, nodes in community_nodes.items():
            if len(nodes) >= min_community_size:
                # Get concept details for each node
                concepts = []
                for node in nodes:
                    concept_data = self.concept_query.get_concept_by_id(node)
                    if concept_data:
                        concepts.append(concept_data)
                
                # Get category distribution
                category_counts = {}
                for concept in concepts:
                    category = concept.get("category", "unknown")
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1
                    
                # Get top concept by reference count
                top_concept = max(concepts, key=lambda x: x.get("reference_count", 0)) if concepts else None
                
                summaries.append({
                    "community_id": community_id,
                    "size": len(nodes),
                    "top_concept": top_concept["name"] if top_concept else None,
                    "top_concept_id": top_concept["id"] if top_concept else None,
                    "category_distribution": category_counts,
                    "concepts": concepts
                })
                
        # Sort by community size
        return sorted(summaries, key=lambda x: x["size"], reverse=True)
    
    def find_paths_between_concepts(self, source_id: int, target_id: int, 
                                  max_paths: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two concepts in the graph
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_paths: Maximum number of paths to find
            
        Returns:
            List of paths, where each path is a list of concepts with relationships
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build_graph() first.")
            
        if source_id not in self.graph or target_id not in self.graph:
            return []
            
        # Find simple paths between source and target
        all_paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=6))[:max_paths]
        
        # Format results
        result_paths = []
        for path in all_paths:
            path_data = []
            
            # Add source node
            source_concept = self.concept_query.get_concept_by_id(path[0])
            if source_concept:
                path_data.append({
                    "id": source_concept["id"],
                    "name": source_concept["name"],
                    "category": source_concept["category"],
                    "relationship": None
                })
                
            # Add intermediate nodes with relationships
            for i in range(1, len(path)):
                prev_id = path[i-1]
                curr_id = path[i]
                
                # Get relationship data
                rel_data = self.rel_query.get_relationship(prev_id, curr_id)
                
                # Get concept data
                concept_data = self.concept_query.get_concept_by_id(curr_id)
                if concept_data:
                    path_data.append({
                        "id": concept_data["id"],
                        "name": concept_data["name"],
                        "category": concept_data["category"],
                        "relationship": rel_data["relationship_type"] if rel_data else "unknown"
                    })
                    
            if len(path_data) > 1:
                result_paths.append(path_data)
                
        return result_paths
    
    def get_concept_neighborhood(self, concept_id: int, 
                              max_distance: int = 2) -> nx.DiGraph:
        """
        Get a subgraph of concepts around a given concept
        
        Args:
            concept_id: Center concept ID
            max_distance: Maximum distance from center concept
            
        Returns:
            NetworkX DiGraph of the neighborhood
        """
        if not self.graph:
            raise ValueError("No graph set for visualization")
            
        if concept_id not in self.graph:
            return nx.DiGraph()
            
        # Get all nodes within max_distance
        neighborhood_nodes = {concept_id}
        current_frontier = {concept_id}
        
        for _ in range(max_distance):
            next_frontier = set()
            
            for node in current_frontier:
                # Add successors (outgoing edges)
                successors = set(self.graph.successors(node))
                next_frontier.update(successors)
                
                # Add predecessors (incoming edges)
                predecessors = set(self.graph.predecessors(node))
                next_frontier.update(predecessors)
                
            # Remove nodes we've already seen
            next_frontier -= neighborhood_nodes
            neighborhood_nodes.update(next_frontier)
            current_frontier = next_frontier
            
        # Create subgraph
        return self.graph.subgraph(neighborhood_nodes).copy()
    
    def analyze_concept(self, concept_id: int) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a concept
        
        Args:
            concept_id: Concept ID to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build_graph() first.")
            
        if concept_id not in self.graph:
            return {"error": f"Concept ID {concept_id} not found in graph"}
            
        # Get basic concept data
        concept_data = self.concept_query.get_concept_by_id(concept_id)
        if not concept_data:
            return {"error": f"Concept ID {concept_id} not found in database"}
            
        # Get graph metrics
        in_degree = self.graph.in_degree(concept_id)
        out_degree = self.graph.out_degree(concept_id)
        
        # Get centrality metrics
        try:
            betweenness = nx.betweenness_centrality(self.graph, k=100)[concept_id]
        except:
            betweenness = 0
            
        try:
            pagerank = nx.pagerank(self.graph)[concept_id]
        except:
            pagerank = 0
            
        # Get related concepts
        related_concepts = self.concept_query.get_related_concepts(concept_id)
        
        # Get content with this concept
        content_items = self.concept_query.get_content_with_concept(concept_id, limit=10)
        
        # Combine everything
        return {
            "concept": concept_data,
            "graph_metrics": {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "betweenness_centrality": betweenness,
                "pagerank": pagerank
            },
            "relationships": {
                "incoming": [r for r in related_concepts if r["direction"] == "incoming"],
                "outgoing": [r for r in related_concepts if r["direction"] == "outgoing"]
            },
            "content": content_items
        }


class GraphVisualizer:
    """Visualize the knowledge graph in various formats"""
    
    def __init__(self, graph: Optional[nx.DiGraph] = None, 
                output_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            graph: NetworkX graph to visualize (optional)
            output_dir: Directory to save visualizations
        """
        self.graph = graph
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check dependencies
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available. Install with 'pip install networkx'")
            
    def set_graph(self, graph: nx.DiGraph) -> None:
        """
        Set the graph to visualize
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        
    def visualize_with_matplotlib(self, output_file: str = "knowledge_graph.png",
                                layout: str = "spring",
                                node_size_by: str = "reference_count",
                                edge_width_by: str = "confidence",
                                show_labels: bool = True,
                                label_font_size: int = 8) -> str:
        """
        Create a static visualization using matplotlib
        
        Args:
            output_file: Filename for the output image
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell')
            node_size_by: Node attribute to scale node sizes by
            edge_width_by: Edge attribute to scale edge widths by
            show_labels: Whether to show node labels
            label_font_size: Font size for node labels
            
        Returns:
            Path to saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install with 'pip install matplotlib'")
            
        if not self.graph:
            raise ValueError("No graph set for visualization")
            
        if self.graph.number_of_nodes() > 100:
            logger.warning(f"Graph has {self.graph.number_of_nodes()} nodes, which may be too many for a clear visualization")
        
        # Create a larger figure
        plt.figure(figsize=(16, 12), dpi=300)
        
        # Define layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == "shell":
            pos = nx.shell_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout algorithm: {layout}")
            
        # Get node sizes based on attribute
        if node_size_by and nx.get_node_attributes(self.graph, node_size_by):
            node_sizes = []
            for node in self.graph.nodes():
                size = self.graph.nodes[node].get(node_size_by, 1)
                # Scale size for visibility
                node_sizes.append(50 + (size * 20))
        else:
            node_sizes = 300
            
        # Get edge widths based on attribute
        edge_widths = []
        if edge_width_by:
            for u, v, data in self.graph.edges(data=True):
                width = data.get(edge_width_by, 1)
                # Scale width for visibility
                edge_widths.append(width * 2)
        
        # Get node colors by category
        node_categories = {}
        category_colors = {}
        color_map = plt.cm.tab20
        
        for node, attrs in self.graph.nodes(data=True):
            category = attrs.get("category", "unknown")
            if category not in node_categories:
                node_categories[category] = []
                category_colors[category] = color_map(len(category_colors) % 20)
            node_categories[category].append(node)
        
        # Draw the graph by category for better visualization
        for i, (category, nodes) in enumerate(node_categories.items()):
            nx.draw_networkx_nodes(
                self.graph, 
                pos, 
                nodelist=nodes,
                node_size=[node_sizes[list(self.graph.nodes()).index(n)] for n in nodes] if isinstance(node_sizes, list) else node_sizes,
                node_color=[category_colors[category]] * len(nodes),
                alpha=0.8,
                label=category
            )
        
        # Draw edges
        if edge_widths:
            nx.draw_networkx_edges(
                self.graph, 
                pos, 
                width=edge_widths,
                alpha=0.5,
                arrowsize=10,
                edge_color='gray'
            )
        else:
            nx.draw_networkx_edges(
                self.graph, 
                pos, 
                alpha=0.5,
                arrowsize=10,
                edge_color='gray'
            )
        
        # Draw labels if requested
        if show_labels:
            # Get node labels (name attribute or node ID)
            labels = {}
            for node in self.graph.nodes():
                name = self.graph.nodes[node].get("name", str(node))
                labels[node] = name
                
            nx.draw_networkx_labels(
                self.graph, 
                pos, 
                labels=labels,
                font_size=label_font_size,
                font_family='sans-serif',
                font_color='black',
                font_weight='bold'
            )
        
        # Add a legend for categories
        plt.legend(scatterpoints=1, loc='lower right')
        
        # Remove axes
        plt.axis('off')
        plt.title(f"Knowledge Graph ({self.graph.number_of_nodes()} concepts, {self.graph.number_of_edges()} relationships)")
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    def visualize_interactive(self, output_file: str = "knowledge_graph.html",
                           layout: str = "force",
                           node_size_by: str = "reference_count",
                           edge_width_by: str = "confidence",
                           show_communities: bool = False,
                           communities: Optional[Dict[int, int]] = None) -> str:
        """
        Create an interactive HTML visualization using Plotly
        
        Args:
            output_file: Filename for the output HTML
            layout: Graph layout algorithm ('force', 'circular', 'random')
            node_size_by: Node attribute to scale node sizes by
            edge_width_by: Edge attribute to scale edge widths by
            show_communities: Whether to color nodes by communities
            communities: Community assignments (if None, uses modularity-based communities)
            
        Returns:
            Path to saved visualization file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available. Install with 'pip install plotly'")
            
        if not self.graph:
            raise ValueError("No graph set for visualization")
        
        # Create undirected graph for layout (but keep track of directed edges)
        G_undirected = self.graph.to_undirected()
        
        # Compute layout positions
        if layout == "force":
            pos = nx.spring_layout(G_undirected, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G_undirected)
        elif layout == "random":
            pos = nx.random_layout(G_undirected, seed=42)
        else:
            raise ValueError(f"Unknown layout algorithm: {layout}")
        
        # Get node communities if requested
        if show_communities:
            if communities is None:
                # Use networkx community detection
                try:
                    import community as community_louvain
                    communities = community_louvain.best_partition(G_undirected)
                except ImportError:
                    # Fall back to built-in method
                    communities = {}
                    for i, comm in enumerate(nx.algorithms.community.greedy_modularity_communities(G_undirected)):
                        for node in comm:
                            communities[node] = i
            
            # Get unique community IDs
            community_ids = set(communities.values())
            # Create a colormap
            colorscale = px.colors.qualitative.Plotly if 'px' in globals() else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            community_colors = {comm_id: colorscale[i % len(colorscale)] for i, comm_id in enumerate(community_ids)}
        
        # Generate the edge traces
        edge_x = []
        edge_y = []
        edge_data = []
        
        for u, v, data in self.graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Create a slightly curved edge by adding an intermediate point
            n_points = 10  # Number of points for the curve
            edge_trace_x = []
            edge_trace_y = []
            
            for i in range(n_points + 1):
                t = i / n_points
                # Straight line: (x0, y0) to (x1, y1)
                x = x0 * (1 - t) + x1 * t
                y = y0 * (1 - t) + y1 * t
                edge_trace_x.append(x)
                edge_trace_y.append(y)
            
            edge_x.extend(edge_trace_x)
            edge_y.extend(edge_trace_y)
            
            # Add None values to create a break in the line
            edge_x.append(None)
            edge_y.append(None)
            
            # Store edge data for hover info
            rel_type = data.get("relationship", "related_to")
            confidence = data.get("confidence", 0.0)
            edge_data.append({
                "source": self.graph.nodes[u].get("name", str(u)),
                "target": self.graph.nodes[v].get("name", str(v)),
                "relationship": rel_type,
                "confidence": confidence
            })
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Generate the node trace
        node_x = []
        node_y = []
        node_data = []
        
        for node, attrs in self.graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            name = attrs.get("name", str(node))
            category = attrs.get("category", "unknown")
            ref_count = attrs.get("reference_count", 1)
            description = attrs.get("description", "")
            
            # Calculate node size based on attribute
            size = 10
            if node_size_by:
                if node_size_by == "reference_count":
                    size = 10 + (ref_count * 2)
                else:
                    size = 10 + (attrs.get(node_size_by, 1) * 2)
            
            # Get community color if applicable
            color = None
            if show_communities and node in communities:
                community_id = communities[node]
                color = community_colors.get(community_id)
            
            node_data.append({
                "id": node,
                "name": name,
                "category": category,
                "reference_count": ref_count,
                "description": description[:100] + "..." if description and len(description) > 100 else description,
                "size": size,
                "color": color,
                "community_id": communities.get(node) if show_communities else None
            })
        
        # Create node markers
        marker_colors = [n["color"] for n in node_data] if show_communities else None
        marker_sizes = [n["size"] for n in node_data]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True if show_communities else False,
                colorscale='YlGnBu' if show_communities else None,
                color=marker_colors,
                size=marker_sizes,
                line=dict(width=2, color='#333')
            )
        )
        
        # Create hover text
        hover_texts = []
        for node in node_data:
            text = f"Name: {node['name']}<br>"
            text += f"Category: {node['category']}<br>"
            text += f"References: {node['reference_count']}<br>"
            if node['description']:
                text += f"Description: {node['description']}<br>"
            if show_communities:
                text += f"Community: {node['community_id']}"
            hover_texts.append(text)
        
        node_trace.hovertext = hover_texts
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title=f'Knowledge Graph ({self.graph.number_of_nodes()} concepts, {self.graph.number_of_edges()} relationships)',
                          titlefont=dict(size=16),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          plot_bgcolor='#fff'
                      ))
        
        # Save as HTML
        output_path = os.path.join(self.output_dir, output_file)
        pio.write_html(fig, file=output_path, auto_open=False)
        
        logger.info(f"Saved interactive visualization to {output_path}")
        return output_path
    
    def export_to_gexf(self, output_file: str = "knowledge_graph.gexf") -> str:
        """
        Export graph to GEXF format for use in Gephi
        
        Args:
            output_file: Filename for the output GEXF file
            
        Returns:
            Path to saved GEXF file
        """
        if not self.graph:
            raise ValueError("No graph set for visualization")
            
        output_path = os.path.join(self.output_dir, output_file)
        nx.write_gexf(self.graph, output_path)
        
        logger.info(f"Exported graph to GEXF format: {output_path}")
        return output_path
    
    def export_to_json(self, output_file: str = "knowledge_graph.json") -> str:
        """
        Export graph to JSON format
        
        Args:
            output_file: Filename for the output JSON file
            
        Returns:
            Path to saved JSON file
        """
        if not self.graph:
            raise ValueError("No graph set for visualization")
            
        # Convert graph to JSON-serializable format
        data = {
            "nodes": [],
            "links": []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                "name": attrs.get("name", str(node)),
                "category": attrs.get("category", "unknown"),
                "reference_count": attrs.get("reference_count", 1)
            }
            
            # Add description if available
            if "description" in attrs and attrs["description"]:
                node_data["description"] = attrs["description"]
                
            data["nodes"].append(node_data)
            
        # Add edges
        for u, v, attrs in self.graph.edges(data=True):
            link_data = {
                "source": u,
                "target": v,
                "relationship": attrs.get("relationship", "related_to"),
                "confidence": attrs.get("confidence", 0.0),
                "reference_count": attrs.get("reference_count", 1)
            }
            
            data["links"].append(link_data)
            
        # Save to file
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported graph to JSON format: {output_path}")
        return output_path
    
    def visualize_concept_neighborhood(self, concept_id: int, output_file: str = None,
                                    max_distance: int = 2, 
                                    interactive: bool = True) -> str:
        """
        Visualize a specific concept and its neighborhood
        
        Args:
            concept_id: Center concept ID to visualize
            output_file: Filename for output (if None, auto-generates)
            max_distance: Maximum distance from center concept
            interactive: Whether to create interactive (True) or static (False) viz
            
        Returns:
            Path to saved visualization file
        """
        if not self.graph:
            raise ValueError("No graph set for visualization")
            
        # Create a knowledge graph instance to use neighborhood function
        kg = KnowledgeGraph()
        kg.graph = self.graph
        
        # Get the concept neighborhood
        neighborhood = kg.get_concept_neighborhood(concept_id, max_distance)
        
        if neighborhood.number_of_nodes() == 0:
            raise ValueError(f"Concept ID {concept_id} not found in graph")
            
        # Get the central concept name
        center_name = self.graph.nodes[concept_id].get("name", str(concept_id))
        
        # Set the subgraph
        self.set_graph(neighborhood)
        
        # Generate filename if not provided
        if output_file is None:
            center_name_safe = center_name.replace(" ", "_").replace("/", "_")
            if interactive:
                output_file = f"concept_{concept_id}_{center_name_safe}.html"
            else:
                output_file = f"concept_{concept_id}_{center_name_safe}.png"
        
        # Create visualization
        if interactive:
            return self.visualize_interactive(output_file=output_file)
        else:
            return self.visualize_with_matplotlib(output_file=output_file)


class KnowledgeGraphManager:
    """Manage and integrate knowledge graph operations"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the knowledge graph manager
        
        Args:
            db_path: Path to SQLite database (defaults to config.DB_PATH)
        """
        self.db_path = db_path or config.DB_PATH
        self.concept_query = ConceptQuery(db_path)
        self.rel_query = RelationshipQuery(db_path)
        
        if NETWORKX_AVAILABLE:
            self.graph_builder = KnowledgeGraph(db_path)
            self.visualizer = GraphVisualizer()
        else:
            self.graph_builder = None
            self.visualizer = None
        
        # Flag to indicate if graph is built
        self.graph_built = False
        
    def check_concepts_available(self) -> bool:
        """
        Check if concepts are available in the database
        
        Returns:
            Boolean indicating if concepts exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if concepts table exists and has data
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name='concepts'
            """)
            
            if cursor.fetchone()[0] == 0:
                logger.warning("Concepts table does not exist in the database")
                return False
                
            # Check if there are any concepts
            cursor.execute("SELECT COUNT(*) FROM concepts")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.warning("No concepts found in the database")
                return False
                
            logger.info(f"Found {count} concepts in the database")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error checking concepts: {str(e)}")
            return False
        finally:
            conn.close()
    
    def build_graph(self, **kwargs) -> bool:
        """
        Build the knowledge graph with the given parameters
        
        Args:
            **kwargs: Parameters to pass to the graph builder
            
        Returns:
            Boolean indicating success
        """
        if not self.graph_builder:
            logger.error("Graph builder not available (NetworkX not installed)")
            return False
            
        if not self.check_concepts_available():
            logger.error("Cannot build graph: No concepts available")
            return False
            
        try:
            self.graph_builder.build_graph(**kwargs)
            self.visualizer.set_graph(self.graph_builder.graph)
            self.graph_built = True
            return True
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            return False
    
    def generate_visualizations(self, output_dir: str = "visualizations") -> Dict[str, str]:
        """
        Generate standard set of visualizations
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of visualization paths
        """
        if not self.graph_built:
            if not self.build_graph():
                return {"error": "Failed to build graph for visualization"}
        
        try:
            self.visualizer.output_dir = output_dir
            
            results = {}
            
            # Standard network graph visualizations
            if MATPLOTLIB_AVAILABLE:
                results["static"] = self.visualizer.visualize_with_matplotlib(
                    output_file="knowledge_graph.png"
                )
            
            if PLOTLY_AVAILABLE:
                results["interactive"] = self.visualizer.visualize_interactive(
                    output_file="knowledge_graph.html"
                )
            
            # Export in different formats
            results["json"] = self.visualizer.export_to_json()
            results["gexf"] = self.visualizer.export_to_gexf()
            
            # Generate community visualization if available
            try:
                communities = self.graph_builder.get_concept_communities()
                if PLOTLY_AVAILABLE:
                    results["communities"] = self.visualizer.visualize_interactive(
                        output_file="knowledge_graph_communities.html",
                        show_communities=True,
                        communities=communities
                    )
                    
                # Generate community summary
                community_summary = self.graph_builder.get_community_summary(communities)
                
                # Save community summary as JSON
                summary_path = os.path.join(output_dir, "community_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(community_summary, f, indent=2)
                    
                results["community_summary"] = summary_path
            except Exception as e:
                logger.warning(f"Could not generate community visualization: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {"error": str(e)}
    
    def search_and_visualize(self, search_term: str, output_dir: str = "visualizations") -> Dict[str, Any]:
        """
        Search for concepts and visualize the neighborhood
        
        Args:
            search_term: Term to search for in concept names
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with search results and visualization paths
        """
        if not self.check_concepts_available():
            return {"error": "No concepts available"}
            
        # Search for matching concepts
        concepts = self.concept_query.search_concepts(search_term, limit=10)
        
        if not concepts:
            return {
                "error": f"No concepts found matching '{search_term}'",
                "suggestions": self.concept_query.get_top_concepts(5)
            }
        
        # Build graph if not already built
        if not self.graph_built:
            if not self.build_graph():
                return {
                    "error": "Failed to build graph for visualization",
                    "concepts": concepts
                }
        
        # Generate visualizations for top matching concept
        top_concept = concepts[0]
        
        try:
            # Ensure output directory exists
            self.visualizer.output_dir = output_dir
            
            # Generate neighborhood visualization
            viz_results = {}
            if MATPLOTLIB_AVAILABLE:
                viz_results["static"] = self.visualizer.visualize_concept_neighborhood(
                    top_concept["id"],
                    interactive=False
                )
                
            if PLOTLY_AVAILABLE:
                viz_results["interactive"] = self.visualizer.visualize_concept_neighborhood(
                    top_concept["id"],
                    interactive=True
                )
                
            # Get related concepts for top match
            related = self.concept_query.get_related_concepts(top_concept["id"])
            
            # Get content with this concept
            content = self.concept_query.get_content_with_concept(top_concept["id"])
            
            # Analyze concept (if graph is available)
            if self.graph_built and top_concept["id"] in self.graph_builder.graph:
                analysis = self.graph_builder.analyze_concept(top_concept["id"])
            else:
                analysis = None
                
            return {
                "query": search_term,
                "concepts": concepts,
                "top_concept": top_concept,
                "related": related,
                "content": content,
                "analysis": analysis,
                "visualizations": viz_results
            }
            
        except Exception as e:
            logger.error(f"Error in search and visualize: {str(e)}")
            return {
                "error": str(e),
                "concepts": concepts
            }
    
    def get_concept_report(self, concept_id: int) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a specific concept
        
        Args:
            concept_id: ID of the concept to report on
            
        Returns:
            Dictionary with concept report data
        """
        # Get basic concept info
        concept = self.concept_query.get_concept_by_id(concept_id)
        if not concept:
            return {"error": f"Concept ID {concept_id} not found"}
            
        # Get related concepts
        related = self.concept_query.get_related_concepts(concept_id)
        
        # Get content with this concept
        content = self.concept_query.get_content_with_concept(concept_id)
        
        # Build graph if not already built and analyze concept
        analysis = None
        if self.graph_builder:
            if not self.graph_built:
                self.build_graph()
                
            if self.graph_built and concept_id in self.graph_builder.graph:
                analysis = self.graph_builder.analyze_concept(concept_id)
                
        return {
            "concept": concept,
            "related_concepts": {
                "outgoing": [r for r in related if r["direction"] == "outgoing"],
                "incoming": [r for r in related if r["direction"] == "incoming"]
            },
            "content": content,
            "analysis": analysis
        }
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Dictionary with graph statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count concepts
            cursor.execute("SELECT COUNT(*) FROM concepts")
            stats["concept_count"] = cursor.fetchone()[0]
            
            # Count relationships
            cursor.execute("SELECT COUNT(*) FROM concept_relationships")
            stats["relationship_count"] = cursor.fetchone()[0]
            
            # Count content with concepts
            cursor.execute("SELECT COUNT(DISTINCT content_id) FROM content_concepts")
            stats["content_with_concepts"] = cursor.fetchone()[0]
            
            # Get top categories
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM concepts
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category
                ORDER BY count DESC
                LIMIT 10
            """)
            stats["top_categories"] = [{"category": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Get top relationship types
            cursor.execute("""
                SELECT relationship_type, COUNT(*) as count
                FROM concept_relationships
                GROUP BY relationship_type
                ORDER BY count DESC
                LIMIT 10
            """)
            stats["top_relationship_types"] = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Get most connected concepts
            cursor.execute("""
                SELECT c.id, c.name, c.category,
                    (SELECT COUNT(*) FROM concept_relationships 
                     WHERE source_concept_id = c.id OR target_concept_id = c.id) as connection_count
                FROM concepts c
                ORDER BY connection_count DESC
                LIMIT 10
            """)
            stats["most_connected_concepts"] = [
                {"id": row[0], "name": row[1], "category": row[2], "connections": row[3]} 
                for row in cursor.fetchall()
            ]
            
            # Build graph if we have NetworkX
            if self.graph_builder and not self.graph_built:
                if stats["concept_count"] > 0 and stats["relationship_count"] > 0:
                    self.build_graph()
            
            # Add graph-specific stats if available
            if self.graph_built:
                graph = self.graph_builder.graph
                stats["graph"] = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": nx.density(graph),
                    "is_directed": graph.is_directed(),
                    "is_connected": nx.is_weakly_connected(graph),
                    "avg_clustering": nx.average_clustering(graph.to_undirected()),
                    "avg_shortest_path": -1  # Placeholder, computed below
                }
                
                # Try to compute average shortest path length (can be expensive)
                try:
                    largest_cc = max(nx.weakly_connected_components(graph), key=len)
                    largest_subgraph = graph.subgraph(largest_cc)
                    stats["graph"]["avg_shortest_path"] = nx.average_shortest_path_length(largest_subgraph)
                except:
                    # This can fail for disconnected graphs or very large graphs
                    pass
                
                # Get central concepts
                try:
                    stats["central_concepts"] = {}
                    for centrality_type in ["degree", "pagerank"]:
                        central = self.graph_builder.get_central_concepts(
                            limit=5, 
                            centrality_type=centrality_type
                        )
                        
                        stats["central_concepts"][centrality_type] = [
                            {"id": c["id"], "name": c["name"], "score": c["centrality_score"]}
                            for c in central
                        ]
                except Exception as e:
                    logger.warning(f"Could not compute centrality: {str(e)}")
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting graph stats: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {"error": str(e)}
        finally:
            conn.close()


def run_concept_extraction(content_id=None, batch=False, batch_size=10, 
                        max_items=None, source_type=None, force=False,
                        db_path=None):
    """
    Run concept extraction from the command line
    
    Args:
        content_id: Specific content ID to process
        batch: Process in batch mode
        batch_size: Size of batches
        max_items: Maximum items to process
        source_type: Type of content to process
        force: Force reprocessing of already processed content
        db_path: Database path
        
    Returns:
        Success or failure message
    """
    from concept_extractor import extract_and_store_from_content, batch_extract_concepts
    
    if content_id:
        # Process a single content item
        success = extract_and_store_from_content(content_id, db_path)
        if success:
            return f"Successfully extracted concepts from content ID {content_id}"
        else:
            return f"Failed to extract concepts from content ID {content_id}"
    
    if batch:
        # Process in batch mode
        count = batch_extract_concepts(
            batch_size=batch_size,
            max_items=max_items,
            source_type=source_type,
            force_update=force,
            db_path=db_path
        )
        return f"Successfully processed {count} content items"
        
    return "No action specified. Use --content-id or --batch."


# Basic usage example if run directly
if __name__ == "__main__":
    import argparse
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Knowledge Graph tools for AI concepts")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Query commands
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--concepts", action="store_true", help="List top concepts")
    query_parser.add_argument("--categories", action="store_true", help="List concept categories")
    query_parser.add_argument("--category", type=str, help="Show concepts in category")
    query_parser.add_argument("--relationships", action="store_true", help="List relationship types")
    query_parser.add_argument("--relationship-type", type=str, help="Show relationships of this type")
    query_parser.add_argument("--search", type=str, help="Search concepts by name/description")
    query_parser.add_argument("--concept-id", type=int, help="Show concept details by ID")
    query_parser.add_argument("--concept-name", type=str, help="Show concept details by name")
    query_parser.add_argument("--related", type=int, help="Show concepts related to a concept ID")
    query_parser.add_argument("--content", type=int, help="Show content with a concept ID")
    query_parser.add_argument("--limit", type=int, default=20, help="Limit results")
    
    # Extract commands
    extract_parser = subparsers.add_parser("extract", help="Extract concepts from content")
    extract_parser.add_argument("--content-id", type=int, help="Extract concepts from specific content item")
    extract_parser.add_argument("--batch", action="store_true", help="Process in batch mode")
    extract_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    extract_parser.add_argument("--max-items", type=int, help="Maximum number of items to process")
    extract_parser.add_argument("--source-type", 
                              choices=["research_paper", "github", "instagram"],
                              help="Only process items of this source type")
    extract_parser.add_argument("--force", action="store_true", 
                              help="Process items even if they already have concepts")
    
    # Visualize commands
    viz_parser = subparsers.add_parser("visualize", help="Visualize the knowledge graph")
    viz_parser.add_argument("--output-dir", type=str, default="visualizations", 
                          help="Output directory for visualizations")
    viz_parser.add_argument("--concept-id", type=int, help="Visualize neighborhood of this concept")
    viz_parser.add_argument("--search", type=str, help="Search and visualize concepts")
    viz_parser.add_argument("--all", action="store_true", help="Generate all standard visualizations")
    viz_parser.add_argument("--min-confidence", type=float, default=0.5, 
                          help="Minimum confidence for relationships")
    viz_parser.add_argument("--min-references", type=int, default=1, 
                          help="Minimum reference count for concepts")
    viz_parser.add_argument("--interactive", action="store_true", 
                          help="Generate interactive visualizations")
    viz_parser.add_argument("--static", action="store_true", 
                          help="Generate static visualizations")
    viz_parser.add_argument("--format", choices=["png", "html", "json", "gexf", "all"], 
                          default="all", help="Output format")
    
    # Analysis commands
    analysis_parser = subparsers.add_parser("analyze", help="Analyze the knowledge graph")
    analysis_parser.add_argument("--stats", action="store_true", help="Show knowledge graph statistics")
    analysis_parser.add_argument("--concept-id", type=int, help="Analyze a specific concept")
    analysis_parser.add_argument("--communities", action="store_true", help="Detect and analyze communities")
    analysis_parser.add_argument("--centrality", action="store_true", help="Analyze concept centrality")
    analysis_parser.add_argument("--output", type=str, help="Output file for analysis results (JSON)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize manager
    manager = KnowledgeGraphManager()
    
    # Handle commands
    if args.command == "query":
        concept_query = ConceptQuery()
        rel_query = RelationshipQuery()
        
        # Basic query commands (from original if __name__ == "__main__" block)
        if args.concepts:
            concepts = concept_query.get_top_concepts(limit=args.limit)
            print(f"Top {len(concepts)} concepts:")
            for concept in concepts:
                print(f"- {concept['name']} ({concept['category']}) - {concept['reference_count']} references")
                
        elif args.categories:
            categories = concept_query.get_concept_categories()
            print(f"Concept categories ({len(categories)}):")
            for category in categories:
                print(f"- {category}")
                
        elif args.category:
            concepts = concept_query.get_concepts_by_category(args.category, limit=args.limit)
            print(f"Concepts in category '{args.category}' ({len(concepts)}):")
            for concept in concepts:
                print(f"- {concept['name']} - {concept['reference_count']} references")
                
        elif args.relationships:
            rel_types = rel_query.get_relationship_types()
            print(f"Relationship types ({len(rel_types)}):")
            for rel_type in rel_types:
                print(f"- {rel_type}")
                
        elif args.relationship_type:
            relationships = rel_query.get_relationships_by_type(args.relationship_type, limit=args.limit)
            print(f"Relationships of type '{args.relationship_type}' ({len(relationships)}):")
            for rel in relationships:
                print(f"- {rel['source_name']} -> {rel['target_name']} (confidence: {rel['confidence_score']:.2f})")
                
        elif args.search:
            concepts = concept_query.search_concepts(args.search, limit=args.limit)
            print(f"Search results for '{args.search}' ({len(concepts)}):")
            for concept in concepts:
                print(f"- {concept['name']} ({concept['category']}) - {concept['reference_count']} references")
                print(f"  {concept['description'][:100]}..." if concept['description'] else "  No description")
                
        elif args.concept_id:
            concept = concept_query.get_concept_by_id(args.concept_id)
            if concept:
                print(f"Concept ID {args.concept_id}:")
                print(f"Name: {concept['name']}")
                print(f"Category: {concept['category']}")
                print(f"Description: {concept['description']}")
                print(f"References: {concept['reference_count']}")
                print(f"First seen: {concept['first_seen_date']}")
                print(f"Last updated: {concept['last_updated']}")
            else:
                print(f"Concept ID {args.concept_id} not found")
                
        elif args.concept_name:
            concept = concept_query.get_concept_by_name(args.concept_name)
            if concept:
                print(f"Concept '{args.concept_name}':")
                print(f"ID: {concept['id']}")
                print(f"Category: {concept['category']}")
                print(f"Description: {concept['description']}")
                print(f"References: {concept['reference_count']}")
                print(f"First seen: {concept['first_seen_date']}")
                print(f"Last updated: {concept['last_updated']}")
            else:
                print(f"Concept '{args.concept_name}' not found")
                
        elif args.related:
            related = concept_query.get_related_concepts(args.related)
            concept = concept_query.get_concept_by_id(args.related)
            if concept:
                print(f"Concepts related to '{concept['name']}' ({len(related)}):")
                outgoing = [r for r in related if r["direction"] == 'outgoing']
                incoming = [r for r in related if r["direction"] == 'incoming']
                
                if outgoing:
                    print("\nOutgoing relationships:")
                    for rel in outgoing:
                        print(f"- {concept['name']} -[{rel['relationship']}]-> {rel['name']} (confidence: {rel['confidence']:.2f})")
                        
                if incoming:
                    print("\nIncoming relationships:")
                    for rel in incoming:
                        print(f"- {rel['name']} -[{rel['relationship']}]-> {concept['name']} (confidence: {rel['confidence']:.2f})")
        else:
            query_parser.print_help()
    
    elif args.command == "extract":
        # Run concept extraction
        result = run_concept_extraction(
            content_id=args.content_id,
            batch=args.batch,
            batch_size=args.batch_size,
            max_items=args.max_items,
            source_type=args.source_type,
            force=args.force
        )
        print(result)
    
    elif args.command == "visualize":
        # Check if we have visualization dependencies
        if not NETWORKX_AVAILABLE:
            print("Error: NetworkX not available. Install with 'pip install networkx'")
            sys.exit(1)
            
        if args.interactive and not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Install with 'pip install plotly'")
            args.interactive = False
            
        if args.static and not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Install with 'pip install matplotlib'")
            args.static = False
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set up KnowledgeGraph instance with parameters
        graph_params = {
            "min_confidence": args.min_confidence,
            "min_references": args.min_references
        }
        
        # Create visualizations
        if args.concept_id:
            # Visualize a specific concept neighborhood
            manager.build_graph(**graph_params)
            
            # Create the visualizer
            visualizer = GraphVisualizer(manager.graph_builder.graph, args.output_dir)
            
            # Generate visualizations
            if args.static or not args.interactive:
                static_path = visualizer.visualize_concept_neighborhood(
                    args.concept_id, 
                    interactive=False
                )
                print(f"Generated static visualization: {static_path}")
                
            if args.interactive or not args.static:
                interactive_path = visualizer.visualize_concept_neighborhood(
                    args.concept_id, 
                    interactive=True
                )
                print(f"Generated interactive visualization: {interactive_path}")
                
        elif args.search:
            # Search and visualize concepts
            result = manager.search_and_visualize(args.search, args.output_dir)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                if "suggestions" in result:
                    print("\nSuggested concepts:")
                    for concept in result["suggestions"]:
                        print(f"- {concept['name']} ({concept['category']})")
            else:
                print(f"Found {len(result['concepts'])} concepts matching '{args.search}'")
                print(f"Top match: {result['top_concept']['name']} ({result['top_concept']['category']})")
                
                if "visualizations" in result:
                    print("\nVisualizations:")
                    for viz_type, path in result["visualizations"].items():
                        print(f"- {viz_type}: {path}")
        
        elif args.all:
            # Generate standard visualizations
            print("Generating standard visualizations...")
            results = manager.generate_visualizations(args.output_dir)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print("\nGenerated visualizations:")
                for viz_type, path in results.items():
                    print(f"- {viz_type}: {path}")
        
        else:
            viz_parser.print_help()
    
    elif args.command == "analyze":
        if args.stats:
            # Show knowledge graph statistics
            stats = manager.get_knowledge_graph_stats()
            
            if "error" in stats:
                print(f"Error: {stats['error']}")
            else:
                print("\nKnowledge Graph Statistics:")
                print(f"Concepts: {stats['concept_count']}")
                print(f"Relationships: {stats['relationship_count']}")
                print(f"Content with concepts: {stats['content_with_concepts']}")
                
                print("\nTop Categories:")
                for cat in stats["top_categories"]:
                    print(f"- {cat['category']}: {cat['count']} concepts")
                
                print("\nTop Relationship Types:")
                for rel in stats["top_relationship_types"]:
                    print(f"- {rel['type']}: {rel['count']} relationships")
                
                print("\nMost Connected Concepts:")
                for concept in stats["most_connected_concepts"]:
                    print(f"- {concept['name']} ({concept['category']}): {concept['connections']} connections")
                
                if "graph" in stats:
                    print("\nGraph Properties:")
                    for key, value in stats["graph"].items():
                        if key != "avg_shortest_path" or value != -1:
                            print(f"- {key}: {value}")
                            
                if "central_concepts" in stats:
                    print("\nMost Central Concepts:")
                    for centrality_type, concepts in stats["central_concepts"].items():
                        print(f"\nBy {centrality_type} centrality:")
                        for c in concepts:
                            print(f"- {c['name']}: {c['score']:.4f}")
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"\nSaved statistics to {args.output}")
        
        elif args.concept_id:
            # Analyze a specific concept
            report = manager.get_concept_report(args.concept_id)
            
            if "error" in report:
                print(f"Error: {report['error']}")
            else:
                concept = report["concept"]
                print(f"\nConcept: {concept['name']} (ID: {concept['id']})")
                print(f"Category: {concept['category']}")
                print(f"References: {concept['reference_count']}")
                print(f"Description: {concept['description']}")
                
                outgoing = report["related_concepts"]["outgoing"]
                incoming = report["related_concepts"]["incoming"]
                
                print(f"\nRelationships ({len(outgoing) + len(incoming)} total):")
                
                if outgoing:
                    print("\nOutgoing:")
                    for rel in outgoing:
                        print(f"- {concept['name']} -[{rel['relationship']}]-> {rel['name']} ({rel['confidence']:.2f})")
                
                if incoming:
                    print("\nIncoming:")
                    for rel in incoming:
                        print(f"- {rel['name']} -[{rel['relationship']}]-> {concept['name']} ({rel['confidence']:.2f})")
                
                if report["content"]:
                    print(f"\nFound in {len(report['content'])} content items:")
                    for item in report["content"]:
                        print(f"- [{item['source_type']}] {item['title']} (importance: {item['importance']})")
                
                if report["analysis"] and "graph_metrics" in report["analysis"]:
                    metrics = report["analysis"]["graph_metrics"]
                    print("\nGraph Metrics:")
                    print(f"- In-degree: {metrics['in_degree']}")
                    print(f"- Out-degree: {metrics['out_degree']}")
                    print(f"- Total degree: {metrics['total_degree']}")
                    print(f"- Betweenness centrality: {metrics['betweenness_centrality']:.4f}")
                    print(f"- PageRank: {metrics['pagerank']:.4f}")
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(report, f, indent=2)
                    print(f"\nSaved concept report to {args.output}")
        
        elif args.communities:
            # Analyze communities in the graph
            if not manager.build_graph():
                print("Error: Could not build graph for community analysis")
                sys.exit(1)
                
            print("Detecting concept communities...")
            communities = manager.graph_builder.get_concept_communities()
            
            community_summary = manager.graph_builder.get_community_summary(communities)
            print(f"Found {len(community_summary)} communities")
            
            # Display top communities
            for i, comm in enumerate(community_summary[:10]):  # Show top 10
                print(f"\nCommunity {comm['community_id']} ({comm['size']} concepts)")
                print(f"Top concept: {comm['top_concept']}")
                
                print("Category distribution:")
                for category, count in comm['category_distribution'].items():
                    print(f"- {category}: {count}")
                
                print("Key concepts:")
                for concept in sorted(comm['concepts'], key=lambda x: x['reference_count'], reverse=True)[:5]:
                    print(f"- {concept['name']} ({concept['reference_count']} references)")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(community_summary, f, indent=2)
                print(f"\nSaved community analysis to {args.output}")
        
        elif args.centrality:
            # Analyze concept centrality
            if not manager.build_graph():
                print("Error: Could not build graph for centrality analysis")
                sys.exit(1)
                
            centrality_results = {}
            
            for centrality_type in ["degree", "betweenness", "pagerank"]:
                print(f"\nAnalyzing {centrality_type} centrality...")
                try:
                    central_concepts = manager.graph_builder.get_central_concepts(
                        limit=20, 
                        centrality_type=centrality_type
                    )
                    
                    centrality_results[centrality_type] = central_concepts
                    
                    print(f"Top concepts by {centrality_type} centrality:")
                    for i, concept in enumerate(central_concepts[:10]):  # Show top 10
                        print(f"{i+1}. {concept['name']} ({concept['category']}) - {concept['centrality_score']:.4f}")
                except Exception as e:
                    print(f"Error computing {centrality_type} centrality: {str(e)}")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(centrality_results, f, indent=2)
                print(f"\nSaved centrality analysis to {args.output}")
        
        else:
            analysis_parser.print_help()
    
    else:
        parser.print_help() 