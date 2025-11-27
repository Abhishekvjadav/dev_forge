"""
Search utilities for vector, graph, and hybrid retrieval.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

from .config import settings
from .embeddings import cosine_similarity, embedding_model
from .models import GraphTraversalResult, HybridSearchResult, Node, VectorSearchResult
from .storage import GraphVectorStore


class SearchService:
    def __init__(self, store: GraphVectorStore) -> None:
        self.store = store

    def add_node(self, text: str, metadata: Dict[str, str]) -> Node:
        node = Node(text=text, metadata=metadata)
        vector = embedding_model.embed(text)
        self.store.upsert_node(node, vector)
        return node

    def update_node(
        self,
        node_id: str,
        text: str,
        metadata: Dict[str, str],
        regenerate_embedding: bool = False,
    ) -> Node:
        existing = self.store.get_node(node_id)
        if not existing:
            raise ValueError("Node does not exist")
        updated = Node(
            id=existing.id,
            text=text,
            metadata=metadata,
            created_at=existing.created_at,
        )
        if regenerate_embedding or text != existing.text:
            vector = embedding_model.embed(text)
        else:
            vector = self.store.node_embeddings().get(node_id)
        self.store.upsert_node(updated, vector)
        return updated

    def vector_search(self, query: str, top_k: int | None = None) -> List[VectorSearchResult]:
        query_vec = embedding_model.embed(query)
        embeddings = self.store.node_embeddings()
        scored: List[VectorSearchResult] = []
        for node_id, vector in embeddings.items():
            score = cosine_similarity(query_vec, vector)
            node = self.store.get_node(node_id)
            if node:
                scored.append(VectorSearchResult(node=node, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        if top_k is None:
            return scored
        return scored[:top_k]

    def graph_traversal(self, start_id: str, depth: int | None = None) -> List[GraphTraversalResult]:
        depth = depth or settings.max_depth
        visited = set([start_id])
        queue = deque([(start_id, 0, [start_id])])
        results: List[GraphTraversalResult] = []
        while queue:
            node_id, current_depth, path = queue.popleft()
            if current_depth > depth:
                continue
            node = self.store.get_node(node_id)
            if node and current_depth > 0:
                results.append(
                    GraphTraversalResult(node=node, depth=current_depth, path=path.copy())
                )
            if current_depth == depth:
                continue
            for edge in self.store.neighbors(node_id):
                neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, current_depth + 1, path + [neighbor_id]))
        return results

    def hybrid_search(
        self,
        query: str,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        top_k: int | None = None,
    ) -> List[HybridSearchResult]:
        vector_weight = vector_weight or settings.default_vector_weight
        graph_weight = graph_weight or settings.default_graph_weight
        top_k = top_k or settings.top_k
        vector_results = self.vector_search(query, top_k=None)
        # build adjacency info for quick lookup
        adjacency_scores: Dict[str, float] = {}
        for vec_result in vector_results[:top_k * 2]:
            traversals = self.graph_traversal(vec_result.node.id, depth=2)
            for traverse in traversals:
                bonus = 1.0 / (traverse.depth + 1)
                adjacency_scores[traverse.node.id] = max(
                    adjacency_scores.get(traverse.node.id, 0.0), bonus
                )
        merged: Dict[str, Tuple[VectorSearchResult, float]] = {}
        for vec_result in vector_results:
            merged[vec_result.node.id] = (vec_result, adjacency_scores.get(vec_result.node.id, 0.0))
        for node_id, graph_score in adjacency_scores.items():
            if node_id not in merged:
                node = self.store.get_node(node_id)
                if node:
                    merged[node_id] = (
                        VectorSearchResult(node=node, score=0.0),
                        graph_score,
                    )

        combined: List[HybridSearchResult] = []
        for node_id, (vec_res, graph_score) in merged.items():
            combined_score = vector_weight * vec_res.score + graph_weight * graph_score
            combined.append(
                HybridSearchResult(
                    node=vec_res.node,
                    vector_score=vec_res.score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                )
            )
        combined.sort(key=lambda r: r.combined_score, reverse=True)
        return combined[:top_k]


