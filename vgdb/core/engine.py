"""
VectorGraphEngine orchestrates embeddings, FAISS vector search, and NetworkX graph traversal.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorGraphEngine:
    """
    Cohesive engine that keeps the vector index and graph representation in sync.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_hops: int = 2,
    ) -> None:
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.doc_map: Dict[int, str] = {}
        self.rev_doc_map: Dict[str, int] = {}
        self.graph = nx.DiGraph()
        self.max_hops = max_hops

    # ------------------------------------------------------------------
    # Node & Edge management
    # ------------------------------------------------------------------
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vector)
        return vector

    def embed(self, text: str) -> np.ndarray:
        vec = self.encoder.encode([text])[0].astype("float32")
        return self._normalize(vec.reshape(1, -1))

    def add_node(self, node_id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        vector = self.embed(text)
        self.index.add(vector)
        internal_id = self.index.ntotal - 1
        self.doc_map[internal_id] = node_id
        self.rev_doc_map[node_id] = internal_id
        self.graph.add_node(node_id, text=text, **metadata)

    def add_nodes(self, nodes: Iterable[Tuple[str, str, Dict[str, Any]]]) -> None:
        for node_id, text, metadata in nodes:
            self.add_node(node_id, text, metadata)

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        if source not in self.graph or target not in self.graph:
            raise ValueError("Both source and target must exist before adding an edge.")
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    def add_edges(self, edges: Iterable[Tuple[str, str, str, float]]) -> None:
        for source, target, relation, weight in edges:
            self.add_edge(source, target, relation, weight)

    # ------------------------------------------------------------------
    # Retrieval logic
    # ------------------------------------------------------------------
    def _vector_candidates(self, query: str, limit: int) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.embed(query)
        k = min(limit, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        candidates: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            candidates.append((self.doc_map[idx], float(score)))
        return candidates

    def _graph_expansion(self, anchor: str) -> List[Tuple[str, int]]:
        if anchor not in self.graph:
            return []
        visited = set([anchor])
        queue = deque([(anchor, 0)])
        expansions: List[Tuple[str, int]] = []
        while queue:
            node_id, depth = queue.popleft()
            if depth >= self.max_hops:
                continue
            for neighbor in set(self.graph.successors(node_id)).union(self.graph.predecessors(node_id)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_depth = depth + 1
                expansions.append((neighbor, next_depth))
                queue.append((neighbor, next_depth))
        return expansions

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        alpha controls the influence of vector vs. graph signals.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        vector_hits = self._vector_candidates(query, limit=max(top_k * 2, 10))
        scores: Dict[str, Dict[str, Any]] = {}

        for node_id, vector_score in vector_hits:
            base = vector_score * alpha
            scores[node_id] = {
                "score": base,
                "vector_score": vector_score,
                "graph_boost": 0.0,
                "reason": ["semantic anchor"],
            }

        # Graph expansion from vector anchors
        for anchor_id, _ in vector_hits[:top_k]:
            expansions = self._graph_expansion(anchor_id)
            for neighbor_id, hop in expansions:
                decay = 1 / (hop + 1)
                boost = (1 - alpha) * decay
                if neighbor_id in scores:
                    scores[neighbor_id]["score"] += boost
                    scores[neighbor_id]["graph_boost"] += boost
                    scores[neighbor_id]["reason"].append(f"graph hop {hop} from {anchor_id}")
                else:
                    scores[neighbor_id] = {
                        "score": boost,
                        "vector_score": 0.0,
                        "graph_boost": boost,
                        "reason": [f"graph hop {hop} from {anchor_id}"],
                    }

        ranked = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for node_id, meta in ranked:
            node_data = self.graph.nodes.get(node_id, {})
            results.append(
                {
                    "id": node_id,
                    "text": node_data.get("text", ""),
                    "metadata": {k: v for k, v in node_data.items() if k != "text"},
                    "score": round(meta["score"], 4),
                    "reason": " + ".join(meta["reason"]),
                    "vector_score": round(meta["vector_score"], 4),
                    "graph_boost": round(meta["graph_boost"], 4),
                }
            )
        return results


