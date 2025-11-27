"""
Data models for nodes, edges, and search results.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    return uuid4().hex


class Node(BaseModel):
    id: str = Field(default_factory=generate_id)
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Edge(BaseModel):
    id: str = Field(default_factory=generate_id)
    source_id: str
    target_id: str
    type: str = "related"
    weight: float = 1.0


class VectorSearchResult(BaseModel):
    node: Node
    score: float


class GraphTraversalResult(BaseModel):
    node: Node
    depth: int
    path: List[str]


class HybridSearchResult(BaseModel):
    node: Node
    vector_score: float
    graph_score: float
    combined_score: float
    extra: Optional[Dict[str, str]] = None


