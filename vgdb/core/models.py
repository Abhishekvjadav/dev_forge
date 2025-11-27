"""
Pydantic models for nodes, edges, and search responses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class NodePayload(BaseModel):
    id: str = Field(..., description="Stable identifier (e.g., slug)")
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgePayload(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    reason: str
    vector_score: float
    graph_boost: float


