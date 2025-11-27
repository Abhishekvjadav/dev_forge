"""
FastAPI service exposing CRUD, search, and hybrid retrieval.
"""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException

from .config import settings
from .models import Edge, Node
from .search import SearchService
from .storage import GraphVectorStore

app = FastAPI(title="Hybrid Vector + Graph DB", version="0.1.0")
store = GraphVectorStore()
search_service = SearchService(store)


@app.post("/nodes", response_model=Node)
def create_node(payload: Dict[str, object]):
    text = payload.get("text")
    metadata = payload.get("metadata", {})
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    text = text.strip()
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    node = search_service.add_node(text=text, metadata=metadata)
    return node


@app.get("/nodes/{node_id}", response_model=Node)
def read_node(node_id: str):
    node = store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.put("/nodes/{node_id}", response_model=Node)
def update_node(node_id: str, payload: Dict[str, object]):
    node = store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    text = payload.get("text", node.text)
    if isinstance(text, str):
        text = text.strip()
    metadata = payload.get("metadata", node.metadata)
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    regenerate = payload.get("regenerate_embedding", False)
    try:
        updated = search_service.update_node(
            node_id=node_id,
            text=text,
            metadata=metadata,
            regenerate_embedding=regenerate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated


@app.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    store.delete_node(node_id)
    return {"status": "ok"}


@app.post("/edges", response_model=Edge)
def create_edge(edge: Edge):
    if not (store.get_node(edge.source_id) and store.get_node(edge.target_id)):
        raise HTTPException(status_code=400, detail="Source or target does not exist")
    store.upsert_edge(edge)
    return edge


@app.get("/edges/{edge_id}", response_model=Edge)
def read_edge(edge_id: str):
    edge = store.get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge


@app.delete("/edges/{edge_id}")
def delete_edge(edge_id: str):
    store.delete_edge(edge_id)
    return {"status": "ok"}


@app.post("/search/vector")
def vector_search(payload: Dict[str, object]):
    query = str(payload.get("query_text", ""))
    raw_top_k = payload.get("top_k")
    top_k = int(raw_top_k) if raw_top_k is not None else settings.top_k
    results = search_service.vector_search(query, top_k=top_k)
    return results


@app.get("/search/graph")
def graph_search(start_id: str, depth: Optional[int] = None):
    if not store.get_node(start_id):
        raise HTTPException(status_code=404, detail="Start node missing")
    return search_service.graph_traversal(start_id, depth)


@app.post("/search/hybrid")
def hybrid_search(payload: Dict[str, object]):
    query = str(payload.get("query_text", ""))
    vector_weight = float(payload.get("vector_weight", settings.default_vector_weight))
    graph_weight = float(payload.get("graph_weight", settings.default_graph_weight))
    top_k = int(payload.get("top_k", settings.top_k))
    return search_service.hybrid_search(
        query=query,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        top_k=top_k,
    )


