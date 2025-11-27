"""
Typer-powered CLI for ingesting data and running searches locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.table import Table

from .search import SearchService
from .storage import GraphVectorStore

app = typer.Typer(add_completion=False, help="Hybrid Vector + Graph DB CLI")
store = GraphVectorStore()
search = SearchService(store)


@app.command()
def ingest(file: Path = typer.Argument(..., help="JSON file with nodes array")):
    """
    Load nodes from a JSON file: [{text: ..., metadata: {...}}]
    """
    data = json.loads(file.read_text(encoding="utf-8"))
    nodes = data if isinstance(data, list) else data.get("nodes", [])
    for entry in nodes:
        text = entry["text"]
        metadata = entry.get("metadata", {})
        node = search.add_node(text, metadata)
        print(f"[green]Added node {node.id}[/green]")


@app.command()
def vector(query: str, top_k: int = 5):
    """
    Run vector search for the given query.
    """
    results = search.vector_search(query, top_k=top_k)
    table = Table("Rank", "Node ID", "Score", "Snippet")
    for idx, result in enumerate(results, start=1):
        table.add_row(str(idx), result.node.id, f"{result.score:.3f}", result.node.text[:60])
    print(table)


@app.command()
def graph(start_id: str, depth: int = 2):
    """
    Traverse graph from start node up to depth.
    """
    results = search.graph_traversal(start_id, depth)
    table = Table("Node ID", "Depth", "Path")
    for res in results:
        table.add_row(res.node.id, str(res.depth), " -> ".join(res.path))
    print(table)


@app.command()
def hybrid(query: str, vector_weight: float = 0.6, graph_weight: float = 0.4, top_k: int = 5):
    """
    Hybrid search combining vector + graph signals.
    """
    results = search.hybrid_search(
        query=query,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        top_k=top_k,
    )
    table = Table("Rank", "Node ID", "Vector", "Graph", "Combined")
    for idx, res in enumerate(results, start=1):
        table.add_row(
            str(idx),
            res.node.id,
            f"{res.vector_score:.3f}",
            f"{res.graph_score:.3f}",
            f"{res.combined_score:.3f}",
        )
    print(table)


if __name__ == "__main__":
    app()


