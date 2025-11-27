"""
Persistence layer built on top of SQLite.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import settings, ensure_directories
from .models import Edge, Node


class GraphVectorStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        ensure_directories()
        self.db_path = db_path or settings.db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT,
                weight REAL,
                FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
            """
        )
        self.conn.commit()

    def upsert_node(self, node: Node, vector: Optional[List[float]] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO nodes (id, text, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                node.id,
                node.text,
                json.dumps(node.metadata),
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
            ),
        )
        if vector is not None:
            cur.execute(
                """
                INSERT INTO embeddings (node_id, vector)
                VALUES (?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    vector=excluded.vector
                """,
                (node.id, json.dumps(vector)),
            )
        self.conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes WHERE id=?", (node_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Node(
            id=row["id"],
            text=row["text"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list_nodes(self) -> List[Node]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes")
        nodes = []
        for row in cur.fetchall():
            nodes.append(
                Node(
                    id=row["id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
            )
        return nodes

    def delete_node(self, node_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self.conn.commit()

    def upsert_edge(self, edge: Edge) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO edges (id, source_id, target_id, type, weight)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id,
                target_id=excluded.target_id,
                type=excluded.type,
                weight=excluded.weight
            """,
            (edge.id, edge.source_id, edge.target_id, edge.type, edge.weight),
        )
        self.conn.commit()

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges WHERE id=?", (edge_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=row["type"],
            weight=row["weight"],
        )

    def delete_edge(self, edge_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM edges WHERE id=?", (edge_id,))
        self.conn.commit()

    def list_edges(self) -> List[Edge]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges")
        return [
            Edge(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                type=row["type"],
                weight=row["weight"],
            )
            for row in cur.fetchall()
        ]

    def node_embeddings(self) -> Dict[str, List[float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT node_id, vector FROM embeddings")
        return {row["node_id"]: json.loads(row["vector"]) for row in cur.fetchall()}

    def neighbors(self, node_id: str) -> List[Edge]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM edges WHERE source_id=? OR target_id=?",
            (node_id, node_id),
        )
        return [
            Edge(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                type=row["type"],
                weight=row["weight"],
            )
            for row in cur.fetchall()
        ]

    def snapshot(self, name: str) -> Path:
        snapshot_path = settings.snapshot_dir / f"{name}.db"
        self.conn.commit()
        dest = sqlite3.connect(snapshot_path)
        self.conn.backup(dest)
        dest.close()
        return snapshot_path

    def load_snapshot(self, snapshot_path: Path) -> None:
        if not snapshot_path.exists():
            raise FileNotFoundError(snapshot_path)
        self.conn.close()
        snapshot_conn = sqlite3.connect(snapshot_path)
        dest = sqlite3.connect(self.db_path)
        snapshot_conn.backup(dest)
        snapshot_conn.close()
        dest.close()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row


