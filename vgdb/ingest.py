"""
Data ingestion script that bootstraps a Marvel-inspired knowledge graph.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List

from .core.engine import VectorGraphEngine

DB_FILE = Path("vgdb/data/marvel_db.pkl")


def sample_dataset() -> Dict[str, List[Dict]]:
    nodes = [
        {
            "id": "iron_man",
            "text": "Tony Stark is a genius billionaire and the armored Avenger Iron Man.",
            "metadata": {"type": "hero", "affiliation": "Avengers"},
        },
        {
            "id": "captain_america",
            "text": "Steve Rogers is Captain America, a super soldier wielding an indestructible shield.",
            "metadata": {"type": "hero", "affiliation": "Avengers"},
        },
        {
            "id": "thor",
            "text": "Thor is the Asgardian god of thunder, wielder of Mjolnir and Stormbreaker.",
            "metadata": {"type": "hero", "affiliation": "Asgard"},
        },
        {
            "id": "thanos",
            "text": "Thanos is the Mad Titan seeking to balance the universe using the Infinity Stones.",
            "metadata": {"type": "villain", "affiliation": "Titans"},
        },
        {
            "id": "infinity_gauntlet",
            "text": "The Infinity Gauntlet channels the combined power of all six Infinity Stones.",
            "metadata": {"type": "artifact"},
        },
        {
            "id": "avengers_endgame",
            "text": "Avengers: Endgame chronicles the final stand against Thanos to reverse the snap.",
            "metadata": {"type": "movie"},
        },
    ]
    edges = [
        {"source": "iron_man", "target": "thanos", "relation": "FOUGHT"},
        {"source": "captain_america", "target": "thanos", "relation": "FOUGHT"},
        {"source": "thor", "target": "thanos", "relation": "FOUGHT"},
        {"source": "thanos", "target": "infinity_gauntlet", "relation": "WIELDS"},
        {"source": "iron_man", "target": "captain_america", "relation": "FRIEND_OF"},
        {"source": "thor", "target": "captain_america", "relation": "ALLY_OF"},
        {"source": "iron_man", "target": "avengers_endgame", "relation": "APPEARED_IN"},
        {"source": "captain_america", "target": "avengers_endgame", "relation": "APPEARED_IN"},
        {"source": "thanos", "target": "avengers_endgame", "relation": "ANTAGONIST_OF"},
    ]
    return {"nodes": nodes, "edges": edges}


def ingest(dataset: Dict[str, List[Dict]] | None = None, persist_path: Path | None = None) -> VectorGraphEngine:
    dataset = dataset or sample_dataset()
    engine = VectorGraphEngine()
    for node in dataset["nodes"]:
        engine.add_node(node["id"], node["text"], node["metadata"])
    for edge in dataset["edges"]:
        engine.add_edge(edge["source"], edge["target"], edge["relation"])
    output_path = persist_path or DB_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(engine, f)
    return engine


if __name__ == "__main__":
    engine = ingest()
    print(f"✅ Hybrid database saved to {DB_FILE.resolve()}")


