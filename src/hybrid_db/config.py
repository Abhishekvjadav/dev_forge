"""
Central configuration for the hybrid database.
"""
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")
    db_path: Path = data_dir / "hybrid.db"
    snapshot_dir: Path = data_dir / "snapshots"
    embedding_dim: int = 384
    default_vector_weight: float = 0.6
    default_graph_weight: float = 0.4
    max_depth: int = 3
    top_k: int = 5


settings = Settings()


def ensure_directories() -> None:
    """
    Create folders for database and snapshots if missing.
    """
    settings.data_dir.mkdir(exist_ok=True)
    settings.snapshot_dir.mkdir(exist_ok=True)


