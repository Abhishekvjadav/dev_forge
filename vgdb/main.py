"""
FastAPI entrypoint exposing hybrid search alongside a static demo UI.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ingest import DB_FILE, ingest

app = FastAPI(title="Vector + Graph Native DB", version="0.2.0")


def load_engine():
    path = Path(DB_FILE)
    if not path.exists():
        ingest()
    with path.open("rb") as f:
        return pickle.load(f)


engine = load_engine()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-form natural language query")
    top_k: int = Field(5, ge=1, le=20)
    alpha: float = Field(0.7, ge=0.0, le=1.0)


@app.post("/search", response_model=dict)
def hybrid_search(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    results = engine.hybrid_search(payload.query, top_k=payload.top_k, alpha=payload.alpha)
    return {"results": results}


app.mount("/", StaticFiles(directory="vgdb/static", html=True), name="static")


