"""
FastAPI adapter for the existing compiler optimization backend.

Run from the project root:
    uvicorn backend.api.server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.api.pipeline import TacParseError, analyze_source, optimize_source, simulate_source

app = FastAPI(
    title="Compiler Intelligence System API",
    description="REST adapter over the existing rule-based TAC optimization engine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TacRequest(BaseModel):
    tac: str = Field(..., description="Three Address Code source")
    strategy: str | None = Field(default=None, description="Optional optimizer id for a single pass")
    max_iterations: int = Field(default=20, ge=1, le=20)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "system": "COMPILER INTELLIGENCE SYSTEM",
        "mode": "rule-based",
        "ready": True,
    }


@app.post("/api/analyze")
def analyze(req: TacRequest):
    try:
        return analyze_source(req.tac)
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Optimization analysis failed.") from None


@app.post("/api/optimize")
def optimize(req: TacRequest):
    try:
        return optimize_source(req.tac, strategy=req.strategy)
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Optimization pass failed.") from None


@app.post("/api/simulate")
def simulate(req: TacRequest):
    try:
        return simulate_source(req.tac, max_iterations=req.max_iterations)
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Optimization simulation failed.") from None
