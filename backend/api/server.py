"""
FastAPI adapter for the existing compiler optimization backend.

Run from the project root:
    uvicorn backend.api.server:app --reload --port 8000

The four original routes are unchanged in shape.  Each accepts two new
*optional* request fields:

    selector      "rule" (default) or "ml" - who chooses the strategy
    include_shap  attach SHAP explanations to ML-driven iteration traces

An explicit ``strategy`` still pins a pass and always wins over ``selector``.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.api.ml_routes import router as ml_router
from backend.api.pipeline import (
    TacParseError,
    analyze_source,
    optimize_source,
    simulate_source,
)
from backend.ml.model import ModelUnavailableError

app = FastAPI(
    title="Compiler Intelligence System API",
    description="REST adapter over the existing TAC optimization engine, its rule-based selector and the trained Random Forest selector.",
    version="2.0.0",
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

app.include_router(ml_router)


class TacRequest(BaseModel):
    """Original request contract plus the optional selector controls."""

    # `forbid` keeps stray client fields from being silently accepted, while
    # the three original fields keep their exact names and defaults.
    model_config = ConfigDict(extra="forbid")

    tac: str = Field(..., description="Three Address Code source")
    strategy: str | None = Field(
        default=None,
        description="Optional optimizer id for a single pass; takes precedence over `selector`",
    )
    max_iterations: int = Field(default=20, ge=1, le=20)
    selector: str = Field(
        default="rule",
        pattern="^(rule|ml)$",
        description="Who chooses the strategy: 'rule' (deterministic scorer) or 'ml' (trained forest)",
    )
    include_shap: bool = Field(
        default=True,
        description="Attach per-iteration SHAP explanations when selector='ml'",
    )
    include_ml: bool = Field(
        default=True,
        description="Include the forest's prediction alongside the rule-based analysis",
    )


def _guard(exc: Exception, fallback: str) -> None:
    """Translate known failures into stable, non-leaky HTTP errors."""
    if isinstance(exc, TacParseError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, ModelUnavailableError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": str(exc),
                "selector": "ml",
                "remedy": "python -m backend.ml.train",
            },
        ) from exc
    # Anything else becomes a generic message: no paths, no tracebacks.
    raise HTTPException(status_code=500, detail=fallback) from None


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
        return analyze_source(req.tac, include_ml=req.include_ml)
    except Exception as exc:
        _guard(exc, "Optimization analysis failed.")


@app.post("/api/optimize")
def optimize(req: TacRequest):
    try:
        return optimize_source(
            req.tac,
            strategy=req.strategy,
            selector=req.selector,
            include_ml=req.include_ml,
        )
    except Exception as exc:
        _guard(exc, "Optimization pass failed.")


@app.post("/api/simulate")
def simulate(req: TacRequest):
    try:
        return simulate_source(
            req.tac,
            max_iterations=req.max_iterations,
            selector=req.selector,
            include_shap=req.include_shap,
        )
    except Exception as exc:
        _guard(exc, "Optimization simulation failed.")
