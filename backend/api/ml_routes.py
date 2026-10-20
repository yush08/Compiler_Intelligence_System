"""FastAPI router for ML, comparison and dataset endpoints.

Added alongside the four original routes so the existing contract is untouched:
``/api/health``, ``/api/analyze``, ``/api/optimize`` and ``/api/simulate`` keep
their request/response shape and gain only optional fields.

Error policy
------------
* ``400`` - the TAC could not be parsed (bad caller input).
* ``503`` - an endpoint that *must* execute an ML decision was asked to do so
  with no trained artifact.  The body names the missing capability and the
  command that fixes it.
* ``200`` with ``available: false`` - read-only ML views (status, metrics,
  predict, explain) and ``/api/compare`` degrade explicitly so the UI can render
  a designed empty state.  ``/api/compare`` still returns the full Rule-Based
  side in that case; it simply declares no winner.
* ``500`` - unexpected failure, with a generic message. Internal paths and raw
  tracebacks never cross the API boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..ml.compare import compare_source
from ..ml.model import ModelUnavailableError, MODEL_DIR, model_status
from ..ml.schemas import (
    CompareRequest,
    CompareResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureImportanceResponse,
    MlMetricsResponse,
    MlStatusResponse,
    PredictResponse,
)
from . import dataset_service
from . import matrix_service
from .pipeline import TacParseError

router = APIRouter(prefix="/api")

ARTIFACT_DIR = MODEL_DIR


def _read_artifact(name: str) -> dict:
    """Load a JSON artifact, returning an explicit unavailable state."""
    path = ARTIFACT_DIR / name
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "available": False,
            "reason": f"Artifact {name} is missing. Run: python -m backend.ml.train",
        }
    except ValueError:
        return {
            "available": False,
            "reason": f"Artifact {name} is unreadable. Retrain with: python -m backend.ml.train",
        }
    payload.setdefault("available", True)
    return payload


def _parse(source: str):
    from .pipeline import parse_source

    return parse_source(source)


# --- model status / metrics -------------------------------------------------

@router.get("/ml/status", response_model=MlStatusResponse)
def ml_status() -> dict:
    """Model readiness, version, supported classes and SHAP availability."""
    status = model_status()

    from ..ml.predict import unsupported_classes

    try:
        from ..ml.model import load_payload

        status["unsupported_classes"] = unsupported_classes(load_payload())
    except ModelUnavailableError:
        status["unsupported_classes"] = []
    except Exception:  # noqa: BLE001 - status must never fail
        status["unsupported_classes"] = []

    return status


@router.get("/ml/metrics", response_model=MlMetricsResponse)
def ml_metrics() -> dict:
    """Persisted evaluation for all three splits, never recomputed here."""
    evaluation = _read_artifact("evaluation.json")
    return {
        "available": bool(evaluation.get("available")),
        "reason": evaluation.get("reason"),
        "evaluation": evaluation if evaluation.get("available") else {},
        "confusion_matrix": _read_artifact("confusion_matrix.json")
        if evaluation.get("available")
        else {},
        "classification_report": _read_artifact("classification_report.json")
        if evaluation.get("available")
        else {},
    }


@router.get("/ml/comparison")
def ml_comparison() -> dict:
    """Held-out Rule-Based vs ML: prediction quality *and* executed outcomes."""
    payload = _read_artifact("comparison.json")
    if not payload.get("available"):
        return {
            "available": False,
            "reason": payload.get("reason"),
            "prediction": {},
            "optimization": {},
            "per_program": [],
        }

    # Recompute the optimization aggregates from the per-program measurements
    # stored in this same file. The inputs are the saved measurements (never
    # re-executed here), so the numbers stay file-backed while always matching
    # the current aggregate_outcome implementation - including its
    # per-strategy breakdown.
    from ..ml.evaluate import aggregate_outcome

    per_program = payload.get("per_program") or []
    rule_rows = [row.get("rule") for row in per_program if row.get("rule")]
    ml_rows = [row.get("ml") for row in per_program if row.get("ml")]
    if rule_rows and ml_rows:
        optimization = dict(payload.get("optimization") or {})
        optimization["rule_based"] = aggregate_outcome(rule_rows)
        optimization["ml"] = aggregate_outcome(ml_rows)
        optimization["aggregation_note"] = (
            "Aggregated by backend.ml.evaluate.aggregate_outcome from the "
            "per-program measurements saved in comparison.json; nothing is "
            "re-executed when this endpoint is read."
        )
        payload = {**payload, "optimization": optimization}
    return payload


@router.get("/ml/complexity")
def ml_complexity() -> dict:
    """Theoretical complexity with real parameters + measured latency/sizes.

    Theoretical bounds (rules, trees, depth, features) are kept strictly
    apart from measured wall-clock and byte figures; every measured block
    carries the method that produced it, and anything unmeasurable reports
    ``available: false`` with a reason.
    """
    from ..ml.complexity import complexity_report

    try:
        return complexity_report()
    except Exception:
        raise HTTPException(status_code=500, detail="Complexity report unavailable.") from None


@router.get("/ml/feature-importance", response_model=FeatureImportanceResponse)
def ml_feature_importance() -> dict:
    """Native RF importance and mean|SHAP| importance, labelled separately."""
    from ..ml.explain import global_importance

    return global_importance()


# --- inference ---------------------------------------------------------------

@router.post("/ml/predict", response_model=PredictResponse)
def ml_predict(request: ExplainRequest) -> dict:
    """Predict a strategy for a TAC program using the persisted forest."""
    try:
        tac = _parse(request.tac)
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from ..ml.predict import predict_from_tac, unavailable_result

    try:
        return predict_from_tac(tac)
    except Exception:  # noqa: BLE001
        return unavailable_result(
            "ML prediction failed. Train the model with: python -m backend.ml.train"
        )


@router.post("/ml/explain", response_model=ExplainResponse)
def ml_explain(request: ExplainRequest) -> dict:
    """SHAP explanation of the forest's prediction for one TAC program.

    Two availability flags come back and they are deliberately independent:

    * ``available``            - did SHAP produce contributions?
    * ``model_available``       - is there a trained forest at all?

    A missing model leaves ``reason`` populated and the contribution lists
    empty; there is never a substituted answer.
    """
    try:
        tac = _parse(request.tac)
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from ..features.extractor import extract_features
    from ..ml.explain import explain_prediction

    features = extract_features(tac)
    prediction = _safe_predict(tac)
    explanation = explain_prediction(features, top_k=request.top_k)

    return {
        "available": bool(explanation.get("available")),
        "reason": explanation.get("reason"),
        "shap": explanation.get("shap") or {},
        "explainer": explanation.get("explainer"),
        "predicted_class": explanation.get("predicted_class"),
        "predicted_class_index": explanation.get("predicted_class_index"),
        "output_scale": explanation.get("output_scale"),
        "probabilities": prediction.get("probabilities") or explanation.get("probabilities") or {},
        "base_value": explanation.get("base_value"),
        "contributions": explanation.get("contributions") or [],
        "ranked_contributions": explanation.get("ranked_contributions") or [],
        "supporting": explanation.get("supporting") or [],
        "opposing": explanation.get("opposing") or [],
        "total_contribution": explanation.get("total_contribution"),
        "reconstructed_probability": explanation.get("reconstructed_probability"),
        "model_probability": explanation.get("model_probability"),
        "note": explanation.get("note"),
        "summary": explanation.get("summary"),
        "disclaimer": explanation.get("disclaimer"),
        # Extras preserved by `extra="allow"` on ExplainResponse.
        "feature_values": {k: float(features.get(k, 0)) for k in features},
        "predicted_strategy": prediction.get("predicted_strategy"),
        "model_version": prediction.get("model_version"),
        "model_available": bool(prediction.get("available")),
        "model_reason": prediction.get("reason"),
    }


def _safe_predict(tac) -> dict:
    from ..ml.predict import predict_from_tac, unavailable_result

    try:
        return predict_from_tac(tac)
    except Exception as exc:  # noqa: BLE001
        return unavailable_result(f"{type(exc).__name__}")


# --- comparison --------------------------------------------------------------

@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> dict:
    """Run both selectors on the same TAC and measure each outcome (§14)."""
    try:
        result = compare_source(
            request.tac,
            max_iterations=request.max_iterations,
            include_shap=request.include_shap,
        )
    except TacParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        # Only reachable if compare escalates; the normal path degrades instead.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Comparison failed.") from None
    return result


# --- dataset -----------------------------------------------------------------

@router.get("/dataset/summary")
def dataset_summary() -> dict:
    """Schema, distributions, feature definitions and quality checks."""
    try:
        return dataset_service.dataset_summary()
    except Exception:
        raise HTTPException(status_code=500, detail="Dataset summary unavailable.") from None


@router.get("/dataset/samples")
def dataset_samples(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=25, ge=1, le=200),
    q: str = Query(default="", max_length=200),
    split: str | None = Query(default=None, pattern="^(train|validation|test)$"),
) -> dict:
    """Paginated, searchable sample list - never the whole CSV."""
    try:
        return dataset_service.dataset_samples(page=page, size=size, q=q, split=split)
    except Exception:
        raise HTTPException(status_code=500, detail="Dataset samples unavailable.") from None


@router.get("/dataset/samples/{sample_id}")
def dataset_sample(sample_id: str) -> dict:
    """Full detail for one sample: features, six measurements and live selectors."""
    try:
        result = dataset_service.dataset_sample(sample_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Dataset sample unavailable.") from None

    if not result.get("available"):
        raise HTTPException(status_code=404, detail=result.get("reason", "Sample not found."))
    return result


@router.get("/optimization-matrix")
def optimization_matrix(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
    q: str = Query(default="", max_length=200),
    split: str | None = Query(default=None, pattern="^(train|validation|test)$"),
) -> dict:
    """Per-pass measurements for every technique on real dataset programs.

    Each cell executes one optimization pass on an independent copy of the
    program's original TAC (never chained onto another pass's output), so the
    matrix compares the six techniques on a common input with the project's
    shared measurement convention.
    """
    try:
        return matrix_service.optimization_matrix(page=page, size=size, q=q, split=split)
    except Exception:
        raise HTTPException(status_code=500, detail="Optimization matrix unavailable.") from None
