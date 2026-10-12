"""Inference against the persisted Random Forest.

Contract
--------
``predict_*`` returns an explicit ``available`` flag plus, when available, the
predicted strategy and the model's raw ``predict_proba`` output.  Three things
are deliberately *not* done here:

* **No fallback to the rule-based selector.**  A missing or corrupt artifact
  yields ``available: false`` with a reason.  Silently returning a rule-based
  answer labelled "ML" would be fabrication (§9/§22).
* **No invented probability for ``strength_reduction``.**  The class has no
  winning examples, so it is absent from ``model.classes_``; it therefore never
  appears in ``probabilities``.  :func:`unsupported_classes` names it explicitly
  so the UI can say so.
* **No confidence claim.**  The probabilities are the forest's raw, uncalibrated
  vote fractions and are reported as such.
"""

from __future__ import annotations

from .dataset import FEATURE_KEYS, STRATEGIES, feature_frame
from .model import ModelUnavailableError, load_metadata, load_payload

__all__ = [
    "predict_from_features",
    "predict_from_tac",
    "predicted_strategy_or_none",
    "unsupported_classes",
    "unavailable_result",
    "UncalibratedNote",
]

UncalibratedNote = (
    "Probabilities are the forest's raw class-vote fractions from "
    "predict_proba. They are not calibrated and should not be read as "
    "guaranteed confidence."
)


def unavailable_result(reason: str) -> dict:
    """Uniform 'no prediction' shape.

    Every key present in a successful response is also present here (as None or
    an empty collection), so callers can read ``result["predicted_strategy"]``
    without branching on ``available`` first.
    """
    return {
        "available": False,
        "selector": "ml",
        "reason": reason,
        "predicted_strategy": None,
        "predicted_probability": None,
        "probabilities": {},
        "probability_ranking": [],
        "classes": [],
        "unsupported_classes": [],
        "feature_values": {},
        "model_version": None,
        "model_type": None,
        "trained_at": None,
        "probabilities_calibrated": False,
        "note": None,
    }


def unsupported_classes(payload: dict) -> list[str]:
    """Strategies the artifact cannot emit, with the reason.

    ``strength_reduction`` lands here: it never wins a label, so the forest has
    no examples of it and its column is absent from ``classes_``.
    """
    model_classes = list(payload.get("class_names") or [])
    return [name for name in STRATEGIES if name not in model_classes]


def predict_from_features(features: dict) -> dict:
    """Predict a strategy from the eight feature values."""
    missing = [k for k in FEATURE_KEYS if k not in features]
    if missing:
        return unavailable_result("Missing feature value(s): " + ", ".join(missing))

    try:
        payload = load_payload()
    except ModelUnavailableError as exc:
        return unavailable_result(str(exc))

    import numpy as np

    model = payload["model"]
    # predict_proba columns follow model.classes_ (sklearn sorts them); using
    # any other ordering would attach each probability to the wrong strategy.
    classes = [str(c) for c in getattr(model, "classes_", [])] or list(
        payload.get("class_names") or []
    )
    row = feature_frame(features)

    try:
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:  # noqa: BLE001 - report, never crash the API
        return unavailable_result(
            f"Model inference failed ({type(exc).__name__}). "
            "The rule-based selector remains available."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted = classes[predicted_index] if predicted_index < len(classes) else None

    # Only classes the forest actually learned are emitted - no placeholder
    # probability for an unseen class.
    emitted = {
        classes[i]: round(float(probabilities[i]), 6)
        for i in range(min(len(classes), len(probabilities)))
    }

    metadata = dict(payload.get("metadata") or {}) or load_metadata()
    ordered = sorted(emitted.items(), key=lambda kv: -kv[1])

    return {
        "available": True,
        "selector": "ml",
        "predicted_strategy": predicted,
        "predicted_probability": round(float(probabilities[predicted_index]), 6),
        "probabilities": emitted,
        "probability_ranking": [{"strategy": k, "probability": v} for k, v in ordered],
        "classes": list(classes),
        "unsupported_classes": unsupported_classes(payload),
        "feature_values": {k: float(features[k]) for k in FEATURE_KEYS},
        "model_version": metadata.get("model_version"),
        "model_type": metadata.get("model_type") or type(model).__name__,
        "trained_at": metadata.get("trained_at"),
        "probabilities_calibrated": False,
        "note": UncalibratedNote,
        "reason": None,
    }


def predict_from_tac(tac) -> dict:
    """Extract the eight features from a TAC, then predict."""
    from ..features.extractor import extract_features

    features = extract_features(tac)
    result = predict_from_features(features)
    result["feature_values"] = {k: float(features.get(k, 0)) for k in FEATURE_KEYS}
    return result


def predicted_strategy_or_none(tac) -> str | None:
    """Strategy id for the engine's ML selector, or ``None`` when unavailable.

    Returning ``None`` (rather than raising) lets the caller produce a
    well-formed "model unavailable" response while the rule-based path keeps
    working.
    """
    result = predict_from_tac(tac)
    if not result.get("available"):
        return None
    return result.get("predicted_strategy")
