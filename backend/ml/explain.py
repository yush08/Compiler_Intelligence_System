"""SHAP explanations for the trained forest - with an honest fallback.

Three rules govern this module:

1. **Only real SHAP values are returned.** Every contribution comes from
   ``shap.TreeExplainer`` applied to the fitted forest.  If SHAP cannot run, the
   status is ``available: false`` with a reason - never invented numbers and
   never a silent substitution of the Random Forest's built-in importances.
2. **Native importance and SHAP importance are different things** and are
   labelled separately everywhere they appear.  Native importance is the forest's
   own mean impurity decrease (it splits credit among correlated features);
   SHAP importance is the mean |contribution| of each feature to the model's
   output.  They may rank features differently.
3. **SHAP explains the prediction, not causality.** Descriptive text produced
   here says "associated with"/"pushed the prediction", never "caused".

SHAP availability is checked lazily and cached: a failed import degrades the
feature to an explicit unavailable state instead of crashing the API.
"""

from __future__ import annotations

import json
import threading

from .dataset import FEATURE_KEYS, feature_frame
from .model import MODEL_DIR

__all__ = [
    "shap_status",
    "global_importance",
    "explain_prediction",
    "shap_summary_text",
]

IMPORTANCE_FILE = MODEL_DIR / "feature_importance.json"

_STATE = {
    "checked": False,
    "available": False,
    "version": None,
    "reason": None,
    "explainer": None,
    "explainer_key": None,
    "error": None,
}
_LOCK = threading.Lock()


def shap_status() -> dict:
    """Availability descriptor for ``GET /api/ml/status``. Never raises."""
    with _LOCK:
        if not _STATE["checked"]:
            try:
                import shap  # noqa: F401
            except Exception as exc:  # noqa: BLE001 - any import failure is "unavailable"
                _STATE.update(
                    available=False,
                    reason=f"SHAP is not installed ({type(exc).__name__}). "
                    "Install it with: pip install shap",
                )
            else:
                import shap as _shap

                _STATE.update(available=True, version=getattr(_shap, "__version__", None), reason=None)
            _STATE["checked"] = True

        return {
            "available": bool(_STATE["available"]),
            "version": _STATE["version"],
            "reason": _STATE["reason"],
            "scope": (
                "SHAP attributions describe how model inputs contributed to one "
                "prediction. They do not establish causation and are not an "
                "optimization algorithm."
            ),
        }


def _get_explainer(payload: dict):
    """Build (and cache) a TreeExplainer for the currently loaded forest."""
    status = shap_status()
    if not status["available"]:
        raise RuntimeError(status["reason"])

    model = payload["model"]
    key = id(model)
    with _LOCK:
        if _STATE["explainer"] is not None and _STATE["explainer_key"] == key:
            return _STATE["explainer"]
        import shap

        explainer = shap.TreeExplainer(model)
        _STATE["explainer"] = explainer
        _STATE["explainer_key"] = key
        return explainer


def _as_per_class(values, n_samples: int, n_features: int, n_classes: int):
    """Normalise SHAP output to a (n_samples, n_features, n_classes) ndarray.

    TreeExplainer has returned three shapes across shap versions: a list of
    per-class arrays, a 3-D array, or a 2-D array for single-output models.
    Normalising once here keeps every caller simple.
    """
    import numpy as np

    if isinstance(values, (list, tuple)):
        stacked = np.stack([np.asarray(v) for v in values], axis=-1)
        return stacked.reshape(n_samples, n_features, len(values))

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if n_classes == 1:
            return arr.reshape(n_samples, n_features, 1)
        raise ValueError(f"unexpected 2-D SHAP output for {n_classes} classes: {arr.shape}")
    if arr.ndim == 3:
        return arr
    raise ValueError(f"unexpected SHAP output shape: {arr.shape}")


def explain_prediction(features: dict, top_k: int = 8) -> dict:
    """Explain one prediction with real per-feature SHAP contributions.

    Args:
        features: mapping of the eight ``FEATURE_KEYS`` to numeric values.
        top_k: how many features to return in the ranked list.

    Returns a dict with the model's probabilities, the predicted class, the
    SHAP contribution of every feature **for the predicted class**, a ranked
    positive/negative split, and a readable summary.  When SHAP or the model is
    unavailable the result is ``available: false`` with a reason.
    """
    from .model import ModelUnavailableError, load_payload

    missing = [k for k in FEATURE_KEYS if k not in features]
    if missing:
        return {
            "available": False,
            "reason": "Missing feature value(s): " + ", ".join(missing),
        }

    status = shap_status()
    if not status["available"]:
        return {"available": False, "reason": status["reason"], "shap": status}

    try:
        payload = load_payload()
    except ModelUnavailableError as exc:
        return {"available": False, "reason": str(exc), "shap": status}

    import numpy as np

    try:
        explainer = _get_explainer(payload)
        row = feature_frame(features)
        raw = explainer.shap_values(row)
    except Exception as exc:  # noqa: BLE001 - degrade, do not crash the API
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "available": False,
            "reason": "SHAP could not explain this prediction "
            f"({type(exc).__name__}). Model predictions remain available.",
            "shap": status,
        }

    model = payload["model"]
    classes = list(payload.get("class_names") or [])
    n_classes = len(classes)

    contributions = _as_per_class(raw, 1, len(FEATURE_KEYS), n_classes)

    probabilities = model.predict_proba(row)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = classes[predicted_index] if predicted_index < n_classes else None

    values = contributions[0, :, predicted_index]

    base_value = None
    expected = getattr(explainer, "expected_value", None)
    if expected is not None:
        expected_arr = np.atleast_1d(np.asarray(expected, dtype=float))
        if expected_arr.size > predicted_index:
            base_value = float(expected_arr[predicted_index])

    reconstruction = (base_value if base_value is not None else 0.0) + float(values.sum())
    model_probability = float(probabilities[predicted_index])

    entries = []
    for index, name in enumerate(FEATURE_KEYS):
        value = float(features[name])
        shap_value = float(values[index])
        entries.append(
            {
                "feature": name,
                "value": value,
                "shap_value": shap_value,
                "mean_abs_rank": 0,
                "direction": "positive" if shap_value > 0 else ("negative" if shap_value < 0 else "neutral"),
            }
        )

    ranked = sorted(entries, key=lambda e: -abs(e["shap_value"]))
    for position, entry in enumerate(ranked, start=1):
        entry["mean_abs_rank"] = position

    supporting = [e for e in ranked if e["shap_value"] > 0][:top_k]
    opposing = [e for e in ranked if e["shap_value"] < 0][:top_k]

    note = None
    if abs(reconstruction - model_probability) > 0.05:
        note = (
            "SHAP contributions sum to a value different from the reported "
            "probability; interpret the magnitudes as relative contributions."
        )

    return {
        "available": True,
        "shap": status,
        "explainer": "TreeExplainer",
        "predicted_class": predicted_class,
        "predicted_class_index": predicted_index,
        # Explicit statement of the scale every number below lives on, so the
        # UI never has to imply whether these are probabilities, raw scores
        # or log-odds.
        "output_scale": {
            "space": "probability",
            "range": "0-1",
            "note": (
                "Output scale: the probability, the base value and every SHAP "
                "contribution are in probability space for the predicted class "
                "(0 to 1) - not log-odds and not a raw decision-function score."
            ),
        },
        "probabilities": {
            classes[i]: round(float(probabilities[i]), 6) for i in range(min(n_classes, len(probabilities)))
        },
        "base_value": base_value,
        "contributions": entries,
        "ranked_contributions": ranked,
        "supporting": supporting,
        "opposing": opposing,
        "total_contribution": float(values.sum()),
        "reconstructed_probability": reconstruction,
        "model_probability": model_probability,
        "note": note,
        "summary": shap_summary_text(predicted_class, supporting, opposing),
        "disclaimer": (
            "SHAP explains how the model's inputs contributed to this prediction. "
            "It does not establish causation."
        ),
    }


def shap_summary_text(predicted_class: str | None, supporting: list[dict], opposing: list[dict]) -> str:
    """One readable sentence describing what moved the prediction."""
    if not predicted_class:
        return "No predicted class available."

    def _fmt(entries: list[dict]) -> str:
        parts = []
        for entry in entries[:3]:
            parts.append(f"{entry['feature']}={_num(entry['value'])} (+{entry['shap_value']:.4f})")
        return ", ".join(parts)

    def _fmt_neg(entries: list[dict]) -> str:
        parts = []
        for entry in entries[:3]:
            parts.append(f"{entry['feature']}={_num(entry['value'])} ({entry['shap_value']:.4f})")
        return ", ".join(parts)

    left = _fmt(supporting) or "no feature pushed strongly towards it"
    right = _fmt_neg(opposing)
    sentence = (
        f"The model predicted '{predicted_class}', pushed by {left}"
        + (f", and held back by {right}." if right else ".")
    )
    return sentence


def _num(value: float) -> str:
    if float(value) != int(float(value)):
        return f"{float(value):.4f}"
    return str(int(float(value)))


def global_importance() -> dict:
    """Global feature importance, with native and SHAP rankings kept distinct.

    Reads the values computed and saved during training
    (``artifacts/ml/feature_importance.json``); it does not recompute SHAP on
    every request.
    """
    try:
        saved = json.loads(IMPORTANCE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = None

    if not saved:
        return {
            "available": False,
            "reason": "No saved feature importance. Train the model with: python -m backend.ml.train",
            "native": {
                "available": False,
                "source": "RandomForestClassifier.feature_importances_ (mean impurity decrease)",
                "features": [],
                "note": "Not computed yet.",
            },
            "shap": {
                "available": False,
                "source": "mean(|SHAP value|) over the training split",
                "reason": (
                    "No saved feature importance. Train the model with: "
                    "python -m backend.ml.train"
                ),
                "features": [],
                "note": "Not computed yet.",
            },
            "comparison_note": None,
        }

    native = saved.get("native") or {}
    shap_block = saved.get("shap") or {}

    return {
        "available": True,
        "generated_at": saved.get("generated_at"),
        "model_version": saved.get("model_version"),
        "native": {
            "available": bool(native.get("available")),
            "source": "RandomForestClassifier.feature_importances_ (mean impurity decrease)",
            "features": native.get("features", []),
            "note": (
                "Native importance spreads credit across correlated features and "
                "is not interchangeable with SHAP importance."
            ),
        },
        "shap": {
            "available": bool(shap_block.get("available")),
            "source": "mean(|SHAP value|) over the training split",
            "reason": shap_block.get("reason"),
            "features": shap_block.get("features", []),
            "note": (
                "SHAP importance is the average magnitude of each feature's "
                "contribution to the model's output."
            ),
        },
        "comparison_note": saved.get("comparison_note"),
    }
