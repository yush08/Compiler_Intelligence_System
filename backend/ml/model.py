"""Persisting and loading the trained model artifact.

Design notes
------------
* **One load, many requests.** The estimator is cached at module level so the
  API never re-reads the artifact per request and never retrains on inference
  (§6/§12).  The cache is keyed on the file's mtime+size, so retraining while
  the server is running is picked up without a restart.
* **No filesystem paths cross the API boundary.** Paths live in the artifact and
  in this module; :func:`model_status` deliberately reports only the file *name*
  so a stack trace or an absolute path can never leak to the browser.
* **Missing/corrupt is a first-class state.** A missing artifact returns
  ``available: False`` with a reason the UI can render as an empty state -
  it is never silently substituted with the rule-based result.

Artifact layout (``artifacts/ml/``)::

    model.joblib                estimator + feature/class order + config
    metadata.json               version, timestamps, counts, fingerprint
    evaluation.json             per-split metrics + confusion matrices
    comparison.json             rule-based vs ML: prediction *and* execution
    feature_importance.json     native RF importance + global SHAP importance
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .dataset import FEATURE_KEYS, REPOSITORY_ROOT

__all__ = [
    "MODEL_DIR",
    "MODEL_FILE",
    "METADATA_FILE",
    "save_artifacts",
    "load_payload",
    "load_metadata",
    "model_status",
    "clear_cache",
    "ModelUnavailableError",
]

MODEL_DIR = REPOSITORY_ROOT / "artifacts" / "ml"
MODEL_FILE = MODEL_DIR / "model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

_CACHE: dict = {"key": None, "payload": None}
_LOCK = threading.Lock()


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifact is missing or unreadable.

    Callers translate this into an explicit ``available: false`` state with a
    reason - never into a fallback prediction.
    """


def _stat_key() -> tuple | None:
    try:
        stat = MODEL_FILE.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def clear_cache() -> None:
    """Drop the cached estimator (used by tests and after retraining)."""
    with _LOCK:
        _CACHE["key"] = None
        _CACHE["payload"] = None


def save_artifacts(
    model,
    *,
    feature_names: list[str],
    class_names: list[str],
    config: dict,
    metadata: dict,
    out_dir: str | Path | None = None,
) -> Path:
    """Persist the fitted estimator and its metadata. Returns the model path."""
    directory = Path(out_dir) if out_dir is not None else MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "config": dict(config),
        "metadata": dict(metadata),
    }
    target = directory / MODEL_FILE.name
    joblib.dump(payload, target)

    (directory / METADATA_FILE.name).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    clear_cache()
    return target


def load_payload() -> dict:
    """Load the fitted estimator, using a small mtime-keyed cache."""
    key = _stat_key()
    if key is None:
        raise ModelUnavailableError(
            "Model artifact not found. Train it first with: python -m backend.ml.train"
        )

    with _LOCK:
        if _CACHE["payload"] is not None and _CACHE["key"] == key:
            return _CACHE["payload"]
        try:
            payload = joblib.load(MODEL_FILE)
        except Exception as exc:  # noqa: BLE001 - any load failure is "unavailable"
            raise ModelUnavailableError(
                "Model artifact could not be loaded (retrain it with: python -m backend.ml.train)"
            ) from exc

        if not isinstance(payload, dict) or "model" not in payload:
            raise ModelUnavailableError(
                "Model artifact is malformed (retrain it with: python -m backend.ml.train)"
            )

        # Guard against a feature-order drift between the artifact and the code.
        saved = list(payload.get("feature_names") or [])
        if saved and saved != list(FEATURE_KEYS):
            raise ModelUnavailableError(
                "Model artifact was trained on a different feature ordering; retrain with: "
                "python -m backend.ml.train"
            )

        _CACHE["key"] = key
        _CACHE["payload"] = payload
        return payload


def load_metadata() -> dict:
    """Read ``metadata.json`` if present; returns {} when untrained."""
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Fall back to the copy embedded in the joblib payload.
        try:
            return dict(load_payload().get("metadata") or {})
        except ModelUnavailableError:
            return {}


def model_status() -> dict:
    """Readiness descriptor for ``GET /api/ml/status``.

    Always returns a dict - never raises - so the UI can distinguish
    "not trained yet" from "trained" before it tries to predict.
    """
    from .explain import shap_status  # local import: avoids a circular import

    status: dict = {
        "available": False,
        "ready": False,
        "model_version": None,
        "model_type": None,
        "trained_at": None,
        "training_samples": None,
        "dataset_fingerprint": None,
        "feature_names": list(FEATURE_KEYS),
        "supported_classes": [],
        "config": {},
        "artifact_file": MODEL_FILE.name,
        "reason": None,
        "shap": shap_status(),
    }

    if not MODEL_FILE.is_file():
        status["reason"] = (
            "No trained model artifact. Run: python -m backend.ml.train"
        )
        return status

    try:
        payload = load_payload()
    except ModelUnavailableError as exc:
        status["reason"] = str(exc)
        return status

    metadata = dict(payload.get("metadata") or {}) or load_metadata()
    status.update(
        available=True,
        ready=True,
        model_version=metadata.get("model_version"),
        model_type=metadata.get("model_type") or type(payload["model"]).__name__,
        trained_at=metadata.get("trained_at"),
        training_samples=metadata.get("training_samples"),
        dataset_fingerprint=metadata.get("dataset_fingerprint"),
        supported_classes=list(payload.get("class_names") or []),
        config=dict(payload.get("config") or {}),
        reason=None,
    )
    status["trained_at_iso"] = metadata.get("trained_at")
    status["trained_at_human"] = _human_time(metadata.get("trained_at"))
    status["dataset_rows"] = metadata.get("dataset_rows")
    status["dataset_sha256"] = metadata.get("dataset_sha256")
    return status


def _human_time(value) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
