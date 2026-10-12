"""Shared fixtures for the ML-backed tests.

``ml_artifacts`` performs a **genuine end-to-end training run**: the real
:func:`backend.ml.train.train`, the real 18-point grid, writing to the real
``artifacts/ml`` directory.  It is session-scoped, so the whole suite pays for
one training run and every subsequent test reads exactly the artifact a fresh
clone would produce.

Tests that do not need a model do not request the fixture and stay fast.
"""

import importlib.util

import pytest

_ML_MODULES = ("sklearn", "joblib")

HAVE_ML = all(importlib.util.find_spec(name) for name in _ML_MODULES)

requires_ml = pytest.mark.skipif(
    not HAVE_ML,
    reason="scikit-learn/joblib not installed (pip install scikit-learn joblib)",
)


@pytest.fixture(scope="session")
def ml_artifacts():
    """Train the model once per session and return its status + summary."""
    from backend.ml.model import clear_cache, model_status
    from backend.ml.train import train

    summary = train(verbose=False)
    clear_cache()

    status = model_status()
    assert status["available"], f"model unavailable after training: {status.get('reason')}"
    assert status["ready"] is True
    return {"summary": summary, "status": status}


@pytest.fixture(scope="session")
def model_payload(ml_artifacts):
    """The raw joblib payload produced by the session training run."""
    from backend.ml.model import load_payload

    return load_payload()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.api.server import app

    return TestClient(app)


# Small programs covering distinct opportunity classes.
DEAD_CODE_TAC = "t1 = 5 + 3\nt2 = t1 * 2\nunused = t2 + 1\nt3 = a + 0\n"

FOLDING_TAC = "t1 = 10 + 5\nt2 = t1 * 2\n"

CSE_TAC = "t1 = a + b\nt2 = a + b\nt3 = t1 + t2\n"

NO_OPPORTUNITY_TAC = "x = y\nz = x\n"

LOOP_TAC = """
i = 0
limit = n
loop:
i = i + 1
if i < limit goto loop
"""


@pytest.fixture
def programs():
    return {
        "dead_code": DEAD_CODE_TAC,
        "folding": FOLDING_TAC,
        "cse": CSE_TAC,
        "no_opportunity": NO_OPPORTUNITY_TAC,
        "loop": LOOP_TAC,
    }
