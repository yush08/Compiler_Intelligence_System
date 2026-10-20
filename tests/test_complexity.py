"""Complexity report tests: theoretical vs measured, never invented.

* ``SCORING_RULES`` must match the actual number of scoring formulas in the
  rule scorer's source, so the theoretical ``O(R x F)`` bound uses a real R;
* the ML parameters (trees, deepest fitted tree, features) are read from the
  fitted artifact itself;
* measured latency carries its method and exact sample counts, with positive
  times only inside an ``available`` block;
* byte figures are real file sizes or an explicit unavailable state;
* without a model the ML side reports unavailable with a reason while the
  rule side still measures.
"""

from __future__ import annotations

import inspect
import re

import pytest

from backend.ml import complexity as complexity_mod
from backend.scoring import rule_based as rule_based_module

from conftest import requires_ml

pytestmark = requires_ml


@pytest.fixture(scope="module")
def report() -> dict:
    """One full report per module - the report runs a short benchmark."""
    return complexity_mod.complexity_report()


def test_scoring_rule_count_matches_the_rule_scorer_source():
    source = inspect.getsource(rule_based_module.rule_based_strategy_scores)
    formulas = re.findall(r"^\s*\w+_score = ", source, flags=re.MULTILINE)
    assert len(formulas) == complexity_mod.SCORING_RULES
    assert complexity_mod.SCORING_RULES == 6


def test_theoretical_block_uses_real_parameters_and_stays_out_of_measurements(report):
    theoretical = report["theoretical"]
    rules = theoretical["rule_based"]

    # O(R x F) with R counted from the source and F from the extractor.
    assert rules["expression"] == "O(R x F)"
    assert rules["rules"] == complexity_mod.SCORING_RULES
    assert rules["features"] == 8
    assert rules["rule_feature_evaluations"] == rules["rules"] * rules["features"]
    assert "statement count" in rules["note"]

    # The two families of numbers are kept in separate top-level blocks.
    assert set(report) >= {"available", "theoretical", "measured", "limitations"}
    assert "theoretical" not in report["measured"]
    assert "expression" not in report["measured"]
    assert report["limitations"], "the report must state its limitations"


def test_ml_parameters_read_from_the_fitted_artifact(report):
    ml = report["theoretical"]["ml"]
    if not ml.get("available"):
        pytest.skip(f"model unavailable: {ml.get('reason')}")

    assert ml["expression"] == "O(T x D) per prediction"
    assert ml["trees"] >= 1
    assert ml["max_depth"] >= 1
    assert ml["features"] == 8
    assert ml["tree_node_evaluations_upper_bound"] == ml["trees"] * ml["max_depth"]
    assert "fitted artifact" in ml["parameters_source"]


def test_measured_latency_reports_method_counts_and_positive_times(report):
    latency = report["measured"]["latency"]

    assert "method" in latency and "median of" in latency["method"]
    assert "not an isolated benchmark" in latency["method"]
    assert latency["programs"] >= 3

    rule_stats = latency["rule_select_strategy_us"]
    assert rule_stats is not None, latency.get("reason")
    assert rule_stats["calls"] == rule_stats["batches"] * latency["programs"]
    assert 0 < rule_stats["min_us"] <= rule_stats["median_us"] <= rule_stats["max_us"]

    if latency["ml_predict_us"] is None:
        # An unmeasured selector must explain itself instead of showing a gap.
        assert latency["reason"]
    else:
        ml_stats = latency["ml_predict_us"]
        assert 0 < ml_stats["min_us"] <= ml_stats["median_us"] <= ml_stats["max_us"]
        assert latency["ratio_rule_over_ml"] > 0


def test_measured_sizes_are_real_bytes_or_explicit_unavailable(report):
    sizes = report["measured"]["sizes"]

    artifact = sizes["model_artifact"]
    if "bytes" in artifact:
        assert artifact["bytes"] > 1000
        assert artifact["file"] == "model.joblib"
    else:
        assert artifact.get("available") is False and artifact.get("reason")

    rules = sizes["rule_scorer_source"]
    if "bytes" in rules:
        assert rules["bytes"] > 0
        assert rules["file"] == "rule_based.py"
    else:
        assert rules.get("available") is False and rules.get("reason")

    footprint = sizes["model_deserialized"]
    if footprint["available"]:
        assert "tracemalloc" in footprint["method"]
        assert footprint["current_bytes"] > 0
        assert footprint["peak_bytes"] >= footprint["current_bytes"]
    else:
        assert footprint.get("reason")


def test_complexity_degrades_honestly_without_a_model(monkeypatch):
    from pathlib import Path

    from backend.ml import model as model_mod

    monkeypatch.setattr(model_mod, "MODEL_FILE", Path("no", "such", "model.joblib"))
    try:
        model_mod.clear_cache()
        report = complexity_mod.complexity_report()
    finally:
        model_mod.clear_cache()

    ml = report["theoretical"]["ml"]
    assert ml["available"] is False
    assert ml["reason"]
    # The rule side needs no model and still reports its real parameters.
    assert report["theoretical"]["rule_based"]["rules"] == complexity_mod.SCORING_RULES

    latency = report["measured"]["latency"]
    assert latency["rule_select_strategy_us"] is not None
    assert latency["ml_predict_us"] is None
    assert latency["reason"]


def test_complexity_endpoint_returns_the_report(client):
    response = client.get("/api/ml/complexity")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["theoretical"]["rule_based"]["expression"] == "O(R x F)"
    assert payload["measured"]["latency"]["available"] is True
    # No filesystem path may cross the API boundary.
    text = response.text
    assert "D:\\PROJECTS" not in text and "artifacts\\ml" not in text
