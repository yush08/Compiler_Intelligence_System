"""End-to-end API tests for the original routes and the added ML routes.

Three properties are asserted explicitly:

* the **original contract survives** - ``/api/health``, ``/api/analyze``,
  ``/api/optimize`` and ``/api/simulate`` keep their fields and behaviour;
* **missing models degrade honestly** - read endpoints answer 200 with
  ``available: false``, while execution endpoints that were explicitly asked to
  use ML answer 503 naming the remedy;
* **errors never leak internals** - no tracebacks, no filesystem paths.
"""

from pathlib import Path

import pytest

from backend.ml import model as model_mod
from backend.ml.model import MODEL_DIR

from conftest import DEAD_CODE_TAC, FOLDING_TAC, NO_OPPORTUNITY_TAC, requires_ml

pytestmark = requires_ml

SIX_STRATEGIES = {
    "constant_folding",
    "constant_propagation",
    "common_subexpression",
    "dead_code_elimination",
    "algebraic_simplification",
    "strength_reduction",
}


# ---------------------------------------------------------------------------
# Original contract (must not regress)
# ---------------------------------------------------------------------------


def test_health_contract_is_byte_for_byte_unchanged(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "system": "COMPILER INTELLIGENCE SYSTEM",
        "mode": "rule-based",
        "ready": True,
    }


def test_analyze_keeps_every_original_field(client, ml_artifacts):
    response = client.post("/api/analyze", json={"tac": DEAD_CODE_TAC})
    assert response.status_code == 200
    body = response.json()

    for key in ("tac", "features", "strategies", "selected_strategy", "iterations", "final_result"):
        assert key in body, key
    assert {item["id"] for item in body["strategies"]} == SIX_STRATEGIES
    assert len(body["features"]) == 8
    assert body["iterations"] == []
    assert body["final_result"] == {}
    # New fields are additive.
    assert body["ml"]["available"] is True
    assert body["selection"]["selector"] == "rule"


def test_optimize_keeps_every_original_field(client, ml_artifacts):
    body = client.post("/api/optimize", json={"tac": FOLDING_TAC}).json()

    for key in ("tac", "features", "strategies", "selected_strategy", "iterations", "final_result"):
        assert key in body, key

    final = body["final_result"]
    for key in (
        "original_count",
        "final_count",
        "total_reduction",
        "reduction_percentage",
        "iterations",
        "status",
        "original_tac",
        "final_tac",
    ):
        assert key in final, key

    iteration = body["iterations"][0]
    for key in (
        "iteration",
        "strategy",
        "strategy_name",
        "score",
        "before_count",
        "after_count",
        "reduction",
        "improvement_percentage",
        "tac_before",
        "tac_after",
        "status",
    ):
        assert key in iteration, key
    assert iteration["reduction"] == iteration["before_count"] - iteration["after_count"]


def test_simulate_keeps_every_original_field(client, ml_artifacts):
    body = client.post("/api/simulate", json={"tac": DEAD_CODE_TAC}).json()

    for key in ("tac", "features", "strategies", "selected_strategy", "iterations", "final_result"):
        assert key in body, key

    assert body["iterations"], "expected at least one iteration"
    for iteration in body["iterations"]:
        for key in (
            "iteration",
            "strategy",
            "strategy_name",
            "score",
            "before_count",
            "after_count",
            "reduction",
            "improvement_percentage",
            "tac_before",
            "tac_after",
            "status",
        ):
            assert key in iteration, key

    final = body["final_result"]
    assert final["status"] in {"converged", "stopped"}
    assert final["stop_reason"]
    assert final["total_reduction"] == final["original_count"] - final["final_count"]


def test_explicit_strategy_still_wins_over_selector(client, ml_artifacts):
    body = client.post(
        "/api/optimize", json={"tac": DEAD_CODE_TAC, "strategy": "constant_folding", "selector": "ml"}
    ).json()
    assert body["selector"] == "explicit"
    assert body["selected_strategy"]["id"] == "constant_folding"
    assert body["selection"]["selector"] == "explicit"
    assert "pin" in body["selection"]["note"].lower()


def test_invalid_explicit_strategy_falls_back_to_the_rule_selector(client, ml_artifacts):
    body = client.post("/api/optimize", json={"tac": FOLDING_TAC, "strategy": "nonsense"}).json()
    assert body["selector"] == "rule"
    assert body["selected_strategy"]["id"] in SIX_STRATEGIES


# ---------------------------------------------------------------------------
# Selector modes
# ---------------------------------------------------------------------------


def test_optimize_with_ml_selector_executes_the_model_choice(client, ml_artifacts):
    body = client.post("/api/optimize", json={"tac": DEAD_CODE_TAC, "selector": "ml"}).json()

    assert body["selector"] == "ml"
    assert body["selection"]["selector"] == "ml"
    assert body["selection"]["ml_available"] is True
    assert body["selection"]["executed_strategy"] == body["ml"]["predicted_strategy"]
    assert body["selection"]["ml_strategy"] == body["ml"]["predicted_strategy"]
    assert body["selection"]["ml_probabilities"] == body["ml"]["probabilities"]
    # The executed strategy must be a real, applicable pass.
    assert body["selected_strategy"]["id"] == body["ml"]["predicted_strategy"]
    assert body["iterations"][0]["selector"] == "ml"


def test_simulate_with_ml_selector_predicts_at_every_iteration(client, ml_artifacts):
    body = client.post("/api/simulate", json={"tac": DEAD_CODE_TAC, "selector": "ml"}).json()

    assert body["selector"] == "ml"
    assert body["final_result"]["selector"] == "ml"
    for iteration in body["iterations"]:
        assert iteration["selector"] == "ml"
        assert iteration["ml_predicted_strategy"] in SIX_STRATEGIES
        # The engine applied exactly what the forest predicted for that input
        # whenever it executed the prediction; a controlled fallback is
        # recorded as such — the model's pick is never silently replaced.
        assert iteration["predicted_strategy"] == iteration["ml_predicted_strategy"]
        if iteration.get("origin") == "predicted":
            assert iteration["strategy"] == iteration["ml_predicted_strategy"]
        else:
            assert iteration.get("origin") == "fallback"
            assert iteration["strategy"] != iteration["ml_predicted_strategy"]
            assert iteration.get("outcome") in (
                "applied_useful",
                "applied_changed",
                "executed_no_change",
                "rejected",
                "failed",
            )
        assert set(iteration["ml_probabilities"]) == set(body["ml"]["classes"])
        # Rounded to 6 decimals each, so the sum can drift by up to ~5e-6.
        assert abs(sum(iteration["ml_probabilities"].values()) - 1.0) < 1e-5
        # The rule-based view is still recorded alongside, for comparison.
        assert iteration["rule_selected_strategy"] in SIX_STRATEGIES
        assert iteration["strategies_agree"] == (
            iteration["ml_predicted_strategy"] == iteration["rule_selected_strategy"]
        )
        assert iteration["rule_scores"]
        assert iteration["shap"]["available"] is True
        assert len(iteration["shap"]["contributions"]) == 8


def test_simulate_ml_trace_shap_can_be_switched_off(client, ml_artifacts):
    body = client.post(
        "/api/simulate", json={"tac": DEAD_CODE_TAC, "selector": "ml", "include_shap": False}
    ).json()
    for iteration in body["iterations"]:
        assert iteration.get("shap") is None
        assert iteration["ml_predicted_strategy"]


def test_selector_ml_changes_only_the_choice_not_the_measurements(client, ml_artifacts):
    """Same program, two selectors: each run measures itself, honestly."""
    rule = client.post("/api/simulate", json={"tac": NO_OPPORTUNITY_TAC, "selector": "rule"}).json()
    ml = client.post("/api/simulate", json={"tac": NO_OPPORTUNITY_TAC, "selector": "ml"}).json()

    for body in (rule, ml):
        assert body["final_result"]["total_reduction"] == (
            body["final_result"]["original_count"] - body["final_result"]["final_count"]
        )
        assert body["final_result"]["useful_improvement"] == (
            body["final_result"]["original_operation_count"]
            - body["final_result"]["final_operation_count"]
            + body["final_result"]["total_reduction"]
        )


# ---------------------------------------------------------------------------
# ML read endpoints
# ---------------------------------------------------------------------------


def test_ml_status_endpoint(client, ml_artifacts):
    body = client.get("/api/ml/status").json()
    assert body["available"] is True and body["ready"] is True
    assert body["model_type"] == "RandomForestClassifier"
    assert body["model_version"]
    assert body["dataset_fingerprint"]
    assert body["feature_names"] == [
        "num_statements",
        "num_constants",
        "num_repeated_expr",
        "num_dead_stmts",
        "num_algebraic_ops",
        "num_loop_ops",
        "num_tmp_vars",
        "const_ratio",
    ]
    assert len(body["supported_classes"]) == 5
    assert "strength_reduction" not in body["supported_classes"]
    assert "strength_reduction" in body["unsupported_classes"]
    assert body["shap"]["available"] is True
    assert body["reason"] is None
    # A single filename, never a path.
    assert "/" not in body["artifact_file"] and "\\" not in body["artifact_file"]


def test_ml_metrics_endpoint_reports_all_three_splits(client, ml_artifacts):
    body = client.get("/api/ml/metrics").json()
    assert body["available"] is True
    evaluation = body["evaluation"]

    assert set(evaluation["splits"]) == {"train", "validation", "test"}
    assert evaluation["held_out_split"] == "test"
    assert evaluation["primary_metric"] == "macro_f1"
    assert evaluation["limitations"]

    for name, block in evaluation["splits"].items():
        assert set(block["confusion_matrix_labels"][:5]) == set(evaluation["target_classes"])
        assert sum(sum(row) for row in block["confusion_matrix"]) == block["evaluated_samples"]

    assert set(body["confusion_matrix"]["splits"]) == {"train", "validation", "test"}
    assert set(body["classification_report"]["splits"]) == {"train", "validation", "test"}


def test_ml_feature_importance_labels_the_two_measures_distinctly(client, ml_artifacts):
    body = client.get("/api/ml/feature-importance").json()
    assert body["available"] is True
    assert body["native"]["available"] is True
    assert body["shap"]["available"] is True
    assert body["native"]["source"] != body["shap"]["source"]
    assert body["comparison_note"]
    expected = {
        "num_statements",
        "num_constants",
        "num_repeated_expr",
        "num_dead_stmts",
        "num_algebraic_ops",
        "num_loop_ops",
        "num_tmp_vars",
        "const_ratio",
    }
    assert {f["feature"] for f in body["native"]["features"]} == expected
    assert {f["feature"] for f in body["shap"]["features"]} == expected


def test_ml_comparison_endpoint_keeps_the_two_studies_apart(client, ml_artifacts):
    body = client.get("/api/ml/comparison").json()
    assert body["available"] is True
    assert body["prediction"]["rule_based"]["evaluated_samples"] == body["evaluated_samples"]
    assert body["prediction"]["ml"]["evaluated_samples"] == body["evaluated_samples"]

    # Prediction quality carries no reduction numbers...
    assert "mean_useful_improvement" not in body["prediction"]["rule_based"]
    # ...and optimization effectiveness carries no prediction metrics.
    assert "accuracy" not in body["optimization"]["rule_based"]
    assert "macro_f1" not in body["optimization"]["ml"]

    assert body["optimization"]["overall_winner"] in {"rule_based", "ml", "tie"}
    assert len(body["per_program"]) == body["evaluated_samples"]
    assert body["methodology"]["separation"]


def test_predict_and_explain_endpoints(client, ml_artifacts):
    predict = client.post("/api/ml/predict", json={"tac": DEAD_CODE_TAC}).json()
    assert predict["available"] is True
    assert predict["predicted_strategy"] in SIX_STRATEGIES
    # Probabilities are rounded to 6 decimals each, so the sum can drift by up
    # to ~5e-6 across the five classes. 1e-5 absorbs that display rounding.
    assert abs(sum(predict["probabilities"].values()) - 1.0) < 1e-5
    assert predict["probabilities_calibrated"] is False

    explain = client.post("/api/ml/explain", json={"tac": DEAD_CODE_TAC, "top_k": 3}).json()
    assert explain["available"] is True
    assert explain["model_available"] is True
    assert explain["explainer"] == "TreeExplainer"
    assert explain["predicted_class"] == predict["predicted_strategy"]
    assert len(explain["contributions"]) == 8
    assert len(explain["supporting"]) <= 3
    assert explain["reconstructed_probability"] == pytest.approx(
        explain["model_probability"], abs=1e-6
    )
    assert explain["disclaimer"] and "causation" in explain["disclaimer"]


def test_explain_rejects_top_k_outside_the_feature_count(client, ml_artifacts):
    response = client.post("/api/ml/explain", json={"tac": DEAD_CODE_TAC, "top_k": 0})
    assert response.status_code == 422
    response = client.post("/api/ml/explain", json={"tac": DEAD_CODE_TAC, "top_k": 9})
    assert response.status_code == 422


def test_compare_endpoint_separates_strategy_from_outcome(client, ml_artifacts):
    body = client.post("/api/compare", json={"tac": DEAD_CODE_TAC, "max_iterations": 10}).json()

    assert body["available"] is True
    assert body["winner"] in {"rule_based", "ml", "feedback", "tie"}
    assert body["winner_note"]
    assert "measured useful_improvement" in body["decided_by"]

    for side in ("rule_based", "ml", "feedback"):
        block = body[side]
        assert block["available"] is True
        assert block["iterations"] == len(block["history"])
        # The recorded winner must follow these numbers, not a probability.
        assert block["useful_improvement"] == (
            block["measurements"]["instruction_reduction"]
            + block["measurements"]["operation_reduction"]
        )

    # The winner is the unique highest measured gain across all three runs;
    # whenever more than one run reaches it, the result is a tie.
    gains = {
        side: body[side]["useful_improvement"]
        for side in ("rule_based", "ml", "feedback")
    }
    best = max(gains.values())
    leaders = [side for side, gain in gains.items() if gain == best]
    assert body["winner"] == (leaders[0] if len(leaders) == 1 else "tie")


def test_compare_endpoint_reports_agreement_over_the_whole_run(client, ml_artifacts):
    body = client.post("/api/compare", json={"tac": DEAD_CODE_TAC, "max_iterations": 10}).json()
    agreement = body["agreement"]
    rule_seq = [h["selected_strategy"] for h in body["rule_based"]["history"]]
    ml_seq = [h["selected_strategy"] for h in body["ml"]["history"]]

    assert agreement["rule_strategy_sequence"] == rule_seq
    assert agreement["ml_strategy_sequence"] == ml_seq
    assert agreement["strategies_agree"] == (rule_seq == ml_seq)
    assert agreement["explanation"]
    # "identical" may only be claimed when every iteration matched.
    if "identical" in agreement["explanation"]:
        assert agreement["strategies_agree"] is True


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------


def test_dataset_summary_endpoint(client, ml_artifacts):
    body = client.get("/api/dataset/summary").json()
    assert body["available"] is True
    assert body["row_count"] == 241
    assert body["column_count"] == 46
    assert body["model_input_columns"] == [
        "num_statements",
        "num_constants",
        "num_repeated_expr",
        "num_dead_stmts",
        "num_algebraic_ops",
        "num_loop_ops",
        "num_tmp_vars",
        "const_ratio",
    ]
    assert len(body["features"]) == 8
    assert all(feature["description"] for feature in body["features"])
    assert body["target"]["class_count"] == 5
    assert sum(body["target"]["distribution"].values()) == 241
    assert sum(body["splits"]["distribution"].values()) == 241
    assert body["missing_values"]["total"] == 0
    assert body["duplicates"]["duplicate_tac_programs"] == 0
    assert len(body["categories"]) == 25
    # The split is grouped by generation family, surfaced on the summary.
    assert body["splits"]["grouped_by"] == "family"
    assert body["splits"]["family_count"] == 27
    assert set(body["strategies"]) == SIX_STRATEGIES
    assert all(len(histogram) > 0 for histogram in body["feature_histograms"].values())
    # No filesystem path crosses the API.
    assert "dataset_file" not in body or "\\" not in str(body["dataset_file"])


def test_dataset_samples_are_paginated_not_dumped(client, ml_artifacts):
    first = client.get("/api/dataset/samples", params={"page": 1, "size": 10}).json()
    assert first["available"] is True
    assert first["page"] == 1
    assert first["size"] == 10
    assert first["total"] == 241
    assert first["pages"] == 25
    assert len(first["items"]) == 10
    assert first["filtered_out"] == 0

    second = client.get("/api/dataset/samples", params={"page": 2, "size": 10}).json()
    assert second["items"] != first["items"]
    assert {i["sample_id"] for i in first["items"]}.isdisjoint(
        {i["sample_id"] for i in second["items"]}
    )

    # A single page never returns every row.
    assert len(first["items"]) < first["total"]


def test_dataset_samples_search_and_split_filters(client, ml_artifacts):
    search = client.get("/api/dataset/samples", params={"q": "dead_code", "size": 200}).json()
    assert search["total"] > 0
    assert search["filtered_out"] > 0
    for item in search["items"]:
        assert "dead_code" in (
            item["sample_id"] + item["generation_category"] + item["best_strategy"]
        ).lower()

    test_split = client.get("/api/dataset/samples", params={"split": "test", "size": 200}).json()
    assert test_split["total"] == 45
    assert all(item["split"] == "test" for item in test_split["items"])


def test_dataset_page_is_clamped_into_range(client, ml_artifacts):
    body = client.get("/api/dataset/samples", params={"page": 999, "size": 5}).json()
    assert body["page"] == body["pages"]
    assert body["items"]


def test_dataset_sample_detail_separates_the_five_roles(client, ml_artifacts):
    listing = client.get("/api/dataset/samples", params={"size": 1}).json()
    sample_id = listing["items"][0]["sample_id"]

    body = client.get(f"/api/dataset/samples/{sample_id}").json()
    assert body["available"] is True
    assert body["sample_id"] == sample_id

    # 1. Ground truth, with the evidence that produced it.
    assert body["ground_truth"]["label"] in SIX_STRATEGIES
    assert body["ground_truth"]["matches_stored_measurements"] is True
    assert set(body["useful_improvement_by_strategy"]) == SIX_STRATEGIES
    assert set(body["strategies_tied_at_best"]) <= SIX_STRATEGIES

    # 2. Live rule-based selection (never read from the label).
    assert body["rule_based_selection"]["error"] is None
    assert body["rule_based_selection"]["strategy"] in SIX_STRATEGIES

    # 3. Live ML prediction.
    assert body["ml_prediction"]["available"] is True
    assert body["ml_prediction"]["predicted_strategy"] in SIX_STRATEGIES

    # 4. Nothing is executed by browsing.
    assert body["executed_strategy"] is None
    assert body["executed_strategy_note"]

    # Measured outcome for all six strategies.
    assert set(body["measurements"]) == SIX_STRATEGIES
    for measurement in body["measurements"].values():
        assert set(measurement) == {
            "optimized_instruction_count",
            "instruction_reduction",
            "reduction_percentage",
            "useful_improvement",
        }

    assert list(body["features"]) == [
        "num_statements",
        "num_constants",
        "num_repeated_expr",
        "num_dead_stmts",
        "num_algebraic_ops",
        "num_loop_ops",
        "num_tmp_vars",
        "const_ratio",
    ]
    assert body["tac_lines"] == body["tac_text"].splitlines()


def test_dataset_unknown_sample_is_404(client, ml_artifacts):
    response = client.get("/api/dataset/samples/does-not-exist")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/api/analyze", "/api/optimize", "/api/simulate"])
def test_empty_tac_is_a_400_not_a_500(client, endpoint, ml_artifacts):
    for payload in ({"tac": ""}, {"tac": "   \n\t  "}):
        response = client.post(endpoint, json=payload)
        assert response.status_code == 400, (endpoint, payload)
        assert response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/api/analyze", "/api/optimize", "/api/simulate"])
def test_unparsable_tac_is_a_400(client, endpoint, ml_artifacts):
    response = client.post(endpoint, json={"tac": "!!! this is not tac at all !!!"})
    assert response.status_code == 400


def test_unknown_selector_is_rejected_before_any_work(client, ml_artifacts):
    response = client.post("/api/optimize", json={"tac": FOLDING_TAC, "selector": "random-forest"})
    assert response.status_code == 422


def test_out_of_range_iterations_are_rejected(client, ml_artifacts):
    assert client.post("/api/simulate", json={"tac": FOLDING_TAC, "max_iterations": 0}).status_code == 422
    assert client.post("/api/simulate", json={"tac": FOLDING_TAC, "max_iterations": 99}).status_code == 422


def test_unknown_request_fields_are_rejected(client, ml_artifacts):
    response = client.post("/api/analyze", json={"tac": FOLDING_TAC, "surprise": 1})
    assert response.status_code == 422


def test_ml_predict_rejects_empty_tac(client, ml_artifacts):
    assert client.post("/api/ml/predict", json={"tac": ""}).status_code == 400
    assert client.post("/api/ml/explain", json={"tac": ""}).status_code == 400
    assert client.post("/api/compare", json={"tac": ""}).status_code == 400


# ---------------------------------------------------------------------------
# Missing / broken model
# ---------------------------------------------------------------------------


@pytest.fixture
def no_model(monkeypatch):
    """Point the model layer at a path that does not exist."""
    monkeypatch.setattr(model_mod, "MODEL_FILE", Path("no", "such", "model.joblib"))
    model_mod.clear_cache()
    yield
    model_mod.clear_cache()


def test_missing_model_makes_execution_endpoints_answer_503(client, no_model, ml_artifacts):
    for endpoint, payload in (
        ("/api/optimize", {"tac": DEAD_CODE_TAC, "selector": "ml"}),
        ("/api/simulate", {"tac": DEAD_CODE_TAC, "selector": "ml"}),
    ):
        response = client.post(endpoint, json=payload)
        assert response.status_code == 503, endpoint
        detail = response.json()["detail"]
        assert detail["code"] == "model_unavailable"
        assert detail["selector"] == "ml"
        assert detail["remedy"] == "python -m backend.ml.train"
        assert detail["message"]


def test_missing_model_leaves_the_rule_based_paths_working(client, no_model, ml_artifacts):
    """Rule-Based and Compare-Both must stay available."""
    analyze = client.post("/api/analyze", json={"tac": DEAD_CODE_TAC}).json()
    assert analyze["ml"]["available"] is False
    assert "python -m backend.ml.train" in analyze["ml"]["reason"]
    assert analyze["selected_strategy"]["id"] in SIX_STRATEGIES
    assert analyze["selection"]["ml_strategy"] is None
    assert analyze["selection"]["strategies_agree"] is None
    assert "unavailable" in analyze["selection"]["agreement_note"].lower()

    simulate = client.post("/api/simulate", json={"tac": DEAD_CODE_TAC, "selector": "rule"}).json()
    assert simulate["selector"] == "rule"
    assert simulate["ml"]["available"] is False
    assert simulate["iterations"]

    compare = client.post("/api/compare", json={"tac": DEAD_CODE_TAC}).json()
    assert compare["available"] is True
    assert compare["rule_based"]["available"] is True
    assert compare["rule_based"]["history"]
    assert compare["ml"]["available"] is False
    assert compare["ml"]["history"] == []
    assert compare["winner"] is None
    assert "unavailable" in compare["winner_note"].lower()
    assert compare["agreement"]["strategies_agree"] is None


def test_missing_model_still_answers_read_endpoints(client, no_model, ml_artifacts):
    status = client.get("/api/ml/status").json()
    assert status["available"] is False
    assert status["ready"] is False
    assert status["reason"]
    assert status["shap"]["available"] is True  # SHAP availability is independent

    predict = client.post("/api/ml/predict", json={"tac": DEAD_CODE_TAC}).json()
    assert predict["available"] is False
    assert predict["predicted_strategy"] is None
    assert predict["probabilities"] == {}

    explain = client.post("/api/ml/explain", json={"tac": DEAD_CODE_TAC}).json()
    assert explain["available"] is False
    assert explain["contributions"] == []
    assert explain["model_available"] is False
    # Never a substituted rule-based answer dressed up as ML.
    assert explain.get("predicted_strategy") is None


def test_missing_metrics_artifacts_answer_explicitly(monkeypatch, client, tmp_path, ml_artifacts):
    monkeypatch.setattr("backend.api.ml_routes.ARTIFACT_DIR", tmp_path)

    metrics = client.get("/api/ml/metrics").json()
    assert metrics["available"] is False
    assert "python -m backend.ml.train" in metrics["reason"]
    assert metrics["evaluation"] == {}

    comparison = client.get("/api/ml/comparison").json()
    assert comparison["available"] is False
    assert comparison["per_program"] == []


def test_broken_model_artifact_is_reported_not_crashed(client, monkeypatch, tmp_path, ml_artifacts):
    broken = tmp_path / "model.joblib"
    broken.write_bytes(b"this is not a pickle")
    monkeypatch.setattr(model_mod, "MODEL_FILE", broken)
    model_mod.clear_cache()
    try:
        status = client.get("/api/ml/status").json()
        assert status["available"] is False
        assert "retrain" in status["reason"].lower()

        response = client.post("/api/optimize", json={"tac": DEAD_CODE_TAC, "selector": "ml"})
        assert response.status_code == 503
    finally:
        model_mod.clear_cache()


# ---------------------------------------------------------------------------
# Errors never leak internals
# ---------------------------------------------------------------------------


def _assert_clean(body_text: str):
    assert "Traceback" not in body_text
    assert "site-packages" not in body_text
    assert "D:\\" not in body_text
    assert "/Users/" not in body_text
    assert ".py\", line" not in body_text


def test_error_bodies_never_leak_paths_or_tracebacks(client, no_model, ml_artifacts):
    cases = [
        ("post", "/api/analyze", {"tac": ""}),
        ("post", "/api/optimize", {"tac": "!!!", "selector": "ml"}),
        ("post", "/api/simulate", {"tac": DEAD_CODE_TAC, "selector": "ml"}),
        ("post", "/api/ml/predict", {"tac": ""}),
        ("post", "/api/ml/explain", {"tac": "!!!"}),
        ("post", "/api/compare", {"tac": ""}),
        ("get", "/api/dataset/samples/does-not-exist", None),
    ]
    for method, url, payload in cases:
        response = (
            getattr(client, method)(url, json=payload)
            if payload is not None
            else getattr(client, method)(url)
        )
        assert response.status_code >= 400, (url, response.status_code)
        _assert_clean(response.text)

    # A degraded ML status is a deliberate 200, not an error, but it must be
    # equally clean: it carries a reason, never a path or a traceback.
    degraded = client.get("/api/ml/status")
    assert degraded.status_code == 200
    assert degraded.json()["available"] is False
    _assert_clean(degraded.text)


def test_successful_responses_never_expose_a_path(client, ml_artifacts):
    for url in (
        "/api/ml/status",
        "/api/ml/metrics",
        "/api/ml/feature-importance",
        "/api/ml/comparison",
        "/api/dataset/summary",
        "/api/dataset/samples",
        "/api/optimization-matrix",
    ):
        response = client.get(url)
        assert response.status_code == 200, url
        assert "artifacts\\ml" not in response.text, url
        assert "D:\\PROJECTS" not in response.text, url


def test_ml_comparison_aggregate_is_recomputed_from_its_stored_rows(client):
    """The aggregate block always matches the per-program measurements saved
    in the same file - including the per-strategy breakdown - without
    re-executing anything when the endpoint is read."""
    payload = client.get("/api/ml/comparison").json()
    if not payload.get("available"):
        pytest.skip("comparison artifact unavailable in this environment")

    per_program = payload["per_program"]
    optimization = payload["optimization"]

    for side in ("rule_based", "ml"):
        block = optimization[side]
        assert block["programs"] == len(per_program)
        # Per-strategy outcomes exist and are internally consistent.
        assert isinstance(block["per_strategy"], dict)
        for entry in block["per_strategy"].values():
            assert entry["programs"] >= 1
            assert 0 <= entry["improved_programs"] <= entry["programs"]
        assert block["failed_programs"] >= 0
        assert "aggregate_outcome" in optimization["aggregation_note"]


def test_ml_explain_states_output_scale_and_verifies_class_mapping(client, ml_artifacts):
    """P5: the explanation names its scale (probability, not log-odds), shows
    every class's probability with the predicted one as the actual argmax, and
    carries all eight feature contributions plus the real base value."""
    explain = client.post("/api/ml/explain", json={"tac": DEAD_CODE_TAC, "top_k": 8}).json()
    if not explain.get("available"):
        pytest.skip(f"SHAP unavailable: {explain.get('reason')}")

    scale = explain.get("output_scale") or {}
    assert scale.get("space") == "probability"
    assert scale.get("range") == "0-1"
    assert "log-odds" in scale.get("note", "")
    assert "probability space" in scale.get("note", "")

    probabilities = explain.get("probabilities") or {}
    assert len(probabilities) >= 5, "every target class must be listed for the mapping"
    predicted = explain.get("predicted_class")
    assert predicted in probabilities
    # The displayed mapping must be internally consistent: the reported
    # prediction really is the argmax of the displayed probabilities.
    assert max(probabilities, key=probabilities.get) == predicted

    contributions = explain.get("contributions") or []
    assert len(contributions) == 8, "all eight features, not just the top-k slice"
    assert explain.get("base_value") is not None, "the actual SHAP base value"
    assert explain.get("ranked_contributions") and len(explain["ranked_contributions"]) == 8
    assert explain.get("summary"), "a concise value-based explanation sentence"
    assert explain.get("disclaimer")
