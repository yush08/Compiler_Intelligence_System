"""Training, persistence, prediction and SHAP explainability tests.

The session fixture ``ml_artifacts`` performs one genuine end-to-end training
run (real ``train()``, real grid, real artifact directory), so these tests read
exactly the model a fresh clone would produce.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.features.extractor import FEATURE_KEYS, FEATURE_DESCRIPTIONS
from backend.ml import dataset as ds
from backend.ml import explain as explain_mod
from backend.ml import model as model_mod
from backend.ml import train as train_mod
from backend.ml.evaluate import aggregate_outcome, classification_metrics
from backend.ml.explain import explain_prediction, global_importance, shap_status
from backend.ml.model import model_status
from backend.ml.predict import predict_from_features, predict_from_tac
from backend.tac.parser import parse_tac

from conftest import DEAD_CODE_TAC, requires_ml

pytestmark = requires_ml


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------


def test_dataset_validation_passes(ml_artifacts):
    report = ds.validate_dataset(ds.load_dataset())
    assert report["passed"] is True, report["errors"]
    assert report["rows"] == 241
    assert report["columns"] == 46


def test_feature_order_is_identical_everywhere():
    """One ordering, three owners: generator, extractor and ML pipeline."""
    import sys
    from pathlib import Path as _Path

    experiments = _Path(train_mod.__file__).resolve().parents[2] / "experiments"
    sys.path.insert(0, str(experiments))
    try:
        from generate_dataset import FEATURE_KEYS as GENERATOR_KEYS
    finally:
        sys.path.pop(0)

    assert GENERATOR_KEYS == FEATURE_KEYS
    assert ds.FEATURE_KEYS == FEATURE_KEYS
    assert list(FEATURE_DESCRIPTIONS) == FEATURE_KEYS


def test_only_the_eight_features_reach_the_model(ml_artifacts):
    from backend.ml.model import load_payload

    loaded = load_payload()
    assert loaded["feature_names"] == FEATURE_KEYS
    assert loaded["model"].n_features_in_ == len(FEATURE_KEYS)
    assert list(loaded["model"].feature_names_in_) == FEATURE_KEYS


def test_no_leaking_column_is_a_model_input():
    assert not (set(ds.FEATURE_KEYS) & ds.LEAKY_COLUMNS)
    for name in ("best_strategy", "sample_id", "split", "tac_program"):
        assert name not in ds.FEATURE_KEYS


def test_label_matches_stored_measurements():
    """Every stored best_strategy is the argmax of the six useful_improvement values."""
    df = ds.load_dataset()
    mismatches = 0
    for _, row in df.iterrows():
        values = {
            name: {"useful_improvement": int(row[ds.label_measurement_column(name)])}
            for name in ds.STRATEGIES
        }
        from backend.optimizer.measure import best_strategy_from
        from backend.scoring.rule_based import STRATEGY_TIE_ORDER

        if best_strategy_from(values, STRATEGY_TIE_ORDER) != row[ds.TARGET_COLUMN]:
            mismatches += 1
    assert mismatches == 0


def test_feature_matrix_is_restricted_to_the_eight_columns():
    df = ds.load_dataset()
    X, y = ds.feature_matrix(df)
    assert list(X.columns) == FEATURE_KEYS
    assert X.shape == (241, 8)
    assert set(y.unique()) <= set(ds.STRATEGIES)


# ---------------------------------------------------------------------------
# Training and artifacts
# ---------------------------------------------------------------------------


def test_training_produces_every_artifact(ml_artifacts, model_payload):
    directory = model_mod.MODEL_DIR
    for name in (
        "model.joblib",
        "metadata.json",
        "evaluation.json",
        "confusion_matrix.json",
        "classification_report.json",
        "feature_importance.json",
        "comparison.json",
    ):
        assert (directory / name).is_file(), f"missing artifact: {name}"
        assert (directory / name).stat().st_size > 0


def test_artifact_records_feature_and_class_order(model_payload, ml_artifacts):
    classes = [str(c) for c in model_payload["model"].classes_]
    assert model_payload["class_names"] == classes
    assert model_payload["feature_names"] == FEATURE_KEYS
    metadata = model_payload["metadata"]
    assert metadata["feature_names"] == FEATURE_KEYS
    assert metadata["seed"] == train_mod.DEFAULT_SEED
    assert metadata["n_estimators"] == train_mod.N_ESTIMATORS
    assert metadata["training_command"] == "python -m backend.ml.train"


def test_five_classes_and_no_strength_reduction(model_payload, ml_artifacts):
    df = ds.load_dataset()
    vocabulary = ds.discover_target_classes(df)
    classes = [str(c) for c in model_payload["model"].classes_]

    # Discovered from the data - never hard-coded to six.
    assert len(vocabulary) == 5
    assert "strength_reduction" not in vocabulary
    assert set(classes) <= set(vocabulary)
    assert "strength_reduction" not in classes


def test_metrics_reported_for_all_three_splits(ml_artifacts):
    evaluation = json.loads(
        (model_mod.MODEL_DIR / "evaluation.json").read_text(encoding="utf-8")
    )
    assert set(evaluation["splits"]) == {"train", "validation", "test"}
    assert evaluation["held_out_split"] == "test"
    assert evaluation["primary_metric"] == "macro_f1"
    for name in ("train", "validation", "test"):
        block = evaluation["splits"][name]
        for key in ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"):
            assert key in block, f"{name} missing {key}"
        assert 0.0 <= block["accuracy"] <= 1.0
        assert block["evaluated_samples"] == evaluation["sample_counts"][name]
    assert evaluation["limitations"], "limitations must be documented"


def test_tuning_never_sees_the_test_split(ml_artifacts):
    metadata = json.loads(
        (model_mod.MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    tuning = metadata["tuning"]
    assert tuning["candidates"] == len(train_mod.GRID)
    for candidate in tuning["candidate_results"]:
        # Only configuration + validation score: the test split is absent.
        assert set(candidate) == {"config", "validation_macro_f1"}
    assert set(tuning["selected"]) == {"max_depth", "min_samples_leaf", "class_weight"}


def test_confusion_matrix_cells_sum_to_evaluated_samples(ml_artifacts):
    payload = json.loads(
        (model_mod.MODEL_DIR / "confusion_matrix.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (model_mod.MODEL_DIR / "evaluation.json").read_text(encoding="utf-8")
    )
    for split, block in payload["splits"].items():
        total = sum(sum(row) for row in block["matrix"])
        assert total == evaluation["splits"][split]["evaluated_samples"], split
        assert len(block["matrix"]) == len(block["labels"])


def test_training_is_reproducible(ml_artifacts, model_payload):
    """Same seed + same config => identical predictions."""
    from sklearn.ensemble import RandomForestClassifier

    df = ds.load_dataset()
    frames = ds.split_frames(df)
    X_train, y_train = ds.feature_matrix(frames["train"])
    X_test, _ = ds.feature_matrix(frames["test"])
    config = model_payload["config"]

    def fit():
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight=config["class_weight"],
            random_state=config["random_state"],
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf

    first, second = fit(), fit()
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
    assert np.allclose(first.feature_importances_, second.feature_importances_)


def test_metadata_fingerprint_matches_the_dataset_used(ml_artifacts):
    metadata = json.loads(
        (model_mod.MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    df = ds.load_dataset()
    assert metadata["dataset_fingerprint"] == ds.dataset_fingerprint(df)
    assert metadata["dataset_rows"] == len(df)
    assert len(metadata["dataset_sha256"]) == 64


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_predict_probabilities_are_a_valid_distribution(ml_artifacts):
    result = predict_from_tac(parse_tac(DEAD_CODE_TAC))
    assert result["available"] is True
    assert result["predicted_strategy"] in result["classes"]
    total = sum(result["probabilities"].values())
    # The API rounds each probability to 6 decimals for display, so the sum can
    # drift by up to ~5e-6 (6 classes x 5e-7 rounding). 1e-6 was tighter than
    # that rounding can satisfy; 1e-5 checks the distribution without failing on
    # a display artifact.
    assert abs(total - 1.0) < 1e-5, total
    assert result["predicted_probability"] == pytest.approx(
        max(result["probabilities"].values()), abs=1e-9
    )


def test_predict_never_invents_a_strength_reduction_probability(ml_artifacts):
    result = predict_from_tac(parse_tac("i = 0\nloop:\ni = i + 1\nif i < n goto loop\n"))
    assert result["available"] is True
    assert "strength_reduction" not in result["probabilities"]
    assert "strength_reduction" in result["unsupported_classes"]


def test_probabilities_are_not_labelled_as_calibrated(ml_artifacts):
    result = predict_from_tac(parse_tac(DEAD_CODE_TAC))
    assert result["probabilities_calibrated"] is False
    assert "not calibrated" in result["note"]


def test_prediction_matches_a_direct_model_call(ml_artifacts, model_payload):
    features = {k: float(v) for k, v in zip(FEATURE_KEYS, [6, 4, 1, 2, 1, 0, 3, 0.66])}
    result = predict_from_features(features)

    frame = ds.feature_frame(features)
    direct = model_payload["model"].predict_proba(frame)[0]
    classes = [str(c) for c in model_payload["model"].classes_]

    assert result["predicted_strategy"] == classes[int(np.argmax(direct))]
    for index, name in enumerate(classes):
        assert result["probabilities"][name] == pytest.approx(float(direct[index]), abs=1e-6)


def test_probability_columns_follow_model_classes_not_a_sorted_list(ml_artifacts, model_payload):
    """Guard against attaching a probability to the wrong strategy."""
    classes = [str(c) for c in model_payload["model"].classes_]
    features = {k: 1.0 for k in FEATURE_KEYS}
    result = predict_from_features(features)
    assert list(result["probabilities"]) == classes


def test_missing_feature_is_reported_not_imputed(ml_artifacts):
    result = predict_from_features({"num_statements": 3})
    assert result["available"] is False
    assert "Missing feature value(s)" in result["reason"]


def test_predict_reports_unavailable_when_artifact_missing(monkeypatch, ml_artifacts):
    monkeypatch.setattr(model_mod, "MODEL_FILE", Path("no", "such", "model.joblib"))
    try:
        status = model_mod.model_status()
        assert status["available"] is False
        assert status["ready"] is False
        assert "python -m backend.ml.train" in status["reason"]

        result = predict_from_features({k: 1.0 for k in FEATURE_KEYS})
        assert result["available"] is False
        assert result["predicted_strategy"] is None
        assert "python -m backend.ml.train" in result["reason"]
    finally:
        model_mod.clear_cache()


def test_malformed_artifact_is_reported_not_crashed(monkeypatch, ml_artifacts, tmp_path):
    broken = tmp_path / "model.joblib"
    broken.write_bytes(b"not a pickle")
    monkeypatch.setattr(model_mod, "MODEL_FILE", broken)
    try:
        result = predict_from_features({k: 1.0 for k in FEATURE_KEYS})
        assert result["available"] is False
        assert "retrain" in result["reason"].lower()
    finally:
        model_mod.clear_cache()


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


def test_shap_is_available_in_this_environment(ml_artifacts):
    status = shap_status()
    assert status["available"] is True, status["reason"]
    assert status["version"]
    assert "causation" in status["scope"]


def test_shap_explains_the_predicted_class_with_all_eight_features(ml_artifacts):
    features = ds.feature_matrix(ds.load_dataset())[0].iloc[0].to_dict()
    result = explain_prediction(features)

    assert result["available"] is True, result.get("reason")
    assert result["explainer"] == "TreeExplainer"
    assert len(result["contributions"]) == len(FEATURE_KEYS)
    assert [c["feature"] for c in result["contributions"]] == FEATURE_KEYS
    # The explanation is bound to the predicted class specifically.
    assert result["predicted_class"] in result["probabilities"]
    assert len(result["probabilities"]) == 5
    assert result["predicted_class_index"] == list(result["probabilities"]).index(
        result["predicted_class"]
    )
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_shap_contributions_reconstruct_the_model_probability(ml_artifacts, model_payload):
    """The authenticity check: base + sum(SHAP) must equal predict_proba."""
    features = {k: float(v) for k, v in zip(FEATURE_KEYS, [6, 4, 1, 2, 1, 0, 3, 0.66])}
    result = explain_prediction(features)

    assert result["available"] is True
    assert result["base_value"] is not None
    assert result["reconstructed_probability"] == pytest.approx(
        result["model_probability"], abs=1e-6
    )

    # And it must agree with a direct predict_proba call.
    frame = ds.feature_frame(features)
    classes = [str(c) for c in model_payload["model"].classes_]
    direct = model_payload["model"].predict_proba(frame)[0]
    index = classes.index(result["predicted_class"])
    assert result["model_probability"] == pytest.approx(float(direct[index]), abs=1e-9)


def test_shap_ranks_features_by_absolute_contribution(ml_artifacts):
    features = {k: 0.0 for k in FEATURE_KEYS}
    features["num_dead_stmts"] = 3.0
    result = explain_prediction(features)

    magnitudes = [abs(c["shap_value"]) for c in result["ranked_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [c["mean_abs_rank"] for c in result["ranked_contributions"]] == list(
        range(1, len(FEATURE_KEYS) + 1)
    )
    assert result["supporting"] and all(e["shap_value"] > 0 for e in result["supporting"])
    assert all(e["shap_value"] < 0 for e in result["opposing"])
    assert result["disclaimer"] and "causation" in result["disclaimer"]


def test_shap_summary_is_readable_text(ml_artifacts):
    result = explain_prediction({k: 2.0 for k in FEATURE_KEYS})
    summary = result["summary"]
    assert isinstance(summary, str) and summary
    assert result["predicted_class"] in summary


def test_global_importance_labels_native_and_shap_separately(ml_artifacts):
    payload = global_importance()
    assert payload["available"] is True
    assert payload["native"]["available"] is True
    assert payload["shap"]["available"] is True

    for block in (payload["native"], payload["shap"]):
        features = block["features"]
        assert [f["feature"] for f in features] == FEATURE_KEYS or set(
            f["feature"] for f in features
        ) == set(FEATURE_KEYS)
        assert sorted(f["rank"] for f in features) == list(range(1, len(FEATURE_KEYS) + 1))

    assert "impurity" in payload["native"]["source"].lower()
    assert "shap" in payload["shap"]["source"].lower()
    assert payload["comparison_note"]

    native_keys = {f["feature"] for f in payload["native"]["features"]}
    shap_keys = {f["feature"] for f in payload["shap"]["features"]}
    assert native_keys == shap_keys == set(FEATURE_KEYS)


def test_shap_falls_back_to_an_explicit_unavailable_state(monkeypatch, ml_artifacts):
    monkeypatch.setitem(explain_mod._STATE, "checked", True)
    monkeypatch.setitem(explain_mod._STATE, "available", False)
    monkeypatch.setitem(explain_mod._STATE, "reason", "SHAP is not installed (ImportError).")
    monkeypatch.setitem(explain_mod._STATE, "explainer", None)

    status = shap_status()
    assert status["available"] is False
    assert "not installed" in status["reason"]

    result = explain_prediction({k: 1.0 for k in FEATURE_KEYS})
    assert result["available"] is False
    assert "not installed" in result["reason"]
    # Crucially: no fabricated numbers.
    assert "contributions" not in result
    assert "ranked_contributions" not in result
    assert result.get("summary") is None
    assert result.get("total_contribution") is None

    # Native importance keeps working and is never relabelled as SHAP.
    payload = global_importance()
    assert payload["native"]["available"] is True
    assert payload["native"]["features"]
    assert "impurity" in payload["native"]["source"].lower()

    # Global SHAP importance is read from the file written during training, so
    # values that were really computed remain valid even if SHAP is unavailable
    # at inference time. The live status is reported separately via /api/ml/status.
    assert payload["shap"]["available"] is True
    assert payload["shap"]["features"]

    # Model readiness is unaffected: only SHAP degrades.
    status = model_status()
    assert status["available"] is True
    assert status["shap"]["available"] is False


def test_global_importance_reports_unavailable_when_never_computed(
    monkeypatch, ml_artifacts, tmp_path
):
    monkeypatch.setattr(explain_mod, "IMPORTANCE_FILE", tmp_path / "feature_importance.json")

    payload = global_importance()
    assert payload["available"] is False
    assert "python -m backend.ml.train" in payload["reason"]
    assert payload["native"]["available"] is False
    assert payload["shap"]["available"] is False
    assert payload["shap"]["features"] == []


def test_global_importance_distinguishes_a_shap_free_training_run(
    monkeypatch, ml_artifacts, tmp_path
):
    """If training ran without SHAP, the file says so instead of showing zeros."""
    path = tmp_path / "feature_importance.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00",
                "model_version": "rf-v1-test",
                "native": {"available": True, "features": [{"feature": "num_statements", "importance": 1.0, "rank": 1}]},
                "shap": {"available": False, "features": [], "reason": "SHAP is not installed."},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(explain_mod, "IMPORTANCE_FILE", path)

    payload = global_importance()
    assert payload["available"] is True
    assert payload["native"]["available"] is True
    assert payload["shap"]["available"] is False
    assert payload["shap"]["reason"] == "SHAP is not installed."
    assert payload["shap"]["features"] == []


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def test_classification_metrics_handles_absent_classes():
    labels = ["a", "b", "c"]
    metrics = classification_metrics(["a", "a", "b"], ["a", "b", "b"], labels)
    # Class "c" has no support: reported as absent, not silently dropped.
    assert metrics["per_class"]["c"]["support"] == 0
    assert metrics["per_class"]["c"]["evaluated"] is False
    assert metrics["absent_classes"] == ["c"]
    assert metrics["evaluated_samples"] == 3
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == 3


def test_classification_metrics_flags_predictions_outside_the_vocabulary():
    labels = ["a", "b"]
    metrics = classification_metrics(["a", "b"], ["a", "zzz"], labels)
    assert metrics["predicted_outside_vocabulary"] == ["zzz"]
    # The out-of-vocabulary prediction still counts as an error and still
    # appears in the matrix, so no evaluated sample is lost.
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == 2
    assert "zzz" in metrics["confusion_matrix_labels"]
    assert metrics["per_class"]["b"]["recall"] == 0.0


def test_aggregate_outcome_reports_ties_and_failures():
    rows = [
        {"strategy": "x", "useful_improvement": 3, "instruction_reduction": 1, "operation_reduction": 2},
        {"strategy": "y", "useful_improvement": 0, "instruction_reduction": 0, "operation_reduction": 0},
        {"strategy": "y", "useful_improvement": -1, "instruction_reduction": 0, "operation_reduction": -1},
        {"strategy": "x", "useful_improvement": 0, "failed": True},
    ]
    outcome = aggregate_outcome(rows)
    assert outcome["programs"] == 4
    assert outcome["improved_programs"] == 1
    assert outcome["regressed_programs"] == 1
    assert outcome["failed_programs"] == 1
    assert outcome["unchanged_programs"] == 1
    assert outcome["strategy_distribution"] == {"x": 2, "y": 2}


def test_empty_aggregate_outcome_is_safe():
    outcome = aggregate_outcome([])
    assert outcome["programs"] == 0
    assert outcome["improved_pct"] == 0.0
