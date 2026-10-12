"""Train the Random Forest strategy selector.

Run from the project root::

    python -m backend.ml.train

What it does
------------
1. Loads ``dataset/processed/optimization_dataset.csv`` and refuses to proceed
   if validation reports an error.
2. Respects the dataset's **existing** train/validation/test assignments.  No
   sample is ever moved between splits.
3. Selects hyperparameters from a small, fixed grid using **validation**
   macro-F1 only.  The test split is touched exactly once, at the end.
4. Fits the final forest on the **training** split.
5. Evaluates that one model on all three splits (validation is post-selection
   and therefore mildly optimistic - stated in the output).
6. Persists the estimator, its metadata, the evaluation, the feature
   importance (native and SHAP, kept distinct) and the Rule-Based vs ML
   comparison.

The final model is deliberately *not* refitted on train+validation: keeping it
trained on exactly the 124 training rows means every reported metric describes
the artifact that was saved, with no train/validation ambiguity.

Reproducibility: a single fixed seed drives the split, the grid search and the
forest, so two runs on the same dataset produce byte-identical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.extractor import FEATURE_KEYS
from .dataset import (
    DEFAULT_DATASET_PATH,
    TARGET_COLUMN,
    discover_target_classes,
    feature_matrix,
    load_dataset,
    split_frames,
    validate_dataset,
)
from .evaluate import classification_metrics

__all__ = ["train", "DEFAULT_SEED", "MODEL_VERSION_PREFIX"]

DEFAULT_SEED = 20261005
MODEL_VERSION_PREFIX = "rf-v1"

# Trees only reduce variance; more of them cannot overfit. Not worth tuning.
N_ESTIMATORS = 300

# A small fixed grid - not a random or Bayesian search. The dataset has 124
# training rows, so anything larger would be fitting noise.
GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
    for depth in (None, 8, 12)
    for leaf in (1, 2, 4)
    for weight in (None, "balanced")
]

LIMITATIONS = [
    "177 programs total, 124 used for training: every reported metric is "
    "preliminary and generalization to unseen TAC programs is not established.",
    "The corpus is generator-derived: 15 of the 17 generation categories carry "
    "exactly one winning label, and every category present in the test split "
    "also appears in the training split. A high test score therefore shows "
    "that the forest learned the generator's opportunity patterns - it is not "
    "evidence that the model generalizes to arbitrary handwritten TAC.",
    "Held-out category coverage is incomplete: 4 of the 17 generation "
    "categories (algebraic_simplification, common_subexpression, "
    "dead_code_elimination, no_opportunity) contribute zero test samples, so "
    "those generation patterns are never evaluated directly. All five target "
    "labels still appear in the test split with support between 2 and 8, but "
    "per-category behaviour on those four is unmeasured.",
    "num_dead_stmts is the strongest predictor (SHAP rank 1). It is a static "
    "liveness count computed from the instruction stream, not the output of "
    "running the dead-code pass, but it does directly measure the opportunity "
    "that dead_code_elimination exploits.",
    "Classes are imbalanced (59 vs 18 in the largest/smallest class), so "
    "macro-F1 is the primary metric rather than accuracy.",
    "strength_reduction has no winning examples, so it is absent from the "
    "target vocabulary and the model can never predict it.",
    "strength_reduction is currently a documented no-op in the engine, which "
    "is why it never wins a label.",
    "The 10 no_opportunity programs all share the tie-break label chosen when "
    "every pass scores zero, so part of the learned mapping reproduces the "
    "deterministic tie rule rather than an optimization insight.",
    "Ground-truth labels are defined by optimizer.measure's "
    "useful_improvement (statement reduction + operation reduction), so "
    "'correct' means 'agrees with that definition', not 'provably optimal'.",
    "Only 26 held-out test samples exist, so the confusion matrix is sparse "
    "and per-class recall on the test split is unstable (support ranges from "
    "2 to 8).",
    "Validation metrics are reported for the configuration selected on that "
    "same split, so they are mildly optimistic; the test split is untouched.",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_key(candidate: dict) -> tuple:
    """Deterministic ordering: validation macro-F1, then accuracy, then simpler."""
    depth_rank = 999 if candidate["config"]["max_depth"] is None else candidate["config"]["max_depth"]
    return (
        -candidate["validation"]["macro_f1"],
        -candidate["validation"]["accuracy"],
        depth_rank,
        -candidate["config"]["min_samples_leaf"],
        0 if candidate["config"]["class_weight"] is None else 1,
        str(candidate["config"]),
    )


def _fit(seed: int, config: dict):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        class_weight=config["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def _evaluate_splits(model, frames, labels: list[str]) -> dict:
    from .dataset import FEATURE_KEYS as _KEYS  # explicit: only these inputs

    results = {}
    for name, frame in frames.items():
        X, y = feature_matrix(frame)
        X = X.loc[:, _KEYS]
        predictions = [str(p) for p in model.predict(X)]
        results[name] = classification_metrics(list(y.astype(str)), predictions, labels)
        results[name]["samples"] = int(len(frame))
    return results


def _native_importance(model) -> dict:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"available": False, "features": [], "reason": "Model exposes no feature_importances_."}

    ranked = sorted(
        (
            {"feature": name, "importance": round(float(value), 6), "rank": 0}
            for name, value in zip(FEATURE_KEYS, importances)
        ),
        key=lambda item: -item["importance"],
    )
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return {"available": True, "features": ranked}


def _shap_importance(model, X_train: pd.DataFrame) -> dict:
    """Global mean|SHAP| per feature over the training split.

    Returns an explicit ``available: false`` state if SHAP cannot run - the
    native importances are never relabelled as SHAP values.
    """
    from .explain import shap_status

    status = shap_status()
    if not status["available"]:
        return {"available": False, "features": [], "per_class": {}, "reason": status["reason"]}

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_train)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the training run
        return {
            "available": False,
            "features": [],
            "per_class": {},
            "reason": f"SHAP could not run ({type(exc).__name__}: {exc}).",
        }

    classes = [str(c) for c in getattr(model, "classes_", [])]
    array = _stack_shap(values, len(X_train), len(FEATURE_KEYS), len(classes))

    per_class: dict = {}
    for index, class_name in enumerate(classes):
        mean_abs = np.abs(array[:, :, index]).mean(axis=0)
        per_class[class_name] = _ranked(mean_abs)

    overall = np.abs(array).mean(axis=2).mean(axis=0)
    return {
        "available": True,
        "reason": None,
        "explainer": "TreeExplainer",
        "computed_over": f"training split ({len(X_train)} rows)",
        "aggregation": "mean(|SHAP value|) per feature, averaged across target classes",
        "features": _ranked(overall),
        "per_class": per_class,
    }


def _stack_shap(values, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        stacked = np.stack([np.asarray(v, dtype=float) for v in values], axis=-1)
        return stacked.reshape(n_samples, n_features, len(values))
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array.reshape(n_samples, n_features, 1)
    if array.ndim == 3:
        return array
    raise ValueError(f"unexpected SHAP output shape: {array.shape}")


def _ranked(values: np.ndarray) -> list[dict]:
    ranked = sorted(
        (
            {"feature": name, "mean_abs_shap": round(float(value), 6), "rank": 0}
            for name, value in zip(FEATURE_KEYS, values)
        ),
        key=lambda item: -item["mean_abs_shap"],
    )
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def train(
    dataset_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    skip_comparison: bool = False,
    verbose: bool = True,
) -> dict:
    """Full training pipeline. Returns the training summary dict."""
    from .model import MODEL_DIR, save_artifacts

    from .compare import compare_on_dataset

    started = datetime.now(timezone.utc)
    dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    out_path = Path(out_dir) if out_dir else MODEL_DIR

    def say(message: str = "") -> None:
        if verbose:
            print(message)

    say("=" * 78)
    say("COMPILER INTELLIGENCE SYSTEM - Random Forest strategy selector")
    say("=" * 78)

    # --- 1. load + validate ------------------------------------------------
    try:
        df = load_dataset(dataset_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    report = validate_dataset(df)
    say(f"Dataset       : {dataset_path.name}")
    say(f"Rows/columns  : {report['rows']} x {report['columns']}")
    say(f"Fingerprint   : {report.get('fingerprint')}")
    say(f"Validation    : {'PASSED' if report['passed'] else 'FAILED'}")
    for warning in report["warnings"]:
        say(f"  warning     : {warning}")

    if not report["passed"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}", file=sys.stderr)
        print("Dataset validation failed; not training.", file=sys.stderr)
        raise SystemExit(1)

    labels = discover_target_classes(df)
    frames = split_frames(df)
    say(f"Target classes: {labels} ({len(labels)} discovered from the data)")
    say(
        "Splits        : "
        + ", ".join(f"{name}={len(frame)}" for name, frame in frames.items())
    )

    # --- 2. inputs ---------------------------------------------------------
    X_all, y_all = feature_matrix(df)
    X_train, y_train = feature_matrix(frames["train"])
    X_val, y_val = feature_matrix(frames["validation"])
    X_test, y_test = feature_matrix(frames["test"])

    train_classes = sorted(set(y_train.astype(str)))
    missing_in_train = [name for name in labels if name not in train_classes]
    if missing_in_train:
        say(
            "  warning     : class(es) absent from the training split, so the "
            f"model cannot learn them: {', '.join(missing_in_train)}"
        )

    # --- 3. small grid search on VALIDATION macro-F1 only ------------------
    say("")
    say(f"Hyperparameter search: {len(GRID)} fixed configurations, "
        "selected by validation macro-F1")
    say("-" * 78)
    say(f"{'max_depth':>10} {'min_leaf':>9} {'class_weight':>14} {'val macro-F1':>13} {'val acc':>9}")
    candidates = []
    for config in GRID:
        model = _fit(seed, config)
        model.fit(X_train, y_train)
        predictions = [str(p) for p in model.predict(X_val)]
        metrics = classification_metrics(list(y_val.astype(str)), predictions, labels)
        candidate = {"config": dict(config), "validation": metrics}
        candidates.append(candidate)
        say(
            f"{str(config['max_depth']):>10} {config['min_samples_leaf']:>9} "
            f"{str(config['class_weight']):>14} {metrics['macro_f1']:>13.4f} "
            f"{metrics['accuracy']:>9.4f}"
        )

    candidates.sort(key=_rank_key)
    best = candidates[0]
    say("-" * 78)
    say(f"Selected      : {best['config']}")

    # --- 4. fit the final model on the TRAINING split ----------------------
    model = _fit(seed, best["config"])
    model.fit(X_train, y_train)

    # The probability columns are ordered by model.classes_; saving anything
    # else would silently mislabel every prediction.
    class_names = [str(c) for c in model.classes_]
    assert class_names == sorted(labels) or set(class_names) <= set(labels), (
        f"model classes {class_names} disagree with the dataset vocabulary {labels}"
    )

    fingerprint = report["fingerprint"]
    model_version = f"{MODEL_VERSION_PREFIX}-{fingerprint}"
    trained_at = started.isoformat(timespec="seconds")

    # --- 5. evaluate on all three splits -----------------------------------
    split_metrics = _evaluate_splits(model, frames, labels)

    # --- 6. feature importance (native and SHAP, kept distinct) ------------
    native = _native_importance(model)
    shap_block = _shap_importance(model, X_train)

    importance_payload = {
        "generated_at": trained_at,
        "model_version": model_version,
        "native": native,
        "shap": shap_block,
        "comparison_note": (
            "Native importance is the forest's mean impurity decrease; SHAP "
            "importance is mean |contribution| to the model output. They are "
            "different measures and may rank features differently."
        ),
    }

    # --- 7. persist --------------------------------------------------------
    metadata = {
        "model_version": model_version,
        "model_type": type(model).__name__,
        "trained_at": trained_at,
        "seed": seed,
        "n_estimators": N_ESTIMATORS,
        "best_config": best["config"],
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "dataset_fingerprint": fingerprint,
        "dataset_sha256": _file_sha256(dataset_path),
        "dataset_file": dataset_path.name,
        "target_classes": labels,
        "model_classes": class_names,
        "feature_names": list(FEATURE_KEYS),
        "primary_metric": "macro_f1",
        "held_out_split": "test",
        "tuning": {
            "strategy": "fixed grid, selected by validation macro-F1",
            "candidates": len(GRID),
            "candidate_results": [
                {"config": c["config"], "validation_macro_f1": c["validation"]["macro_f1"]}
                for c in candidates
            ],
            "selected": best["config"],
        },
        "limitations": LIMITATIONS,
        "training_command": "python -m backend.ml.train",
    }

    config = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": best["config"]["max_depth"],
        "min_samples_leaf": best["config"]["min_samples_leaf"],
        "class_weight": best["config"]["class_weight"],
        "random_state": seed,
        "n_jobs": -1,
    }

    save_artifacts(
        model,
        feature_names=list(FEATURE_KEYS),
        class_names=class_names,
        config=config,
        metadata=metadata,
        out_dir=out_path,
    )

    evaluation = {
        "model_version": model_version,
        "generated_at": trained_at,
        "held_out_split": "test",
        "primary_metric": "macro_f1",
        "target_classes": labels,
        "model_classes": class_names,
        "sample_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "splits": split_metrics,
        "tuning_note": (
            "The selected configuration was chosen on the validation split, so "
            "validation metrics are mildly optimistic. The test split was used "
            "exactly once, for final evaluation."
        ),
        "metrics_definition": {
            "accuracy": "fraction of correct predictions",
            "macro_f1": "unweighted mean of per-class F1; primary metric under imbalance",
            "weighted_f1": "support-weighted mean of per-class F1",
        },
        "limitations": LIMITATIONS,
    }

    confusion = {
        "model_version": model_version,
        "generated_at": trained_at,
        "splits": {
            name: {
                "labels": metrics["confusion_matrix_labels"],
                "matrix": metrics["confusion_matrix"],
                "note": metrics["confusion_matrix_note"],
            }
            for name, metrics in split_metrics.items()
        },
    }

    classification_report_payload = {
        "model_version": model_version,
        "generated_at": trained_at,
        "splits": {name: metrics["per_class"] for name, metrics in split_metrics.items()},
        "note": (
            "A class with support 0 in a split is flagged evaluated=false and "
            "was not meaningfully assessed in that split."
        ),
    }

    out_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "evaluation.json", evaluation)
    _write_json(out_path / "confusion_matrix.json", confusion)
    _write_json(out_path / "classification_report.json", classification_report_payload)
    _write_json(out_path / "feature_importance.json", importance_payload)

    # --- 8. Rule-Based vs ML on the held-out split --------------------------
    comparison = None
    if not skip_comparison:
        comparison = compare_on_dataset(df, split="test", payload={
            "model": model,
            "feature_names": list(FEATURE_KEYS),
            "class_names": class_names,
            "config": config,
            "metadata": metadata,
        })
        comparison["model_version"] = model_version
        _write_json(out_path / "comparison.json", comparison)

    # --- 9. report ---------------------------------------------------------
    say("")
    say("Results (train / validation / held-out test)")
    say("-" * 78)
    say(f"{'split':<12} {'n':>4} {'accuracy':>10} {'macro-F1':>10} {'weighted-F1':>13}")
    for name in ("train", "validation", "test"):
        metrics = split_metrics[name]
        say(
            f"{name:<12} {metrics['evaluated_samples']:>4} {metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} {metrics['weighted_f1']:>13.4f}"
        )

    say("")
    say("Held-out test - per-class F1")
    say("-" * 78)
    say(f"{'class':<28} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}")
    for name in labels:
        cell = split_metrics["test"]["per_class"][name]
        say(
            f"{name:<28} {cell['precision']:>10.4f} {cell['recall']:>8.4f} "
            f"{cell['f1']:>8.4f} {cell['support']:>8}"
        )

    say("")
    say("Held-out test - confusion matrix")
    say("-" * 78)
    cm_labels = split_metrics["test"]["confusion_matrix_labels"]
    matrix = split_metrics["test"]["confusion_matrix"]
    width = max((len(name) for name in cm_labels), default=8)
    say(" " * (width + 2) + " ".join(f"{i:>3}" for i in range(len(cm_labels))))
    for index, row in enumerate(matrix):
        say(f"{index:>3} {cm_labels[index]:<{width}} " + " ".join(f"{v:>3}" for v in row))
    say("    (" + ", ".join(f"{i}={name}" for i, name in enumerate(cm_labels)) + ")")

    if shap_block.get("available"):
        say("")
        say("Global feature importance")
        say("-" * 78)
        say(f"{'rank':>4} {'feature':<20} {'SHAP mean|v|':>14} {'RF importance':>14}")
        native_map = {item["feature"]: item["importance"] for item in native.get("features", [])}
        for item in shap_block["features"][: len(FEATURE_KEYS)]:
            say(
                f"{item['rank']:>4} {item['feature']:<20} {item['mean_abs_shap']:>14.6f} "
                f"{native_map.get(item['feature'], 0.0):>14.6f}"
            )
    else:
        say("")
        say(f"SHAP importance unavailable: {shap_block.get('reason')}")

    if comparison and comparison.get("available"):
        say("")
        say("Rule-Based vs ML on the held-out test split")
        say("-" * 78)
        pred = comparison["prediction"]
        opt = comparison["optimization"]
        say(
            f"{'selector':<14} {'accuracy':>10} {'macro-F1':>10} {'mean instr red':>15} "
            f"{'mean useful':>12} {'% improved':>11}"
        )
        for key, label in (("rule_based", "Rule-Based"), ("ml", "ML")):
            say(
                f"{label:<14} {pred[key]['accuracy']:>10.4f} {pred[key]['macro_f1']:>10.4f} "
                f"{opt[key]['mean_instruction_reduction']:>15.4f} "
                f"{opt[key]['mean_useful_improvement']:>12.4f} "
                f"{opt[key]['improved_pct']:>10.2f}%"
            )
        say(
            f"Disagreements: {pred['disagreement_count']} of "
            f"{comparison['evaluated_samples']} programs"
        )
        say(f"Optimization winner: {opt['overall_winner'].upper()} - {opt['overall_note']}")

    say("")
    say("Limitations")
    say("-" * 78)
    for limitation in LIMITATIONS:
        say(f"  - {limitation}")

    say("")
    say(f"Artifacts written to: {out_path}")
    for name in sorted(p.name for p in out_path.glob("*")):
        say(f"  - {name}")
    say("")
    say(f"Completed in {(datetime.now(timezone.utc) - started).total_seconds():.1f}s")

    summary = {
        "model_version": model_version,
        "dataset_fingerprint": fingerprint,
        "config": config,
        "selected_config": best["config"],
        "candidates_evaluated": len(GRID),
        "class_names": class_names,
        "target_classes": labels,
        "split_metrics": split_metrics,
        "feature_importance": importance_payload,
        "comparison": comparison,
        "metadata": metadata,
        "artifact_dir": str(out_path),
    }
    return summary


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.ml.train",
        description="Train the Random Forest strategy selector on the existing dataset.",
    )
    parser.add_argument("--dataset", default=None, help="Path to optimization_dataset.csv")
    parser.add_argument("--out", default=None, help="Artifact output directory (default: artifacts/ml)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 20261005)")
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the Rule-Based vs ML evaluation on the test split",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args(argv)

    try:
        train(
            dataset_path=args.dataset,
            out_dir=args.out,
            seed=args.seed,
            skip_comparison=args.no_comparison,
            verbose=not args.quiet,
        )
    except SystemExit as exc:
        return int(exc.code or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
