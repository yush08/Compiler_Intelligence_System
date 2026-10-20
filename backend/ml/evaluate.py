"""Classification metrics shared by training, evaluation and comparison.

One implementation is used everywhere so the numbers in `evaluation.json`, the
comparison screen and the tests can never drift apart.

Handling of absent classes
--------------------------
The target vocabulary is five classes (``strength_reduction`` has no winning
examples) and the held-out split holds only 26 rows, so a class can easily be
missing from ``y_true``, ``y_pred``, or both.  Every metric is therefore computed
against an explicit ``labels`` list taken from the dataset, which means:

* a class absent from the data scores 0.0 rather than silently vanishing from
  the macro average;
* :func:`classification_metrics` reports which classes were actually observed
  so the UI can say "not evaluated" instead of implying a real score;
* a prediction outside the vocabulary is counted as wrong instead of crashing.

``macro-F1`` is the number to trust here: with a 3.3x imbalance, accuracy is
dominated by the majority class and weighted-F1 tracks it.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

__all__ = [
    "classification_metrics",
    "aggregate_outcome",
]


def _safe(report_dict: dict, label: str, key: str, field: str) -> float:
    """Read one cell of sklearn's report dict without KeyError-ing."""
    row = report_dict.get(label) or {}
    value = row.get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Compute the full metric set for one selector against one split.

    Args:
        y_true: ground-truth labels actually present in this split.
        y_pred: predicted labels.  May contain classes outside ``labels``; those
            count as errors rather than raising.
        labels: the dataset's full target vocabulary, in stable order.  Every
            metric is reported for every entry, including absent classes.

    Returns a dict with accuracy, macro/weighted F1, per-class
    precision/recall/F1/support, a confusion matrix aligned to ``labels``, the
    sklearn classification report, evaluated counts and the label
    distributions - plus ``absent_classes``, naming classes with no ground-truth
    support in this split (they are *not* meaningfully evaluated).
    """
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]

    if not labels:
        raise ValueError("labels vocabulary must not be empty")

    evaluated = len(y_true)
    if evaluated != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) differ in length")

    label_set = set(labels)

    # Predictions the model/selectors never emits are legitimate outcomes of a
    # multi-class argmax, but a label outside the declared vocabulary means the
    # evaluator and the producer disagree - surface it explicitly.
    predicted_outside = sorted({p for p in y_pred if p not in label_set})
    true_outside = sorted({t for t in y_true if t not in label_set})

    # sklearn needs every class it is asked to score to appear in y_true's
    # encoding; unknown labels are mapped to a sentinel that cannot match, so
    # they are simply counted as errors.
    unknown = "__unknown__"
    while unknown in label_set:
        unknown += "_"
    scoring_labels = list(labels) + ([unknown] if predicted_outside or true_outside else [])

    y_true_enc = [t if t in label_set else unknown for t in y_true]
    y_pred_enc = [p if p in label_set else unknown for p in y_pred]

    accuracy = float(accuracy_score(y_true_enc, y_pred_enc))
    macro_f1 = float(f1_score(y_true_enc, y_pred_enc, labels=scoring_labels, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true_enc, y_pred_enc, labels=scoring_labels, average="weighted", zero_division=0)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=scoring_labels, zero_division=0
    )

    # Confusion matrix. Every label that appears in y_true or y_pred gets a
    # row/column, so the cells always sum to `evaluated` - a rule-based selector
    # can legitimately emit `strength_reduction`, which is outside the target
    # vocabulary, and dropping that column would silently lose those errors.
    # Rows/columns beyond `labels` are appended in `confusion_matrix_labels`.
    cm_labels = list(labels) + [
        name
        for name in sorted(set(y_true) | set(y_pred))
        if name not in label_set
    ]
    y_true_cm = [t if t in cm_labels else unknown for t in y_true]
    y_pred_cm = [p if p in cm_labels else unknown for p in y_pred]
    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=cm_labels).tolist()

    report = classification_report(
        y_true_enc, y_pred_enc, labels=scoring_labels, zero_division=0, output_dict=True
    )

    per_class = {}
    for index, name in enumerate(labels):
        support_count = int(support[index])
        per_class[name] = {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": support_count,
            # Absent from this split's ground truth => the recall/precision pair
            # is degenerate.  Flagged so the UI never presents it as measured.
            "evaluated": support_count > 0,
        }

    def _distribution(values: list[str]) -> dict:
        out: dict = {name: 0 for name in labels}
        extra: dict = {}
        for v in values:
            if v in out:
                out[v] += 1
            else:
                extra[v] = extra.get(v, 0) + 1
        return {"by_class": out, "outside_vocabulary": extra}

    return {
        "evaluated_samples": evaluated,
        "labels": list(labels),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
        "confusion_matrix_note": (
            "Rows/columns are `labels` in order; any label outside the target "
            "vocabulary (e.g. a selector emitting strength_reduction) is "
            "appended so the cells sum to evaluated_samples."
            if len(cm_labels) > len(labels)
            else "Rows/columns follow the target vocabulary in order."
        ),
        "classification_report": {
            name: {
                "precision": round(_safe(report, name, name, "precision"), 6),
                "recall": round(_safe(report, name, name, "recall"), 6),
                "f1-score": round(_safe(report, name, name, "f1-score"), 6),
                "support": round(_safe(report, name, name, "support"), 6),
            }
            for name in labels
        },
        "absent_classes": [name for name in labels if per_class[name]["support"] == 0],
        "predicted_outside_vocabulary": predicted_outside,
        "true_outside_vocabulary": true_outside,
        "actual_distribution": _distribution(y_true),
        "predicted_distribution": _distribution(y_pred),
    }


def aggregate_outcome(rows: list[dict]) -> dict:
    """Summarise per-program *optimization* outcomes (not prediction quality).

    Prediction accuracy and optimization effectiveness answer different
    questions, so they are computed from different inputs and reported
    separately; this helper only ever sees executed-run measurements.

    Expects each row to carry ``useful_improvement``, ``instruction_reduction``
    and ``operation_reduction``.  ``applicable``/``failed``/``changed`` are
    optional flags defaulting to sensible values.
    """
    if not rows:
        return {
            "programs": 0,
            "mean_instruction_reduction": 0.0,
            "mean_operation_reduction": 0.0,
            "mean_useful_improvement": 0.0,
            "improved_programs": 0,
            "improved_pct": 0.0,
            "unchanged_programs": 0,
            "unchanged_pct": 0.0,
            "regressed_programs": 0,
            "failed_programs": 0,
            "strategy_distribution": {},
            "per_strategy": {},
        }

    n = len(rows)
    improved = [r for r in rows if int(r.get("useful_improvement", 0)) > 0]
    regressed = [r for r in rows if int(r.get("useful_improvement", 0)) < 0]
    failed = [r for r in rows if r.get("failed")]
    unchanged = [
        r
        for r in rows
        if int(r.get("useful_improvement", 0)) == 0 and not r.get("failed")
    ]

    distribution: dict = {}
    for r in rows:
        key = r.get("strategy") or "unknown"
        distribution[key] = distribution.get(key, 0) + 1

    # Per-strategy measured outcomes: for the strategies the selector actually
    # executed, how much work each one removed on average. Strategies the
    # selector never chose are absent - not zero - so "never used" is not
    # confused with "used and measured 0".
    per_strategy: dict = {}
    for r in rows:
        key = r.get("strategy") or "unknown"
        bucket = per_strategy.setdefault(key, {"programs": 0, "useful_sum": 0, "improved_programs": 0})
        useful = int(r.get("useful_improvement", 0) or 0)
        bucket["programs"] += 1
        bucket["useful_sum"] += useful
        if useful > 0 and not r.get("failed"):
            bucket["improved_programs"] += 1
    per_strategy = {
        key: {
            "programs": value["programs"],
            "mean_useful_improvement": round(value["useful_sum"] / value["programs"], 6),
            "improved_programs": value["improved_programs"],
        }
        for key, value in sorted(per_strategy.items())
    }

    def _mean(key: str) -> float:
        return round(sum(float(r.get(key, 0) or 0) for r in rows) / n, 6)

    return {
        "programs": n,
        "mean_instruction_reduction": _mean("instruction_reduction"),
        "mean_operation_reduction": _mean("operation_reduction"),
        "mean_useful_improvement": _mean("useful_improvement"),
        "improved_programs": len(improved),
        "improved_pct": round(len(improved) / n * 100, 4),
        "unchanged_programs": len(unchanged),
        "unchanged_pct": round(len(unchanged) / n * 100, 4),
        "regressed_programs": len(regressed),
        "failed_programs": len(failed),
        "strategy_distribution": dict(sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_strategy": per_strategy,
    }
