"""Rule-Based vs ML - two *separate* comparisons that must never be merged.

Prediction quality and optimization effectiveness answer different questions:

* **Prediction quality** - how often does each selector name the dataset's
  ``best_strategy`` label on the held-out split?  Ground truth here is the
  stored label.
* **Optimization effectiveness** - when each selector's chosen strategy is
  actually *executed* against the same original TAC, how much work disappears?
  Ground truth here is the measured ``useful_improvement``, not the label.

A selector can predict the label frequently yet pick strategies that help less,
or the reverse, so :func:`compare_on_dataset` reports the two side by side and
keeps them in distinct blocks.  Neither block ever uses the stored label as a
selector's prediction.

Independence guarantee: every run starts from a ``deepcopy`` of the same
original TAC and goes through the same :func:`measure_pass`, so one selector's
mutations can never reach the other's run.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter

from ..api.pipeline import parse_source
from ..api.pipeline import _feedback_guided_selector
from ..api.serialize import STRATEGY_DISPLAY, tac_to_lines, tac_to_source
from ..features.extractor import extract_features
from ..optimizer.diff import tac_diff
from ..optimizer.correctness import check_semantic_equivalence
from ..optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from ..optimizer.measure import measure_pass, operation_count, work_remaining
from ..scoring.rule_based import rule_based_strategy_scores, select_strategy, strategy_reason
from .dataset import (
    FEATURE_KEYS,
    STRATEGIES,
    TARGET_COLUMN,
    TAC_COLUMN,
    ID_COLUMN,
    SPLIT_COLUMN,
    discover_target_classes,
    feature_frame,
    feature_matrix,
    split_frames,
    load_dataset,
)
from .evaluate import aggregate_outcome, classification_metrics

__all__ = [
    "apply_strategy_once",
    "compare_on_dataset",
    "compare_source",
    "TIE",
    "RULE_WINS",
    "ML_WINS",
    "FEEDBACK_WINS",
]

TIE = "tie"
RULE_WINS = "rule_based"
ML_WINS = "ml"
FEEDBACK_WINS = "feedback"


def apply_strategy_once(original_tac, strategy: str) -> dict:
    """Execute exactly one pass on an independent copy of ``original_tac``.

    Returns the executed strategy, whether it was applicable/failed/changed, the
    measured outcome and the real before/after trace.  Never raises: a pass that
    blows up is reported as a failed run so the aggregate counts it honestly
    instead of dropping it.
    """
    fn = OPTIMIZATION_FUNCTIONS.get(strategy)

    record: dict = {
        "strategy": strategy,
        "applicable": fn is not None,
        "failed": False,
        "changed": False,
        "reason": None,
        "before_lines": tac_to_lines(original_tac),
        "before_text": tac_to_source(original_tac),
        "after_lines": tac_to_lines(original_tac),
        "after_text": tac_to_source(original_tac),
        "measurements": measure_pass(original_tac, original_tac),
        "diff": tac_diff(original_tac, original_tac),
    }

    if fn is None:
        record["reason"] = f"Unknown optimization strategy: {strategy}"
        record["failed"] = True
        return record

    try:
        optimized = fn(deepcopy(original_tac))
    except Exception as exc:  # noqa: BLE001 - a crashing pass is a failed run
        record["failed"] = True
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record

    if optimized is None:
        # The pass declined to transform this program: a legitimate no-op, not
        # an error. The engine treats None as "stop" for the same reason.
        record["reason"] = "Pass produced no output for this program."
        return record

    if tac_to_source(optimized) == record["before_text"]:
        record["reason"] = "Strategy executed but the TAC was unchanged."
        record["after_lines"] = tac_to_lines(optimized)
        record["after_text"] = tac_to_source(optimized)
        return record

    record["changed"] = True
    record["after_lines"] = tac_to_lines(optimized)
    record["after_text"] = tac_to_source(optimized)
    record["measurements"] = measure_pass(original_tac, optimized)
    record["diff"] = tac_diff(original_tac, optimized)
    record["reason"] = "TAC transformed."
    return record


def _one_program_outcome(original_tac, strategy: str) -> dict:
    """Flatten a single execution into the row shape ``aggregate_outcome`` wants."""
    run = apply_strategy_once(original_tac, strategy)
    measurements = run["measurements"]
    return {
        "strategy": strategy,
        "applicable": run["applicable"],
        "failed": run["failed"],
        "changed": run["changed"],
        "original_instruction_count": int(measurements["original_instruction_count"]),
        "original_operation_count": int(
            operation_count(original_tac)
        ),
        "instruction_reduction": int(measurements["instruction_reduction"]),
        "operation_reduction": int(measurements["operation_reduction"]),
        "useful_improvement": int(measurements["useful_improvement"]),
        "reduction_percentage": float(measurements["reduction_percentage"]),
    }


def _predict_ml(features: dict, payload: dict | None) -> str | None:
    """Single prediction from the loaded forest, or ``None`` if unavailable."""
    if not payload:
        return None
    import numpy as np

    model = payload["model"]
    # predict_proba columns follow model.classes_; never index them with a
    # different ordering.
    classes = [str(c) for c in getattr(model, "classes_", [])] or list(
        payload.get("class_names") or []
    )
    row = feature_frame(features)
    probabilities = model.predict_proba(row)[0]
    index = int(np.argmax(probabilities))
    return classes[index] if index < len(classes) else None


def compare_on_dataset(df=None, split: str = "test", payload: dict | None = None) -> dict:
    """Evaluate both selectors on the dataset's held-out split (§10).

    Prediction block: both selectors scored against the stored
    ``best_strategy`` label with the real scoring/model code - never the label
    itself.

    Optimization block: each selector's chosen strategy is executed once on an
    independent copy of the same original TAC and measured with the same
    :func:`measure_pass`.

    Returns ``{"available": False, "reason": ...}`` when the model is missing,
    so the evaluation screen never shows a fabricated comparison.
    """
    from .dataset import load_dataset
    from .model import ModelUnavailableError, load_payload

    if df is None:
        df = load_dataset()

    if payload is None:
        try:
            payload = load_payload()
        except ModelUnavailableError as exc:
            return {"available": False, "reason": str(exc)}

    frames = split_frames(df)
    if split not in frames:
        return {"available": False, "reason": f"Dataset has no {split!r} split."}
    frame = frames[split]

    labels = discover_target_classes(df)

    rule_predictions: list[str] = []
    ml_predictions: list[str] = []
    ground_truth: list[str] = []

    rule_runs: list[dict] = []
    ml_runs: list[dict] = []
    paired: list[dict] = []

    parse_failures: list[dict] = []
    feature_mismatches: list[dict] = []

    from ..features.extractor import extract_features
    from ..tac.parser import parse_tac

    for _, row in frame.iterrows():
        sample_id = row[ID_COLUMN]
        truth = str(row[TARGET_COLUMN])

        try:
            tac = parse_tac(str(row[TAC_COLUMN]))
            if not tac.instructions:
                raise ValueError("no instructions parsed")
        except Exception as exc:  # noqa: BLE001 - record and continue
            parse_failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        # Features come from the real backend, not from the stored columns.
        features = extract_features(tac)
        stored = {k: row[k] for k in FEATURE_KEYS if k in row}
        drift = {
            k: {"extracted": float(features[k]), "stored": float(stored[k])}
            for k in FEATURE_KEYS
            if k in stored and abs(float(features[k]) - float(stored[k])) > 1e-9
        }
        if drift:
            feature_mismatches.append({"sample_id": sample_id, "drift": drift})

        rule_strategy = select_strategy(tac)
        ml_strategy = _predict_ml(features, payload)

        ground_truth.append(truth)
        rule_predictions.append(rule_strategy)
        ml_predictions.append(ml_strategy if ml_strategy is not None else "__ml_unavailable__")

        rule_run = _one_program_outcome(tac, rule_strategy)
        ml_run = (
            _one_program_outcome(tac, ml_strategy)
            if ml_strategy is not None
            else {
                "strategy": None,
                "applicable": False,
                "failed": True,
                "changed": False,
                "instruction_reduction": 0,
                "operation_reduction": 0,
                "useful_improvement": 0,
                "reduction_percentage": 0.0,
            }
        )
        rule_runs.append(rule_run)
        ml_runs.append(ml_run)

        rule_gain = rule_run["useful_improvement"]
        ml_gain = ml_run["useful_improvement"]
        if rule_gain > ml_gain:
            program_winner = RULE_WINS
        elif ml_gain > rule_gain:
            program_winner = ML_WINS
        else:
            program_winner = TIE

        paired.append(
            {
                "sample_id": sample_id,
                "category": row.get("generation_category"),
                "ground_truth": truth,
                "rule_strategy": rule_strategy,
                "ml_strategy": ml_strategy,
                "strategies_match": rule_strategy == ml_strategy,
                "rule_correct": rule_strategy == truth,
                "ml_correct": ml_strategy == truth,
                "rule": rule_run,
                "ml": ml_run,
                "program_winner": program_winner,
            }
        )

    if not paired:
        return {
            "available": False,
            "reason": "No sample in the split could be parsed, so nothing was evaluated.",
            "parse_failures": parse_failures,
        }

    rule_pred_metrics = classification_metrics(ground_truth, rule_predictions, labels)
    ml_pred_metrics = classification_metrics(ground_truth, ml_predictions, labels)

    rule_agg = aggregate_outcome(rule_runs)
    ml_agg = aggregate_outcome(ml_runs)

    wins = {"rule_based": 0, "ml": 0, "tie": 0}
    for item in paired:
        wins[item["program_winner"]] += 1

    if rule_agg["mean_useful_improvement"] > ml_agg["mean_useful_improvement"]:
        overall, overall_note = RULE_WINS, "Rule-Based achieved a higher mean useful improvement."
    elif ml_agg["mean_useful_improvement"] > rule_agg["mean_useful_improvement"]:
        overall, overall_note = ML_WINS, "ML achieved a higher mean useful improvement."
    else:
        overall, overall_note = TIE, "Both selectors achieved the same mean useful improvement."

    same_strategy_count = sum(1 for item in paired if item["strategies_match"])

    return {
        "available": True,
        "split": split,
        "evaluated_samples": len(paired),
        "target_classes": labels,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methodology": {
            "prediction": (
                "Each selector's decision is compared with the stored "
                "best_strategy label. The label is never used as either "
                "selector's prediction."
            ),
            "optimization": (
                "Each selector's chosen strategy is applied once to an "
                "independent deepcopy of the same original TAC and measured "
                "with optimizer.measure.measure_pass."
            ),
            "separation": (
                "Prediction quality and optimization effectiveness are "
                "reported separately and are never combined into one number."
            ),
            "rule_based_selector": "backend.scoring.rule_based.select_strategy (deterministic argmax + tie order)",
            "ml_selector": f"{type(payload['model']).__name__}.predict over the eight features",
        },
        "prediction": {
            "rule_based": rule_pred_metrics,
            "ml": ml_pred_metrics,
            "same_prediction_count": sum(
                1 for r, m in zip(rule_predictions, ml_predictions) if r == m
            ),
            "disagreement_count": sum(
                1 for r, m in zip(rule_predictions, ml_predictions) if r != m
            ),
        },
        "optimization": {
            "rule_based": rule_agg,
            "ml": ml_agg,
            "program_wins": wins,
            "overall_winner": overall,
            "overall_note": overall_note,
            "same_strategy_programs": same_strategy_count,
            "different_strategy_programs": len(paired) - same_strategy_count,
        },
        "per_program": paired,
        "data_quality": {
            "parse_failures": parse_failures,
            "feature_mismatches": feature_mismatches,
        },
        "limitations": [
            f"Only {len(paired)} held-out samples were evaluated; every metric is preliminary.",
            "The forest has no winning examples for strength_reduction, so the ML "
            "selector cannot emit it; a Rule-Based prediction of "
            "strength_reduction is outside the target vocabulary and counts as "
            "an error while being reported explicitly.",
            "Labels come from the useful_improvement definition in "
            "optimizer.measure, so 'correct' means 'matches that definition'.",
            "strength_reduction is a documented no-op, so runs selecting it "
            "record zero improvement by construction.",
        ],
    }


def _comparison_iteration_context(include_shap: bool):
    """Attach *both* selectors' opinions to every iteration of a compare run.

    Each run already records the strategy it executed; this adds what the other
    selector would have chosen on the same input, so a timeline can show exactly
    where two runs start to diverge instead of only comparing their first pick.
    """

    def context(tac, features: dict, selected: str) -> dict:
        from .predict import predict_from_features

        rule_pick = select_strategy(tac)
        prediction = predict_from_features(features)
        ml_pick = prediction.get("predicted_strategy") if prediction.get("available") else None

        block = {
            "rule_selected_strategy": rule_pick,
            "ml_predicted_strategy": ml_pick,
            "ml_probabilities": prediction.get("probabilities"),
            "ml_predicted_probability": prediction.get("predicted_probability"),
            "ml_model_version": prediction.get("model_version"),
            "strategies_agree": (ml_pick == rule_pick) if ml_pick else None,
        }
        if include_shap:
            from .explain import explain_prediction

            block["shap"] = explain_prediction(features)
        return block

    return context


def _agreement_block(
    tac,
    ml_available: bool,
    rule_run: dict,
    ml_run: dict | None,
    feedback_run: dict | None,
    rule_strategy: str,
    ml_strategy: str | None,
) -> dict:
    """Compare the runs' *full* strategy sequences, not just the first pick."""
    rule_sequence = [item.get("selected_strategy") for item in rule_run.get("history", [])]
    ml_sequence = (
        [item.get("selected_strategy") for item in ml_run.get("history", [])]
        if ml_run is not None
        else []
    )
    feedback_sequence = (
        [item.get("selected_strategy") for item in feedback_run.get("history", [])]
        if feedback_run is not None
        else []
    )

    if not ml_available:
        return {
            "initial_strategies_agree": None,
            "strategies_agree": None,
            "rule_strategy": rule_strategy,
            "ml_strategy": None,
            "feedback_strategy": None,
            "rule_strategy_sequence": rule_sequence,
            "ml_strategy_sequence": [],
            "feedback_strategy_sequence": [],
            "common_prefix_length": 0,
            "diverged_at_iteration": None,
            "explanation": "ML unavailable - only the Rule-Based outcome is shown.",
            "note": "Agreement is only claimed when all runs actually executed.",
        }

    # Find common prefix across all three sequences
    limit = min(len(rule_sequence), len(ml_sequence), len(feedback_sequence))
    common = 0
    while common < limit and rule_sequence[common] == ml_sequence[common] == feedback_sequence[common]:
        common += 1

    max_len = max(len(rule_sequence), len(ml_sequence), len(feedback_sequence))
    diverged_at = common + 1 if common < max_len else None
    initial_agree = rule_strategy == ml_strategy == feedback_sequence[0] if feedback_sequence else rule_strategy == ml_strategy
    sequences_match = rule_sequence == ml_sequence == feedback_sequence

    if sequences_match:
        explanation = (
            "All three selectors chose the same strategy at every iteration, so the "
            "runs produced identical transformations."
        )
    elif initial_agree:
        explanation = (
            f"All selectors opened with '{rule_strategy}', but they diverged at "
            f"iteration {diverged_at}. The measured outcomes below show "
            "the real consequence of each choice."
        )
    else:
        explanation = (
            f"The selectors disagree from the first iteration. The measured "
            "outcomes below show the real consequence of each choice."
        )

    return {
        "initial_strategies_agree": initial_agree,
        "strategies_agree": sequences_match,
        "rule_strategy": rule_strategy,
        "ml_strategy": ml_strategy,
        "feedback_strategy": feedback_sequence[0] if feedback_sequence else None,
        "rule_strategy_sequence": rule_sequence,
        "ml_strategy_sequence": ml_sequence,
        "feedback_strategy_sequence": feedback_sequence,
        "common_prefix_length": common,
        "diverged_at_iteration": diverged_at,
        "explanation": explanation,
        "note": (
            "strategies_agree is true only when every iteration matched; "
            "initial_strategies_agree covers the first choice alone."
        ),
    }


def run_useful_improvement(original_tac, run: dict) -> int:
    """Total work an engine run removed, measured with the shared implementation."""
    return int(measure_pass(original_tac, run["final_tac"])["useful_improvement"])


def _correctness_block(original_tac, final_tac: dict) -> dict:
    """Run the bounded semantic checker and shape its verdict for the UI.

    Returns the checker's status verbatim (``equivalent`` /
    ``not_equivalent`` / ``inconclusive``) with its detail string and the
    variables that were actually compared.  An unexpected checker failure is
    reported as ``inconclusive`` rather than being swallowed or promoted to a
    pass, so a bug in the checker can never masquerade as "the optimizer was
    right."
    """
    from ..optimizer.correctness import check_semantic_equivalence

    try:
        verdict = check_semantic_equivalence(original_tac, final_tac)
    except Exception as exc:  # noqa: BLE001 - checker bugs must not claim success
        return {
            "status": "inconclusive",
            "note": f"Checker raised {type(exc).__name__}: {exc}",
            "observed_vars": [],
            "outputs_compared": False,
        }
    return {
        "status": verdict["status"],
        "note": verdict["detail"],
        "observed_vars": verdict["observed_vars"],
        "outputs_compared": verdict.get("outputs_compared", False),
    }


def _run_block(original_tac, run: dict, runtime_ms: float | None = None) -> dict:
    """Common shape for one selector's executed run in the compare screen.

    ``runtime_ms`` is the wall-clock time this single run took, measured by
    the caller with ``time.perf_counter`` around exactly that run; ``None``
    means the arm never executed and no time is claimed.  Pass counts are
    derived from the same history the trace renders, so they can never
    disagree with it.
    """
    from .evaluate import aggregate_outcome

    iterations = []
    for item in run.get("history", []):
        before = item.get("tac_before")
        after = item.get("tac_after")
        iterations.append(
            {
                "iteration": item.get("iteration"),
                "selected_strategy": item.get("selected_strategy"),
                "strategy_name": STRATEGY_DISPLAY.get(item.get("selected_strategy"), item.get("selected_strategy")),
                "selector": item.get("selector"),
                # Selection vs execution (§2.7): what the selector predicted
                # for this program state, what the engine actually ran, and
                # whether that run was the prediction or a controlled
                # fallback attempt.  The outcome names what the pass really
                # did (applied_useful / applied_changed / executed_no_change /
                # rejected / failed / unsupported).
                "predicted_strategy": item.get("predicted_strategy"),
                "origin": item.get("origin"),
                "outcome": item.get("outcome"),
                "rejection_reason": item.get("rejection_reason"),
                "correctness_note": item.get("correctness_note"),
                "rule_scores": item.get("scores"),
                "rule_selected_strategy": item.get("rule_selected_strategy"),
                "ml_predicted_strategy": item.get("ml_predicted_strategy"),
                "ml_probabilities": item.get("ml_probabilities"),
                "strategies_agree": item.get("strategies_agree"),
                "shap": item.get("shap"),
                "before_count": item.get("before_count"),
                "after_count": item.get("after_count"),
                "improvement": item.get("improvement"),
                "operation_count_before": item.get("operation_count_before"),
                "operation_count_after": item.get("operation_count_after"),
                "useful_improvement": item.get("useful_improvement"),
                "changed": item.get("changed"),
                "accepted": item.get("accepted"),
                "rationale": item.get("rationale"),
                "tac_before": tac_to_lines(before),
                "tac_after": tac_to_lines(after),
                "diff": tac_diff(before, after) if before is not None and after is not None else None,
            }
        )

    measurements = measure_pass(original_tac, run["final_tac"])
    return {
        "history": iterations,
        "final_lines": tac_to_lines(run["final_tac"]),
        "final_text": tac_to_source(run["final_tac"]),
        "measurements": measurements,
        "diff": tac_diff(original_tac, run["final_tac"]),
        "stop_reason": run.get("stop_reason"),
        "total_reduction": run["total_reduction"],
        "iterations": run["total_iterations"],
        "converged": run["converged"],
        "useful_improvement": int(measurements["useful_improvement"]),
        "original_instruction_count": int(measurements["original_instruction_count"]),
        "final_instruction_count": int(measurements["optimized_instruction_count"]),
        "original_operation_count": operation_count(original_tac),
        "final_operation_count": operation_count(run["final_tac"]),
        "changed": run["total_iterations"] > 0
        and tac_to_source(original_tac) != tac_to_source(run["final_tac"]),
        # Optimization-metrics accounting (§7): attempted vs actually changed
        # vs accepted, all read from the same history shown in the trace.
        "passes_attempted": len(iterations),
        "passes_changed": sum(1 for item in iterations if item.get("changed")),
        "passes_unchanged": sum(1 for item in iterations if not item.get("changed")),
        "passes_accepted": sum(1 for item in iterations if item.get("accepted")),
        # Attempts the semantic acceptance check rejected (kept out of the
        # final program) and attempts made by the controlled fallback rather
        # than the selector's own prediction.
        "passes_rejected": sum(1 for item in iterations if item.get("outcome") == "rejected"),
        "fallback_attempts": sum(1 for item in iterations if item.get("origin") == "fallback"),
        "runtime_ms": runtime_ms,
        # Semantic validation (§P9 regression): the final program is executed
        # against the original under backend.tac.interpreter and compared on
        # the values of every variable the optimized program still reads, plus
        # termination behavior.  "equivalent" is bounded evidence over the
        # checker's input vectors - never a claim of correctness for all
        # inputs; see backend.optimizer.correctness for the exact contract.
        "correctness_checked": True,
        "correctness": _correctness_block(original_tac, run["final_tac"]),
    }


def compare_source(source: str, max_iterations: int = 20, include_shap: bool = True) -> dict:
    """Live side-by-side comparison of three selectors on one TAC.

    All runs start from the *same* parsed TAC, each is deep-copied internally
    by :func:`run_iterative_optimization`, and all are measured with the same
    implementation.  The winner is decided by measured improvement - never by
    prediction probability.  Agreement is computed over the *whole* strategy
    sequence, because selectors can agree on the first pick and still
    diverge later.  When the model is unavailable the ML and Feedback blocks
    report that explicitly and no winner is declared.
    """
    from .model import ModelUnavailableError, load_payload
    from .predict import predict_from_tac

    tac = parse_source(source)
    features = extract_features(tac)

    rule_scores = rule_based_strategy_scores(tac)
    rule_strategy = select_strategy(tac)

    try:
        payload = load_payload()
        ml_result = predict_from_tac(tac)
    except ModelUnavailableError as exc:
        payload = None
        ml_result = {"available": False, "selector": "ml", "reason": str(exc)}

    ml_available = bool(ml_result.get("available")) and payload is not None
    ml_strategy = ml_result.get("predicted_strategy") if ml_available else None

    def _ml_selector(tac_slice, _payload=payload):
        return _predict_ml(extract_features(tac_slice), _payload)

    def _feedback_selector(tac_slice, _payload=payload):
        # Clear rejected cache for each new comparison run
        if hasattr(_feedback_guided_selector, "_rejected_cache"):
            _feedback_guided_selector._rejected_cache.clear()
        return _feedback_guided_selector(tac_slice)

    # Independent copies of the same original TAC: neither run can observe the
    # other's mutations. All carry the shared context so each timeline shows
    # what the *other* selectors would have chosen at that same point.
    shared_context = _comparison_iteration_context(include_shap)
    # One-time warm-up (first module imports, first model prediction, first
    # SHAP background) runs BEFORE any arm is timed, so the arm that happens
    # to run first does not absorb setup costs the other arms skip.  Nothing
    # here is measured: each runtime_ms below still covers exactly one engine
    # run (feature extraction + selection + passes), and correctness checking
    # happens later, outside every timing.
    shared_context(tac, features, rule_strategy)
    # Wall-clock time of each run, measured around exactly that run.
    rule_t0 = perf_counter()
    rule_run = run_iterative_optimization(
        deepcopy(tac),
        max_iterations=max_iterations,
        iteration_context=shared_context,
        selector_name="rule",
    )
    rule_runtime_ms = round((perf_counter() - rule_t0) * 1000, 3)

    if ml_available:
        ml_t0 = perf_counter()
        ml_run = run_iterative_optimization(
            deepcopy(tac),
            max_iterations=max_iterations,
            selector=_ml_selector,
            iteration_context=shared_context,
            selector_name="ml",
        )
        ml_runtime_ms = round((perf_counter() - ml_t0) * 1000, 3)
        ml_block = {
            "available": True,
            "strategy": ml_strategy,
            "reason": None,
            **_run_block(tac, ml_run, runtime_ms=ml_runtime_ms),
        }
        ml_block["ml_probabilities"] = ml_result.get("probabilities")
        ml_block["predicted_probability"] = ml_result.get("predicted_probability")
        ml_block["model_version"] = ml_result.get("model_version")

        # Feedback-Guided ML run
        fb_t0 = perf_counter()
        feedback_run = run_iterative_optimization(
            deepcopy(tac),
            max_iterations=max_iterations,
            selector=_feedback_selector,
            iteration_context=shared_context,
            selector_name="feedback",
        )
        feedback_runtime_ms = round((perf_counter() - fb_t0) * 1000, 3)
        feedback_block = {
            "available": True,
            "strategy": None,  # No single strategy - varies per iteration
            "reason": None,
            **_run_block(tac, feedback_run, runtime_ms=feedback_runtime_ms),
        }
        feedback_block["ml_probabilities"] = ml_result.get("probabilities")
        feedback_block["predicted_probability"] = ml_result.get("predicted_probability")
        feedback_block["model_version"] = ml_result.get("model_version")
    else:
        ml_run = None
        ml_block = {
            "available": False,
            "strategy": None,
            "reason": ml_result.get("reason") or "Model unavailable.",
            "history": [],
            "final_lines": tac_to_lines(tac),
            "final_text": tac_to_source(tac),
            "measurements": measure_pass(tac, tac),
            "diff": tac_diff(tac, tac),
            "stop_reason": None,
            "total_reduction": 0,
            "iterations": 0,
            "converged": False,
            "useful_improvement": 0,
            "original_instruction_count": len(tac),
            "final_instruction_count": len(tac),
            "original_operation_count": operation_count(tac),
            "final_operation_count": operation_count(tac),
            "changed": False,
            # Never-executed arm: no passes, no measured time, no claims.
            "passes_attempted": 0,
            "passes_changed": 0,
            "passes_unchanged": 0,
            "passes_accepted": 0,
            "runtime_ms": None,
            "correctness_checked": False,
            "correctness": {
                "status": "not_run",
                "note": (
                    "This arm did not run, so no time, no pass counts and no "
                    "semantic check are reported for it."
                ),
                "observed_vars": [],
                "outputs_compared": False,
            },
        }
        feedback_run = None
        feedback_block = {
            "available": False,
            "strategy": None,
            "reason": ml_result.get("reason") or "Model unavailable.",
            "history": [],
            "final_lines": tac_to_lines(tac),
            "final_text": tac_to_source(tac),
            "measurements": measure_pass(tac, tac),
            "diff": tac_diff(tac, tac),
            "stop_reason": None,
            "total_reduction": 0,
            "iterations": 0,
            "converged": False,
            "useful_improvement": 0,
            "original_instruction_count": len(tac),
            "final_instruction_count": len(tac),
            "original_operation_count": operation_count(tac),
            "final_operation_count": operation_count(tac),
            "changed": False,
            # Never-executed arm: no passes, no measured time, no claims.
            "passes_attempted": 0,
            "passes_changed": 0,
            "passes_unchanged": 0,
            "passes_accepted": 0,
            "runtime_ms": None,
            "correctness_checked": False,
            "correctness": {
                "status": "not_run",
                "note": (
                    "This arm did not run, so no time, no pass counts and no "
                    "semantic check are reported for it."
                ),
                "observed_vars": [],
                "outputs_compared": False,
            },
        }

    rule_block = {
        "available": True,
        "selected_strategy": rule_strategy,
        "strategy_name": STRATEGY_DISPLAY.get(rule_strategy, rule_strategy),
        "rationale": strategy_reason(tac, rule_strategy),
        "scores": {k: round(float(v), 4) for k, v in rule_scores.items()},
        **_run_block(tac, rule_run, runtime_ms=rule_runtime_ms),
    }

    # Add attempted candidates to feedback block for traceability
    if feedback_block["available"] and feedback_run:
        # Extract attempted candidates from the run history
        attempted = []
        for item in feedback_run.get("history", []):
            attempted.append({
                "strategy": item.get("selected_strategy"),
                "predicted_probability": item.get("ml_predicted_probability"),
                "outcome": "accepted" if item.get("accepted") else "rejected",
                "improvement": item.get("improvement", 0),
                "reason": item.get("rationale") or item.get("stop_reason") or "no improvement",
            })
        feedback_block["attempted_candidates"] = attempted

    rule_gain = rule_block["useful_improvement"]
    ml_gain = ml_block["useful_improvement"] if ml_block["available"] else None
    feedback_gain = feedback_block["useful_improvement"] if feedback_block["available"] else None

    if not ml_available:
        # ML (and therefore Feedback-Guided ML) never ran, so there is only
        # one measured outcome: naming Rule-Based the "winner" would be
        # winning a race with a single runner. The blocks above already
        # report the unavailability explicitly, so no winner is declared.
        winner = None
        winner_note = (
            "ML is unavailable, so only the Rule-Based outcome was measured — "
            "no winner is declared."
        )
    else:
        # Determine winner among all three
        gains = {
            "rule_based": rule_gain,
            "ml": ml_gain if ml_gain is not None else -1,
            "feedback": feedback_gain if feedback_gain is not None else -1,
        }
        max_gain = max(gains.values())
        winners = [k for k, v in gains.items() if v == max_gain]

        if len(winners) > 1:
            winner = TIE
            winner_note = "Multiple selectors produced the same measured improvement."
        elif winners[0] == "rule_based":
            winner = RULE_WINS
            winner_note = "Rule-Based removed more work overall."
        elif winners[0] == "ml":
            winner = ML_WINS
            winner_note = "ML removed more work overall."
        else:
            winner = FEEDBACK_WINS
            winner_note = "Feedback-Guided ML removed more work overall."

    agreement = _agreement_block(
        tac, ml_available, rule_run, ml_run, feedback_run, rule_strategy, ml_strategy
    )

    return {
        "available": True,
        "original_lines": tac_to_lines(tac),
        "original_text": tac_to_source(tac),
        "original_instruction_count": len(tac),
        "original_operation_count": operation_count(tac),
        "original_work_remaining": work_remaining(tac),
        "features": {k: float(features.get(k, 0)) for k in FEATURE_KEYS},
        "rule_based": rule_block,
        "ml": ml_block,
        "feedback": feedback_block,
        "agreement": agreement,
        "winner": winner,
        "winner_note": winner_note,
        "decided_by": "measured useful_improvement of all executed runs",
    }
