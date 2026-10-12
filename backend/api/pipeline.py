"""
API orchestration over existing backend modules.

Does not reimplement parser, features, scoring, or optimizers.

Selector modes
--------------
Every entry point accepts ``selector``:

* ``"rule"`` (default) - the existing deterministic scorer, unchanged.
* ``"ml"`` - the trained forest chooses the strategy.  Raises
  :class:`~backend.ml.model.ModelUnavailableError` when no artifact exists, so
  the caller gets an explicit unavailable state instead of a silently
  substituted rule-based answer.

An explicit ``strategy`` argument always wins over ``selector``: pinning a pass
is an instruction to run that pass, not a request for a selection.
"""

from copy import deepcopy

from backend.api.serialize import STRATEGY_DISPLAY, tac_to_lines
from backend.features.extractor import extract_features
from backend.optimizer.diff import tac_diff
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from backend.optimizer.measure import measure_pass, operation_count, work_remaining
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy, strategy_reason
from backend.tac.parser import parse_tac
from backend.ml.model import ModelUnavailableError


class TacParseError(ValueError):
    """Raised when TAC source is empty or cannot be parsed into instructions."""


def parse_source(source: str):
    if source is None or not str(source).strip():
        raise TacParseError("TAC source is empty.")
    tac = parse_tac(source)
    if not tac.instructions:
        raise TacParseError("No valid TAC instructions were parsed from the supplied source.")
    return tac


def _strategy_reason_for(name: str, features: dict) -> str:
    reasons = {
        "constant_folding": (
            f"Multiple constant expressions detected ({features['num_constants']} constants)."
            if features["num_constants"]
            else "Limited constant expressions detected."
        ),
        "constant_propagation": (
            "Constant assignments can propagate into dependent expressions."
            if features["num_constants"]
            else "Few constant assignments available for propagation."
        ),
        "common_subexpression": (
            f"Repeated expression opportunities detected ({features['num_repeated_expr']})."
            if features["num_repeated_expr"]
            else "No repeated subexpressions detected."
        ),
        "dead_code_elimination": (
            f"Unused assignments detected ({features['num_dead_stmts']})."
            if features["num_dead_stmts"]
            else "No unused assignments detected."
        ),
        "algebraic_simplification": (
            f"Algebraic identity opportunities detected ({features['num_algebraic_ops']})."
            if features["num_algebraic_ops"]
            else "Limited algebraic simplification opportunities."
        ),
        "strength_reduction": (
            f"Loop-related multiplication patterns detected ({features['num_loop_ops']})."
            if features["num_loop_ops"]
            else "Limited loop-related opportunities."
        ),
    }
    return reasons.get(name, "Strategy scored from the current TAC feature vector.")


def _features_payload(features: dict) -> dict:
    return {
        "num_statements": features["num_statements"],
        "num_constants": features["num_constants"],
        "num_repeated_expr": features["num_repeated_expr"],
        "num_dead_stmts": features["num_dead_stmts"],
        "num_algebraic_ops": features["num_algebraic_ops"],
        "num_loop_ops": features["num_loop_ops"],
        "num_tmp_vars": features["num_tmp_vars"],
        "const_ratio": features["const_ratio"],
    }


def _strategies_payload(tac, features: dict, scores: dict, selected: str) -> list[dict]:
    items = []
    for name, display in STRATEGY_DISPLAY.items():
        score = float(scores.get(name, 0.0))
        items.append(
            {
                "id": name,
                "name": display,
                "score": round(score, 4),
                "reason": _strategy_reason_for(name, features),
                "opportunity": score > 0.05,
                "selected": name == selected,
            }
        )
    items.sort(key=lambda item: item["score"], reverse=True)
    return items


def _selected_payload(tac, scores: dict, selected: str) -> dict:
    return {
        "id": selected,
        "name": STRATEGY_DISPLAY.get(selected, selected),
        "score": round(float(scores.get(selected, 0.0)), 4),
        # The reason must describe the strategy being reported here, not the
        # one strategy_reason() would re-select on its own — otherwise an
        # explicitly requested pass comes back with another pass's reason.
        "reason": strategy_reason(tac, selected),
    }


# --- selector plumbing ------------------------------------------------------

def _ml_prediction(tac) -> dict:
    """Live forest prediction for ``tac``.

    Always returns a dict: a missing artifact becomes ``available: false`` with
    a reason, so read-only endpoints can degrade to an explicit empty state
    instead of failing the whole request.
    """
    from backend.ml.predict import predict_from_tac, unavailable_result

    try:
        return predict_from_tac(tac)
    except Exception as exc:  # noqa: BLE001 - report, never crash the API
        return unavailable_result(
            f"ML prediction failed ({type(exc).__name__})."
        )


def _feedback_guided_selector(tac) -> str:
    """Feedback-Guided ML selector.

    Uses the trained Random Forest to rank eligible strategies by predicted
    probability, then attempts them in order until one produces a measurable
    improvement that passes correctness validation. Rejected candidates are
    recorded and not retried for the same TAC state, but may be reconsidered
    after the TAC changes. Does not retrain the model.
    """
    from backend.ml.predict import predict_from_tac, unavailable_result
    from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS
    from copy import deepcopy

    prediction = predict_from_tac(tac)
    if not prediction.get("available"):
        raise ModelUnavailableError(
            prediction.get("reason") or "Model artifact is unavailable."
        )

    # Get eligible strategies sorted by predicted probability (highest first)
    probabilities = prediction.get("probabilities", {})
    if not probabilities:
        # Fallback to rule-based if no probabilities available
        from backend.scoring.rule_based import select_strategy
        return select_strategy(tac)

    # Sort strategies by probability (descending)
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    eligible = [name for name, _ in ranked if name in OPTIMIZATION_FUNCTIONS]

    # Track rejected candidates for this TAC state
    tac_key = str(tac)
    if not hasattr(_feedback_guided_selector, "_rejected_cache"):
        _feedback_guided_selector._rejected_cache = {}
    if tac_key not in _feedback_guided_selector._rejected_cache:
        _feedback_guided_selector._rejected_cache[tac_key] = set()

    rejected = _feedback_guided_selector._rejected_cache[tac_key]

    # Try candidates in order of predicted probability
    for strategy_name in eligible:
        if strategy_name in rejected:
            continue
        # Test the candidate on a copy
        test_tac = deepcopy(tac)
        opt_fn = OPTIMIZATION_FUNCTIONS[strategy_name]
        result_tac = opt_fn(test_tac)
        
        # Measure improvement
        before_count = len(tac)
        after_count = len(result_tac)
        improvement = before_count - after_count
        
        if improvement > 0:
            # Check if output actually changed
            if str(tac) != str(result_tac):
                return strategy_name
        
        # Candidate was ineffective - reject it for this TAC state
        rejected.add(strategy_name)

    # No eligible candidate improved the program - fall back to rule-based
    from backend.scoring.rule_based import select_strategy
    return select_strategy(tac)


def _require_ml(tac) -> dict:
    """ML prediction or a hard failure - used when ML execution was requested."""
    from backend.ml.model import ModelUnavailableError

    result = _ml_prediction(tac)
    if not result.get("available"):
        raise ModelUnavailableError(
            result.get("reason") or "Model artifact is unavailable."
        )
    return result


def _selection_payload(
    selector: str, executed: str | None, rule_strategy: str, ml_result: dict
) -> dict:
    """States who chose what, so the UI never has to guess."""
    ml_strategy = ml_result.get("predicted_strategy") if ml_result.get("available") else None
    ml_available = bool(ml_result.get("available"))

    if selector == "ml":
        note = "The trained forest chose the strategy for this run."
    elif selector == "explicit":
        note = "The caller pinned this strategy; selector choice was not consulted."
    else:
        note = "The deterministic rule-based scorer chose the strategy."

    if not ml_available:
        agree_note = "ML unavailable, so agreement cannot be determined."
    elif ml_strategy == rule_strategy:
        agree_note = "Rule-Based and ML predict the same strategy."
    else:
        agree_note = "Rule-Based and ML predict different strategies."

    return {
        "selector": selector,
        "executed_strategy": executed,
        "rule_based_strategy": rule_strategy,
        "ml_strategy": ml_strategy,
        "ml_available": ml_available,
        "ml_reason": ml_result.get("reason"),
        "ml_probabilities": ml_result.get("probabilities"),
        "ml_predicted_probability": ml_result.get("predicted_probability"),
        "model_version": ml_result.get("model_version"),
        "strategies_agree": (ml_strategy == rule_strategy) if ml_strategy else None,
        "note": note,
        "agreement_note": agree_note,
    }


def _ml_iteration_context(include_shap: bool):
    """Per-iteration trace enrichment for runs driven by the ML selector.

    Attaches what the forest predicted on this iteration's input, what the
    rule-based scorer would have picked on the same input (so a disagreement is
    visible inside the timeline), and - when requested and available - the SHAP
    explanation of the decision.
    """

    def context(tac, features: dict, selected: str) -> dict:
        from backend.ml.predict import predict_from_features

        rule_pick = select_strategy(tac)
        prediction = predict_from_features(features)
        ml_pick = prediction.get("predicted_strategy") if prediction.get("available") else None

        block = {
            "rule_selected_strategy": rule_pick,
            "ml_predicted_strategy": ml_pick,
            "ml_probabilities": prediction.get("probabilities") or {},
            "ml_predicted_probability": prediction.get("predicted_probability"),
            "ml_model_version": prediction.get("model_version"),
            # Whether the two selectors agreed *at this exact input* - the
            # timeline can therefore show where a divergence starts.
            "strategies_agree": (ml_pick == rule_pick) if ml_pick else None,
        }
        if include_shap:
            from backend.ml.explain import explain_prediction

            block["shap"] = explain_prediction(features)
        return block

    return context


def _iteration_payload(item: dict) -> dict:
    """Map one engine history entry onto the API's iteration shape.

    Existing keys keep their exact meaning (``score`` is still the rule-based
    score of the executed strategy); the added keys carry the measurements and
    trace fields the history view needs.
    """
    strategy = item["selected_strategy"]
    reduction = item["improvement"]
    payload = {
        "iteration": item["iteration"],
        "selector": item.get("selector", "rule"),
        "strategy": strategy,
        "strategy_name": STRATEGY_DISPLAY.get(strategy, strategy),
        "score": round(float(item.get("confidence", 0.0)), 4),
        "rule_scores": {k: round(float(v), 4) for k, v in (item.get("scores") or {}).items()},
        "rationale": item.get("rationale"),
        "before_count": item["before_count"],
        "after_count": item["after_count"],
        "reduction": reduction,
        "improvement_percentage": item["improvement_percentage"],
        "operation_count_before": item.get("operation_count_before"),
        "operation_count_after": item.get("operation_count_after"),
        "operation_reduction": item.get("operation_reduction"),
        "useful_improvement": item.get("useful_improvement"),
        "changed": item.get("changed"),
        "accepted": item.get("accepted"),
        "will_continue": item.get("will_continue"),
        "stop_reason": item.get("stop_reason"),
        "tac_before": tac_to_lines(item.get("tac_before")),
        "tac_after": tac_to_lines(item.get("tac_after")),
        "diff": item.get("diff"),
        "status": "converged" if reduction == 0 else "applied",
    }
    # Optional ML/SHAP enrichment present only on ML-driven runs.
    for key in (
        "rule_selected_strategy",
        "ml_predicted_strategy",
        "ml_probabilities",
        "ml_predicted_probability",
        "strategies_agree",
        "ml_model_version",
        "shap",
    ):
        if key in item:
            payload[key] = item[key]
    return payload


def analyze_source(source: str, include_ml: bool = True) -> dict:
    tac = parse_source(source)

    features = extract_features(tac)
    scores = rule_based_strategy_scores(tac)
    selected = select_strategy(tac)
    # The forest is queried alongside the rule-based view so the workbench can
    # show both selectors for the same TAC in one round trip. A missing
    # artifact degrades to an explicit state rather than failing the analysis.
    ml = _ml_prediction(tac) if include_ml else {
        "available": False,
        "selector": "ml",
        "reason": "ML prediction not requested.",
    }
    return {
        "tac": tac_to_lines(tac),
        "features": _features_payload(features),
        "strategies": _strategies_payload(tac, features, scores, selected),
        "selected_strategy": _selected_payload(tac, scores, selected),
        "selector": "rule",
        "ml": ml,
        "selection": _selection_payload("rule", selected, selected, ml),
        "iterations": [],
        "final_result": {},
    }


def optimize_source(
    source: str,
    strategy: str | None = None,
    selector: str = "rule",
    include_ml: bool = True,
) -> dict:
    """Apply exactly one pass and measure it.

    ``strategy`` pins a pass and wins over ``selector``.  Otherwise ``selector``
    decides: ``"rule"`` reproduces the original behaviour, ``"ml"`` lets the
    forest choose, ``"feedback"`` uses measured outcomes to guide the next
    candidate (all raising :class:`ModelUnavailableError` when untrained).
    """
    tac = parse_source(source)

    features = extract_features(tac)
    scores = rule_based_strategy_scores(tac)
    rule_strategy = select_strategy(tac)

    ml = _ml_prediction(tac) if include_ml else {
        "available": False,
        "selector": "ml",
        "reason": "ML prediction not requested.",
    }

    if strategy is not None and strategy in OPTIMIZATION_FUNCTIONS:
        selected = strategy
        selector_used = "explicit"
    elif selector == "ml":
        ml = _require_ml(tac)
        selected = ml["predicted_strategy"]
        selector_used = "ml"
    elif selector == "feedback":
        ml = _require_ml(tac)
        selected = _feedback_guided_selector(tac)
        selector_used = "feedback"
    else:
        selected = rule_strategy
        selector_used = "rule"

    opt_fn = OPTIMIZATION_FUNCTIONS[selected]
    before = deepcopy(tac)
    after = opt_fn(tac)
    before_count = len(before)
    after_count = len(after)
    reduction = before_count - after_count
    pct = (reduction / before_count * 100) if before_count else 0.0

    measurements = measure_pass(before, after)
    changed = tac_to_lines(before) != tac_to_lines(after)
    change_diff = tac_diff(before, after) if changed else None
    rationale = strategy_reason(tac, selected)

    iteration = {
        "iteration": 1,
        "selector": selector_used,
        "strategy": selected,
        "strategy_name": STRATEGY_DISPLAY.get(selected, selected),
        "score": round(float(scores.get(selected, 0.0)), 4),
        "rule_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "rationale": rationale,
        "before_count": before_count,
        "after_count": after_count,
        "reduction": reduction,
        "improvement_percentage": round(pct, 2),
        "operation_count_before": operation_count(before),
        "operation_count_after": operation_count(after),
        "operation_reduction": int(measurements["operation_reduction"]),
        "useful_improvement": int(measurements["useful_improvement"]),
        "changed": changed,
        "accepted": True,
        "tac_before": tac_to_lines(before),
        "tac_after": tac_to_lines(after),
        "diff": change_diff,
        "status": "applied" if reduction > 0 else "no_change",
    }
    # features / strategies / selected_strategy must all describe the SAME TAC:
    # the one the caller submitted. Scoring them on `after` made a single
    # response report two different scores for the same strategy (the
    # selected_strategy score came from `tac` while strategies came from
    # `after`), and made the reason claim "no repeated subexpressions" for a
    # TAC that was only repeated before the pass ran.
    return {
        "tac": tac_to_lines(after),
        "features": _features_payload(features),
        "strategies": _strategies_payload(tac, features, scores, rule_strategy),
        "selected_strategy": _selected_payload(tac, scores, selected),
        "selector": selector_used,
        "ml": ml,
        "selection": _selection_payload(selector_used, selected, rule_strategy, ml),
        "iterations": [iteration],
        "final_result": {
            "original_count": before_count,
            "final_count": after_count,
            "total_reduction": reduction,
            "reduction_percentage": round(pct, 2),
            "iterations": 1,
            "status": "applied" if reduction > 0 else "no_change",
            "original_tac": tac_to_lines(before),
            "final_tac": tac_to_lines(after),
            "original_operation_count": operation_count(before),
            "final_operation_count": operation_count(after),
            "operation_reduction": int(measurements["operation_reduction"]),
            "useful_improvement": int(measurements["useful_improvement"]),
            "changed": changed,
            "diff": change_diff,
            "rationale": rationale,
        },
    }


def simulate_source(
    source: str,
    max_iterations: int = 20,
    selector: str = "rule",
    include_shap: bool = True,
) -> dict:
    """Run the engine to convergence and return the full trace.

    ``selector="ml"`` makes the forest choose at *every* iteration, and each
    recorded iteration carries the rule-based scores, the forest's prediction
    and (when available) the SHAP explanation for the input it saw.
    
    ``selector="feedback"`` uses Feedback-Guided ML: the forest ranks candidates
    by predicted probability, then each candidate is tested on a copy of the
    current TAC. The first candidate that produces measurable improvement is
    accepted. Rejected candidates are not retried for the same TAC state.
    The engine's convergence safeguard is untouched.
    """
    tac = parse_source(source)

    initial_features = extract_features(tac)
    initial_scores = rule_based_strategy_scores(tac)
    initial_rule_selected = select_strategy(tac)
    ml = _ml_prediction(tac)

    if selector == "ml":
        # Raises ModelUnavailableError when the artifact is missing, so the
        # caller receives an explicit unavailable state instead of a run that
        # quietly fell back to the rule-based selector.
        _require_ml(tac)
        result = run_iterative_optimization(
            tac,
            max_iterations=max_iterations,
            selector=lambda t: _require_ml(t)["predicted_strategy"],
            iteration_context=_ml_iteration_context(include_shap),
            selector_name="ml",
        )
        selector_used = "ml"
    elif selector == "feedback":
        _require_ml(tac)
        # Clear any stale rejected cache for the new run
        if hasattr(_feedback_guided_selector, "_rejected_cache"):
            _feedback_guided_selector._rejected_cache.clear()
        result = run_iterative_optimization(
            tac,
            max_iterations=max_iterations,
            selector=_feedback_guided_selector,
            iteration_context=_ml_iteration_context(include_shap),
            selector_name="feedback",
        )
        selector_used = "feedback"
    else:
        result = run_iterative_optimization(tac, max_iterations=max_iterations)
        selector_used = "rule"

    initial_selected = (
        ml.get("predicted_strategy") if selector_used == "ml" else initial_rule_selected
    )

    iterations = [_iteration_payload(item) for item in result["history"]]

    original_count = len(tac)
    final_tac = result["final_tac"]
    final_count = len(final_tac)
    total_reduction = result["total_reduction"]
    pct = (total_reduction / original_count * 100) if original_count else 0.0
    status = "converged" if result["converged"] or (iterations and iterations[-1]["reduction"] == 0) else "stopped"

    original_measure = measure_pass(tac, final_tac)

    return {
        "tac": tac_to_lines(final_tac),
        "features": _features_payload(initial_features),
        "strategies": _strategies_payload(tac, initial_features, initial_scores, initial_rule_selected),
        "selected_strategy": _selected_payload(tac, initial_scores, initial_selected),
        "selector": selector_used,
        "ml": ml,
        "selection": _selection_payload(selector_used, initial_selected, initial_rule_selected, ml),
        "iterations": iterations,
        "final_result": {
            "original_count": original_count,
            "final_count": final_count,
            "total_reduction": total_reduction,
            "reduction_percentage": round(pct, 2),
            "iterations": len([i for i in iterations if i["status"] == "applied"]),
            "status": status,
            "original_tac": tac_to_lines(tac),
            "final_tac": tac_to_lines(final_tac),
            "original_operation_count": operation_count(tac),
            "final_operation_count": operation_count(final_tac),
            "operation_reduction": int(original_measure["operation_reduction"]),
            "useful_improvement": int(original_measure["useful_improvement"]),
            "stop_reason": result["stop_reason"],
            "converged": result["converged"],
            "selector": selector_used,
            "original_instruction_count": original_count,
            "final_instruction_count": final_count,
        },
    }
