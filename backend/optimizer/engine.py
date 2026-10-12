"""
Iterative Optimization Engine.

Implements the core optimization loop:
    Analyze
    ↓
    Extract Features
    ↓
    Calculate Strategy Scores
    ↓
    Select Strategy
    ↓
    Apply Strategy
    ↓
    Measure Improvement
    ↓
    Update TAC
    ↓
    Repeat

Stops when no further improvement is detected or maximum iterations reached.
"""

from copy import deepcopy

from ..tac.instruction import TAC

from ..api.serialize import tac_to_source
from ..features.extractor import extract_features
from ..scoring.rule_based import (
    rule_based_strategy_scores,
    select_strategy,
    strategy_reason,
)
from ..optimizations.constant_folding import constant_folding
from ..optimizations.constant_propagation import constant_propagation
from ..optimizations.common_subexpression import common_subexpression_elimination
from ..optimizations.dead_code import dead_code_elimination
from ..optimizations.algebraic_simplification import algebraic_simplification
from ..optimizations.strength_reduction import strength_reduction

from .diff import tac_diff
from .measure import measure_pass, operation_count, work_remaining


# Strategy map: name -> optimization function
OPTIMIZATION_FUNCTIONS = {
    "constant_folding": constant_folding,
    "constant_propagation": constant_propagation,
    "common_subexpression": common_subexpression_elimination,
    "dead_code_elimination": dead_code_elimination,
    "algebraic_simplification": algebraic_simplification,
    "strength_reduction": strength_reduction,
}


MAX_ITERATIONS = 20


def run_iterative_optimization(
    tac: TAC,
    max_iterations: int = MAX_ITERATIONS,
    selector=None,
    iteration_context=None,
    selector_name: str = "rule_based",
):
    """
    Run the iterative optimization engine.

    Args:
        tac: Initial TAC to optimize
        max_iterations: Maximum number of iterations (default: 20)
        selector: Optional ``callable(tac) -> strategy_id`` replacing the
            default rule-based choice.  The ML mode passes a selector backed by
            the trained forest; the engine itself is unchanged and is never
            duplicated.  ``None`` preserves the original rule-based behaviour.
        iteration_context: Optional ``callable(tac_before, features, selected) -> dict``
            whose entries are merged into that iteration's record.  It receives
            the TAC the features were extracted from (the input to this
            iteration), so attached explanations describe the decision that was
            actually made.  Used to attach ML probabilities and SHAP
            contributions on request without making them mandatory for
            rule-based runs.
        selector_name: Label recorded on every iteration so the trace states
            which selector drove it (``rule_based`` / ``ml``).

    Returns:
        dict with keys:
            "history": list of iteration result dicts
            "final_tac": the optimized TAC
            "total_reduction": total instruction reduction
            "total_iterations": number of iterations performed
            "converged": whether optimization converged
            "stop_reason": why the loop ended
            "selector": which selector drove the run
    """
    history = []
    current_tac = deepcopy(tac)
    total_before = len(current_tac)

    choose = selector if selector is not None else select_strategy
    stop_reason = "max_iterations_reached"

    for iteration_num in range(1, max_iterations + 1):
        # 1. Analyze: extract features
        features = extract_features(current_tac)

        # 2. Calculate strategy scores (the rule-based view is always recorded
        #    so a run driven by another selector can still be compared against
        #    what the default selector would have chosen).
        scores = rule_based_strategy_scores(current_tac)

        # 3. Select strategy
        selected_strategy = choose(current_tac)

        # 4. Apply strategy
        opt_fn = OPTIMIZATION_FUNCTIONS.get(selected_strategy)
        if opt_fn is None:
            stop_reason = "no_applicable_optimization"
            break

        # Snapshot BEFORE the pass runs so the recorded "TAC before" can never
        # reflect a partially-transformed program, even if a future pass
        # mutates its input in place.
        tac_before = deepcopy(current_tac)

        before_count = len(current_tac)
        before_ops = operation_count(current_tac)
        new_tac = opt_fn(current_tac)
        after_count = len(new_tac)
        after_ops = operation_count(new_tac)

        # 5. Measure improvement (statement reduction is the engine's own
        #    convergence metric; useful_improvement also credits operations
        #    that disappeared without a statement being removed).
        improvement = before_count - after_count
        improvement_percentage = (improvement / before_count * 100) if before_count > 0 else 0
        measurements = measure_pass(tac_before, new_tac)
        useful_improvement = int(measurements["useful_improvement"])

        # 6. Update TAC
        current_tac = new_tac

        changed = tac_to_source(tac_before) != tac_to_source(current_tac)

        # Stop when the selected strategy yields no further instruction
        # reduction.  The safeguard itself is unchanged from the original
        # engine; only the reason recorded for it is more precise.
        if improvement == 0:
            stop_reason = (
                "no_instruction_reduction"
                if useful_improvement > 0
                else "no_further_useful_improvement"
            )
            will_continue = False
        else:
            will_continue = True

        # Record the iteration result
        result = {
            "iteration": iteration_num,
            "selector": selector_name,
            "features": features,
            "scores": scores,
            "selected_strategy": selected_strategy,
            "confidence": scores.get(selected_strategy, 0),
            "rationale": strategy_reason(current_tac, selected_strategy),
            "before_count": before_count,
            "after_count": after_count,
            "improvement": improvement,
            "improvement_percentage": round(improvement_percentage, 2),
            "operation_count_before": before_ops,
            "operation_count_after": after_ops,
            "operation_reduction": int(measurements["operation_reduction"]),
            "useful_improvement": useful_improvement,
            "changed": changed,
            # The engine keeps every transformation it receives; a no-change
            # result is still "accepted" but stops the loop.
            "accepted": True,
            "will_continue": will_continue,
            "stop_reason": None if will_continue else stop_reason,
            "diff": tac_diff(tac_before, current_tac) if changed else None,
            "tac_before": tac_before,
            "tac_after": current_tac,
        }
        if iteration_context is not None:
            # The context explains the decision just made, so it receives the
            # TAC the features were extracted from - not the already-transformed
            # result.
            extra = iteration_context(tac_before, features, selected_strategy)
            if extra:
                result.update(extra)
        history.append(result)

        if not will_continue:
            break

    total_after = len(current_tac)
    total_reduction = total_before - total_after
    total_iterations = len(history)
    converged = (
        total_iterations > 0
        and history[-1]["improvement"] == 0
        if history
        else False
    )

    # Trim converged flag if we stopped due to max iterations
    if total_iterations >= max_iterations:
        converged = False  # Maxed out, not truly converged
        stop_reason = "max_iterations_reached"

    return {
        "history": history,
        "final_tac": current_tac,
        "original_tac": deepcopy(tac),
        "total_reduction": total_reduction,
        "total_iterations": total_iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "selector": selector_name,
        "original_instruction_count": total_before,
        "final_instruction_count": total_after,
        "original_operation_count": operation_count(tac),
        "final_operation_count": operation_count(current_tac),
        "useful_improvement": int(work_remaining(tac) - work_remaining(current_tac)),
    }