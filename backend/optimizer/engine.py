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

from ..features.extractor import extract_features
from ..scoring.rule_based import rule_based_strategy_scores, select_strategy
from ..optimizations.constant_folding import constant_folding
from ..optimizations.constant_propagation import constant_propagation
from ..optimizations.common_subexpression import common_subexpression_elimination
from ..optimizations.dead_code import dead_code_elimination
from ..optimizations.algebraic_simplification import algebraic_simplification
from ..optimizations.strength_reduction import strength_reduction


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


def run_iterative_optimization(tac: TAC, max_iterations: int = MAX_ITERATIONS):
    """
    Run the iterative optimization engine.

    Args:
        tac: Initial TAC to optimize
        max_iterations: Maximum number of iterations (default: 20)

    Returns:
        dict with keys:
            "history": list of iteration result dicts
            "final_tac": the optimized TAC
            "total_reduction": total instruction reduction
            "total_iterations": number of iterations performed
            "converged": whether optimization converged
    """
    history = []
    current_tac = deepcopy(tac)
    total_before = len(current_tac)

    for iteration_num in range(1, max_iterations + 1):
        # 1. Analyze: extract features
        features = extract_features(current_tac)

        # 2. Calculate strategy scores
        scores = rule_based_strategy_scores(current_tac)

        # 3. Select strategy
        selected_strategy = select_strategy(current_tac)

        # 4. Apply strategy
        opt_fn = OPTIMIZATION_FUNCTIONS.get(selected_strategy)
        if opt_fn is None:
            break

        before_count = len(current_tac)
        new_tac = opt_fn(current_tac)
        after_count = len(new_tac)

        # 5. Measure improvement
        improvement = before_count - after_count
        improvement_percentage = (improvement / before_count * 100) if before_count > 0 else 0

        # Store TAC before this iteration
        tac_before = deepcopy(current_tac)

        # 6. Update TAC
        current_tac = new_tac

        # Record the iteration result
        result = {
            "iteration": iteration_num,
            "features": features,
            "scores": scores,
            "selected_strategy": selected_strategy,
            "confidence": scores.get(selected_strategy, 0),
            "before_count": before_count,
            "after_count": after_count,
            "improvement": improvement,
            "improvement_percentage": round(improvement_percentage, 2),
            "tac_before": tac_before,
            "tac_after": current_tac,
        }
        history.append(result)

        # Stop when the selected strategy yields no further instruction reduction.
        if improvement == 0:
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

    return {
        "history": history,
        "final_tac": current_tac,
        "total_reduction": total_reduction,
        "total_iterations": total_iterations,
        "converged": converged,
    }