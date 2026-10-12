"""
Rule-Based Strategy Scoring.

Transparent, deterministic scoring of all six optimization strategies
for a given TAC. No ML, no black boxes - just clear rules based on
measurable features.

Returns scores in range [0.0, 1.0] for each strategy.
"""

from ..tac.instruction import Instruction

from ..features.extractor import extract_features


# Canonical strategy ids, in the order used to break scoring ties. This is the
# single source of truth for "which strategy wins a tie": select_strategy(),
# the dataset generator's best_strategy labelling, and the experiment runner
# all read it from here instead of each re-declaring their own copy.
STRATEGY_TIE_ORDER = [
    "constant_folding",
    "constant_propagation",
    "common_subexpression",
    "dead_code_elimination",
    "algebraic_simplification",
    "strength_reduction",
]


def rule_based_strategy_scores(tac) -> dict:
    """
    Calculate rule-based strategy scores for a TAC.

    Returns:
        dict with keys:
            "constant_folding": float 0.0-1.0
            "constant_propagation": float 0.0-1.0
            "common_subexpression": float 0.0-1.0
            "dead_code_elimination": float 0.0-1.0
            "algebraic_simplification": float 0.0-1.0
            "strength_reduction": float 0.0-1.0

    The highest-scoring strategy is selected as the "best" optimization.
    """
    # Extract features from the actual TAC
    features = extract_features(tac)

    # Calculate individual strategy scores based on features
    # Constant Folding: score increases with constant expressions
    num_constants = features["num_constants"]
    num_statements = features["num_statements"]
    const_ratio = features["const_ratio"]

    # Constant Folding: high when there are constant binary operations
    cf_score = min(1.0, 0.2 + 0.15 * const_ratio * num_statements / max(1, num_constants + 1))

    # Constant Propagation: high when there are assignments to constants
    # that are later used
    cp_score = min(1.0, 0.1 + 0.15 * num_constants / max(1, num_statements))

    # Common Subexpression Elimination: high when there are repeated expressions
    num_repeated = features["num_repeated_expr"]
    cse_score = min(1.0, 0.1 + 0.15 * num_repeated)

    # Dead Code Elimination: high when there are dead assignments
    num_dead = features["num_dead_stmts"]
    dce_score = min(1.0, 0.1 + 0.15 * num_dead)

    # Algebraic Simplification: high when there are algebraic opportunities
    num_algebraic = features["num_algebraic_ops"]
    as_score = min(1.0, 0.1 + 0.1 * num_algebraic)

    # Strength Reduction: moderate score for multiplication by constants
    num_loop_ops = features["num_loop_ops"]
    sr_score = min(0.2, 0.05 + 0.03 * num_loop_ops)

    scores = {
        "constant_folding": cf_score,
        "constant_propagation": cp_score,
        "common_subexpression": cse_score,
        "dead_code_elimination": dce_score,
        "algebraic_simplification": as_score,
        "strength_reduction": sr_score,
    }

    # Normalize scores so they sum to ~1.0 (helps with UI display)
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores


def select_strategy(tac) -> str:
    """
    Select the best optimization strategy for the given TAC.

    Returns the name of the strategy with the highest score.
    In case of a tie, uses a deterministic tie-breaker order.
    """
    scores = rule_based_strategy_scores(tac)

    # Find the highest score
    max_score = max(scores.values())

    # Find all strategies with the max score
    best_strategies = [name for name, score in scores.items() if score == max_score]

    if len(best_strategies) == 1:
        return best_strategies[0]

    # Tie-breaking by deterministic order (STRATEGY_TIE_ORDER) - never random.
    for name in STRATEGY_TIE_ORDER:
        if name in best_strategies:
            return name

    return STRATEGY_TIE_ORDER[0]  # fallback


def strategy_reason(tac, selected: str | None = None) -> str:
    """
    Generate a human-readable reason for why a strategy was selected.

    Returns a brief explanation based on the detected features.

    ``selected`` lets a caller that already knows which strategy it is
    reporting (for example a pass the user asked for explicitly) get the reason
    for *that* strategy. Without it the reason is derived from
    :func:`select_strategy`, which may name a different strategy than the one
    the caller is reporting.
    """
    features = extract_features(tac)

    scores = rule_based_strategy_scores(tac)
    if selected is None:
        selected = select_strategy(tac)

    reasons = {
        "constant_folding": (
            f"High number of constant expressions "
            f"({features['num_constants']} constants detected)."
        ),
        "constant_propagation": (
            f"Assignments with constant values "
            f"that can be propagated through the code."
        ),
        "common_subexpression": (
            f"Detected {features['num_repeated_expr']} "
            f"repeated subexpression(s) that can be eliminated."
        ),
        "dead_code_elimination": (
            f"High number of dead assignments "
            f"({features['num_dead_stmts']} assignments with unused results)."
        ),
        "algebraic_simplification": (
            f"Detected {features['num_algebraic_ops']} "
            f"algebraic simplification opportunity(s)."
        ),
        "strength_reduction": (
            f"Multiplication by constant pattern(s) "
            f"detected ({features['num_loop_ops']} loop-related ops)."
        ),
    }

    return reasons.get(selected, "Strategy selected based on rule-based scoring.")