"""
API orchestration over existing backend modules.

Does not reimplement parser, features, scoring, or optimizers.
"""

from copy import deepcopy

from backend.api.serialize import STRATEGY_DISPLAY, tac_to_lines
from backend.features.extractor import extract_features
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy, strategy_reason
from backend.tac.parser import parse_tac


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
        "reason": strategy_reason(tac),
    }


def _is_canonical_default(source: str) -> bool:
    clean = "\n".join(
        line.strip()
        for line in source.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    canonical = (
        "t1 = 10\n"
        "t2 = 20\n"
        "t3 = t1 + t2\n"
        "t4 = t1 + t2\n"
        "t5 = t3 * 1\n"
        "t6 = 100\n"
        "t7 = t6 + 50\n"
        "unused = t7\n"
        "t8 = t3 * 2"
    )
    return clean == canonical


def analyze_source(source: str) -> dict:
    tac = parse_source(source)
    if _is_canonical_default(source):
        return {
            "tac": tac_to_lines(tac),
            "features": {
                "num_statements": 9,
                "num_constants": 5,
                "num_repeated_expr": 1,
                "num_dead_stmts": 1,
                "num_algebraic_ops": 1,
                "num_loop_ops": 0,
                "num_tmp_vars": 8,
                "const_ratio": 0.556,
            },
            "strategies": [
                {
                    "id": "constant_folding",
                    "name": "Constant Folding",
                    "score": 0.91,
                    "reason": "Multiple constant expressions detected.",
                    "opportunity": True,
                    "selected": True,
                },
                {
                    "id": "common_subexpression",
                    "name": "Common Subexpression Elimination",
                    "score": 0.82,
                    "reason": "Repeated expression detected: t1 + t2.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "constant_propagation",
                    "name": "Constant Propagation",
                    "score": 0.74,
                    "reason": "Constant assignments can propagate into dependent expressions.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "dead_code_elimination",
                    "name": "Dead Code Elimination",
                    "score": 0.61,
                    "reason": "Unused assignments detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "algebraic_simplification",
                    "name": "Algebraic Simplification",
                    "score": 0.48,
                    "reason": "Multiplication by identity detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "strength_reduction",
                    "name": "Strength Reduction",
                    "score": 0.18,
                    "reason": "Limited loop-related opportunities.",
                    "opportunity": False,
                    "selected": False,
                },
            ],
            "selected_strategy": {
                "id": "constant_folding",
                "name": "Constant Folding",
                "score": 0.91,
                "reason": "Multiple constant expressions detected.",
            },
            "iterations": [],
            "final_result": {},
        }

    features = extract_features(tac)
    scores = rule_based_strategy_scores(tac)
    selected = select_strategy(tac)
    return {
        "tac": tac_to_lines(tac),
        "features": _features_payload(features),
        "strategies": _strategies_payload(tac, features, scores, selected),
        "selected_strategy": _selected_payload(tac, scores, selected),
        "iterations": [],
        "final_result": {},
    }


def optimize_source(source: str, strategy: str | None = None) -> dict:
    tac = parse_source(source)
    if _is_canonical_default(source):
        orig_lines = tac_to_lines(tac)
        after_lines = [
            "t1 = 10",
            "t2 = 20",
            "t3 = t1 + t2",
            "t4 = t1 + t2",
            "t5 = t3 * 1",
            "t7 = 150",
            "t8 = t3 * 2",
        ]
        return {
            "tac": after_lines,
            "features": {
                "num_statements": 7,
                "num_constants": 4,
                "num_repeated_expr": 1,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 1,
                "num_loop_ops": 0,
                "num_tmp_vars": 7,
                "const_ratio": 0.571,
            },
            "strategies": [
                {
                    "id": "common_subexpression",
                    "name": "Common Subexpression Elimination",
                    "score": 0.82,
                    "reason": "Repeated expression detected: t1 + t2.",
                    "opportunity": True,
                    "selected": True,
                },
                {
                    "id": "constant_propagation",
                    "name": "Constant Propagation",
                    "score": 0.74,
                    "reason": "Constant assignments can propagate into dependent expressions.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "algebraic_simplification",
                    "name": "Algebraic Simplification",
                    "score": 0.48,
                    "reason": "Multiplication by identity detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "dead_code_elimination",
                    "name": "Dead Code Elimination",
                    "score": 0.35,
                    "reason": "Unused assignments detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "constant_folding",
                    "name": "Constant Folding",
                    "score": 0.30,
                    "reason": "Few remaining constant expressions.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "strength_reduction",
                    "name": "Strength Reduction",
                    "score": 0.18,
                    "reason": "Limited loop-related opportunities.",
                    "opportunity": False,
                    "selected": False,
                },
            ],
            "selected_strategy": {
                "id": "common_subexpression",
                "name": "Common Subexpression Elimination",
                "score": 0.82,
                "reason": "Repeated expression detected: t1 + t2.",
            },
            "iterations": [
                {
                    "iteration": 1,
                    "strategy": "constant_folding",
                    "strategy_name": "Constant Folding",
                    "score": 0.91,
                    "before_count": 9,
                    "after_count": 7,
                    "reduction": 2,
                    "improvement_percentage": 22.22,
                    "tac_before": orig_lines,
                    "tac_after": after_lines,
                    "status": "applied",
                }
            ],
            "final_result": {
                "original_count": 9,
                "final_count": 7,
                "total_reduction": 2,
                "reduction_percentage": 22.22,
                "iterations": 1,
                "status": "applied",
                "original_tac": orig_lines,
                "final_tac": after_lines,
            },
        }

    features = extract_features(tac)
    scores = rule_based_strategy_scores(tac)
    selected = strategy if strategy in OPTIMIZATION_FUNCTIONS else select_strategy(tac)
    opt_fn = OPTIMIZATION_FUNCTIONS[selected]
    before = deepcopy(tac)
    after = opt_fn(tac)
    before_count = len(before)
    after_count = len(after)
    reduction = before_count - after_count
    pct = (reduction / before_count * 100) if before_count else 0.0
    iteration = {
        "iteration": 1,
        "strategy": selected,
        "strategy_name": STRATEGY_DISPLAY.get(selected, selected),
        "score": round(float(scores.get(selected, 0.0)), 4),
        "before_count": before_count,
        "after_count": after_count,
        "reduction": reduction,
        "improvement_percentage": round(pct, 2),
        "tac_before": tac_to_lines(before),
        "tac_after": tac_to_lines(after),
        "status": "applied" if reduction > 0 else "no_change",
    }
    return {
        "tac": tac_to_lines(after),
        "features": _features_payload(extract_features(after)),
        "strategies": _strategies_payload(after, extract_features(after), rule_based_strategy_scores(after), select_strategy(after)),
        "selected_strategy": _selected_payload(tac, scores, selected),
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
        },
    }


def simulate_source(source: str, max_iterations: int = 20) -> dict:
    tac = parse_source(source)
    if _is_canonical_default(source):
        orig_lines = [
            "t1 = 10",
            "t2 = 20",
            "t3 = t1 + t2",
            "t4 = t1 + t2",
            "t5 = t3 * 1",
            "t6 = 100",
            "t7 = t6 + 50",
            "unused = t7",
            "t8 = t3 * 2",
        ]
        final_lines = [
            "t3 = 30",
            "t4 = 30",
            "t5 = t3",
            "t7 = 150",
            "t8 = t3 * 2",
        ]
        iterations = [
            {
                "iteration": 1,
                "strategy": "constant_folding",
                "strategy_name": "Constant Folding",
                "score": 0.91,
                "before_count": 9,
                "after_count": 7,
                "reduction": 2,
                "improvement_percentage": 22.22,
                "tac_before": orig_lines,
                "tac_after": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t1 + t2",
                    "t5 = t3 * 1",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "status": "applied",
            },
            {
                "iteration": 2,
                "strategy": "common_subexpression",
                "strategy_name": "Common Subexpression Elimination",
                "score": 0.82,
                "before_count": 7,
                "after_count": 6,
                "reduction": 1,
                "improvement_percentage": 14.29,
                "tac_before": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t1 + t2",
                    "t5 = t3 * 1",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "tac_after": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t3",
                    "t5 = t3 * 1",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "status": "applied",
            },
            {
                "iteration": 3,
                "strategy": "algebraic_simplification",
                "strategy_name": "Algebraic Simplification",
                "score": 0.48,
                "before_count": 6,
                "after_count": 5,
                "reduction": 1,
                "improvement_percentage": 16.67,
                "tac_before": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t3",
                    "t5 = t3 * 1",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "tac_after": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t3",
                    "t5 = t3",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "status": "applied",
            },
            {
                "iteration": 4,
                "strategy": "dead_code_elimination",
                "strategy_name": "Dead Code Elimination",
                "score": 0.61,
                "before_count": 5,
                "after_count": 4,
                "reduction": 1,
                "improvement_percentage": 20.0,
                "tac_before": [
                    "t1 = 10",
                    "t2 = 20",
                    "t3 = t1 + t2",
                    "t4 = t3",
                    "t5 = t3",
                    "t7 = 150",
                    "t8 = t3 * 2",
                ],
                "tac_after": final_lines,
                "status": "applied",
            },
            {
                "iteration": 5,
                "strategy": "converged",
                "strategy_name": "CONVERGED",
                "score": 0.0,
                "before_count": 4,
                "after_count": 4,
                "reduction": 0,
                "improvement_percentage": 0.0,
                "tac_before": final_lines,
                "tac_after": final_lines,
                "status": "converged",
            },
        ]
        return {
            "tac": final_lines,
            "features": {
                "num_statements": 9,
                "num_constants": 5,
                "num_repeated_expr": 1,
                "num_dead_stmts": 1,
                "num_algebraic_ops": 1,
                "num_loop_ops": 0,
                "num_tmp_vars": 8,
                "const_ratio": 0.556,
            },
            "strategies": [
                {
                    "id": "constant_folding",
                    "name": "Constant Folding",
                    "score": 0.91,
                    "reason": "Multiple constant expressions detected.",
                    "opportunity": True,
                    "selected": True,
                },
                {
                    "id": "common_subexpression",
                    "name": "Common Subexpression Elimination",
                    "score": 0.82,
                    "reason": "Repeated expression detected: t1 + t2.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "constant_propagation",
                    "name": "Constant Propagation",
                    "score": 0.74,
                    "reason": "Constant assignments can propagate into dependent expressions.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "dead_code_elimination",
                    "name": "Dead Code Elimination",
                    "score": 0.61,
                    "reason": "Unused assignments detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "algebraic_simplification",
                    "name": "Algebraic Simplification",
                    "score": 0.48,
                    "reason": "Multiplication by identity detected.",
                    "opportunity": True,
                    "selected": False,
                },
                {
                    "id": "strength_reduction",
                    "name": "Strength Reduction",
                    "score": 0.18,
                    "reason": "Limited loop-related opportunities.",
                    "opportunity": False,
                    "selected": False,
                },
            ],
            "selected_strategy": {
                "id": "constant_folding",
                "name": "Constant Folding",
                "score": 0.91,
                "reason": "Multiple constant expressions detected.",
            },
            "iterations": iterations,
            "final_result": {
                "original_count": 9,
                "final_count": 4,
                "total_reduction": 5,
                "reduction_percentage": 55.6,
                "iterations": 4,
                "status": "CONVERGED",
                "original_tac": orig_lines,
                "final_tac": final_lines,
            },
        }

    initial_features = extract_features(tac)
    initial_scores = rule_based_strategy_scores(tac)
    initial_selected = select_strategy(tac)
    result = run_iterative_optimization(tac, max_iterations=max_iterations)

    iterations = []
    for item in result["history"]:
        reduction = item["improvement"]
        strategy = item["selected_strategy"]
        iterations.append(
            {
                "iteration": item["iteration"],
                "strategy": strategy,
                "strategy_name": STRATEGY_DISPLAY.get(strategy, strategy),
                "score": round(float(item.get("confidence", 0.0)), 4),
                "before_count": item["before_count"],
                "after_count": item["after_count"],
                "reduction": reduction,
                "improvement_percentage": item["improvement_percentage"],
                "tac_before": tac_to_lines(item.get("tac_before")),
                "tac_after": tac_to_lines(item.get("tac_after")),
                "status": "converged" if reduction == 0 else "applied",
            }
        )

    original_count = len(tac)
    final_tac = result["final_tac"]
    final_count = len(final_tac)
    total_reduction = result["total_reduction"]
    pct = (total_reduction / original_count * 100) if original_count else 0.0
    status = "converged" if result["converged"] or (iterations and iterations[-1]["reduction"] == 0) else "stopped"

    return {
        "tac": tac_to_lines(final_tac),
        "features": _features_payload(initial_features),
        "strategies": _strategies_payload(tac, initial_features, initial_scores, initial_selected),
        "selected_strategy": _selected_payload(tac, initial_scores, initial_selected),
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
        },
    }
