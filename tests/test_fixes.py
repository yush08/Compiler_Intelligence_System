"""
Regression tests for the feature extractor, strategy scoring, and the
optimization-pass correctness fixes.

Covers the feature verification cases A-H (constant expressions, constant
propagation, common subexpressions, dead code, algebraic opportunities,
loop-related TAC, temporaries, mixed) and every bug fixed in the backend
cleanup pass.
"""

import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.api.serialize import instruction_to_line, tac_to_lines
from backend.features.extractor import extract_features
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from backend.optimizations.algebraic_simplification import algebraic_simplification
from backend.optimizations.common_subexpression import common_subexpression_elimination
from backend.optimizations.constant_folding import constant_folding
from backend.optimizations.constant_propagation import constant_propagation
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy
from backend.tac.instruction import Instruction
from backend.tac.parser import parse_tac

STRATEGIES = [
    "constant_folding",
    "constant_propagation",
    "common_subexpression",
    "dead_code_elimination",
    "algebraic_simplification",
    "strength_reduction",
]


def feats(source: str) -> dict:
    return extract_features(parse_tac(source))


# ---------------------------------------------------------------------------
# Section 4 - feature verification cases A-H
# ---------------------------------------------------------------------------


def test_features_a_constant_expressions():
    f = feats("t1 = 10 + 20\nt2 = t1 * 2")
    assert f == {
        "num_statements": 2,
        "num_constants": 2,
        "num_repeated_expr": 0,
        "num_dead_stmts": 1,
        "num_algebraic_ops": 0,
        "num_loop_ops": 0,
        "num_tmp_vars": 2,
        "const_ratio": 1.0,
    }


def test_features_b_constant_propagation():
    f = feats("a = 10\nb = a\nc = b + 5")
    assert f["num_statements"] == 3
    assert f["num_constants"] == 2
    assert f["num_repeated_expr"] == 0
    assert f["num_dead_stmts"] == 1
    assert f["num_algebraic_ops"] == 0
    assert f["num_loop_ops"] == 0
    assert f["num_tmp_vars"] == 0
    assert abs(f["const_ratio"] - 2 / 3) < 1e-9


def test_features_c_common_subexpressions():
    f = feats("t1 = a + b\nt2 = a + b")
    assert f["num_repeated_expr"] == 1
    assert f["num_constants"] == 0
    assert f["const_ratio"] == 0.0
    assert f["num_tmp_vars"] == 2


def test_features_d_dead_code():
    f = feats("t1 = 10\nunused = 999\nresult = t1 + 2")
    # `unused` and `result` are never read; `t1` is read by the third line.
    assert f["num_dead_stmts"] == 2
    assert f["num_constants"] == 3
    assert f["num_tmp_vars"] == 1


def test_features_e_algebraic_opportunities():
    f = feats("t1 = a + 0\nt2 = b * 1\nt3 = c - 0")
    assert f["num_algebraic_ops"] == 3
    assert f["num_dead_stmts"] == 3


def test_features_f_loop_related():
    f = feats("L1:\nif i < n goto L2\ngoto L3\nL2:\ni = i + 1\ngoto L1\nL3:")
    assert f["num_statements"] == 7
    # 3 labels + 2 gotos + 1 conditional
    assert f["num_loop_ops"] == 6
    assert f["num_dead_stmts"] == 0  # control flow is never dead
    assert f["num_tmp_vars"] == 0
    assert abs(f["const_ratio"] - 1 / 7) < 1e-9


def test_features_g_temporary_variables():
    src = "\n".join(
        [
            "t1 = a",
            "t2 = b",
            "t3 = t1 + t2",
            "t4 = t3 * 2",
            "t5 = t4 + t1",
            "t6 = t5 + t2",
            "t7 = t6 * 3",
            "t8 = t7 + t3",
            "t9 = t8 + t4",
            "t1 = t9 + t1",
        ]
    )
    f = feats(src)
    # distinct t1..t9, occurrences not double counted, `a`/`b` are not temps
    assert f["num_tmp_vars"] == 9
    assert f["num_statements"] == 10
    assert f["num_constants"] == 2  # `t3 * 2` and `t6 * 3`
    assert f["num_dead_stmts"] == 0  # every t1..t9 is consumed
    assert f["num_loop_ops"] == 0
    assert f["num_algebraic_ops"] == 0


def test_features_h_mixed_and_no_false_positives():
    src = (
        "t1 = 5 + 3\n"
        "t2 = a + b\n"
        "t3 = a + b\n"
        "t4 = t1 * 1\n"
        "unused = 999\n"
        "t5 = t4 + 0\n"
        "x = t5 + t3"
    )
    f = feats(src)
    assert f["num_statements"] == 7
    assert f["num_repeated_expr"] == 1  # exactly one repeated expression
    assert f["num_algebraic_ops"] == 2  # t1 * 1 and t4 + 0
    assert f["num_loop_ops"] == 0  # no control flow present
    assert f["num_tmp_vars"] == 5  # t1..t5, not `unused` or `x`
    assert f["num_constants"] == 4  # 5+3, t1*1, 999, t4+0
    assert f["num_dead_stmts"] == 3  # t2, unused, x
    assert abs(f["const_ratio"] - 4 / 7) < 1e-9


def test_features_no_opportunity_program_reports_nothing():
    f = feats("t1 = a + b\nresult = t1")
    assert f["num_algebraic_ops"] == 0
    assert f["num_loop_ops"] == 0
    assert f["num_repeated_expr"] == 0
    assert f["num_constants"] == 0


def test_features_empty_and_comments_only():
    assert extract_features(parse_tac(""))["num_statements"] == 0
    only_comments = extract_features(parse_tac("# note\n\n   "))
    assert only_comments["num_statements"] == 0
    assert only_comments["const_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Constant propagation correctness
# ---------------------------------------------------------------------------


def test_constant_propagation_respects_redefinition():
    """The env used to be built in a pre-pass, so `x = a` saw the later `a = 99`."""
    out = tac_to_lines(constant_propagation(parse_tac("a = 10\nx = a\na = 99\ny = a")))
    assert out == ["a = 10", "x = 10", "a = 99", "y = 99"]


def test_constant_propagation_clears_stale_binding():
    """A variable rebound to a non-constant must lose its constant binding."""
    out = tac_to_lines(constant_propagation(parse_tac("a = 10\na = x\nb = a")))
    assert out == ["a = 10", "a = x", "b = a"]


def test_constant_propagation_single_pass_is_ordered():
    out = tac_to_lines(
        constant_propagation(parse_tac("a = 1\nb = a\na = 2\nc = a\nd = b"))
    )
    assert out == ["a = 1", "b = 1", "a = 2", "c = 2", "d = 1"]


def test_constant_propagation_folds_when_one_operand_was_a_variable():
    """`p = c + 3` must collapse: the old test demanded BOTH operands change,
    so propagation fired but the operation stayed."""
    out = tac_to_lines(constant_propagation(parse_tac("c = 12\np = c + 3")))
    assert out == ["c = 12", "p = 15"]


def test_constant_propagation_does_not_swallow_constant_folding():
    """`f = 5 + 3` has no propagated operand, so folding remains CF's job."""
    out = tac_to_lines(constant_propagation(parse_tac("f = 5 + 3")))
    assert out == ["f = 5 + 3"]


def test_constant_propagation_leaves_division_by_zero_alone():
    out = tac_to_lines(constant_propagation(parse_tac("c = 5\np = c / 0")))
    assert out == ["c = 5", "p = 5 / 0"]


# ---------------------------------------------------------------------------
# Common subexpression elimination correctness
# ---------------------------------------------------------------------------


def test_cse_respects_operand_redefinition():
    out = tac_to_lines(
        common_subexpression_elimination(parse_tac("t1 = a + b\na = 5\nt2 = a + b"))
    )
    assert out == ["t1 = a + b", "a = 5", "t2 = a + b"]


def test_cse_respects_result_redefinition():
    out = tac_to_lines(
        common_subexpression_elimination(parse_tac("t1 = a + b\nt1 = 7\nt2 = a + b"))
    )
    assert out == ["t1 = a + b", "t1 = 7", "t2 = a + b"]


def test_cse_still_merges_genuine_repeats():
    out = tac_to_lines(
        common_subexpression_elimination(parse_tac("t1 = a + b\nt2 = a + b"))
    )
    assert out == ["t1 = a + b", "t2 = t1"]


def test_cse_handles_self_referential_expression():
    # `t1 = t1 + 1` must not be cached as meaning "t1 + 1 == t1".
    out = tac_to_lines(
        common_subexpression_elimination(parse_tac("t1 = t1 + 1\nt2 = t1 + 1"))
    )
    assert out == ["t1 = t1 + 1", "t2 = t1 + 1"]


# ---------------------------------------------------------------------------
# Algebraic simplification
# ---------------------------------------------------------------------------


def test_algebraic_zero_times_variable():
    """`0 * x -> 0` is counted by the extractor and the score but was never applied."""
    out = tac_to_lines(algebraic_simplification(parse_tac("t1 = 0 * x\nt2 = y * 0")))
    assert out == ["t1 = 0", "t2 = 0"]


def test_algebraic_applies_every_documented_identity():
    out = tac_to_lines(
        algebraic_simplification(
            parse_tac("t1 = a + 0\nt2 = 0 + b\nt3 = c - 0\nt4 = d * 1\nt5 = 1 * e\nt6 = f / 1")
        )
    )
    assert out == ["t1 = a", "t2 = b", "t3 = c", "t4 = d", "t5 = e", "t6 = f"]


# ---------------------------------------------------------------------------
# Parser correctness
# ---------------------------------------------------------------------------


def test_parser_supports_modulo():
    tac = parse_tac("t1 = a % b\nt2 = t1 % 2")
    assert [i.opcode for i in tac.instructions] == [Instruction.MOD, Instruction.MOD]
    assert tac[1].arg2 == "2"
    # the constant operand is now visible to the feature extractor
    assert extract_features(tac)["num_constants"] == 1


def test_constant_folding_folds_modulo():
    out = tac_to_lines(constant_folding(parse_tac("t1 = 7 % 3")))
    assert out == ["t1 = 1"]


def test_conditional_keeps_relational_operator():
    tac = parse_tac("L1:\nif i < n goto L2\ngoto L3\nL2:\ni = i + 1\ngoto L1\nL3:")
    branch = [i for i in tac.instructions if i.opcode == Instruction.IF][0]
    assert branch.relop == "<"
    assert instruction_to_line(branch) == "if i < n goto L2"


def test_serialized_conditional_round_trips():
    src = "L1:\nif i < n goto L2\ngoto L3\nL2:\ni = i + 1\ngoto L1\nL3:"
    lines = tac_to_lines(parse_tac(src))
    assert len(parse_tac("\n".join(lines))) == len(parse_tac(src))


def test_repr_is_valid_tac():
    """CONST/UMINUS/IF used to render as `t4 = 0 const None`, `t1 = -5 uminus None`
    and `if i if n goto L2`."""
    tac = parse_tac("t1 = -5\nt2 = 7\nt3 = t2 + 1\nif t3 > 0 goto L2\nL2:")
    rendered = [str(i) for i in tac.instructions]
    assert rendered[0] == "t1 = -5"
    assert rendered[1] == "t2 = 7"
    assert rendered[3] == "if t3 > 0 goto L2"


def test_lowercase_labels_and_jumps_are_not_dropped():
    """`[A-Za_]` excluded b-z, so `done:` / `goto done` silently vanished."""
    src = "start:\nif x > 0 goto done\ngoto done\ndone:\nresult = x"
    tac = parse_tac(src)
    assert len(tac) == 5
    assert tac_to_lines(tac) == [
        "start:",
        "if x > 0 goto done",
        "goto done",
        "done:",
        "result = x",
    ]


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def test_constant_folding_never_assumes_addition():
    """The removed fallback computed val1 + val2 for any operator."""
    tac = parse_tac("t1 = 5 - 3\nt2 = 6 * 7\nt3 = 8 % 5")
    assert tac_to_lines(constant_folding(tac)) == ["t1 = 2", "t2 = 42", "t3 = 3"]


# ---------------------------------------------------------------------------
# Strategy scoring
# ---------------------------------------------------------------------------


def test_scores_are_normalised_and_bounded():
    for source in ("t1 = 5 + 3\nt2 = a + b\nt2 = a + b", "L1:\ngoto L1", ""):
        scores = rule_based_strategy_scores(parse_tac(source))
        assert set(scores) == set(STRATEGIES)
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_selection_is_argmax_with_deterministic_tie_break():
    sources = [
        "t1 = 5 + 3\nt2 = t1 * 2",
        "t1 = a + b\nt2 = a + b",
        "t1 = 10\nunused = 999\nresult = t1 + 2",
        "t1 = a + 0\nt2 = b * 1",
        "",
    ]
    for source in sources:
        tac = parse_tac(source)
        scores = rule_based_strategy_scores(tac)
        best = max(scores.values())
        expected = next(s for s in STRATEGIES if scores[s] == best)
        # deterministic across repeated calls
        assert select_strategy(tac) == expected
        assert select_strategy(tac) == expected


# ---------------------------------------------------------------------------
# Optimization engine
# ---------------------------------------------------------------------------


def test_engine_terminates_and_never_grows_the_tac():
    for source in (
        "t1 = 5 + 3\nt2 = a + b\nt2 = a + b\nunused = 999\nt3 = t1 + 0",
        "L1:\nif i < n goto L2\ngoto L1\nL2:",
        "x = a + b",
        "",
    ):
        tac = parse_tac(source)
        result = run_iterative_optimization(tac, max_iterations=20)
        assert len(result["history"]) <= 20
        counts = [len(tac)] + [h["after_count"] for h in result["history"]]
        assert all(after <= before for before, after in zip(counts, counts[1:]))
        # a zero-improvement step must end the loop, never be followed by more
        improvements = [h["improvement"] for h in result["history"]]
        for index, improvement in enumerate(improvements):
            if improvement == 0:
                assert index == len(improvements) - 1, (
                    "engine kept running after a pass that changed nothing"
                )


def test_every_pass_keeps_tac_reparsable():
    source = "t1 = 5 + 3\nt2 = a + b\nt2 = a + b\nunused = 999\nt3 = t1 * 1\nt4 = 0 * z"
    for name, fn in OPTIMIZATION_FUNCTIONS.items():
        tac = parse_tac(source)
        out = fn(deepcopy(tac))
        text = "\n".join(tac_to_lines(out))
        assert len(parse_tac(text)) == len(out), name
        assert len(out) <= len(tac), name
