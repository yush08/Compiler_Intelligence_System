"""Optimization-engine fallback, pass outcomes, convergence, and robustness.

Covers the final-review requirements on the engine itself:

* §2 — a no-effect prediction does not end the run: the controlled fallback
  executes the rule-ranked next *eligible* strategy, predicted vs executed
  stay distinguishable, features/eligibility are re-extracted after every
  accepted change, and no strategy is retried on an unchanged program state.
* §3 — every attempt lands in exactly one documented outcome
  (applied_useful / applied_changed / executed_no_change / rejected /
  failed / unsupported).
* §4 — termination: cycle guard, iteration cap, and stop reasons that
  describe the real ending condition (CONVERGED vs ITERATION LIMIT vs
  FAILED).
* §6 — semantic acceptance: unsafe transformations are rejected and rolled
  back; faulting programs never produce a false equivalence claim.
* §7 — robustness with faculty-entered programs: observable outputs are
  established by running the ORIGINAL program on the interpreter, then
  preserved through optimization — expected finals are never hardcoded.

No test in this module hardcodes an expected optimized TAC; expectations are
observable behavior plus invariants (output preservation, bounded counts,
honest records).
"""

from copy import deepcopy

import pytest

from backend.api.pipeline import TacParseError, parse_source
from backend.api.serialize import tac_to_lines
from backend.features.extractor import extract_features
from backend.optimizer import engine as engine_mod
from backend.optimizer.correctness import check_semantic_equivalence
from backend.optimizer.engine import (
    OPTIMIZATION_FUNCTIONS,
    PASS_OUTCOMES,
    eligible_strategies,
    run_iterative_optimization,
)
from backend.optimizer.measure import measure_pass
from backend.tac.interpreter import execute_tac
from backend.tac.parser import parse_tac

# The two programs from the review request that exposed early stopping.
EXAMPLE_A = "a = 5\nb = a\nc = b + 3\nprint c"
EXAMPLE_B = "a = 10\nb = 20\nc = a + 5\nprint c"

EXAMPLE_TEN = """t1 = 4
t2 = 6
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
unused = 999
t6 = 20 + 22
t7 = t6 - 2
result = t5 + 2
print result"""


# ---------------------------------------------------------------------------
# §2 Fallback after a no-effect pass
# ---------------------------------------------------------------------------


def test_constant_propagation_is_tried_after_a_no_effect_fold():
    """Example A: CF predicts, does nothing, and the run must still reach
    constant propagation instead of giving up (§2 required behavior)."""
    tac = parse_tac(EXAMPLE_A)
    run = run_iterative_optimization(tac)

    first = run["history"][0]
    assert first["origin"] == "predicted"
    assert first["selected_strategy"] == "constant_folding"
    assert first["outcome"] == "executed_no_change"
    assert first["changed"] is False
    assert first["will_continue"] is True  # the no-op did NOT end the run
    assert first["stop_reason"] is None

    # The fallback executed the eligible propagation strategy…
    cp_steps = [i for i in run["history"] if i["origin"] == "fallback"]
    assert cp_steps
    assert cp_steps[0]["selected_strategy"] == "constant_propagation"
    assert cp_steps[0]["predicted_strategy"] == "constant_folding"
    assert cp_steps[0]["accepted"] is True

    # …and the program converged to the observable answer 8.
    assert run["converged"] is True
    assert run["stop_reason"] in {"no_further_useful_improvement", "no_instruction_reduction"}
    final_run = execute_tac(run["final_tac"])
    assert final_run.output == [8.0]
    verdict = check_semantic_equivalence(tac, run["final_tac"])
    assert verdict["status"] == "equivalent", verdict["detail"]


def test_dead_assignment_is_removed_after_a_no_effect_fold():
    """Example B: CF alone cannot remove `b = 20`; the fallback must reach
    dead-code elimination (§2 required behavior)."""
    tac = parse_tac(EXAMPLE_B)
    run = run_iterative_optimization(tac)

    assert run["history"][0]["selected_strategy"] == "constant_folding"
    assert run["history"][0]["outcome"] == "executed_no_change"
    dce_steps = [
        i for i in run["history"]
        if i["origin"] == "fallback" and i["selected_strategy"] == "dead_code_elimination"
    ]
    assert dce_steps, "dead-code elimination was never offered as a fallback"
    assert any(i["accepted"] for i in dce_steps)

    assert run["converged"] is True
    final_run = execute_tac(run["final_tac"])
    assert final_run.output == [15.0]
    assert check_semantic_equivalence(tac, run["final_tac"])["status"] == "equivalent"


def test_fallback_ranking_comes_from_the_existing_rule_scores():
    """§2.5: fallback candidates reuse the rule-based ranking — on Example B
    DCE (the higher-scoring eligible strategy) is tried before CP."""
    run = run_iterative_optimization(parse_tac(EXAMPLE_B))
    fallback_sequence = [i["selected_strategy"] for i in run["history"] if i["origin"] == "fallback"]
    assert fallback_sequence  # a fallback happened
    assert fallback_sequence[0] == "dead_code_elimination"
    assert "Fallback" in next(
        i["rationale"] for i in run["history"] if i["origin"] == "fallback"
    )


def test_no_strategy_is_retried_while_the_program_state_is_unchanged():
    """§2.12: within one program state every strategy runs at most once; the
    state key resets only when an accepted transformation changes the TAC."""
    run = run_iterative_optimization(parse_tac(EXAMPLE_B))
    seen_state = None
    tried_on_state: set[str] = set()
    for item in run["history"]:
        state = "\n".join(tac_to_lines(item["tac_before"]))
        if state != seen_state:
            seen_state = state
            tried_on_state = set()
        assert item["selected_strategy"] not in tried_on_state, (
            f"state {state!r} retried {item['selected_strategy']} without an "
            "intervening accepted change"
        )
        tried_on_state.add(item["selected_strategy"])


def test_features_and_eligibility_are_recomputed_from_every_recorded_input():
    """§2.9/§2.10: each attempt records features and eligible strategies
    measured from the exact program it was decided on."""
    run = run_iterative_optimization(parse_tac(EXAMPLE_A))
    assert run["total_iterations"] > 1  # the fallback really ran
    for item in run["history"]:
        state_tac = parse_tac("\n".join(tac_to_lines(item["tac_before"])))
        assert item["features"] == extract_features(state_tac)
        assert item["features"]["num_statements"] == item["before_count"]
        assert item["eligible_strategies"] == eligible_strategies(state_tac)


def test_strategies_without_an_opportunity_are_reported_as_ineligible():
    """§5.5: a program with no opportunity for any pass reports an empty
    eligible set — and the engine does not pretend to select work."""
    tac = parse_tac("print result")
    assert eligible_strategies(tac) == []
    run = run_iterative_optimization(tac)
    assert run["history"][0]["eligible_strategies"] == []
    assert run["total_iterations"] == 1  # exactly the prediction, no blind runs
    assert run["converged"] is True


def test_eligibility_mirrors_the_passes_trigger_conditions():
    """Each eligibility scan must agree with the pass it gates: eligible
    means the pass really has work to do (or, for strength reduction, its
    documented detection condition fires)."""
    # CF: literal-literal foldable expression…
    assert "constant_folding" in eligible_strategies(parse_tac("t1 = 5 + 3\nprint t1"))
    # …but not for a variable operand, and not for division by a zero literal.
    assert "constant_folding" not in eligible_strategies(parse_tac("t1 = a + 3\nprint t1"))
    assert "constant_folding" not in eligible_strategies(parse_tac("t1 = 10 / 0\nprint t1"))
    # CP: a constant definition read later; not for free variables only.
    assert "constant_propagation" in eligible_strategies(parse_tac(EXAMPLE_A))
    assert "constant_propagation" not in eligible_strategies(parse_tac("b = a / 0\nprint b"))
    # DCE / CSE / AS track their features directly.
    assert "dead_code_elimination" in eligible_strategies(parse_tac(EXAMPLE_B))
    assert "dead_code_elimination" not in eligible_strategies(parse_tac("a = 1\nprint a"))
    assert "common_subexpression" in eligible_strategies(
        parse_tac("c = a + b\nd = a + b\nprint d")
    )
    assert "algebraic_simplification" in eligible_strategies(parse_tac("t = a * 1\nprint t"))
    # SR detection (MUL by integer constant >= 2) — the pass itself is a
    # documented conservative no-op, which the history records honestly.
    assert "strength_reduction" in eligible_strategies(parse_tac("t = i * 8\nprint t"))
    assert "strength_reduction" not in eligible_strategies(parse_tac("t = i * 1\nprint t"))


# ---------------------------------------------------------------------------
# §3 Pass outcomes
# ---------------------------------------------------------------------------


def test_outcomes_cover_useful_change_noop_and_stay_in_the_documented_set():
    """One program drives all three non-error outcomes, and every attempt
    names exactly one of the documented outcomes."""
    run = run_iterative_optimization(parse_tac("a = 9\nb = a * 1\nc = b + 0\nprint c"))
    outcomes = [item["outcome"] for item in run["history"]]
    assert outcomes
    assert set(outcomes) <= set(PASS_OUTCOMES)
    # Useful work, a neutral rewrite, and a plain no-op all occurred.
    assert "applied_useful" in outcomes
    assert "applied_changed" in outcomes
    assert "executed_no_change" in outcomes
    # A no-op removed nothing but was still executed and recorded.
    noop = next(item for item in run["history"] if item["outcome"] == "executed_no_change")
    assert noop["changed"] is False
    assert noop["improvement"] == 0
    assert noop["useful_improvement"] == 0
    assert noop["accepted"] is True  # identity: the program is kept as-is
    # An applied_changed rewrite really changed text without shrinking counts.
    changed = next(item for item in run["history"] if item["outcome"] == "applied_changed")
    assert changed["changed"] is True
    assert changed["improvement"] == 0
    assert changed["accepted"] is True
    assert run["converged"] is True


def test_semantically_unsafe_transformation_is_rejected_and_rolled_back(monkeypatch):
    """§6.5: a pass whose rewrite would change observable output is
    rejected — the program the engine keeps is the pre-pass program."""
    source = "t1 = a + b\nprint t1"

    def wrong_transform(_tac):
        return parse_tac("t1 = a - b\nprint t1")

    monkeypatch.setitem(OPTIMIZATION_FUNCTIONS, "strength_reduction", wrong_transform)
    run = run_iterative_optimization(
        parse_tac(source), selector=lambda _: "strength_reduction"
    )

    rejected = [item for item in run["history"] if item["outcome"] == "rejected"]
    assert len(rejected) == 1
    item = rejected[0]
    assert item["accepted"] is False
    assert item["changed"] is True  # the attempt's rewrite is recorded…
    assert item["correctness_status"] == "not_equivalent"
    assert item["rejection_reason"]  # …with the checker's actual finding
    # …but it never became the program.
    assert tac_to_lines(run["final_tac"]) == tac_to_lines(parse_tac(source))
    assert run["passes_rejected"] == 1
    assert run["converged"] is True  # exhaustion, not a crash


def test_crashing_pass_is_recorded_as_failed_and_kept_out(monkeypatch):
    """§3 'failed because of an error' — reported, not raised, and the
    program is untouched."""

    def exploding(_tac):
        raise RuntimeError("boom")

    monkeypatch.setitem(OPTIMIZATION_FUNCTIONS, "strength_reduction", exploding)
    run = run_iterative_optimization(
        parse_tac("t1 = a + b\nprint t1"), selector=lambda _: "strength_reduction"
    )

    failed = [item for item in run["history"] if item["outcome"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["accepted"] is False
    assert "RuntimeError: boom" in failed[0]["rejection_reason"]
    assert tac_to_lines(failed[0]["tac_after"]) == tac_to_lines(failed[0]["tac_before"])
    assert tac_to_lines(run["final_tac"]) == ["t1 = a + b", "print t1"]
    assert run["converged"] is True
    assert run["stop_reason"] == "no_further_useful_improvement"


def test_unsupported_strategy_outcome_is_part_of_the_vocabulary():
    """§3 'unsupported for the current input' — recorded, stop = FAILED."""
    assert "unsupported" in PASS_OUTCOMES
    run = run_iterative_optimization(
        parse_tac("t1 = 5 + 3"), selector=lambda _: "not_a_pass"
    )
    assert run["history"][0]["outcome"] == "unsupported"
    assert run["stop_reason"] == "no_applicable_optimization"
    assert run["converged"] is False


# ---------------------------------------------------------------------------
# §4 Convergence, termination, and stop reasons
# ---------------------------------------------------------------------------


def test_cyclic_rewrites_terminate_via_the_repeated_state_guard(monkeypatch):
    """Two semantics-preserving passes that undo each other must not loop
    forever: the second visit to a program state is rejected (§4.4/§4.5)."""
    state_a = parse_tac("t = a + b\nprint t")
    state_b = parse_tac("t = b + a\nprint t")

    def toggle(tac):
        # A real toggle: A -> B, B -> A, either way semantics-preserving.
        if "\n".join(tac_to_lines(tac)) == "\n".join(tac_to_lines(state_a)):
            return deepcopy(state_b)
        return deepcopy(state_a)

    monkeypatch.setitem(OPTIMIZATION_FUNCTIONS, "constant_folding", toggle)
    run = run_iterative_optimization(deepcopy(state_a), selector=lambda _: "constant_folding")

    assert len(run["history"]) == 2  # A -> B accepted, B -> A rejected
    accepted_first = run["history"][0]
    assert accepted_first["accepted"] is True
    assert tac_to_lines(accepted_first["tac_after"]) == tac_to_lines(state_b)
    cycle_rejected = run["history"][1]
    assert cycle_rejected["outcome"] == "rejected"
    assert "previously visited program state" in cycle_rejected["rejection_reason"]
    assert cycle_rejected["accepted"] is False
    # Terminated by exhaustion with the accepted state as the final program.
    assert run["converged"] is True
    assert tac_to_lines(run["final_tac"]) == tac_to_lines(state_b)
    # The kept program still behaves like the original (commuted operands).
    assert execute_tac(run["final_tac"]).output == execute_tac(state_a).output


def test_iteration_limit_still_bounds_a_run_that_would_keep_improving():
    """§2.13/§4.6: the cap is a real limit — reported as ITERATION LIMIT,
    never dressed up as convergence."""
    run = run_iterative_optimization(parse_tac(EXAMPLE_TEN), max_iterations=3)
    assert run["total_iterations"] == 3
    assert run["stop_reason"] == "max_iterations_reached"
    assert run["converged"] is False
    assert run["history"][-1]["will_continue"] is True  # it wanted more


def test_exhaustion_reports_convergence_and_the_reason_matches_the_last_change():
    """§4.6: convergence reasons distinguish 'nothing ever changed', 'only
    operations disappeared', and 'real reductions then nothing left'."""
    # Nothing ever changed:
    unchanged = run_iterative_optimization(parse_tac("print result"))
    assert unchanged["converged"] is True
    assert unchanged["stop_reason"] == "no_further_useful_improvement"
    # The last accepted change reduced instructions:
    shrunk = run_iterative_optimization(parse_tac(EXAMPLE_TEN))
    assert shrunk["converged"] is True
    assert shrunk["stop_reason"] == "no_further_useful_improvement"
    assert shrunk["total_reduction"] > 0
    # A run whose last accepted change only removed operations reports the
    # historical instruction-count reason:
    op_only = run_iterative_optimization(parse_tac("t1 = 5 + 3\nprint t1"))
    assert op_only["converged"] is True
    assert op_only["stop_reason"] == "no_instruction_reduction"
    assert op_only["total_reduction"] == 0
    assert op_only["useful_improvement"] > 0


def test_final_tac_is_exactly_the_replay_of_accepted_transformations():
    """§4.8: the final program is the input with every accepted attempt
    applied in order — rejected attempts contribute nothing."""
    run = run_iterative_optimization(parse_tac(EXAMPLE_TEN))
    current = parse_tac(EXAMPLE_TEN)
    for item in run["history"]:
        assert tac_to_lines(current) == tac_to_lines(item["tac_before"])
        replayed = OPTIMIZATION_FUNCTIONS[item["selected_strategy"]](current)
        assert tac_to_lines(replayed) == tac_to_lines(item["tac_after"])
        if item["accepted"]:
            current = replayed
    assert tac_to_lines(current) == tac_to_lines(run["final_tac"])


# ---------------------------------------------------------------------------
# §6/§7 Faculty-entered program robustness
# ---------------------------------------------------------------------------

ROBUSTNESS_CASES = [
    ("constant folding", "t1 = 10 + 20\nt2 = t1 * 2\nprint t2", [60.0]),
    ("constant propagation", EXAMPLE_A, [8.0]),
    ("dead code elimination", EXAMPLE_B, [15.0]),
    ("common subexpression", "a = 4\nb = 6\nc = a + b\nd = a + b\nprint d", [10.0]),
    ("algebraic simplification", "a = 9\nb = a * 1\nc = b + 0\nprint c", [9.0]),
    ("multiple observable outputs", "a = 3\nb = 4\nprint a\nprint b", [3.0, 4.0]),
]


@pytest.mark.parametrize(
    "name,source,expected_output",
    ROBUSTNESS_CASES,
    ids=[case[0] for case in ROBUSTNESS_CASES],
)
def test_faculty_programs_keep_their_observable_output(name, source, expected_output):
    """§7: the expected observable behavior is established by running the
    ORIGINAL program on the interpreter — the optimized TAC is never
    hardcoded. The final program must print the same sequence, converge, and
    carry a real checker verdict over its printed output."""
    tac = parse_tac(source)

    original_run = execute_tac(tac)
    assert original_run.error is None, name
    assert original_run.output == expected_output, name

    run = run_iterative_optimization(tac)
    assert run["converged"] is True, run["stop_reason"]
    assert set(i["outcome"] for i in run["history"]) <= set(PASS_OUTCOMES)

    # Bounded semantic check over the real checker, plus an independent
    # interpreter run of the final program agreeing with the original.
    verdict = check_semantic_equivalence(tac, run["final_tac"])
    assert verdict["status"] == "equivalent", (name, verdict["detail"])
    assert verdict["outputs_compared"] is True, name
    final_run = execute_tac(run["final_tac"])
    assert final_run.error is None, name
    assert final_run.output == original_run.output, name

    # Optimization never *grows* the program or loses the print line.
    assert len(run["final_tac"]) <= len(tac), name
    if "print" in source:
        assert any(line.startswith("print") for line in tac_to_lines(run["final_tac"])), name
    # The dashboard numbers derive from the measured programs.
    m = measure_pass(tac, run["final_tac"])
    assert m["instruction_reduction"] == len(tac) - len(run["final_tac"]), name


def test_division_by_zero_program_never_claims_successful_equivalence():
    """§7 error handling: execution fails, so no equivalence may be claimed
    and the run reports the inconclusive checker state honestly."""
    source = "a = 10\nb = a / 0\nprint b"
    tac = parse_tac(source)

    original_run = execute_tac(tac)
    assert original_run.error is not None  # the fault is real

    run = run_iterative_optimization(tac)
    verdict = check_semantic_equivalence(tac, run["final_tac"])
    assert verdict["status"] == "inconclusive", verdict["detail"]
    assert verdict["outputs_compared"] is True

    # The propagation rewrite (a -> 10) cannot be shown safe because the
    # program faults either way — so it was rejected and rolled back.
    assert run["passes_rejected"] >= 1
    assert tac_to_lines(run["final_tac"]) == tac_to_lines(parse_tac(source))
    assert run["converged"] is True


def test_literal_division_by_zero_is_never_folded():
    """CF refuses `10 / 0` by design, and the run records that as an honest
    no-effect attempt rather than inventing a value."""
    tac = parse_tac("t1 = 10 / 0\nprint t1")
    run = run_iterative_optimization(tac)
    assert tac_to_lines(run["final_tac"]) == ["t1 = 10 / 0", "print t1"]
    assert check_semantic_equivalence(tac, run["final_tac"])["status"] == "inconclusive"


def test_negative_and_decimal_constants_optimize_with_output_preserved():
    source = "t1 = 2.5 + 0.5\nt2 = -4 * 2\nprint t1\nprint t2"
    tac = parse_tac(source)
    original_run = execute_tac(tac)
    assert original_run.error is None
    assert original_run.output == [3.0, -8.0]

    run = run_iterative_optimization(tac)
    assert any(
        i["selected_strategy"] == "constant_folding" and i["accepted"]
        for i in run["history"]
    ), "literal folding was expected to apply here"
    assert execute_tac(run["final_tac"]).output == [3.0, -8.0]
    assert check_semantic_equivalence(tac, run["final_tac"])["status"] == "equivalent"


def test_variable_reassignment_is_not_stripped_by_any_pass():
    """Reassignment within the supported grammar: the observable final value
    (99) must survive; the conservative passes leave both definitions."""
    source = "a = 5\na = 99\nprint a"
    tac = parse_tac(source)
    assert execute_tac(tac).output == [99.0]

    run = run_iterative_optimization(tac)
    assert execute_tac(run["final_tac"]).output == [99.0]
    assert check_semantic_equivalence(tac, run["final_tac"])["status"] == "equivalent"
    # DCE must not remove either definition: `a` is read by the print.
    assert "a = 99" in tac_to_lines(run["final_tac"])


def test_labels_gotos_and_conditions_survive_optimization():
    """Loops/branches are part of the documented grammar; the engine must
    not crash on them or rewrite the control flow unverified."""
    source = (
        "i = 0\n"
        "total = 0\n"
        "loop:\n"
        "total = total + i\n"
        "i = i + 1\n"
        "if i < 5 goto loop\n"
        "print total"
    )
    tac = parse_tac(source)
    original_run = execute_tac(tac)
    assert original_run.error is None
    assert original_run.output == [10.0]

    run = run_iterative_optimization(tac)
    final_lines = tac_to_lines(run["final_tac"])
    assert "loop:" in final_lines
    assert any(line.startswith("if ") for line in final_lines)
    assert run["converged"] is True
    assert execute_tac(run["final_tac"]).output == [10.0]
    assert check_semantic_equivalence(tac, run["final_tac"])["status"] == "equivalent"


def test_empty_and_malformed_input_never_reach_the_engine():
    """§7: the API layer rejects empty/malformed input with an understandable
    error before any optimization runs; the engine also survives an empty TAC
    if given one directly."""
    for bad in ("", "   \n# only a comment\n", "= = =\n@@@\nnot a tac line"):
        with pytest.raises(TacParseError):
            parse_source(bad)

    run = run_iterative_optimization(parse_tac(""))
    assert run["converged"] is True
    assert run["total_reduction"] == 0
    assert run["total_iterations"] <= 1
