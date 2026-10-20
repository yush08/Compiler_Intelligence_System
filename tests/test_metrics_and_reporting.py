"""Metric, convergence, runtime, and reporting audit tests (§13).

Every expectation is derived from the documented metric definitions and the
actual programs under test — never from hardcoded dashboard numbers:

* instruction counting — what counts as an instruction (print yes, comments
  and blank lines no), the same counting on both sides of a measurement,
  and the empty-input guard;
* operation counting — exactly the documented opcode set, before vs after,
  and the useful_improvement identity (instructions + operations);
* pass history — recorded before/after programs, chained iterations, honest
  changed/accepted flags, and strategy-selection reporting from both
  selectors;
* convergence — unchanged programs, one-change programs, multi-pass runs,
  the iteration limit, convergence exactly at the limit, and failure stops;
* runtime — measured for every executed arm and excluding correctness
  verification from the timed boundary; model payload caching;
* diff counts — deletion, insertion, replacement, reordering, duplicates,
  and agreement with the measurement block;
* audit regressions — replaying every recorded pass reproduces the reported
  final TAC for both strategies, the CF/CP labels describe the actual
  transformations (literal folding vs variable-fed propagation), and the
  combined diff attributes removed/modified to the statements that really
  changed instead of a positional alignment artifact;
* observable behavior — the §10 example program end to end for both
  strategies, plus checker regression cases (nontermination, unsupported
  instructions).
"""

import math
import os
import time

import pytest

from backend.api.serialize import tac_to_lines
from backend.ml import model as model_mod
from backend.ml.compare import compare_source
from backend.ml.predict import predict_from_tac
from backend.optimizer.correctness import check_semantic_equivalence
from backend.optimizer.diff import tac_diff
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from backend.optimizer.measure import measure_pass, operation_count
from backend.scoring.rule_based import select_strategy
from backend.tac.instruction import Instruction, TAC
from backend.tac.interpreter import execute_tac
from backend.tac.parser import parse_tac

# The §10 example: constant folding opportunities, repeated expressions,
# algebraic simplification, dead assignments, constant propagation, and an
# observable print (expected output: 12).
EXAMPLE = """t1 = 4
t2 = 6
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
unused = 999
t6 = 20 + 22
t7 = t6 - 2
result = t5 + 2
print result"""


@pytest.fixture(scope="module")
def example_compare(ml_artifacts):
    """One compare run of the example program, shared by the read-only tests."""
    return compare_source(EXAMPLE)


# ---------------------------------------------------------------------------
# §2 Instruction counting
# ---------------------------------------------------------------------------


def test_instruction_counting_includes_print_and_skips_comments_and_blank_lines():
    tac = parse_tac("# a comment\n\nt1 = 5\n   \nprint t1\n# trailing comment")
    # print is a real instruction of the dialect; comments and whitespace are
    # not instructions under any circumstances.
    assert len(tac) == 2


def test_malformed_lines_never_count_as_instructions():
    tac = parse_tac("= = =\nnot a tac line\n@@@")
    assert len(tac) == 0
    assert operation_count(tac) == 0


def test_measure_pass_uses_the_same_counting_on_both_sides_and_guards_empty():
    original = parse_tac("t1 = 5 + 3\nprint t1")
    optimized = parse_tac("t1 = 8\nprint t1")
    m = measure_pass(original, optimized)
    assert m["original_instruction_count"] == len(original) == 2
    assert m["optimized_instruction_count"] == len(optimized) == 2
    assert m["instruction_reduction"] == 0
    assert m["reduction_percentage"] == 0.0

    empty = measure_pass(parse_tac(""), parse_tac(""))
    assert empty["original_instruction_count"] == 0
    assert empty["optimized_instruction_count"] == 0
    assert empty["instruction_reduction"] == 0
    assert empty["reduction_percentage"] == 0.0  # guarded, never ZeroDivisionError


def test_both_arms_are_measured_against_the_same_original(example_compare):
    parsed = parse_tac(EXAMPLE)
    assert example_compare["original_instruction_count"] == len(parsed) == 10
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        assert block["original_instruction_count"] == len(parsed)
        # The counts come from the actual final TAC that arm returned.
        final = parse_tac("\n".join(block["final_lines"]))
        assert block["final_instruction_count"] == len(final)


# ---------------------------------------------------------------------------
# §3 Operation counting
# ---------------------------------------------------------------------------


def test_operation_counting_follows_the_documented_opcode_definition():
    # Arithmetic operations count.
    assert operation_count(parse_tac("t1 = a + b")) == 1
    assert operation_count(parse_tac("t1 = a - b\nt2 = a * b\nt3 = a / b\nt4 = a % b")) == 4
    assert operation_count(parse_tac("t1 = -a")) == 1  # unary minus
    # Constants, copies, and print do NOT count (assignments and I/O are not
    # arithmetic — this is the documented definition in backend.optimizer.measure).
    assert operation_count(parse_tac("t3 = 10")) == 0  # constant assignment
    assert operation_count(parse_tac("x = y")) == 0  # variable assignment
    assert operation_count(parse_tac("print result")) == 0  # observable output
    # Labels, jumps, and comparisons are control flow, not arithmetic.
    assert operation_count(parse_tac("L1:")) == 0
    assert operation_count(parse_tac("goto L1")) == 0
    assert operation_count(parse_tac("if a > 0 goto L1")) == 0
    # Counting is opcode-based, not liveness-based: a dead arithmetic
    # statement still contains the operation it would perform.
    assert operation_count(parse_tac("dead = a + b")) == 1
    # A folded expression (`t5 = t1 * 1` → `t5 = 10`) is a constant
    # assignment, i.e. exactly the "t3 = 10" case above: 0.
    assert operation_count(parse_tac("t5 = 10")) == 0
    # Empty programs have no operations.
    assert operation_count(parse_tac("")) == 0


def test_useful_improvement_is_instructions_removed_plus_operations_removed():
    original = parse_tac("t1 = 5 + 3\nt2 = t1\nunused = 9")
    optimized = parse_tac("t1 = 8\nt2 = t1")
    m = measure_pass(original, optimized)
    assert m["instruction_reduction"] == 1
    assert m["operation_reduction"] == 1
    assert m["useful_improvement"] == 2
    # The definition is an identity, not a weighting scheme.
    assert m["useful_improvement"] == m["instruction_reduction"] + m["operation_reduction"]


def test_useful_improvement_is_negative_when_a_pass_adds_work():
    m = measure_pass(parse_tac("t1 = 5"), parse_tac("t1 = 5 + 0"))
    assert m["instruction_reduction"] == 0
    assert m["operation_reduction"] == -1
    assert m["useful_improvement"] == -1  # zero instructions, +1 operation


def test_both_arms_use_the_same_useful_improvement_definition(example_compare):
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        m = block["measurements"]
        # The block-level metric is the measurement of its own before/after.
        assert block["useful_improvement"] == m["useful_improvement"]
        assert block["original_instruction_count"] == m["original_instruction_count"]
        assert block["original_operation_count"] == operation_count(parse_tac(EXAMPLE))
        assert block["final_operation_count"] == operation_count(parse_tac("\n".join(block["final_lines"])))
        # …and equals the visible before/after split the UI shows.
        assert block["useful_improvement"] == (
            (block["original_instruction_count"] - block["final_instruction_count"])
            + (block["original_operation_count"] - block["final_operation_count"])
        )


# ---------------------------------------------------------------------------
# §5 Pass history and strategy-selection reporting
# ---------------------------------------------------------------------------


def test_pass_history_records_the_real_before_and_after_programs(example_compare):
    block = example_compare["rule_based"]
    history = block["history"]
    assert history

    previous_after = None
    for item in history:
        assert len(item["tac_before"]) == item["before_count"]
        assert len(item["tac_after"]) == item["after_count"]
        if previous_after is not None:
            assert item["tac_before"] == previous_after  # passes chain in order
        previous_after = item["tac_after"]
        assert item["changed"] is (item["tac_before"] != item["tac_after"])
        assert item["accepted"] is True
        # Each recorded improvement matches that pass's own counts.
        assert item["useful_improvement"] == item["improvement"] + (
            item["operation_count_before"] - item["operation_count_after"]
        )

    # The final TAC is exactly the last pass's output.
    assert previous_after == block["final_lines"]
    assert sum(item["improvement"] for item in history) == block["measurements"]["instruction_reduction"]
    assert block["passes_attempted"] == len(history)
    assert block["passes_changed"] == sum(1 for item in history if item["changed"])
    assert block["passes_unchanged"] == sum(1 for item in history if not item["changed"])
    assert block["passes_accepted"] == sum(1 for item in history if item["accepted"])


def test_a_pass_with_no_effect_is_recorded_as_executed_not_skipped():
    """A predicted pass that does nothing is still executed and recorded —
    and, since the fallback redesign, it no longer ends the whole run: the
    engine tries the next eligible strategy instead (§2)."""
    run = run_iterative_optimization(parse_tac("x = y"), selector=lambda _: "constant_folding")
    # CF has nothing literal to fold; the fallback then runs the one strategy
    # with a real opportunity here (DCE removes the unread assignment), and a
    # final CF attempt on the empty program records the no-op.
    assert run["total_iterations"] == 3
    first = run["history"][0]
    assert first["selected_strategy"] == "constant_folding"
    assert first["origin"] == "predicted"
    assert first["outcome"] == "executed_no_change"
    assert first["changed"] is False
    assert first["improvement"] == 0
    assert tac_to_lines(first["tac_before"]) == tac_to_lines(first["tac_after"])
    # The no-op did NOT stop the run: the controlled fallback executed the
    # eligible dead-code pass and the transformation was accepted.
    second = run["history"][1]
    assert second["origin"] == "fallback"
    assert second["predicted_strategy"] == "constant_folding"
    assert second["selected_strategy"] == "dead_code_elimination"
    assert second["outcome"] == "applied_useful"
    assert second["changed"] is True and second["accepted"] is True
    # Only exhaustion stops the run now, and only the final entry says so.
    assert run["history"][-1]["will_continue"] is False
    assert all(item["stop_reason"] is None for item in run["history"][:-1])
    assert run["stop_reason"] == "no_further_useful_improvement"
    assert run["converged"] is True
    assert tac_to_lines(run["final_tac"]) == []


def test_rule_rationale_quotes_the_features_of_the_input_it_decided_on(example_compare):
    first = example_compare["rule_based"]["history"][0]
    assert first["selected_strategy"] == "dead_code_elimination"
    # Three assignments are dead in the *input* (t4, unused, t7); after the
    # pass only t6 remains dead. The rationale must describe the program the
    # decision was made from, not the already-transformed one.
    assert "dead assignments (3" in first["rationale"]
    assert "dead assignments (1" not in first["rationale"]


def test_rule_arm_first_pick_is_the_rule_decision(example_compare):
    first = example_compare["rule_based"]["history"][0]
    assert first["selected_strategy"] == select_strategy(parse_tac(EXAMPLE))
    assert first["selected_strategy"] == example_compare["rule_based"]["selected_strategy"]


def test_ml_arm_first_pick_is_the_models_own_prediction(example_compare, ml_artifacts):
    ml = example_compare["ml"]
    assert ml["available"]
    expected = predict_from_tac(parse_tac(EXAMPLE))
    assert expected["available"] is True
    assert ml["history"][0]["selected_strategy"] == expected["predicted_strategy"]


def test_each_iteration_records_which_selector_drove_it(example_compare):
    assert all(item.get("selector") == "rule" for item in example_compare["rule_based"]["history"])
    assert all(item.get("selector") == "ml" for item in example_compare["ml"]["history"])


# ---------------------------------------------------------------------------
# §6 Convergence and stopping
# ---------------------------------------------------------------------------


def test_unchanged_program_converges_on_the_first_pass():
    run = run_iterative_optimization(parse_tac("print result"), selector=lambda _: "constant_propagation")
    assert run["total_iterations"] == 1
    assert run["history"][0]["improvement"] == 0
    assert run["history"][0]["changed"] is False
    assert run["stop_reason"] == "no_further_useful_improvement"
    assert run["converged"] is True


def test_program_that_changes_once_stops_on_the_next_no_change_pass():
    run = run_iterative_optimization(
        parse_tac("t1 = a + b\nunused = 5\nprint t1"),
        selector=lambda _: "dead_code_elimination",
    )
    assert run["total_iterations"] == 2
    assert [item["changed"] for item in run["history"]] == [True, False]
    assert [item["improvement"] for item in run["history"]] == [1, 0]
    assert run["stop_reason"] == "no_further_useful_improvement"
    assert run["converged"] is True
    assert tac_to_lines(run["final_tac"]) == ["t1 = a + b", "print t1"]


def test_multi_pass_run_terminates_on_a_real_stopping_condition():
    run = run_iterative_optimization(parse_tac(EXAMPLE))
    assert run["total_iterations"] >= 2
    assert run["stop_reason"] in {"no_instruction_reduction", "no_further_useful_improvement"}
    assert run["converged"] is True
    assert tac_to_lines(run["final_tac"]) == tac_to_lines(run["history"][-1]["tac_after"])
    # The reason is recorded on the entry that stopped the run; earlier
    # entries promise to continue and carry no reason.
    assert run["history"][-1]["stop_reason"] == run["stop_reason"]
    assert all(item["stop_reason"] is None for item in run["history"][:-1])


def test_iteration_limit_is_reported_only_when_it_really_ends_the_run():
    run = run_iterative_optimization(parse_tac(EXAMPLE), max_iterations=1)
    assert run["total_iterations"] == 1
    assert run["history"][-1]["improvement"] != 0  # it would have continued
    assert run["stop_reason"] == "max_iterations_reached"
    assert run["converged"] is False


def test_convergence_on_the_cap_iteration_is_still_convergence():
    # Regression: the limit used to overwrite the recorded reason even when
    # the loop actually ended on the no-improvement condition.
    run = run_iterative_optimization(
        parse_tac("print result"), max_iterations=1, selector=lambda _: "constant_folding"
    )
    assert run["total_iterations"] == 1
    assert run["stop_reason"] == "no_further_useful_improvement"
    assert run["converged"] is True


def test_unknown_strategy_reports_failure_not_convergence():
    run = run_iterative_optimization(parse_tac("t1 = 5 + 3"), selector=lambda _: "no_such_strategy")
    # The unsupported attempt is recorded as part of the history…
    assert run["total_iterations"] == 1
    item = run["history"][0]
    assert item["outcome"] == "unsupported"
    assert item["accepted"] is False
    assert item["changed"] is False
    assert item["selected_strategy"] == "no_such_strategy"
    assert "not one of the six optimization passes" in item["rationale"]
    assert tac_to_lines(item["tac_after"]) == tac_to_lines(item["tac_before"])
    # …and the run stops FAILED without the engine guessing another strategy
    # behind the selector's back; the program itself is untouched.
    assert run["stop_reason"] == "no_applicable_optimization"
    assert run["converged"] is False
    assert tac_to_lines(run["final_tac"]) == ["t1 = 5 + 3"]


# ---------------------------------------------------------------------------
# §7 Runtime measurement
# ---------------------------------------------------------------------------


def test_runtime_is_measured_for_every_executed_arm(example_compare):
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        assert block["available"] is True
        runtime = block["runtime_ms"]
        assert runtime is not None and math.isfinite(runtime) and runtime > 0.0
    feedback = example_compare.get("feedback")
    if feedback is not None and feedback.get("available"):
        assert feedback["runtime_ms"] is not None and feedback["runtime_ms"] > 0.0
    else:
        assert feedback is None or feedback["runtime_ms"] is None


def test_runtime_excludes_correctness_verification(monkeypatch, ml_artifacts):
    """Correctness runs after the timing boundary, for every arm alike."""
    import backend.ml.compare as compare_mod

    def slow_stub(original_tac, final_tac):
        time.sleep(0.6)
        return {"status": "inconclusive", "note": "stubbed", "observed_vars": [], "outputs_compared": False}

    monkeypatch.setattr(compare_mod, "_correctness_block", slow_stub)

    started = time.perf_counter()
    result = compare_source("t1 = a + b\nresult = t1 * 2\nprint result")
    elapsed = time.perf_counter() - started

    executed = [
        arm for arm in ("rule_based", "ml", "feedback") if result.get(arm) and result[arm]["available"]
    ]
    assert len(executed) >= 2  # rule + ml at minimum
    # The stub really ran once per executed arm…
    assert elapsed >= 0.6 * len(executed) - 0.05
    for arm in executed:
        block = result[arm]
        # …but none of that time is billed to the arm's runtime: if
        # correctness were inside the timing, every runtime would be >= 600 ms.
        assert block["runtime_ms"] is not None
        assert block["runtime_ms"] < 500.0
        # The stub's verdict still propagates — correctness is not swallowed.
        assert block["correctness"]["status"] == "inconclusive"


def test_model_payload_is_cached_until_the_artifact_changes(ml_artifacts):
    model_mod.clear_cache()
    try:
        first = model_mod.load_payload()
        second = model_mod.load_payload()
        assert first is second  # one load per process, predictions reuse it

        # The cache is keyed on the artifact's mtime+size, so an updated
        # artifact is picked up without a restart.
        path = str(model_mod.MODEL_FILE)
        stat = os.stat(path)
        try:
            os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))
            third = model_mod.load_payload()
            assert third is not second
        finally:
            os.utime(path, (stat.st_atime, stat.st_mtime))
    finally:
        model_mod.clear_cache()


# ---------------------------------------------------------------------------
# §9 Diff counts
# ---------------------------------------------------------------------------


def test_diff_counts_cover_deletion_insertion_replacement_reordering_and_duplicates():
    cases = [
        # (before, after, exact expectation or None when only the invariants
        # are contractual — difflib may express a reorder either way)
        (["a = 1", "b = 2"], ["a = 1"], {"added": 0, "removed": 1, "modified": 0, "unchanged": 1}),  # deletion
        (["a = 1"], ["a = 1", "b = 2"], {"added": 1, "removed": 0, "modified": 0, "unchanged": 1}),  # insertion
        (["t1 = 5 + 3"], ["t1 = 8"], {"added": 0, "removed": 0, "modified": 1, "unchanged": 0}),  # folding
        (["x = 1", "x = 1", "y = 2"], ["x = 1", "y = 2"], {"added": 0, "removed": 1, "modified": 0, "unchanged": 2}),  # duplicates
        (["a = 1", "b = 2"], ["b = 2", "a = 1"], None),  # reordering
    ]
    for before_lines, after_lines, expected in cases:
        diff = tac_diff(parse_tac("\n".join(before_lines)), parse_tac("\n".join(after_lines)))
        if expected is not None:
            assert {key: diff[key] for key in expected} == expected
        # Invariants for every shape: each side's statements are accounted for
        # exactly once, and removed-added equals the instruction-count delta.
        assert diff["unchanged"] + diff["modified"] + diff["removed"] == len(before_lines)
        assert diff["unchanged"] + diff["modified"] + diff["added"] == len(after_lines)
        assert diff["removed"] - diff["added"] == len(before_lines) - len(after_lines)
        assert diff["changed"] is (before_lines != after_lines)


def test_compare_diff_counts_agree_with_the_measurement_block(example_compare):
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        diff = block["diff"]
        m = block["measurements"]
        assert diff["before_instruction_count"] == m["original_instruction_count"]
        assert diff["after_instruction_count"] == m["optimized_instruction_count"]
        assert diff["removed"] - diff["added"] == m["instruction_reduction"]
        assert diff["unchanged"] + diff["modified"] + diff["removed"] == m["original_instruction_count"]
        assert diff["unchanged"] + diff["modified"] + diff["added"] == m["optimized_instruction_count"]
        assert diff["changed"] == (block["final_lines"] != example_compare["original_lines"])


# ---------------------------------------------------------------------------
# Audit regressions: input identity, pass replay, transformation semantics,
# and diff attribution on the §10 screenshot program
# ---------------------------------------------------------------------------


def test_both_arms_iterate_from_the_byte_identical_original_program(example_compare):
    """Item 1 of the audit: neither arm may start from a rewritten original."""
    original_lines = tac_to_lines(parse_tac(EXAMPLE))
    assert example_compare["original_lines"] == original_lines
    assert example_compare["original_instruction_count"] == len(original_lines)
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        first = block["history"][0]
        assert first["tac_before"] == original_lines
        assert block["original_instruction_count"] == len(original_lines)
        assert block["original_operation_count"] == operation_count(parse_tac(EXAMPLE))


def test_replaying_every_recorded_pass_reproduces_each_reported_final(example_compare):
    """Item 2 of the audit: the history is a log, not a story.

    Each recorded transformation, applied to its own recorded input, must
    reproduce its recorded output — and the chained replay over the accepted
    attempts must reproduce the final TAC the comparison page displays. If
    any step were fabricated or reordered, either equality would fail. A
    rejected attempt is replayable but must never move the chain: it was
    rolled back, and the chain proves it.
    """
    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        current = parse_tac(EXAMPLE)
        for step in block["history"]:
            assert tac_to_lines(current) == step["tac_before"], (
                f"{arm} iteration {step['iteration']}: replay input diverged "
                "from the recorded input"
            )
            replayed = OPTIMIZATION_FUNCTIONS[step["selected_strategy"]](current)
            assert tac_to_lines(replayed) == step["tac_after"], (
                f"{arm} iteration {step['iteration']} "
                f"({step['selected_strategy']}) does not reproduce its "
                "recorded output when replayed"
            )
            if step["accepted"]:
                # Only accepted attempts move the program forward; a no-change
                # attempt replays to the same program it received.
                current = replayed
            else:
                # A rejected attempt is recorded but never applied.
                assert tac_to_lines(current) == step["tac_before"]
        assert tac_to_lines(current) == block["final_lines"], (
            f"{arm}: chained replay does not equal the reported final TAC"
        )


def test_rule_constant_folding_folds_only_literal_expressions(example_compare):
    """Items 3-4 of the audit: the CF label must describe what ran.

    CF folds `t6 = 20 + 22` because both operands are literals. On its own it
    must leave `t5 = t3 * 1` and `result = t5 + 2` alone — t3/t5 are
    variables, so CF cannot know their values; algebraic simplification (which
    could turn `t5 = t3 * 1` into `t5 = t3`) never runs in this arm. Since
    the fallback redesign the run continues past folding into propagation and
    dead-code elimination, so the *final* program is the fully optimized
    fixed point — the per-attempt claims stay individually verifiable from
    the recorded history.
    """
    block = example_compare["rule_based"]
    history = block["history"]
    cf_steps = [s for s in history if s["selected_strategy"] == "constant_folding"]
    assert cf_steps
    # The first CF attempt folds exactly the literal expression — its only
    # modified pair, nothing deleted, nothing added.
    cf = cf_steps[0]
    assert cf["diff"]["removed"] == cf["diff"]["added"] == 0
    assert [
        (e["before_text"], e["after_text"])
        for e in cf["diff"]["lines"]
        if e["status"] == "modified"
    ] == [("t6 = 20 + 22", "t6 = 42")]
    assert "t5 = t3 * 1" in [
        e["before_text"] for e in cf["diff"]["lines"] if e["status"] == "unchanged"
    ]
    # No algebraic simplification ever ran in this arm, so CF's identity
    # (`x * 1` is not CF's job) held throughout…
    assert "algebraic_simplification" not in [s["selected_strategy"] for s in history]
    # …and the run converged to the fully optimized fixed point instead of
    # stopping after the first folding pass.
    assert block["final_lines"] == ["result = 12", "print result"]
    assert block["converged"] is True


def test_ml_constant_propagation_propagates_variables_but_folds_nothing(example_compare):
    """Items 3-4 of the audit: the CP label must describe what ran.

    CP substitutes the known values of t1/t2 down the chain (t3, t5, result
    become literals) but never folds the purely literal `t6 = 20 + 22` — that
    is constant folding's job. Since the fallback redesign a later CF attempt
    (recorded separately, with its own diff) performs that fold, and the run
    converges to the shared fixed point.
    """
    block = example_compare["ml"]
    history = block["history"]
    assert "constant_propagation" in [s["selected_strategy"] for s in history]
    cp = next(s for s in history if s["selected_strategy"] == "constant_propagation")
    assert cp["diff"]["removed"] == cp["diff"]["added"] == 0
    assert [
        (e["before_text"], e["after_text"])
        for e in cp["diff"]["lines"]
        if e["status"] == "modified"
    ] == [
        ("t3 = t1 + t2", "t3 = 10"),
        ("t5 = t3 * 1", "t5 = 10"),
        ("result = t5 + 2", "result = 12"),
    ]
    # The literal-only expression is never modified by CP…
    assert "t6 = 20 + 22" in [
        e["before_text"] for e in cp["diff"]["lines"] if e["status"] == "unchanged"
    ]
    # …and no attempt in this arm ever folds it: dead-code elimination had
    # already deleted `t7 = t6 - 2` in round 1 (making t6 unread), so a later
    # DCE attempt removes the dead `t6 = 20 + 22` instead — deletion, not
    # folding — and the final program contains neither form.
    assert not any(
        s["selected_strategy"] == "constant_folding"
        and s["diff"] is not None
        and any(
            e["status"] == "modified" and e["before_text"] == "t6 = 20 + 22"
            for e in s["diff"]["lines"]
        )
        for s in history
    )
    assert any(
        s["selected_strategy"] == "dead_code_elimination"
        and s["diff"] is not None
        and any(
            e["status"] == "removed" and e["before_text"] == "t6 = 20 + 22"
            for e in s["diff"]["lines"]
        )
        for s in history
    )
    assert "t6 = 20 + 22" not in block["final_lines"]
    assert "t6 = 42" not in block["final_lines"]
    assert block["final_lines"] == ["result = 12", "print result"]
    assert block["converged"] is True


def test_example_diff_attributes_changes_to_the_statements_that_really_changed(example_compare):
    """Item 7 of the audit: removed vs modified must name the true statements.

    Both arms now run to the shared fixed point `result = 12 / print result`:
    eight assignments are deleted, `result = t5 + 2` is rewritten in place to
    `result = 12`, and `print result` is untouched. A positional (LCS)
    pairing would misattribute these — e.g. claim an early deletion swallowed
    `result`'s rewrite because the rows shifted — while keeping the totals
    deceptively consistent, which is exactly the misleading match the diff
    contract forbids.
    """
    expected_removed = [
        "t1 = 4",
        "t2 = 6",
        "t3 = t1 + t2",
        "t4 = t1 + t2",
        "t5 = t3 * 1",
        "unused = 999",
        "t6 = 20 + 22",
        "t7 = t6 - 2",
    ]
    for arm in ("rule_based", "ml"):
        diff = example_compare[arm]["diff"]
        assert [e["before_text"] for e in diff["lines"] if e["status"] == "removed"] == expected_removed
        assert [
            (e["before_text"], e["after_text"])
            for e in diff["lines"]
            if e["status"] == "modified"
        ] == [("result = t5 + 2", "result = 12")]
        assert [e for e in diff["lines"] if e["status"] == "added"] == []
        assert [e["before_text"] for e in diff["lines"] if e["status"] == "unchanged"] == ["print result"]
        assert (diff["added"], diff["removed"], diff["modified"], diff["unchanged"]) == (0, 8, 1, 1)


def test_diff_pairs_a_fold_with_its_own_statement_even_after_earlier_deletions():
    """The alignment artifact in isolation: a deletion ahead of a fold must
    not make the fold pair with the deleted line."""
    diff = tac_diff(
        parse_tac("a = 1\nu = 9\nt = 2 + 3"),
        parse_tac("t = 5"),
    )
    modified = [(e["before_text"], e["after_text"]) for e in diff["lines"] if e["status"] == "modified"]
    assert modified == [("t = 2 + 3", "t = 5")]
    assert [e["before_text"] for e in diff["lines"] if e["status"] == "removed"] == ["a = 1", "u = 9"]
    # Counts are identical to a purely positional pairing: re-pairing only
    # changes the attribution, never the totals.
    assert (diff["added"], diff["removed"], diff["modified"], diff["unchanged"]) == (0, 2, 1, 0)


def test_diff_pairs_positionally_when_no_result_target_matches():
    """A statement whose result target itself changed still pairs as a rewrite
    (positional fallback over the leftovers), so renames render red/green."""
    diff = tac_diff(parse_tac("a = 1\nb = 2"), parse_tac("c = 3\nd = 4"))
    assert diff["modified"] == 2
    assert diff["added"] == diff["removed"] == diff["unchanged"] == 0


# ---------------------------------------------------------------------------
# §10 Observable behavior on the example program
# ---------------------------------------------------------------------------


def test_example_program_preserves_observable_behavior_for_both_arms(example_compare):
    original = parse_tac(EXAMPLE)
    original_run = execute_tac(original)
    assert original_run.error is None
    assert original_run.output == [12.0]  # the program's documented output

    for arm in ("rule_based", "ml"):
        block = example_compare[arm]
        assert block["available"] is True
        final = parse_tac("\n".join(block["final_lines"]))
        final_run = execute_tac(final)
        assert final_run.error is None
        assert final_run.output == original_run.output == [12.0]

        # The checker's own verdict for this pair.
        assert block["correctness"]["status"] == "equivalent"
        assert block["correctness"]["outputs_compared"] is True

        # The observable print survives; genuinely dead statements do not.
        assert "print result" in block["final_lines"]
        assert not any(line.startswith(("t4 =", "unused =", "t7 =")) for line in block["final_lines"])

        # Every displayed metric is derived from these very programs.
        m = block["measurements"]
        assert m["original_instruction_count"] == 10
        assert m["optimized_instruction_count"] == block["final_instruction_count"] == len(final)
        assert m["instruction_reduction"] == 10 - len(final)
        assert m["reduction_percentage"] == round((10 - len(final)) / 10 * 100, 4)
        assert block["original_operation_count"] == operation_count(original) == 6
        assert block["final_operation_count"] == operation_count(final)
        assert block["useful_improvement"] == m["instruction_reduction"] + m["operation_reduction"]


# ---------------------------------------------------------------------------
# §8 Correctness regression cases
# ---------------------------------------------------------------------------


def test_checker_marks_two_nonterminating_programs_inconclusive():
    program = "n = a\nloop:\ngoto loop"
    verdict = check_semantic_equivalence(parse_tac(program), parse_tac(program))
    assert verdict["status"] == "inconclusive"
    assert "Both programs exceeded the" in verdict["detail"]
    assert "no equivalence claim" in verdict["detail"]


def test_checker_marks_an_unsupported_instruction_inconclusive():
    original = parse_tac("x = a + 1")
    # The bogus opcode must *read* a variable, otherwise the optimized program
    # observes nothing and the by-construction branch would fire first.
    unsupported = TAC([Instruction("frobnicate", result="x", arg1="a")])
    verdict = check_semantic_equivalence(original, unsupported)
    assert verdict["status"] == "inconclusive"
    assert "unknown opcode" in verdict["detail"]
    # The interpreter itself reports the fault rather than raising.
    assert "unknown opcode" in (execute_tac(unsupported).error or "")
