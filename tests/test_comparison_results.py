"""Regression tests for the comparison results the screen reports.

Three groups:

1. The exact 14-instruction review example, run end-to-end through
   :func:`backend.ml.compare.compare_source`.  Observable outputs are
   established by *executing the original program* with
   ``backend.tac.interpreter`` inside the test - never copied from an
   expected optimized listing - and every metric assertion recomputes the
   backend's own formulas from the two measured programs, so the displayed
   numbers must keep matching the input and the final TAC rather than a
   value written down here.
2. Diff classification: a folded / propagated statement must be reported as
   one ``modified`` pair keyed by its result target (never a removal plus an
   addition), unchanged ``print`` statements must keep their original order,
   and the row list must read in original-program order - the exact
   regressions the review screenshot exposed.
3. Targeted programs for CSE, dead code elimination, constant propagation,
   algebraic simplification, multiple print statements and division by
   zero, each checked against the output of the *unoptimized* program.
"""

from __future__ import annotations

import pytest

from backend.api.serialize import tac_to_lines
from backend.ml.compare import compare_source
from backend.optimizer.measure import operation_count, work_remaining
from backend.optimizations.algebraic_simplification import algebraic_simplification
from backend.optimizations.common_subexpression import common_subexpression_elimination
from backend.optimizations.constant_propagation import constant_propagation
from backend.optimizations.dead_code import dead_code_elimination
from backend.tac.interpreter import execute_tac
from backend.tac.parser import parse_tac

# The exact example program from the review: 14 instructions, 9 arithmetic
# operations, a repeated expression, two dead assignments, x1/+0 identities,
# constant arithmetic and two prints at the end.
EXAMPLE_SOURCE = """\
t1 = 12
t2 = 3
t3 = t1 / t2
t4 = t1 / t2
t5 = t3 * 1
t6 = t5 + 0
t7 = 100 - 40
t8 = t7 / 2
t9 = t6 + t8
unused1 = 999
unused2 = t1 * 100
t10 = t9 + t2
print t9
print t10"""

MULTI_PRINT_SOURCE = """\
a = 5
b = 6
unused = 9
t = a + b
print a
print t
print b
print a"""


def _output(source_or_lines: str | list[str]) -> list[float]:
    """Execute a program and return what it printed (fails loudly on errors)."""
    source = source_or_lines if isinstance(source_or_lines, str) else "\n".join(source_or_lines)
    result = execute_tac(parse_tac(source))
    assert result.error is None, f"program must run cleanly, got error: {result.error}"
    return list(result.output)


def _arms(result: dict) -> list[tuple[str, dict]]:
    """Every selector arm that actually ran in this response."""
    return [
        (name, result[name])
        for name in ("rule_based", "ml", "feedback")
        if isinstance(result.get(name), dict) and result[name].get("available")
    ]


@pytest.fixture(scope="module")
def original_output() -> list[float]:
    """Ground truth for the example: what the ORIGINAL program prints."""
    return _output(EXAMPLE_SOURCE)


@pytest.fixture(scope="module")
def example_result() -> dict:
    return compare_source(EXAMPLE_SOURCE, include_shap=False)


# --------------------------------------------------------------------- input


def test_example_input_and_observed_output(example_result, original_output):
    original = parse_tac(EXAMPLE_SOURCE)
    # The input's own shape, pinned because the example IS these numbers.
    assert len(original.instructions) == 14
    assert operation_count(original) == 9
    assert work_remaining(original) == 23
    # Executing the input program itself - independently of any optimizer -
    # establishes what must survive: t9=34, t10=37.
    assert original_output == [34.0, 37.0]
    # The response's description of the input matches the parsed source.
    assert example_result["original_lines"] == tac_to_lines(original)
    assert example_result["original_instruction_count"] == 14
    assert example_result["original_operation_count"] == 9
    assert example_result["original_work_remaining"] == work_remaining(original)


def test_every_arm_preserves_observable_output(
    example_result, original_output
):
    arms = _arms(example_result)
    assert any(name == "rule_based" for name, _ in arms), "rule arm must run"
    for name, block in arms:
        assert _output(block["final_lines"]) == original_output, f"{name} changed the prints"
        # VERIFIED may only appear because the checker really ran and passed.
        assert block["correctness_checked"] is True
        assert block["correctness"]["status"] == "equivalent"
        assert block["correctness"]["outputs_compared"] is True
        # The verdict must describe tested inputs as evidence, not a proof.
        assert "not a proof" in block["correctness"]["note"].lower()


def test_reported_metrics_match_the_actual_input_and_final_tac(example_result):
    original = parse_tac(EXAMPLE_SOURCE)
    arms = _arms(example_result)
    assert arms
    for name, block in arms:
        final = parse_tac(block["final_text"])
        assert block["final_lines"] == tac_to_lines(final)
        # Counts are the two measured programs, nothing else.
        assert block["original_instruction_count"] == len(original.instructions)
        assert block["final_instruction_count"] == len(final.instructions)
        assert block["original_operation_count"] == operation_count(original)
        assert block["final_operation_count"] == operation_count(final)
        # Instruction reduction = (original - final), recomputed here.
        removed = len(original.instructions) - len(final.instructions)
        assert block["total_reduction"] == removed
        assert block["measurements"]["instruction_reduction"] == removed
        assert block["measurements"]["reduction_percentage"] == pytest.approx(
            removed / len(original.instructions) * 100
        )
        # Work removed = work(original) - work(final), recomputed here from
        # the programs themselves instead of trusting the displayed value.
        assert block["useful_improvement"] == work_remaining(original) - work_remaining(final)
        assert block["measurements"]["useful_improvement"] == block["useful_improvement"]


# ------------------------------------------------------------ diff honesty


def test_diff_pairs_rewrites_by_result_target(example_result):
    original_lines = example_result["original_lines"]
    arms = _arms(example_result)
    assert arms
    for name, block in arms:
        diff, final_lines = block["diff"], block["final_lines"]
        # Every statement of both sides is accounted for exactly once.
        assert diff["unchanged"] + diff["modified"] + diff["removed"] == len(original_lines)
        assert diff["unchanged"] + diff["modified"] + diff["added"] == len(final_lines)
        # This program only removes or rewrites statements; a rewrite shown
        # as "removed + added" was the misleading-diff regression.
        assert diff["added"] == 0
        assert diff["modified"] == 2

        pairs = [entry for entry in diff["lines"] if entry["status"] == "modified"]
        assert {entry["before_text"] for entry in pairs} == {"t9 = t6 + t8", "t10 = t9 + t2"}
        for entry in pairs:
            # The pair must keep the statement's result target: the new text
            # assigns the same variable the old text assigned.
            before_stmt = parse_tac(entry["before_text"]).instructions
            after_stmt = parse_tac(entry["after_text"]).instructions
            assert len(before_stmt) == len(after_stmt) == 1
            assert after_stmt[0].result == before_stmt[0].result

        unchanged = [
            entry["before_text"] for entry in diff["lines"] if entry["status"] == "unchanged"
        ]
        assert unchanged == ["print t9", "print t10"]  # original order kept

        rewritten_or_kept = {"t9 = t6 + t8", "t10 = t9 + t2", "print t9", "print t10"}
        removed = [entry["before_text"] for entry in diff["lines"] if entry["status"] == "removed"]
        assert removed == [line for line in original_lines if line not in rewritten_or_kept]

        # Rows read top-to-bottom in original-program order: every row that
        # comes from an original statement carries a strictly increasing
        # before-index, covering the whole original program exactly once.
        before_indexes = [
            entry["before_index"]
            for entry in diff["lines"]
            if entry["before_index"] is not None
        ]
        assert before_indexes == sorted(before_indexes)
        assert before_indexes == list(range(len(original_lines)))

        # Nothing that was removed still exists in the final program under
        # another name (no invented, no duplicated result targets).
        removed_targets = {
            parse_tac(text).instructions[0].result for text in removed
        }
        final_targets = {
            stmt.result
            for stmt in parse_tac(block["final_text"]).instructions
            if stmt.result is not None
        }
        assert removed_targets.isdisjoint(final_targets)


# -------------------------------------------------------- pass accounting


def test_history_separates_attempts_from_transformations(example_result):
    arms = _arms(example_result)
    assert arms
    for name, block in arms:
        history = block["history"]
        assert len(history) == block["iterations"] == block["passes_attempted"]
        # This example stops by exhausting useful work, not by the cap.
        assert block["converged"] is True
        assert block["stop_reason"] == "no_further_useful_improvement"

        changed = [item for item in history if item["changed"]]
        unchanged = [item for item in history if not item["changed"]]
        assert block["passes_changed"] == len(changed)
        assert block["passes_unchanged"] == len(unchanged)
        assert block["passes_changed"] + block["passes_unchanged"] == block["passes_attempted"]
        # The example must exercise both kinds of attempt for this contract
        # to mean anything: real transformations AND executed no-effect runs.
        assert changed and unchanged
        assert block["passes_rejected"] == 0
        assert block["passes_accepted"] == block["passes_attempted"]

        # An executed pass that changed nothing is never a transformation.
        for item in unchanged:
            assert item["outcome"] == "executed_no_change"
            assert item["accepted"] is True
            assert item["tac_before"] == item["tac_after"]
        for item in changed:
            assert item["outcome"] in {"applied_useful", "applied_changed"}
            assert item["accepted"] is True
            assert item["tac_before"] != item["tac_after"]

        # The per-attempt counts chain: every attempt starts where the
        # previous one ended, and the final count is the last attempt's.
        assert history[0]["before_count"] == block["original_instruction_count"]
        for previous, current in zip(history, history[1:]):
            assert previous["after_count"] == current["before_count"]
        assert history[-1]["after_count"] == block["final_instruction_count"]
        assert block["final_instruction_count"] == len(block["final_lines"])

        # Every attempt records what was predicted and what origin it ran as.
        for item in history:
            assert item["origin"] in {"predicted", "fallback"}
            assert item["predicted_strategy"] is not None
            assert item["strategy_name"]


# ------------------------------------------------- multi-print preservation


def test_multiple_print_statements_survive_in_order():
    original = parse_tac(MULTI_PRINT_SOURCE)
    original_lines = tac_to_lines(original)
    expected_output = _output(MULTI_PRINT_SOURCE)
    print_lines = [line for line in original_lines if line.startswith("print")]
    assert len(print_lines) == 4  # includes the repeated `print a`
    assert expected_output == [5.0, 11.0, 6.0, 5.0]

    result = compare_source(MULTI_PRINT_SOURCE, include_shap=False)
    arms = _arms(result)
    assert arms
    for name, block in arms:
        assert _output(block["final_lines"]) == expected_output, f"{name} changed the prints"
        # Every print kept, in its original relative order - dead-code
        # elimination must not eat a print or reorder the sequence.
        final_prints = [line for line in block["final_lines"] if line.startswith("print")]
        assert final_prints == print_lines
        unchanged = [
            entry["before_text"]
            for entry in block["diff"]["lines"]
            if entry["status"] == "unchanged"
        ]
        for line in print_lines:
            assert line in unchanged, f"{line} not reported as unchanged in {name}"
        # The dead assignment is gone; the live ones survived.
        assert not any(line.startswith("unused") for line in block["final_lines"])


# ------------------------------------------------------ single-pass regressions


def test_cse_computes_the_repeated_expression_once():
    source = "a = 10\nb = 20\nt1 = a + b\nt2 = a + b\nr = t1 + t2\nprint r"
    original = parse_tac(source)
    expected_output = _output(source)
    final = common_subexpression_elimination(original)
    final_lines = tac_to_lines(final)

    assert _output(final_lines) == expected_output
    # CSE is a 1->1 rewrite: statement count is untouched, but the repeated
    # expression now has exactly one definition.
    assert len(final.instructions) == len(original.instructions)
    right_sides = [
        line.split("=", 1)[1].strip() for line in final_lines if "=" in line
    ]
    assert right_sides.count("a + b") == 1


def test_dead_code_elimination_removes_only_the_unused_assignment():
    source = "t1 = 5 + 3\nunused = t1 * 2\nprint t1"
    original = parse_tac(source)
    expected_output = _output(source)
    final = dead_code_elimination(original)
    final_lines = tac_to_lines(final)

    assert _output(final_lines) == expected_output
    assert "unused = t1 * 2" not in final_lines
    assert "t1 = 5 + 3" in final_lines
    assert "print t1" in final_lines
    assert len(final.instructions) == len(original.instructions) - 1


def test_constant_propagation_removes_reads_of_the_constant():
    source = "a = 10\nt = a + 1\nprint t"
    original = parse_tac(source)
    expected_output = _output(source)
    final = constant_propagation(original)
    final_lines = tac_to_lines(final)

    assert _output(final_lines) == expected_output
    # After propagation no other statement still READS `a` as an operand.
    for line in final_lines:
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        if lhs.strip() == "a":
            continue
        assert "a" not in rhs.split()
    assert "print t" in final_lines


def test_algebraic_simplification_removes_identity_operations():
    source = "t = a * 1\nu = t + 0\nprint u"
    original = parse_tac(source)
    expected_output = _output(source)
    final = algebraic_simplification(original)
    final_lines = tac_to_lines(final)

    assert _output(final_lines) == expected_output
    # The x1 and +0 identities are gone from the rendered program.
    assert not any("* 1" in line for line in final_lines)
    assert not any("+ 0" in line for line in final_lines)
    assert "print u" in final_lines


# ------------------------------------------------------- conservative corners


def test_division_by_zero_is_never_folded_into_a_literal():
    source = "t = 10 / 0\nprint t"
    result = compare_source(source, include_shap=False)
    arms = _arms(result)
    assert arms
    for name, block in arms:
        # Folding 10 / 0 would invent a value the original faults on, so no
        # arm may report a plain literal assignment for it...
        assert "t = 10" not in block["final_lines"]
        # ...and the division the original performs is still there.
        assert any("/" in line for line in block["final_lines"])
        # If the arm changed nothing, equivalence may only be reported for
        # what the checker actually compared.
        if not block["changed"]:
            assert block["correctness"]["status"] in {"equivalent", "inconclusive"}
