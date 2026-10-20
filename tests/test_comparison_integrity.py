"""Comparison-integrity and observable-output regression tests.

These tests exist because of a discovered issue: ``print result`` was not part
of the TAC dialect at all. The parser silently dropped the line (unmatched
lines returned ``None``), so a demo program whose point was its observable
output lost that output at parse time - ``result`` then read as a dead
assignment, dead-code elimination legitimately removed the whole chain, and
the optimizer reported an empty final program with 100% reduction. The root
cause was missing observable-output support, not an over-eager pass.

The fix is the dialect's smallest justified I/O extension: a ``print x``
instruction that the parser accepts, the serializer renders, the interpreter
executes (recording the emitted values), the correctness checker compares as
observable output, and dead-code elimination treats as an unremovable,
live-operand consumer.

Covered requirements:
  1. observable results survive optimization;
  2. constant folding and propagation preserve observable values;
  3. CSE and algebraic simplification preserve semantics;
  4. truly dead assignments can still be removed;
  5. original and optimized programs check as equivalent when supported;
  6. unsupported correctness checks report inconclusive / not-run (never a pass);
  7. Rule-Based and ML begin from the same original TAC and run independently;
  9. the comparison shows actual final TAC and consistent diff counts;
 11. an identical final TAC reports zero statement reduction.

(Items 8, 10 and 12 are UI behaviour and are covered by the frontend verify
harnesses - verify-render.mjs / verify-integration.mjs - plus the manual
end-to-end check in the Comparison screen.)
"""

import pytest

from backend.api.serialize import tac_to_lines, tac_to_source
from backend.ml import compare as compare_module
from backend.ml.compare import _correctness_block, compare_source
from backend.optimizer.correctness import (
    check_semantic_equivalence,
    has_print_output,
    read_variables,
)
from backend.optimizer.diff import tac_diff
from backend.optimizer.measure import measure_pass
from backend.optimizations.algebraic_simplification import algebraic_simplification
from backend.optimizations.common_subexpression import common_subexpression_elimination
from backend.optimizations.constant_folding import constant_folding
from backend.optimizations.constant_propagation import constant_propagation
from backend.optimizations.dead_code import dead_code_elimination
from backend.tac.instruction import Instruction, TAC
from backend.tac.interpreter import execute_tac
from backend.tac.parser import parse_tac

# The Load Demo program. Kept in sync with DEMO_TAC in the Comparison screen
# (src/views/CompareView.tsx); verify-integration.mjs cross-checks the two.
DEMO_TAC = """t1 = 4
t2 = 6
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
unused = 999
result = t5 + 2
print result"""


def _demo() -> TAC:
    return parse_tac(DEMO_TAC)


# ---------------------------------------------------------------------------
# print parses, serializes and executes
# ---------------------------------------------------------------------------


def test_print_statement_parses_serializes_and_round_trips():
    tac = _demo()
    lines = tac_to_lines(tac)
    # The print line used to be silently dropped by the parser; it must now
    # survive parse -> serialize -> parse unchanged.
    assert len(lines) == 8
    assert lines[-1] == "print result"
    reparsed = parse_tac("\n".join(lines))
    assert tac_to_lines(reparsed) == lines
    assert any(instr.opcode == Instruction.PRINT for instr in tac.instructions)


def test_print_of_a_literal_also_parses():
    lines = tac_to_lines(parse_tac("print 42"))
    assert lines == ["print 42"]


def test_print_executes_as_observable_output():
    run = execute_tac(_demo())
    assert run.terminated
    assert run.error is None
    assert run.output == [12.0]
    assert run.final_state["result"] == 12.0


# ---------------------------------------------------------------------------
# 1 + 5: observable results survive optimization and check as equivalent
# ---------------------------------------------------------------------------


def test_observable_result_survives_every_selector_run(ml_artifacts):
    result = compare_source(DEMO_TAC)
    original_output = execute_tac(_demo()).output
    assert original_output == [12.0]

    executed = [result["rule_based"]]
    if result["ml"]["available"]:
        executed.append(result["ml"])
    if result["feedback"]["available"]:
        executed.append(result["feedback"])
    assert len(executed) >= 1

    for block in executed:
        # The output statement itself is never removed...
        assert "print result" in block["final_lines"], block["final_lines"]
        # ...and the program it feeds still prints the same values.
        final_run = execute_tac(parse_tac(block["final_text"]))
        assert final_run.terminated
        assert final_run.output == original_output
        assert block["correctness"]["status"] == "equivalent"
        assert block["correctness"]["outputs_compared"] is True
        assert "printed identical output" in block["correctness"]["note"]


def test_checker_reports_equivalence_with_output_compared():
    original = _demo()
    optimized = dead_code_elimination(original)
    verdict = check_semantic_equivalence(original, optimized)
    assert verdict["status"] == "equivalent"
    assert verdict["outputs_compared"] is True
    assert "result" in verdict["observed_vars"]
    assert "printed identical output" in verdict["detail"]


def test_checker_fails_when_a_print_is_dropped():
    original = _demo()
    without_print = TAC(
        [instr for instr in original.instructions if instr.opcode != Instruction.PRINT]
    )
    verdict = check_semantic_equivalence(original, without_print)
    assert verdict["status"] == "not_equivalent"
    assert "printed" in verdict["detail"]


def test_empty_result_is_not_equivalent_when_the_original_prints():
    original = parse_tac("unused = 999\nprint unused")
    verdict = check_semantic_equivalence(original, TAC())
    assert verdict["status"] == "not_equivalent"
    assert verdict["outputs_compared"] is True


# ---------------------------------------------------------------------------
# 2 + 3: individual passes preserve observable values
# ---------------------------------------------------------------------------


def test_constant_folding_preserves_observable_value():
    original = parse_tac("t1 = 5 + 3\nresult = t1\nprint result")
    folded = constant_folding(original)
    assert tac_to_lines(folded)[0] == "t1 = 8"
    assert execute_tac(folded).output == [8.0]
    verdict = check_semantic_equivalence(original, folded)
    assert verdict["status"] == "equivalent"
    assert verdict["outputs_compared"] is True


def test_constant_propagation_preserves_observable_value():
    original = parse_tac("t1 = 8\nresult = t1\nprint result")
    propagated = constant_propagation(original)
    assert execute_tac(propagated).output == [8.0]
    verdict = check_semantic_equivalence(original, propagated)
    assert verdict["status"] == "equivalent"
    assert verdict["outputs_compared"] is True


def test_cse_preserves_observable_value():
    original = parse_tac("t1 = a + b\nt2 = a + b\nresult = t2\nprint result")
    cse = common_subexpression_elimination(original)
    assert tac_to_lines(cse) == ["t1 = a + b", "t2 = t1", "result = t2", "print result"]
    assert execute_tac(cse).output == execute_tac(original).output
    verdict = check_semantic_equivalence(original, cse)
    assert verdict["status"] == "equivalent"
    assert verdict["outputs_compared"] is True


def test_algebraic_simplification_preserves_observable_value():
    original = parse_tac("t5 = t3 * 1\nresult = t5 + 2\nprint result")
    simplified = algebraic_simplification(original)
    assert tac_to_lines(simplified)[0] == "t5 = t3"
    assert execute_tac(simplified).output == execute_tac(original).output
    verdict = check_semantic_equivalence(original, simplified)
    assert verdict["status"] == "equivalent"


# ---------------------------------------------------------------------------
# 4: truly dead assignments can still be removed (while print stays live)
# ---------------------------------------------------------------------------


def test_dead_assignments_removed_but_print_chain_kept():
    optimized = dead_code_elimination(_demo())
    lines = tac_to_lines(optimized)
    # t4 (duplicate expression) and unused are genuinely dead: removed.
    assert "t4 = t1 + t2" not in lines
    assert "unused = 999" not in lines
    # The chain feeding print result stays, and so does the print.
    assert "result = t5 + 2" in lines
    assert lines[-1] == "print result"
    assert len(lines) == 6
    assert execute_tac(optimized).output == [12.0]


def test_dce_fixpoint_never_empties_a_printing_program():
    original = parse_tac("t1 = 10\nt2 = t1 + 5\nunused = t2\nprint t2")
    optimized = original
    for _ in range(10):
        next_tac = dead_code_elimination(optimized)
        if tac_to_source(next_tac) == tac_to_source(optimized):
            break
        optimized = next_tac
    # unused is dead and goes; t1/t2 and the print are live and stay.
    assert tac_to_lines(optimized) == ["t1 = 10", "t2 = t1 + 5", "print t2"]
    assert execute_tac(optimized).output == execute_tac(original).output
    assert has_print_output(optimized)
    assert "t2" in read_variables(optimized)


def test_all_dead_fixture_without_print_still_empties():
    """The documented empty-result case is unchanged: without any observable,
    the all-dead fixture legitimately optimizes to empty."""
    original = parse_tac(
        "t1 = 10\nt2 = 20\nt3 = t1 + t2\nt4 = t1 + t2\nunused = t3"
    )
    optimized = original
    for _ in range(10):
        next_tac = dead_code_elimination(optimized)
        if tac_to_source(next_tac) == tac_to_source(optimized):
            break
        optimized = next_tac
    assert optimized.instructions == []
    verdict = check_semantic_equivalence(original, optimized)
    assert verdict["status"] == "equivalent"
    assert verdict["outputs_compared"] is False


# ---------------------------------------------------------------------------
# 6: unsupported checks never claim a pass
# ---------------------------------------------------------------------------


def test_checker_exception_reports_inconclusive_never_a_pass(monkeypatch):
    import backend.optimizer.correctness as correctness_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(correctness_module, "check_semantic_equivalence", boom)
    block = _correctness_block(_demo(), _demo())
    assert block["status"] == "inconclusive"
    assert "RuntimeError" in block["note"]
    assert block["outputs_compared"] is False


def test_not_run_arm_shape_reports_no_check():
    """A never-executed arm must carry not_run, never a verdict."""
    result = compare_source(DEMO_TAC, include_shap=False)
    for block in (result["rule_based"], result["ml"], result["feedback"]):
        if not block["available"]:
            assert block["correctness"]["status"] == "not_run"
            assert block["correctness"]["outputs_compared"] is False
        else:
            assert block["correctness"]["status"] in {
                "equivalent",
                "not_equivalent",
                "inconclusive",
            }


# ---------------------------------------------------------------------------
# 7: same original TAC, independent runs
# ---------------------------------------------------------------------------


def test_rule_and_ml_start_from_the_same_original_and_run_independently(monkeypatch, ml_artifacts):
    recorded: list[tuple[int, list[str]]] = []
    real = compare_module.run_iterative_optimization

    def spy(tac, **kwargs):
        # Capture the input BEFORE the run so engine mutation cannot mask it.
        recorded.append((id(tac), tac_to_lines(tac)))
        return real(tac, **kwargs)

    monkeypatch.setattr(compare_module, "run_iterative_optimization", spy)

    result = compare_source(DEMO_TAC)
    expected = tac_to_lines(_demo())

    # One recorded start per executed arm (rule, and ml/feedback when available).
    expected_runs = 1 + int(result["ml"]["available"]) + int(result["feedback"]["available"])
    assert len(recorded) == expected_runs

    # Every arm received the same original program...
    for _tid, lines in recorded:
        assert lines == expected
    # ...as a distinct object, so no arm can observe another's mutations.
    ids = [tid for tid, _ in recorded]
    assert len(set(ids)) == len(ids)

    # Both executed arms report the same original statement count.
    assert result["rule_based"]["original_instruction_count"] == 8
    if result["ml"]["available"]:
        assert result["ml"]["original_instruction_count"] == 8


# ---------------------------------------------------------------------------
# 9 + 11: actual final TAC, consistent diff counts, zero on identical input
# ---------------------------------------------------------------------------


def test_compare_reports_actual_final_tac_and_consistent_counts(ml_artifacts):
    result = compare_source(DEMO_TAC)
    original_count = result["original_instruction_count"]
    assert original_count == 8

    for name in ("rule_based", "ml", "feedback"):
        block = result[name]
        if not block["available"]:
            continue
        # The final TAC shown is the actual program the run produced: it
        # round-trips through the parser and matches final_text exactly.
        assert block["final_text"] == "\n".join(block["final_lines"])
        assert tac_to_lines(parse_tac(block["final_text"])) == block["final_lines"]

        diff = block["diff"]
        m = block["measurements"]
        final_count = block["final_instruction_count"]

        # Count invariants: modified lines are counted as changed, never as
        # removed; removed - added is exactly the statement-count delta.
        assert diff["removed"] - diff["added"] == original_count - final_count
        assert diff["unchanged"] + diff["modified"] + diff["removed"] == original_count
        assert diff["unchanged"] + diff["modified"] + diff["added"] == final_count
        assert diff["before_instruction_count"] == original_count
        assert diff["after_instruction_count"] == final_count

        # The measurement block matches the same counts, with the documented
        # reduction formula.
        assert m["original_instruction_count"] == original_count
        assert m["optimized_instruction_count"] == final_count
        assert m["instruction_reduction"] == original_count - final_count
        assert m["reduction_percentage"] == round(
            (original_count - final_count) / original_count * 100, 4
        )


def test_identical_final_tac_reports_zero_reduction():
    tac = _demo()
    m = measure_pass(tac, tac)
    assert m["instruction_reduction"] == 0
    assert m["reduction_percentage"] == 0.0
    assert m["operation_reduction"] == 0
    assert m["useful_improvement"] == 0

    d = tac_diff(tac, tac)
    assert d["changed"] is False
    assert d["added"] == d["removed"] == d["modified"] == 0
    assert d["unchanged"] == len(tac)

    # Through the whole compare on a program no pass can change: labels and
    # jumps are never removed by any of the six passes, so every executed arm
    # ends where it started and every reduction number is zero.
    result = compare_source("L1:\ngoto L1")
    for name in ("rule_based", "ml", "feedback"):
        block = result[name]
        if not block["available"]:
            continue
        assert block["final_lines"] == result["original_lines"]
        assert block["measurements"]["instruction_reduction"] == 0
        assert block["measurements"]["reduction_percentage"] == 0.0
        assert block["diff"]["added"] == 0
        assert block["diff"]["removed"] == 0
        assert block["diff"]["changed"] is False


def test_demo_rule_and_ml_finals_are_measured_not_reused(ml_artifacts):
    """Both arms run on the same input; whatever finals they actually reach
    are each measured from that arm's own program (no stale numbers)."""
    result = compare_source(DEMO_TAC)
    rule = result["rule_based"]
    assert rule["available"]
    assert rule["final_lines"] != result["original_lines"]  # DCE really ran
    assert rule["measurements"]["instruction_reduction"] > 0
    if result["ml"]["available"]:
        ml = result["ml"]
        # The ML arm's numbers come from the ML arm's own final program.
        assert ml["measurements"] == measure_pass(
            _demo(), parse_tac(ml["final_text"])
        )
