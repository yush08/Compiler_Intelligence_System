"""Semantic-preservation regression tests.

These tests exist because of a discovered issue: ``constant_propagation``
carried its constant environment *across labels*, so on a counted loop it
rewrote ``acc = acc + i`` to ``acc = 0`` and ``i = i + 1`` to ``i = 1`` -
silently converting a terminating program (result 45) into an infinite loop
with a zeroed accumulator.  ``common_subexpression_elimination`` had the same
class of bug: its expression cache survived across labels, so a back edge
could re-execute an eliminated occurrence with stale operands.

The fix in both passes is the same invariant: a forward fact (known constant,
cached expression) is only valid inside a label-free straight-line stretch,
because every join point in this dialect is a label instruction - jumps can
only target labels.  Both passes therefore clear their state at every label.

Validation model (backend.optimizer.correctness): the observable behavior of
a program is the sequence of values it ``print``s plus the final values of
the variables the *optimized* program still reads, plus whether it
terminates.  All are compared by executing original and optimized programs
under backend.tac.interpreter on deterministic input vectors.
"""

import pytest

from backend.api.serialize import tac_to_lines, tac_to_source
from backend.optimizer.correctness import (
    check_semantic_equivalence,
    read_variables,
)
from backend.optimizer.engine import run_iterative_optimization
from backend.optimizations.algebraic_simplification import algebraic_simplification
from backend.optimizations.common_subexpression import common_subexpression_elimination
from backend.optimizations.constant_folding import constant_folding
from backend.optimizations.constant_propagation import constant_propagation
from backend.optimizations.dead_code import dead_code_elimination
from backend.tac.interpreter import execute_tac, input_value
from backend.tac.parser import parse_tac

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

# The counted loop whose semantics the old constant propagation destroyed.
COUNTED_LOOP = """n = 10
i = 0
acc = 0
loop_head:
if i >= n goto loop_end
acc = acc + i
i = i + 1
goto loop_head
loop_end:
result = acc"""

# CSE variant: the cached expression's operand is redefined inside the loop
# body, so a back edge brings different values to the second occurrence.
CSE_LOOP_REDEF = """t = a + b
loop_head:
u = a + b
a = a + 1
if u < 100 goto loop_head"""

# Forward-branch join: x is 5 on the goto path and 7 on the fall-through
# path, so nothing may propagate x's constant into the joined stretch.
FORWARD_JOIN = """x = 5
if c < 10 goto Lx
x = 7
Lx:
y = x + 1
acc = y + zin
acc = acc + ztail"""

STRAIGHT_LINE = """c0 = 12
p0 = c0 + 3
r0a = u0 + v0
r0b = v0 + u0
acc = p0 + r0a
acc = acc + r0b
acc = acc + ztail"""

ALL_DEAD = """t1 = 10
t2 = 20
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
t6 = 100
t7 = t6 + 50
unused = t7
t8 = t3 * 2"""

BRANCHY = """x = a + 0
if x > 5 goto big
y = x + 1
goto done
big:
y = x + 2
done:
acc = y * 1
acc = acc + ztail"""

PASSES = [
    ("constant_folding", constant_folding),
    ("constant_propagation", constant_propagation),
    ("common_subexpression", common_subexpression_elimination),
    ("dead_code_elimination", dead_code_elimination),
    ("algebraic_simplification", algebraic_simplification),
]

BATTERY = {
    "counted_loop": COUNTED_LOOP,
    "cse_loop_redef": CSE_LOOP_REDEF,
    "forward_join": FORWARD_JOIN,
    "straight_line": STRAIGHT_LINE,
    "all_dead": ALL_DEAD,
    "branchy": BRANCHY,
}


# ---------------------------------------------------------------------------
# Interpreter semantics
# ---------------------------------------------------------------------------


def test_interpreter_runs_counted_loop_to_the_right_answer():
    result = execute_tac(parse_tac(COUNTED_LOOP))
    assert result.terminated is True
    assert result.error is None
    # 0 + 1 + ... + 9
    assert result.final_state["acc"] == 45.0
    assert result.final_state["i"] == 10.0


def test_interpreter_step_limit_bounds_a_non_terminating_loop():
    infinite = parse_tac("loop:\ni = i + 1\ngoto loop")
    result = execute_tac(infinite, step_limit=500)
    assert result.terminated is False
    assert result.steps == 500


def test_interpreter_conditional_branches_both_ways():
    taken = execute_tac(parse_tac("x = 7\ngoto skip\nx = 9\nskip:\nacc = x + 0"))
    assert taken.final_state["acc"] == 7.0
    # Not-taken: x stays 1 (its defined value), so `x + 2` = 3, then `+ zt`.
    not_taken = execute_tac(parse_tac("x = 1\nif x > 5 goto far\nacc = x + 2\ngoto end\nfar:\nacc = 0\nend:\nacc = acc + zt"))
    assert not_taken.final_state["acc"] == 3.0 + input_value("zt")


def test_interpreter_comparison_as_value_never_jumps():
    # The parser's `t = a > b` form reuses the IF opcode with a result var.
    result = execute_tac(parse_tac("t = a > b\nacc = t + 1\nacc = acc + zt"), inputs={"a": 3.0, "b": 7.0})
    assert result.terminated is True
    assert result.final_state["t"] == 0.0
    assert result.final_state["acc"] == 4.0


def test_interpreter_reports_faults_instead_of_raising():
    assert execute_tac(parse_tac("t = 5 / 0")).error == "division by zero"
    assert execute_tac(parse_tac("t = 5 % 0")).error == "modulo by zero"
    assert "unknown label" in execute_tac(parse_tac("goto nowhere")).error


def test_interpreter_inputs_are_deterministic_and_vector_dependent():
    # With only nine possible values in [2, 10], name collisions are expected;
    # determinism (same name -> same value) is the property that matters, and
    # a different input *vector* always shifts the value.
    assert input_value("alpha") == input_value("alpha")
    assert input_value("beta") == input_value("beta")
    assert input_value("alpha", 0) != input_value("alpha", 1)
    assert 2.0 <= input_value("alpha", 0) <= 10.0


# ---------------------------------------------------------------------------
# The discovered issue: flow-blind state must not cross a label
# ---------------------------------------------------------------------------


def test_constant_propagation_is_sound_on_loops():
    """Regression: CP used to zero the accumulator and pin `i = 1`, turning
    this terminating loop into an infinite one."""
    original = parse_tac(COUNTED_LOOP)
    optimized = constant_propagation(original)

    check = check_semantic_equivalence(original, optimized)
    assert check["status"] == "equivalent", check["detail"]
    # The loop body must survive verbatim: nothing in it is a must-constant.
    lines = tac_to_lines(optimized)
    assert "acc = acc + i" in lines
    assert "i = i + 1" in lines


def test_cse_is_sound_when_operands_change_in_a_loop_body():
    """Regression: CSE used to freeze `u` at the pre-loop value of a+b."""
    original = parse_tac(CSE_LOOP_REDEF)
    optimized = common_subexpression_elimination(original)

    check = check_semantic_equivalence(original, optimized)
    assert check["status"] == "equivalent", check["detail"]
    lines = tac_to_lines(optimized)
    assert "u = a + b" in lines, "the occurrence inside the loop must not be eliminated"


def test_constant_propagation_does_not_cross_a_forward_branch_join():
    """x is 5 on one path into Lx and 7 on the other; neither may be chosen."""
    original = parse_tac(FORWARD_JOIN)
    optimized = constant_propagation(original)

    check = check_semantic_equivalence(original, optimized)
    assert check["status"] == "equivalent", check["detail"]
    lines = __import__("backend.api.serialize", fromlist=["tac_to_lines"]).tac_to_lines(optimized)
    assert "y = x + 1" in lines


def test_passes_still_optimize_straight_line_code():
    """The soundness fix must not blind the passes to ordinary opportunities."""
    straight = parse_tac(STRAIGHT_LINE)
    assert "p0 = 15" in tac_to_source(constant_propagation(straight))
    assert "r0b = r0a" in tac_to_source(common_subexpression_elimination(straight))


def test_engine_run_on_the_counted_loop_stays_semantic():
    """Engine-level regression: the full fixpoint loop, both selectors'
    shared machinery, must end in a program that still behaves like the
    original."""
    original = parse_tac(COUNTED_LOOP)
    run = run_iterative_optimization(original)
    check = check_semantic_equivalence(original, run["final_tac"])
    assert check["status"] == "equivalent", check["detail"]
    final = execute_tac(run["final_tac"])
    assert final.terminated is True


# ---------------------------------------------------------------------------
# Every pass preserves observable behavior on every battery program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pass_name", [name for name, _ in PASSES])
@pytest.mark.parametrize("program_name", sorted(BATTERY))
def test_every_pass_preserves_observable_behavior(pass_name, program_name):
    fn = dict(PASSES)[pass_name]
    original = parse_tac(BATTERY[program_name])
    optimized = fn(original)
    check = check_semantic_equivalence(original, optimized)
    assert check["status"] == "equivalent", (
        f"{pass_name} on {program_name}: {check['detail']}"
    )


def test_removing_an_all_dead_program_is_equivalent_not_mysterious():
    """The empty result for the all-dead demo is legitimate and the checker
    says so precisely: nothing the optimized program does is observable.

    One DCE wave removes only the statements whose results nothing reads
    (t4, t5, t8, unused); the surviving t1/t2/t3/t6/t7 chain then becomes
    newly dead, so removal is a fixpoint - the same cascade the engine runs."""
    original = parse_tac(ALL_DEAD)
    optimized = original
    for _ in range(10):
        next_tac = dead_code_elimination(optimized)
        if tac_to_source(next_tac) == tac_to_source(optimized):
            break
        optimized = next_tac
    assert len(optimized.instructions) == 0

    check = check_semantic_equivalence(original, optimized)
    assert check["status"] == "equivalent"
    assert check["observed_vars"] == []
    assert check["outputs_compared"] is False
    assert "neither program prints anything" in check["detail"]


def test_checker_catches_value_divergence():
    original = parse_tac("acc = a + b\nacc = acc + zt")
    sabotaged = parse_tac("acc = a - b\nacc = acc + zt")
    check = check_semantic_equivalence(original, sabotaged)
    assert check["status"] == "not_equivalent"


def test_checker_catches_termination_divergence():
    """The exact shape of the discovered bug: a terminating original against
    an optimized program that loops forever."""
    original = parse_tac(COUNTED_LOOP)
    frozen = parse_tac("""n = 10
i = 0
acc = 0
loop_head:
if i >= n goto loop_end
acc = 0
i = 1
goto loop_head
loop_end:
result = 0""")
    check = check_semantic_equivalence(original, frozen)
    assert check["status"] == "not_equivalent"
    assert "termination" in check["detail"]


def test_checker_reports_faults_as_inconclusive():
    original = parse_tac("acc = a + 1\nacc = acc + zt")
    faulting = parse_tac("acc = a / 0\nacc = acc + zt")
    check = check_semantic_equivalence(original, faulting)
    assert check["status"] == "inconclusive"


def test_read_variables_ignore_labels_and_literals():
    reads = read_variables(parse_tac(COUNTED_LOOP))
    # `result = acc` reads acc; labels and the literal 10 are not variables.
    assert reads == {"acc", "i", "n"}
