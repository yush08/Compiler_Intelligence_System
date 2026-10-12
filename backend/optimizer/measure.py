"""
Measurement helpers for evaluating a single optimization pass.

Used by the dataset generator (experiments/generate_dataset.py) and available
to any future ML evaluation code, so that every consumer measures a pass the
same way.

Why two metrics
---------------
`instruction_reduction` is the engine's own metric and stays the primary,
literal number the API reports.  But five of the six passes are strictly
one-instruction-in / one-instruction-out rewrites: constant folding turns
`t1 = 5 + 3` into `t1 = 8`, CSE turns `t2 = a + b` into `t2 = t1`, and
algebraic simplification turns `t1 = a + 0` into `t1 = a` - each one leaves the
statement count untouched.  Only dead code elimination can drop a statement, so
ranking passes by statement count alone would label almost every program
`dead_code_elimination` and make the dataset unusable for classification.

`useful_improvement` therefore also counts work that disappeared from the
program: an *operation instruction* (ADD/SUB/MUL/DIV/MOD/UMINUS) that became a
plain copy or a literal no longer has to be evaluated at run time.  The measure
is computed purely from what the real pass produced - no pass is credited with
an improvement it did not make.
"""

from ..tac.instruction import Instruction, TAC

# Instructions that still perform a computation at run time. Copies, literal
# definitions, labels and jumps are "already resolved".
OPERATION_OPCODES = frozenset(
    {
        Instruction.ADD,
        Instruction.SUB,
        Instruction.MUL,
        Instruction.DIV,
        Instruction.MOD,
        Instruction.UMINUS,
    }
)


def operation_count(tac: TAC) -> int:
    """Number of instructions in `tac` that still perform a computation."""
    if tac is None or not getattr(tac, "instructions", None):
        return 0
    return sum(1 for instr in tac.instructions if instr.opcode in OPERATION_OPCODES)


def work_remaining(tac: TAC) -> int:
    """Statements plus remaining operations: the work still needed to run `tac`."""
    if tac is None or not getattr(tac, "instructions", None):
        return 0
    return len(tac) + operation_count(tac)


def measure_pass(original: TAC, optimized: TAC) -> dict:
    """Measure what one pass actually did to a TAC.

    Returns:
        dict with:
            original_instruction_count: len(original)
            optimized_instruction_count: len(optimized)
            instruction_reduction:       statements removed (>= 0)
            reduction_percentage:        instruction_reduction / original, 0-100
            operation_reduction:         operations removed (may be negative
                                         if a pass swapped one operation for
                                         another of the same kind)
            useful_improvement:          work_remaining(original)
                                         - work_remaining(optimized)
    """
    original_count = len(original)
    optimized_count = len(optimized)

    instruction_reduction = original_count - optimized_count
    if original_count > 0:
        reduction_percentage = round(instruction_reduction / original_count * 100, 4)
    else:
        # An empty TAC has nothing to reduce; never divide by zero.
        reduction_percentage = 0.0

    useful_improvement = work_remaining(original) - work_remaining(optimized)

    return {
        "original_instruction_count": original_count,
        "optimized_instruction_count": optimized_count,
        "instruction_reduction": instruction_reduction,
        "reduction_percentage": reduction_percentage,
        "operation_reduction": operation_count(original) - operation_count(optimized),
        "useful_improvement": useful_improvement,
    }


def best_strategy_from(measurements: dict, tie_order: list[str]) -> str:
    """Pick the strategy with the largest `useful_improvement`.

    Ties fall back to `tie_order`, which is backend.scoring.rule_based's
    STRATEGY_TIE_ORDER - the same deterministic rule select_strategy() uses, so
    labelling is reproducible run to run and never random.

    When every pass scores 0 (a program with nothing to optimise) the label is
    simply the first strategy in that order.
    """
    if not measurements:
        raise ValueError("no strategy measurements to choose from")

    best_value = max(m["useful_improvement"] for m in measurements.values())

    for name in tie_order:
        if name in measurements and measurements[name]["useful_improvement"] == best_value:
            return name

    # `tie_order` is expected to cover every strategy; fall back to whichever
    # strategy actually reached the maximum rather than dropping the label.
    for name in measurements:
        if measurements[name]["useful_improvement"] == best_value:
            return name

    raise ValueError("no strategy reached the maximum useful_improvement")
