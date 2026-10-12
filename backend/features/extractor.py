"""
Feature extraction from TAC.

extract_features(tac) returns a dictionary of measurable features
calculated from the actual TAC instructions.
"""

import re

from ..tac.instruction import Instruction, TAC


# Opcodes that describe control flow rather than a value computation.
# Matches the set dead_code_elimination never treats as removable.
CONTROL_FLOW_OPCODES = frozenset({Instruction.LABEL, Instruction.GOTO, Instruction.IF})

# The parser's temporaries follow the t<digits> convention (t1, t2, ...).
# Named variables such as `unused`, `a`, `b` are ordinary variables, and
# numeric literals are not variables at all.
_TEMPORARY_NAME = re.compile(r"^t\d+$")

# Canonical name AND ORDER of the eight model inputs. This is the single source
# of truth: the dataset generator, the ML training pipeline, the API and the
# frontend all derive their feature list from here, so a model can never be
# trained on one ordering and scored on another.
FEATURE_KEYS = [
    "num_statements",
    "num_constants",
    "num_repeated_expr",
    "num_dead_stmts",
    "num_algebraic_ops",
    "num_loop_ops",
    "num_tmp_vars",
    "const_ratio",
]

# Human-readable definitions, kept next to the code that computes them so the
# Dataset Explorer can never describe a feature differently from how it is
# actually measured.  Note that num_loop_ops counts control-flow instructions
# (labels/gotos/conditionals), not multiplications.
FEATURE_DESCRIPTIONS = {
    "num_statements": "Total number of TAC instructions.",
    "num_constants": (
        "Statements with a constant operand: CONST assignments plus binary "
        "operations that reference a numeric literal."
    ),
    "num_repeated_expr": (
        "Distinct binary expressions that appear more than once - the "
        "common-subexpression opportunity count."
    ),
    "num_dead_stmts": (
        "Non-control-flow assignments whose result is never read afterwards - "
        "the dead-code opportunity count. Computed by a static liveness scan, "
        "not by running the dead-code pass."
    ),
    "num_algebraic_ops": (
        "Statements matching an algebraic identity such as x+0, x*1, x-0, "
        "x/1 or x*0."
    ),
    "num_loop_ops": (
        "Control-flow instructions (labels, gotos, conditionals) that form "
        "loops and branches."
    ),
    "num_tmp_vars": "Distinct temporary variables named t<number>.",
    "const_ratio": "num_constants divided by num_statements, in the range [0, 1].",
}


def extract_features(tac: TAC) -> dict:
    """
    Extract measurable features from a TAC object.

    Returns:
        dict with keys:
            num_statements: total number of instructions
            num_constants: number of constant assignments
            num_repeated_expr: number of repeated subexpressions
            num_dead_stmts: number of dead assignments
            num_algebraic_ops: number of algebraic simplification opportunities
            num_loop_ops: number of loop-related operations
            num_tmp_vars: number of temporary variables
            const_ratio: ratio of constant assignments (0.0 - 1.0)
    """
    if not tac or not tac.instructions:
        return {
            "num_statements": 0,
            "num_constants": 0,
            "num_repeated_expr": 0,
            "num_dead_stmts": 0,
            "num_algebraic_ops": 0,
            "num_loop_ops": 0,
            "num_tmp_vars": 0,
            "const_ratio": 0.0,
        }

    instructions = tac.instructions
    num_statements = len(instructions)

    # num_constants: count of constant assignments or operations with constants
    # Count instructions where a constant is involved
    num_constants = 0
    for instr in instructions:
        # Count CONST opcode instructions
        if instr.opcode == Instruction.CONST:
            num_constants += 1
        # Count binary operations with constant args
        elif instr.is_binary_op():
            arg1_is_const = instr.arg1 is not None and Instruction._is_constant(instr.arg1)
            arg2_is_const = instr.arg2 is not None and Instruction._is_constant(instr.arg2)
            if arg1_is_const or arg2_is_const:
                num_constants += 1

    # num_repeated_expr: count of expressions that appear more than once
    expr_counts = {}
    for instr in instructions:
        if instr.is_binary_op():
            sig = _expression_signature_for_features(instr)
            if sig is not None:
                expr_counts[sig] = expr_counts.get(sig, 0) + 1
    num_repeated_expr = sum(1 for count in expr_counts.values() if count > 1)

    # num_dead_stmts: count of statements whose results are never used later.
    # This mirrors dead_code_elimination exactly: usage is counted for every
    # defined result, and control flow (labels, gotos, conditionals) is never
    # treated as dead. The old gate used is_assignment(), which only matches
    # the copy opcode, so straight-line statements were never counted.
    result_usage = {}
    for instr in instructions:
        if instr.result is not None:
            result_usage[instr.result] = 0
    for instr in instructions:
        if instr.arg1 is not None and instr.arg1 in result_usage:
            result_usage[instr.arg1] += 1
        if instr.arg2 is not None and instr.arg2 in result_usage:
            result_usage[instr.arg2] += 1
    num_dead_stmts = sum(1 for instr in instructions
                         if instr.result is not None
                         and instr.opcode not in CONTROL_FLOW_OPCODES
                         and result_usage.get(instr.result, 0) == 0)

    # num_algebraic_ops: count of algebraic simplification opportunities
    # (x+0, x*1, x-0, x/1, x*0). The instruction only needs a result var; the
    # old is_assignment() gate required opcode ASSIGN while every check below
    # requires ADD/SUB/MUL/DIV, which made this feature always 0.
    num_algebraic_ops = sum(1 for instr in instructions
                            if _has_algebraic_opportunity(instr))

    # num_loop_ops: control-flow instructions (labels, gotos, conditionals)
    # that actually form loops and branches. Multiplication by a constant is a
    # strength-reduction opportunity, not evidence of a loop, so it is not
    # counted here.
    num_loop_ops = sum(1 for instr in instructions
                       if instr.opcode in CONTROL_FLOW_OPCODES)

    # num_tmp_vars: count of distinct temporary variables (t1, t2, ...).
    # Numeric literals and named variables are excluded; occurrences are not
    # double counted because the names are collected into a set.
    tmp_vars = set()
    for instr in instructions:
        for name in (instr.result, instr.arg1, instr.arg2):
            if name is not None and _TEMPORARY_NAME.match(name):
                tmp_vars.add(name)
    num_tmp_vars = len(tmp_vars)

    # const_ratio: ratio of constant assignments
    const_ratio = num_constants / num_statements if num_statements > 0 else 0.0

    return {
        "num_statements": num_statements,
        "num_constants": num_constants,
        "num_repeated_expr": num_repeated_expr,
        "num_dead_stmts": num_dead_stmts,
        "num_algebraic_ops": num_algebraic_ops,
        "num_loop_ops": num_loop_ops,
        "num_tmp_vars": num_tmp_vars,
        "const_ratio": const_ratio,
    }


def _expression_signature_for_features(instr: Instruction) -> str | None:
    """Create a signature for feature extraction (similar to CSE)."""
    op = instr.opcode
    arg1 = instr.arg1
    arg2 = instr.arg2

    if op in {Instruction.ADD, Instruction.MUL}:
        # Commutative: normalize order
        if arg1 and arg2:
            a1, a2 = sorted([arg1, arg2])
            return f"{op}:{a1}:{a2}"
        return None
    else:
        if arg1 and arg2:
            return f"{op}:{arg1}:{arg2}"
        return None


def _has_algebraic_opportunity(instr: Instruction) -> bool:
    """Check if an instruction has an algebraic simplification opportunity."""
    # Any value-producing statement can carry an algebraic identity. Requiring
    # opcode ASSIGN here was contradictory: every rule below inspects
    # ADD/SUB/MUL/DIV, so the function could never return True.
    if instr.result is None or instr.opcode in CONTROL_FLOW_OPCODES:
        return False

    # Check x + 0 or 0 + x
    if instr.opcode == Instruction.ADD and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
            return True
        if Instruction._is_constant(instr.arg1) and float(instr.arg1) == 0:
            return True

    # Check x - 0
    if instr.opcode == Instruction.SUB and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
            return True

    # Check x * 1 or 1 * x
    if instr.opcode == Instruction.MUL and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 1:
            return True
        if Instruction._is_constant(instr.arg1) and float(instr.arg1) == 1:
            return True

    # Check x / 1
    if instr.opcode == Instruction.DIV and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 1:
            return True

    # Check x * 0 or 0 * x
    if instr.opcode == Instruction.MUL and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
            return True
        if Instruction._is_constant(instr.arg1) and float(instr.arg1) == 0:
            return True

    return False