"""
Feature extraction from TAC.

extract_features(tac) returns a dictionary of measurable features
calculated from the actual TAC instructions.
"""

from ..tac.instruction import Instruction, TAC


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

    # num_dead_stmts: count of assignments whose results are never used
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
                         if instr.is_assignment() and instr.result is not None
                         and result_usage.get(instr.result, 0) == 0)

    # num_algebraic_ops: count of algebraic simplification opportunities
    num_algebraic_ops = 0
    for instr in instructions:
        if instr.is_assignment():
            # Check x+0, x*1, x-0, x/1, x*0
            if _has_algebraic_opportunity(instr):
                num_algebraic_ops += 1

    # num_loop_ops: count of loop-related operations (multiplication by counter, etc.)
    num_loop_ops = 0
    for instr in instructions:
        if instr.opcode == Instruction.MUL:
            if instr.arg2 is not None and Instruction._is_constant(instr.arg2):
                try:
                    val = float(instr.arg2)
                    if val == int(val) and int(val) >= 2:
                        num_loop_ops += 1
                except ValueError:
                    pass

    # num_tmp_vars: count of unique temporary/variable names
    tmp_vars = set()
    for instr in instructions:
        if instr.result is not None:
            tmp_vars.add(instr.result)
        if instr.arg1 is not None:
            tmp_vars.add(instr.arg1)
        if instr.arg2 is not None:
            tmp_vars.add(instr.arg2)
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
    if not instr.is_assignment() or instr.result is None:
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