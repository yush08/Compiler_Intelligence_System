"""
Common Subexpression Elimination (CSE).

Detects repeated expressions and reuses existing results.

Example:
    t1 = a + b
    t2 = a + b        →  t2 = t1

For commutative operators (+, *), treats a+b and b+a as equivalent.
"""

from ..tac.instruction import Instruction, TAC


def common_subexpression_elimination(tac: TAC) -> TAC:
    """
    Apply CSE to the TAC.

    Detects identical (or commutatively equivalent) binary expressions
    and replaces duplicates with references to the original result.

    Example:
        t1 = a + b
        t2 = a + b          →  t2 = t1
        t3 = b + a          →  t3 = t1  (commutative)
    """
    if not tac.instructions:
        return tac

    result_tac = TAC()
    # Map from expression signature -> first temporary result
    # For commutative ops, we normalize the operands
    expr_map = {}  # signature -> result temp

    for instr in tac.instructions:
        new_instr = _cse_instruction(instr, expr_map)
        result_tac.add(new_instr if new_instr else instr)

    return result_tac


def _cse_instruction(instr: Instruction, expr_map: dict) -> Instruction | None:
    """Apply CSE to a single instruction."""
    if not instr.is_binary_op():
        # Non-binary instructions can't be CSE candidates
        return instr

    # Create a signature for the expression
    sig = _expression_signature(instr)

    if sig is None:
        # Can't create a signature, keep as is
        return instr

    if sig in expr_map:
        # We've seen this expression before - replace with existing result
        original_result = expr_map[sig]
        # Return assignment referencing the original
        return Instruction(Instruction.ASSIGN, result=instr.result, arg1=original_result)
    else:
        # First time seeing this expression
        expr_map[sig] = instr.result
        return instr


def _expression_signature(instr: Instruction) -> str | None:
    """
    Create a signature for a binary expression.

    For commutative operators (+, *), normalize operand order
    so that a+b and b+a have the same signature.
    """
    op = instr.opcode
    arg1 = instr.arg1
    arg2 = instr.arg2

    if op == Instruction.ADD or op == Instruction.MUL:
        # Commutative: normalize order
        # Use sorted order for consistency
        if arg1 > arg2:
            arg1, arg2 = arg2, arg1
        return f"{op}:{arg1}:{arg2}"
    else:
        # Non-commutative: preserve order
        return f"{op}:{arg1}:{arg2}"


def common_subexpression_score(tac: TAC) -> float:
    """
    Calculate a score for how much CSE would improve this TAC.

    Score is based on the number of repeated expressions.
    """
    # Count expressions and find repetitions
    expr_counts = {}
    for instr in tac.instructions:
        if instr.is_binary_op():
            sig = _expression_signature(instr)
            if sig is not None:
                expr_counts[sig] = expr_counts.get(sig, 0) + 1

    # Count how many expressions appear more than once
    repeated = sum(1 for count in expr_counts.values() if count > 1)

    if repeated == 0:
        return 0.1  # Very low but non-zero

    # Score based on number of repeated expressions
    # More repetitions → higher score
    return min(1.0, 0.2 + 0.1 * repeated)