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

    A cached expression is dropped as soon as one of its operands (or the
    temporary holding its result) is redefined. Without that invalidation the
    rewrite was unsound: `t1 = a + b / a = 5 / t2 = a + b` became
    `t2 = t1`, which computes with the old `a`.

    The cache is also cleared at every label, for the same must-property
    reason constant propagation clears its environment: inside a label-free
    stretch the only way to re-execute an earlier instruction is to re-enter
    the stretch, and re-entry can only happen through a label.  Across a
    label, a back edge can bring different operand values to a later
    occurrence than the ones the cached temporary holds:
    `t = a + b / loop: u = a + b / a = a + 1 / if u < 100 goto loop` became
    `u = t` - a stale value every iteration, freezing `u` and turning a
    terminating loop into an infinite one.  Checked by
    tests/test_semantic_preservation.py.
    """
    if not tac.instructions:
        return tac

    result_tac = TAC()
    # Map from expression signature -> first temporary result
    expr_map = {}  # signature -> result temp

    for instr in tac.instructions:
        if instr.opcode == Instruction.LABEL:
            # Join point: cached expressions are only valid within this
            # label-free stretch.  Keep the label verbatim.
            expr_map.clear()
            result_tac.add(instr)
            continue
        new_instr, created_sig = _cse_instruction(instr, expr_map)
        result_tac.add(new_instr if new_instr else instr)
        _invalidate(expr_map, instr.result, created_sig)

    return result_tac


def _invalidate(expr_map: dict, defined: str | None, created_sig: str | None) -> None:
    """Drop cached expressions that this instruction just made stale.

    Two distinct things can go wrong when a variable is (re)defined:

    * its OLD value is part of a cached signature, so the cached expression no
      longer denotes what future instructions will compute (`t1 = t1 + 1`
      leaves behind an entry meaning "t1 + 1 == t1" for the *old* t1);
    * the variable itself was the cached alias for some expression, so
      `expr_map[sig] -> t1` points at a value that no longer holds it
      (`t1 = a + b` followed by `t1 = 7`).

    The entry this very instruction just recorded is exempt from the second
    rule only: it is the alias that is still true on the way out.
    """
    if defined is None:
        return

    stale = []
    for sig, result_temp in expr_map.items():
        if defined in sig.split(":"):
            stale.append(sig)  # an operand of the cached expression changed
        elif result_temp == defined and sig != created_sig:
            stale.append(sig)  # the alias itself was overwritten

    for sig in stale:
        del expr_map[sig]


def _cse_instruction(instr: Instruction, expr_map: dict) -> tuple[Instruction | None, str | None]:
    """Apply CSE to a single instruction.

    Returns (replacement or None, the signature this instruction recorded).
    """
    if not instr.is_binary_op():
        # Non-binary instructions can't be CSE candidates
        return instr, None

    # Create a signature for the expression
    sig = _expression_signature(instr)

    if sig is None:
        # Can't create a signature, keep as is
        return instr, None

    if sig in expr_map:
        # We've seen this expression before - replace with existing result
        original_result = expr_map[sig]
        # Return assignment referencing the original
        return (
            Instruction(Instruction.ASSIGN, result=instr.result, arg1=original_result),
            None,
        )
    else:
        # First time seeing this expression
        expr_map[sig] = instr.result
        return instr, sig


def _expression_signature(instr: Instruction) -> str | None:
    """
    Create a signature for a binary expression.

    For commutative operators (+, *), normalize operand order
    so that a+b and b+a have the same signature.
    """
    op = instr.opcode
    arg1 = instr.arg1
    arg2 = instr.arg2

    # Both operands are required: `arg1 > arg2` raised TypeError when either
    # was None, and an expression without both operands cannot be reused anyway.
    if arg1 is None or arg2 is None:
        return None

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