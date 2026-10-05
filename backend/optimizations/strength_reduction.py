"""
Strength Reduction optimization.

Implements conservative strength reduction transformations.

At this stage, strength reduction focuses on detecting patterns where
expensive operations (multiplication) can be replaced with cheaper ones
(addition), particularly in loop-related contexts.

If the TAC does not contain enough loop information, the optimization
documents the limitation rather than creating unsafe transformations.
"""

from ..tac.instruction import Instruction, TAC


def strength_reduction(tac: TAC) -> TAC:
    """
    Apply strength reduction to the TAC.

    Detects opportunities to replace stronger operations (e.g., multiplication)
    with weaker ones (e.g., addition) where safe.

    Note: This is a conservative implementation. Without loop-carrying
    dependency information in the TAC, we can only document limitations
    rather than apply unsafe transformations.
    """
    if not tac.instructions:
        return tac

    # Analyze the TAC for multiplication patterns that could potentially
    # be strengthened-reduced. We look for patterns like:
    #   t = i * 2  where i is a loop variable or counter
    #   t = i * c  where c is a small constant

    result_tac = TAC()
    for instr in tac.instructions:
        new_instr = _strength_reduce_instruction(instr, result_tac)
        result_tac.add(new_instr if new_instr else instr)

    return result_tac


def _strength_reduce_instruction(instr: Instruction, built_tac: TAC) -> Instruction | None:
    """Try to strength-reduce a single multiplication by a constant."""
    if not instr.is_assignment() or instr.result is None:
        return None

    # Only handle multiplication instructions
    if instr.opcode != Instruction.MUL:
        return None

    # Check if arg2 is a constant
    if not Instruction._is_constant(instr.arg2):
        return None

    constant_val = float(instr.arg2)

    # Only reduce certain constants (small integers)
    # Strength reduction is most effective for powers of 2
    if not constant_val.is_integer() or int(constant_val) < 2:
        return None

    const_int = int(constant_val)

    # Check if arg1 is a simple variable/temp (not a complex expression)
    # This is a simplification - in a full implementation we'd check
    # if arg1 is a loop-carrying variable
    if instr.arg1 is None:
        return None

    # For now, document that we can't safely transform without
    # loop-carrying information
    # 
    # Potential strength reduction patterns:
    #   t = i * 2  →  t = i + i   (if i is loop variable)
    #   t = i * 4  →  t = (i + i) + (i + i)
    #   t = i * 8  →  t = (i + i) + (i + i) + (i + i) + (i + i)
    #
    # But without knowing if arg1 is a loop variable, we cannot safely
    # make this transformation.

    # Return None to indicate no transformation applied
    # This is the conservative approach
    return None


def strength_reduction_score(tac: TAC) -> float:
    """
    Calculate a score for how much strength reduction would improve this TAC.

    Score is based on the presence of multiplication by constants
    that could potentially be strength-reduced.

    Note: Without loop information, this score is conservative.
    """
    count = 0
    for instr in tac.instructions:
        if instr.is_binary_op() and instr.opcode == Instruction.MUL:
            if instr.arg2 is not None and Instruction._is_constant(instr.arg2):
                try:
                    val = float(instr.arg2)
                    if val == int(val) and 2 <= int(val) <= 16:
                        # Power of 2 between 2 and 16
                        # or other small constants
                        count += 1
                except ValueError:
                    pass

    if count == 0:
        return 0.05  # Very low score when no opportunities

    # Normalize - small increase for each opportunity
    return min(0.2, 0.05 + 0.03 * count)