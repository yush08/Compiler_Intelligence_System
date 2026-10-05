"""
Dead Code Elimination optimization.

Identifies assignments whose results are never subsequently used.

Does NOT remove:
- Labels
- Jumps (goto)
- Conditional branches (if ... goto)
- Operations with side effects
"""

from ..tac.instruction import Instruction, TAC


def dead_code_elimination(tac: TAC) -> TAC:
    """
    Apply dead code elimination to the TAC.

    Marks assignments whose result is never used later,
    then removes them.

    Control-flow statements (labels, goto, if ... goto) are never removed.
    """
    if not tac.instructions:
        return tac

    # Step 1: Count how many times each result variable is used as an operand
    result_usage_count = {}
    for instr in tac.instructions:
        if instr.result is not None:
            result_usage_count[instr.result] = 0

    for instr in tac.instructions:
        if instr.arg1 is not None and instr.arg1 in result_usage_count:
            result_usage_count[instr.arg1] += 1
        if instr.arg2 is not None and instr.arg2 in result_usage_count:
            result_usage_count[instr.arg2] += 1

    # Step 2: Build result TAC, keeping only live assignments
    result_tac = TAC()
    for instr in tac.instructions:
        # Never remove labels, goto, or if ... goto
        if instr.opcode == Instruction.LABEL:
            result_tac.add(instr)
            continue
        if instr.opcode == Instruction.GOTO:
            result_tac.add(instr)
            continue
        if instr.opcode == Instruction.IF:
            result_tac.add(instr)
            continue

        # For assignments: keep only if result is used later
        if instr.is_assignment() and instr.result is not None:
            # Result is used if its usage count > 0
            if result_usage_count.get(instr.result, 0) > 0:
                result_tac.add(instr)
            # else: dead code, skip it
        else:
            # Other instructions - keep
            result_tac.add(instr)

    return result_tac


def dead_code_elimination_score(tac: TAC) -> float:
    """
    Calculate a score for how much dead code elimination would improve this TAC.

    Score is based on the number of assignments whose results are never used.
    """
    if not tac.instructions:
        return 0.0

    # Count result usages
    result_usage_count = {}
    for instr in tac.instructions:
        if instr.result is not None:
            result_usage_count[instr.result] = 0

    for instr in tac.instructions:
        if instr.arg1 is not None and instr.arg1 in result_usage_count:
            result_usage_count[instr.arg1] += 1
        if instr.arg2 is not None and instr.arg2 in result_usage_count:
            result_usage_count[instr.arg2] += 1

    # Count dead assignments (result usage count is 0)
    dead_assignments = 0
    for instr in tac.instructions:
        if instr.is_assignment() and instr.result is not None:
            if result_usage_count.get(instr.result, 0) == 0:
                dead_assignments += 1

    if dead_assignments == 0:
        return 0.1  # Very low but non-zero

    # Normalize: more dead assignments -> higher score
    return min(1.0, 0.1 + 0.15 * dead_assignments)