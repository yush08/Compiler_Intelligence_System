"""
Algebraic Simplification optimization.

Implements safe algebraic identities to simplify expressions.

Supported rules:
    x + 0 → x
    0 + x → x
    x - 0 → x
    x * 1 → x
    1 * x → x
    x / 1 → x
    x * 0 → 0
    0 * x → 0
"""

from ..tac.instruction import Instruction, TAC


def algebraic_simplification(tac: TAC) -> TAC:
    """
    Apply algebraic simplification to the TAC.

    Replaces expressions with their simplified form using algebraic identities.
    """
    if not tac.instructions:
        return tac

    result_tac = TAC()
    for instr in tac.instructions:
        new_instr = _simplify_instruction(instr)
        result_tac.add(new_instr if new_instr else instr)

    return result_tac


def _simplify_instruction(instr: Instruction) -> Instruction | None:
    """Simplify a single instruction if possible."""
    # Only simplify assignments
    if not instr.is_assignment() or instr.result is None:
        return None

    # Check for x + 0 → x, 0 + x → x
    if instr.opcode == Instruction.ADD and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2):
            if float(instr.arg2) == 0:
                # x + 0 → x, just keep the result var
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg1)
        if Instruction._is_constant(instr.arg1):
            if float(instr.arg1) == 0:
                # 0 + x → x
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg2)

    # Check for x - 0 → x
    if instr.opcode == Instruction.SUB and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2):
            if float(instr.arg2) == 0:
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg1)

    # Check for x * 1 → x, 1 * x → x
    if instr.opcode == Instruction.MUL and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2):
            if float(instr.arg2) == 1:
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg1)
        if Instruction._is_constant(instr.arg1):
            if float(instr.arg1) == 1:
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg2)

    # Check for x / 1 → x
    if instr.opcode == Instruction.DIV and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2):
            if float(instr.arg2) == 1:
                return Instruction(Instruction.ASSIGN, result=instr.result, arg1=instr.arg1)

    # Check for x * 0 → 0, 0 * x → 0
    if instr.opcode == Instruction.MUL and instr.arg2 is not None:
        if Instruction._is_constant(instr.arg2):
            if float(instr.arg2) == 0:
                # Result is 0 - return a constant assignment
                return Instruction(Instruction.CONST, result=instr.result, arg1="0")

    return None


def algebraic_simplification_score(tac: TAC) -> float:
    """
    Calculate a score for how much algebraic simplification would improve this TAC.

    Score is based on the number of algebraic opportunities (x+0, x*1, etc.).
    """
    count = 0
    for instr in tac.instructions:
        # Check x + 0 or 0 + x
        if instr.opcode == Instruction.ADD and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
                count += 1
            elif Instruction._is_constant(instr.arg1) and float(instr.arg1) == 0:
                count += 1
        # Check x - 0
        if instr.opcode == Instruction.SUB and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
                count += 1
        # Check x * 1 or 1 * x
        if instr.opcode == Instruction.MUL and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 1:
                count += 1
            elif Instruction._is_constant(instr.arg1) and float(instr.arg1) == 1:
                count += 1
        # Check x / 1
        if instr.opcode == Instruction.DIV and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 1:
                count += 1
        # Check x * 0 or 0 * x
        if instr.opcode == Instruction.MUL and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg2) and float(instr.arg2) == 0:
                count += 1
            elif Instruction._is_constant(instr.arg1) and float(instr.arg1) == 0:
                count += 1

    if count == 0:
        return 0.1

    # Normalize
    return min(1.0, 0.1 + 0.1 * count)