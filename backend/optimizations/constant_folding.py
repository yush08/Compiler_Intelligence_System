"""
Constant Folding optimization.

Folds constant expressions at compile time.
Example: t1 = 5 + 3  →  t1 = 8

Support: +, -, *, /, %
"""

from ..tac.instruction import Instruction, TAC


def constant_folding(tac: TAC) -> TAC:
    """
    Apply constant folding to the TAC.

    Replace binary operations where both operands are constants
    with a single constant assignment.

    Example:
        t1 = 5 + 3          →  t1 = 8
        t2 = 10 * 2         →  t2 = 20
    """
    if not tac.instructions:
        return tac

    result_tac = TAC()
    for instr in tac.instructions:
        new_instr = _fold_instruction(instr)
        result_tac.add(new_instr if new_instr else instr)

    return result_tac


def _fold_instruction(instr: Instruction) -> Instruction | None:
    """Fold a single instruction if it's a constant expression."""

    # Case 1: Binary operation instruction (ADD, SUB, MUL, DIV, MOD)
    # e.g., t1 = 5 + 3  →  t1 = 8
    if instr.opcode in {Instruction.ADD, Instruction.SUB, Instruction.MUL, Instruction.DIV, Instruction.MOD}:
        # Check if both args are constants
        if instr.arg1 is not None and instr.arg2 is not None:
            if Instruction._is_constant(instr.arg1) and Instruction._is_constant(instr.arg2):
                # Both args are constants - compute the result
                try:
                    val1 = float(instr.arg1)
                    val2 = float(instr.arg2)

                    if instr.opcode == Instruction.ADD:
                        result_val = val1 + val2
                    elif instr.opcode == Instruction.SUB:
                        result_val = val1 - val2
                    elif instr.opcode == Instruction.MUL:
                        result_val = val1 * val2
                    elif instr.opcode == Instruction.DIV:
                        if val2 == 0:
                            return instr  # Don't divide by zero
                        result_val = val1 / val2
                    elif instr.opcode == Instruction.MOD:
                        if val2 == 0:
                            return instr  # Don't mod by zero
                        result_val = val1 % val2

                    # Format the value - remove .0 if it's an integer
                    if result_val == int(result_val):
                        result_str = str(int(result_val))
                    else:
                        result_str = str(result_val)

                    # Return as constant assignment
                    return Instruction(Instruction.CONST, result=instr.result, arg1=result_str)

                except (ValueError, ZeroDivisionError):
                    return instr

    # Case 2: a copy (`t = x`) never has two constant operands, and the parser
    # does not emit ASSIGN with arg2 at all, so this branch was unreachable.
    # It was also wrong: it computed `val1 + val2` regardless of the real
    # operator, which would have silently turned `2 * 3` into `5` had anything
    # ever constructed such an instruction.
    return None