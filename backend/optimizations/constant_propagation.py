"""
Constant Propagation optimization.

Replaces variables that are known to be constants with their constant values.
Example:
    a = 10
    t1 = a + 5          →  t1 = 15
"""

from ..tac.instruction import Instruction, TAC


def constant_propagation(tac: TAC) -> TAC:
    """
    Apply constant propagation to the TAC.

    Uses a simple constant environment to track known constants
    and replaces variable references with their values.
    """
    if not tac.instructions:
        return tac

    # Build a constant environment by scanning instructions
    const_env = {}  # variable -> constant value
    result_tac = TAC()

    # First pass: build the constant environment
    for instr in tac.instructions:
        _update_const_env(instr, const_env)

    # Second pass: propagate constants
    for instr in tac.instructions:
        new_instr = _propagate_instruction(instr, const_env)
        result_tac.add(new_instr if new_instr else instr)

    return result_tac


def _update_const_env(instr: Instruction, const_env: dict):
    """Update the constant environment based on an instruction."""
    if instr.opcode == Instruction.CONST:
        # t = 5  →  const_env[t] = 5
        var = instr.result
        val = instr.arg1
        try:
            const_env[var] = float(val)
        except ValueError:
            pass

    elif instr.is_assignment() and instr.arg2 is None:
        # t = x  where x is already known as constant
        if instr.arg1 in const_env:
            var = instr.result
            const_env[var] = const_env[instr.arg1]

    elif instr.is_binary_op() and instr.has_constant_result():
        # t = 5 + 3  →  we'll handle this in folding, but also track
        # the result as a constant if both args are already known
        # Actually, this case is better handled by constant folding first
        pass


def _propagate_instruction(instr: Instruction, const_env: dict) -> Instruction | None:
    """Propagate constants in a single instruction."""
    if instr.opcode == Instruction.CONST:
        # Already a constant, keep as is
        return instr

    if instr.is_assignment() and instr.arg2 is None:
        # t = x  →  if x is constant, replace with value
        arg1 = instr.arg1
        if arg1 in const_env:
            # Replace with the constant value
            val = const_env[arg1]
            if val == int(val):
                val_str = str(int(val))
            else:
                val_str = str(val)
            return Instruction(Instruction.CONST, result=instr.result, arg1=val_str)
        # Not a constant, keep as is
        return None

    if instr.is_binary_op():
        # t = a + b  →  propagate constants in args
        new_arg1 = _propagate_arg(instr.arg1, const_env)
        new_arg2 = _propagate_arg(instr.arg2, const_env) if instr.arg2 else None

        # Check if both args are now constants
        arg1_is_const = new_arg1 is not None and new_arg1 != instr.arg1
        arg2_is_const = new_arg2 is not None and new_arg2 != instr.arg2

        if arg1_is_const and arg2_is_const:
            # Both args are constants - constant folding would handle this
            # But we can still try to simplify
            try:
                v1 = float(new_arg1)
                v2 = float(new_arg2)
                if instr.opcode == Instruction.ADD:
                    result_val = v1 + v2
                elif instr.opcode == Instruction.SUB:
                    result_val = v1 - v2
                elif instr.opcode == Instruction.MUL:
                    result_val = v1 * v2
                elif instr.opcode == Instruction.DIV:
                    if v2 == 0:
                        return instr
                    result_val = v1 / v2
                elif instr.opcode == Instruction.MOD:
                    if v2 == 0:
                        return instr
                    result_val = v1 % v2

                if result_val == int(result_val):
                    result_str = str(int(result_val))
                else:
                    result_str = str(result_val)
                return Instruction(Instruction.CONST, result=instr.result, arg1=result_str)
            except (ValueError, ZeroDivisionError):
                pass

        # Return modified instruction with propagated args
        new_instr = Instruction(instr.opcode, result=instr.result, arg1=new_arg1, arg2=new_arg2)
        return new_instr

    return None


def _propagate_arg(arg: str, const_env: dict) -> str | None:
    """Propagate a single argument if it's a known constant."""
    if arg in const_env:
        val = const_env[arg]
        if val == int(val):
            return str(int(val))
        return str(val)
    return arg


def constant_propagation_score(tac: TAC) -> float:
    """
    Calculate a score for how much constant propagation would improve this TAC.

    Score is based on the number of assignments where the right-hand side
    is or can become a constant.
    """
    count = 0
    const_vars = set()

    # First, build constant env
    for instr in tac.instructions:
        if instr.opcode == Instruction.CONST:
            try:
                const_vars.add(instr.result)
                float(instr.arg1)
            except ValueError:
                pass
        elif instr.is_assignment() and instr.arg2 is None:
            # Could potentially propagate
            if instr.arg1 in const_vars:
                count += 1

    # Also count assignments where arg1 might become a constant
    # through previous instructions
    for i, instr in enumerate(tac.instructions):
        if instr.is_assignment() and instr.arg2 is None and instr.arg1 not in const_vars:
            # Check if any previous instruction made this a constant
            for prev_instr in tac.instructions[:i]:
                if prev_instr.opcode == Instruction.CONST and prev_instr.result == instr.arg1:
                    const_vars.add(instr.arg1)
                    count += 1
                    break

    if count == 0:
        return 0.1  # Very low but non-zero

    # Normalize
    return min(1.0, 0.1 + 0.15 * count)