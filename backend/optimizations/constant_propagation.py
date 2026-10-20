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

    Walks the TAC once, forwards, keeping a constant environment that is
    updated after every instruction. The previous implementation built the
    environment in a separate pre-pass and then applied it in a second pass,
    so a *later* redefinition of a variable overwrote the value an *earlier*
    read had to see: `a = 10 / x = a / a = 99` propagated 99 into `x`.

    The environment is a *must* fact: "every path reaching this instruction
    has x == 5".  A forward pass establishes that fact exactly while control
    can only arrive by straight-line fall-through, which holds inside a
    label-free stretch - jumps can only target labels, so every join point in
    the program is a label instruction.  The environment is therefore cleared
    at every label.  Without that reset the fact was carried across joins and
    loops, and a constant bound before a label was propagated into code that
    a different path (or a later loop iteration) reached with a different
    value: `i = 0 / loop: i = i + 1 / if i < 10 goto loop` became
    `i = 1 / loop: i = 1 / ...`, turning a terminating loop into an infinite
    one.  Checked by tests/test_semantic_preservation.py.
    """
    if not tac.instructions:
        return tac

    const_env = {}  # variable -> constant value
    result_tac = TAC()

    for instr in tac.instructions:
        if instr.opcode == Instruction.LABEL:
            # Join point: control may arrive here from several paths (or a
            # back edge), so no constant binding survives into the stretch
            # that follows.  The label itself is kept verbatim.
            const_env.clear()
            result_tac.add(instr)
            continue
        new_instr = _propagate_instruction(instr, const_env)
        applied = new_instr if new_instr else instr
        result_tac.add(applied)
        # Record the definition only after the instruction itself has been
        # propagated, so an instruction never sees its own new binding.
        _update_const_env(applied, const_env)

    return result_tac


def _update_const_env(instr: Instruction, const_env: dict):
    """Record what `instr` just defined.

    A definition that is neither a literal nor a copy of a known constant must
    REMOVE the previous binding. Leaving a stale entry in place made
    `a = 10 / a = x / b = a` propagate 10 into `b`.
    """
    var = instr.result
    if var is None:
        return

    if instr.opcode == Instruction.CONST:
        try:
            const_env[var] = float(instr.arg1)
        except (TypeError, ValueError):
            const_env.pop(var, None)
        return

    if instr.is_assignment() and instr.arg2 is None:
        # `t = x`: t inherits x's constant only while x still holds one.
        source = instr.arg1
        if source is not None and source in const_env:
            const_env[var] = const_env[source]
        else:
            const_env.pop(var, None)
        return

    # Binary/unary results and control-flow targets are not known constants.
    const_env.pop(var, None)


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

        # Fold once at least one operand was actually REPLACED by a propagated
        # value and both operands are literals afterwards.
        #
        # The old test required *both* arguments to have changed, so `p = c + 3`
        # (where only `c` was a variable) stayed as `p = 12 + 3` and the pass
        # never removed any work. Folding unconditionally instead would make
        # this pass swallow constant folding - `f0 = 5 + 3` must remain CF's
        # job, which it does because a binary definition never enters the
        # constant environment unless this pass folded it itself.
        propagated = (new_arg1 != instr.arg1) or (new_arg2 != instr.arg2)
        if (
            propagated
            and new_arg1 is not None
            and new_arg2 is not None
            and Instruction._is_constant(new_arg1)
            and Instruction._is_constant(new_arg2)
        ):
            folded = _fold_operands(instr.opcode, new_arg1, new_arg2)
            if folded is not None:
                return Instruction(Instruction.CONST, result=instr.result, arg1=folded)

        # Return modified instruction with propagated args
        return Instruction(instr.opcode, result=instr.result, arg1=new_arg1, arg2=new_arg2)

    return None


def _fold_operands(opcode: str, arg1: str, arg2: str) -> str | None:
    """Evaluate a binary operation on two literals.

    Returns None when the operation is undefined (division or modulo by zero),
    so the caller leaves the original instruction alone instead of crashing.
    """
    try:
        v1 = float(arg1)
        v2 = float(arg2)

        if opcode == Instruction.ADD:
            value = v1 + v2
        elif opcode == Instruction.SUB:
            value = v1 - v2
        elif opcode == Instruction.MUL:
            value = v1 * v2
        elif opcode == Instruction.DIV:
            if v2 == 0:
                return None
            value = v1 / v2
        elif opcode == Instruction.MOD:
            if v2 == 0:
                return None
            value = v1 % v2
        else:
            return None
    except (ValueError, TypeError, OverflowError):
        return None

    if value == int(value):
        return str(int(value))
    return str(value)


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