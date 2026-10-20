"""Execution semantics for this project's TAC dialect.

The dialect has one observable side effect - ``print x`` - and no store or
procedure calls: every other instruction can only define a variable,
transfer control, or do both (``if``).  A computed value is externally
visible only when a later instruction reads it *or* when it reaches a
``print``; this is exactly why dead assignments may be removed while the
chain feeding a ``print`` must be kept.  This module gives the dialect a
concrete small-step semantics so the optimization passes can be checked
against the programs they transform instead of being trusted on
inspection.

Semantics (deterministic, total):

* Variables read before any definition take a deterministic *input value*:
  ``(crc32(name + salt) % 9) + 2`` - always in ``[2, 10]``, so ordinary test
  programs never fault on division by zero by accident.  Callers may override
  inputs explicitly via ``execute_tac(..., inputs={...})``.
* ``print x`` appends the operand's current value to the program's output
  sequence (``ExecutionResult.output``) and defines nothing.  The sequence is
  part of the observable behavior a checker compares.
* ``LABEL`` is a no-op; ``goto L`` and a true branch jump to the first
  instruction labelled ``L``.  Jumping to an unknown label is an execution
  error, not a crash.
* ``if a relop b goto L`` compares the *float* values of its operands.  An
  ``if`` that carries a result (the parser's comparison-as-value form
  ``t = a > b``) defines its result as ``1.0``/``0.0`` and never jumps.
* Arithmetic is IEEE-754 float, matching how the folding passes evaluate
  literals.  Division or modulo by zero is an execution error.
* Execution stops at ``terminated`` or when the step limit is hit
  (``terminated=False``); the step limit bounds non-terminating loops so a
  checker can still reason about them.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from .instruction import Instruction, TAC

DEFAULT_STEP_LIMIT = 100_000

# Deterministic input values stay well away from zero so that a stray
# division cannot fault an ordinary test program by accident.
_INPUT_BASE = 2
_INPUT_SPAN = 9


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running a TAC program under the semantics above.

    ``output`` is the sequence of values ``print`` emitted, in program order -
    the program's observable output.  Programs without ``print`` produce an
    empty list.
    """

    terminated: bool
    steps: int
    final_state: dict = field(default_factory=dict)
    error: str | None = None
    output: list = field(default_factory=list)


def input_value(name: str, vector: int = 0) -> float:
    """Deterministic input value for ``name`` in input vector ``vector``.

    Stable across processes and platforms (crc32, not Python's salted hash),
    so recorded results and tests reproduce exactly.
    """
    digest = zlib.crc32(f"{name}#{vector}".encode("utf-8"))
    return float(_INPUT_BASE + digest % _INPUT_SPAN)


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def _compare(left: float, relop: str | None, right: float) -> bool:
    if relop == ">":
        return left > right
    if relop == "<":
        return left < right
    if relop == ">=":
        return left >= right
    if relop == "<=":
        return left <= right
    if relop == "==":
        return left == right
    if relop == "!=":
        return left != right
    raise ValueError(f"unknown relational operator: {relop!r}")


def execute_tac(
    tac: TAC,
    inputs: dict | None = None,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> ExecutionResult:
    """Run ``tac`` to completion (or the step limit) under the dialect semantics.

    ``inputs`` overrides deterministic input values for named variables;
    ``None`` uses :func:`input_value` for every name that is read before it is
    defined.  Never raises for anything the *program* does: a bad jump or a
    division by zero is reported in ``error``.
    """
    if tac is None or not getattr(tac, "instructions", None):
        return ExecutionResult(terminated=True, steps=0, final_state={})

    labels: dict = {}
    for index, instr in enumerate(tac.instructions):
        if instr.opcode == Instruction.LABEL and instr.result is not None:
            # Duplicate labels: the first definition wins, matching the order
            # a linear scan would encounter them.
            labels.setdefault(instr.result, index)

    provided = dict(inputs or {})
    env: dict = {}
    # Values emitted by `print` so far, in program order.
    emitted: list = []

    def value(token: str | None) -> float:
        if token is None:
            raise ValueError("missing operand")
        if token in env:
            return env[token]
        if _is_number(token):
            return float(token)
        if token in provided:
            return float(provided[token])
        return input_value(token)

    steps = 0
    pc = 0
    instructions = tac.instructions
    while pc < len(instructions):
        if steps >= step_limit:
            return ExecutionResult(
                terminated=False, steps=steps, final_state=dict(env),
                output=list(emitted),
            )
        steps += 1
        instr = instructions[pc]

        try:
            if instr.opcode == Instruction.LABEL:
                pc += 1
                continue

            if instr.opcode == Instruction.GOTO:
                target = labels.get(instr.result)
                if target is None:
                    return ExecutionResult(
                        terminated=False,
                        steps=steps,
                        final_state=dict(env),
                        output=list(emitted),
                        error=f"unknown label: {instr.result!r}",
                    )
                pc = target
                continue

            if instr.opcode == Instruction.IF:
                left = value(instr.arg1)
                right = value(instr.arg2)
                if instr.relop is None:
                    return ExecutionResult(
                        terminated=False,
                        steps=steps,
                        final_state=dict(env),
                        output=list(emitted),
                        error="if without relational operator",
                    )
                decision = _compare(left, instr.relop, right)
                # Disambiguation: a jump target that names a known label makes
                # this the branch form; otherwise `result` is the destination
                # variable of the parser's comparison-as-value form
                # (`t = a > b`), which never jumps.
                if instr.result in labels and decision:
                    # Branch form: jump when the condition holds.
                    pc = labels[instr.result]
                    continue
                if instr.result in labels:
                    # Branch form: condition false, fall through.
                    pc += 1
                    continue
                # Comparison-as-value form: `t = a > b`.
                if instr.result is None:
                    return ExecutionResult(
                        terminated=False,
                        steps=steps,
                        final_state=dict(env),
                        output=list(emitted),
                        error="comparison without result variable",
                    )
                env[instr.result] = 1.0 if decision else 0.0
                pc += 1
                continue

            if instr.opcode == Instruction.CONST:
                if instr.result is None:
                    raise ValueError("constant without result variable")
                env[instr.result] = value(instr.arg1)
                pc += 1
                continue

            if instr.opcode == Instruction.ASSIGN:
                if instr.result is None:
                    raise ValueError("assignment without result variable")
                env[instr.result] = value(instr.arg1)
                pc += 1
                continue

            if instr.opcode == Instruction.UMINUS:
                if instr.result is None:
                    raise ValueError("negation without result variable")
                env[instr.result] = -value(instr.arg1)
                pc += 1
                continue

            if instr.opcode == Instruction.PRINT:
                # Observable output: append the operand's value, define
                # nothing.  A missing operand raises and is reported as an
                # execution error below, exactly like other bad operands.
                emitted.append(value(instr.arg1))
                pc += 1
                continue

            if instr.is_binary_op():
                if instr.result is None:
                    raise ValueError("operation without result variable")
                left = value(instr.arg1)
                right = value(instr.arg2)
                if instr.opcode == Instruction.ADD:
                    computed = left + right
                elif instr.opcode == Instruction.SUB:
                    computed = left - right
                elif instr.opcode == Instruction.MUL:
                    computed = left * right
                elif instr.opcode == Instruction.DIV:
                    if right == 0:
                        return ExecutionResult(
                            terminated=False,
                            steps=steps,
                            final_state=dict(env),
                            output=list(emitted),
                            error="division by zero",
                        )
                    computed = left / right
                else:  # MOD
                    if right == 0:
                        return ExecutionResult(
                            terminated=False,
                            steps=steps,
                            final_state=dict(env),
                            output=list(emitted),
                            error="modulo by zero",
                        )
                    computed = left % right
                env[instr.result] = computed
                pc += 1
                continue

            return ExecutionResult(
                terminated=False,
                steps=steps,
                final_state=dict(env),
                output=list(emitted),
                error=f"unknown opcode: {instr.opcode!r}",
            )
        except ValueError as exc:
            return ExecutionResult(
                terminated=False,
                steps=steps,
                final_state=dict(env),
                output=list(emitted),
                error=str(exc),
            )

    return ExecutionResult(
        terminated=True, steps=steps, final_state=dict(env),
        output=list(emitted),
    )
