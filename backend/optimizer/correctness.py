"""Bounded semantic validation of optimizer output.

A pass claims to optimize a program; this module checks whether the program
it produced still *behaves* like the one it replaced, using the execution
semantics in :mod:`backend.tac.interpreter`.  It exists so the API can report
a real, honestly-scoped status instead of the previous blanket
"no semantic-equivalence checker exists in this project".

What "equivalent" means here - stated precisely, because it is bounded
evidence, not a proof over all inputs:

* The observable behavior of a program under this dialect is (a) the sequence
  of values it prints (``print x`` is the dialect's only output statement) and
  (b) the final values of the variables the **optimized** program still reads
  (as operands or in branch conditions).  A value no surviving instruction
  reads and no ``print`` emits cannot be observed by anything; this is what
  makes removing an all-dead, non-printing program to empty legitimately
  equivalent rather than vacuously "probably fine" - while an optimized
  program that dropped or altered a ``print`` is caught directly.
* Both programs are executed on the same ``k`` deterministic input vectors.
  For every vector both must terminate within the step limit and agree, to
  the last bit of the float, on the printed output sequence and on the final
  value of every observable variable.
* Termination must agree too: one program finishing while the other is still
  running after the step limit is reported as ``not_equivalent`` (that is how
  an optimizer that turns a bounded loop into an infinite loop is caught).
* Two programs that both exceed the step limit, or either program that faults
  (bad jump, division by zero), yield ``inconclusive`` - no claim is made
  when the checker could not actually observe the behavior.

A ``not_equivalent`` verdict is decisive (the checker observed the
divergence).  ``equivalent`` means "no divergence observed on these input
vectors", which is evidence, not a universal proof.
"""

from __future__ import annotations

from ..tac.instruction import Instruction, TAC
from ..tac.interpreter import DEFAULT_STEP_LIMIT, execute_tac

__all__ = [
    "check_semantic_equivalence",
    "read_variables",
    "program_variables",
    "has_print_output",
    "DEFAULT_INPUT_VECTORS",
]

DEFAULT_INPUT_VECTORS = 3


def read_variables(tac: TAC) -> set:
    """Variables the program reads: operands of assignments, operations,
    negations and branch conditions.

    Result names are not reads, and a jump target is a label, not a variable.
    Literal operands are not variables either.  A comparison-as-value ``if``
    (``t = a > b``) reads ``a`` and ``b`` exactly like a binary operation.
    """
    reads: set = set()
    if tac is None or not getattr(tac, "instructions", None):
        return reads
    for instr in tac.instructions:
        for operand in (instr.arg1, instr.arg2):
            if operand is None or Instruction._is_constant(operand):
                continue
            reads.add(operand)
    return reads


def program_variables(tac: TAC) -> set:
    """Every variable name that appears in ``tac`` (read or written)."""
    names: set = set()
    if tac is None or not getattr(tac, "instructions", None):
        return names
    for instr in tac.instructions:
        for token in (instr.arg1, instr.arg2):
            if token is not None and not Instruction._is_constant(token):
                names.add(token)
        # LABEL and GOTO results are label names; every other result is a
        # variable.  Filtering on the label-shaped opcodes keeps labels out
        # of the input vector without needing the label table here.
        if instr.result is not None and instr.opcode not in {
            Instruction.LABEL,
            Instruction.GOTO,
        }:
            names.add(instr.result)
    return names


def has_print_output(tac: TAC) -> bool:
    """True when ``tac`` contains at least one ``print`` (observable output)."""
    if tac is None or not getattr(tac, "instructions", None):
        return False
    return any(instr.opcode == Instruction.PRINT for instr in tac.instructions)


def _vector_inputs(names: set, vector: int) -> dict:
    from ..tac.interpreter import input_value

    return {name: input_value(name, vector) for name in names}


def check_semantic_equivalence(
    original: TAC,
    optimized: TAC,
    input_vectors: int = DEFAULT_INPUT_VECTORS,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> dict:
    """Compare ``optimized`` against ``original`` under the dialect semantics.

    Returns a dict with:
        status:           "equivalent" | "not_equivalent" | "inconclusive"
        detail:           human-readable evidence for the status
        observed_vars:    variables whose final values were compared
        outputs_compared: whether printed output sequences were compared
        checked_vectors:  how many deterministic input vectors ran
        step_limit:       the bound both programs were held to
    """
    observed = sorted(read_variables(optimized))
    names = program_variables(original) | program_variables(optimized)
    outputs_compared = has_print_output(original) or has_print_output(optimized)

    if not observed and not outputs_compared:
        # Nothing either program does can be observed: no surviving
        # instruction reads any variable and neither program prints, so both
        # are extensionally the empty computation.  Say exactly that instead
        # of implying a deep check happened.
        return {
            "status": "equivalent",
            "detail": (
                "The optimized program reads no variables and neither program "
                "prints anything, so nothing either does is observable; the "
                "transform is equivalent by construction."
            ),
            "observed_vars": [],
            "outputs_compared": False,
            "checked_vectors": 0,
            "step_limit": step_limit,
        }

    if input_vectors < 1:
        raise ValueError("input_vectors must be >= 1")

    for vector in range(input_vectors):
        inputs = _vector_inputs(names, vector)
        run_a = execute_tac(original, inputs=inputs, step_limit=step_limit)
        run_b = execute_tac(optimized, inputs=inputs, step_limit=step_limit)

        if run_a.error is not None or run_b.error is not None:
            return {
                "status": "inconclusive",
                "detail": (
                    "A program faulted under the TAC semantics "
                    f"(input vector {vector}: original error={run_a.error!r}, "
                    f"optimized error={run_b.error!r}); behavior before the "
                    "fault cannot be compared, so no equivalence claim is made."
                ),
                "observed_vars": observed,
                "outputs_compared": outputs_compared,
                "checked_vectors": vector + 1,
                "step_limit": step_limit,
            }

        if run_a.terminated and not run_b.terminated:
            return {
                "status": "not_equivalent",
                "detail": (
                    f"On input vector {vector} the original terminated in "
                    f"{run_a.steps} steps but the optimized program was still "
                    f"running after {step_limit} steps - the transform changed "
                    "termination behavior."
                ),
                "observed_vars": observed,
                "outputs_compared": outputs_compared,
                "checked_vectors": vector + 1,
                "step_limit": step_limit,
            }
        if run_b.terminated and not run_a.terminated:
            return {
                "status": "not_equivalent",
                "detail": (
                    f"On input vector {vector} the optimized program terminated "
                    f"in {run_b.steps} steps but the original was still running "
                    f"after {step_limit} steps - the transform changed "
                    "termination behavior."
                ),
                "observed_vars": observed,
                "outputs_compared": outputs_compared,
                "checked_vectors": vector + 1,
                "step_limit": step_limit,
            }
        if not run_a.terminated and not run_b.terminated:
            return {
                "status": "inconclusive",
                "detail": (
                    f"Both programs exceeded the {step_limit}-step limit on "
                    f"input vector {vector}, so their final state cannot be "
                    "compared; no equivalence claim is made."
                ),
                "observed_vars": observed,
                "outputs_compared": outputs_compared,
                "checked_vectors": vector + 1,
                "step_limit": step_limit,
            }

        if outputs_compared and run_a.output != run_b.output:
            return {
                "status": "not_equivalent",
                "detail": (
                    f"On input vector {vector} the original printed "
                    f"{run_a.output!r} while the optimized program printed "
                    f"{run_b.output!r} - the observable output differs."
                ),
                "observed_vars": observed,
                "outputs_compared": outputs_compared,
                "checked_vectors": vector + 1,
                "step_limit": step_limit,
            }

        for name in observed:
            before = run_a.final_state.get(name)
            after = run_b.final_state.get(name)
            if before != after:
                return {
                    "status": "not_equivalent",
                    "detail": (
                        f"On input vector {vector} the original leaves "
                        f"{name}={before!r} while the optimized program leaves "
                        f"{name}={after!r}."
                    ),
                    "observed_vars": observed,
                    "outputs_compared": outputs_compared,
                    "checked_vectors": vector + 1,
                    "step_limit": step_limit,
                }

    if outputs_compared:
        if observed:
            detail = (
                f"Both programs terminated within {step_limit} steps, printed "
                f"identical output sequences on {input_vectors} deterministic "
                "input vectors, and agreed to the last bit on the final "
                f"values of all {len(observed)} observed variables "
                f"({', '.join(observed)}). Bounded evidence, not a proof over "
                "all inputs."
            )
        else:
            detail = (
                f"Both programs terminated within {step_limit} steps and "
                f"printed identical output sequences on {input_vectors} "
                "deterministic input vectors. Bounded evidence, not a proof "
                "over all inputs."
            )
    else:
        detail = (
            f"Both programs terminated within {step_limit} steps with identical "
            f"final values for all {len(observed)} observed variables "
            f"({', '.join(observed)}) on {input_vectors} deterministic input "
            "vectors. Bounded evidence, not a proof over all inputs."
        )

    return {
        "status": "equivalent",
        "detail": detail,
        "observed_vars": observed,
        "outputs_compared": outputs_compared,
        "checked_vectors": input_vectors,
        "step_limit": step_limit,
    }
