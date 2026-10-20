"""
Iterative Optimization Engine.

Implements the core optimization loop:

    Analyze (features + rule scores on the current program)
        ↓
    Select a candidate strategy (the selector's prediction, or a ranked
    fallback when the prediction already had its chance on this program)
        ↓
    Execute that single pass
        ↓
    Measure the outcome and classify it
    (applied_useful / applied_changed / executed_no_change / rejected /
     failed / unsupported)
        ↓
    Accept only transformations that preserve observable behavior
    (bounded check against the ORIGINAL program)
        ↓
    Either the program changed → re-extract features, re-score, re-select;
    or it did not → try the next eligible strategy (controlled fallback)
        ↓
    Stop when no eligible strategy can produce an accepted improvement,
    when a safety limit is reached, or when the selector names a strategy
    the engine does not have.

Selection and fallback stay distinguishable: every attempt records both the
strategy the *selector predicted* for that program state and the strategy
actually *executed* (`origin` is "predicted" or "fallback").  The engine
never silently rewrites an ML (or rule) prediction - a no-effect prediction
is recorded as such and a fallback decision is recorded separately.

Convergence (why optimization stopped):

* ``no_further_useful_improvement`` - every eligible strategy was tried on
  the final program and none could produce an accepted improvement.
* ``no_instruction_reduction`` - same exhaustion, but the last *accepted*
  transformation only removed operations; the instruction count never fell
  to a point where another strategy could act.  (Historical vocabulary,
  kept because the UI reports it.)
* ``max_iterations_reached`` - the documented safety cap ended the run
  while improvements were still being accepted.  ``converged`` is False.
* ``no_applicable_optimization`` - the selector returned a strategy id the
  engine does not have; the run stops without guessing.  ``converged`` is
  False and the attempt is recorded as ``unsupported``.

Termination: a strategy is never retried while it produced no accepted
change on the *same* program state, an accepted transformation that would
reproduce a previously visited program state is rejected (cycle guard),
and ``max_iterations`` bounds every run.
"""

from copy import deepcopy

from ..tac.instruction import Instruction, TAC

from ..api.serialize import tac_to_source
from ..features.extractor import extract_features
from ..scoring.rule_based import (
    STRATEGY_TIE_ORDER,
    rule_based_strategy_scores,
    select_strategy,
    strategy_reason,
)
from ..optimizations.constant_folding import constant_folding
from ..optimizations.constant_propagation import constant_propagation
from ..optimizations.common_subexpression import common_subexpression_elimination
from ..optimizations.dead_code import dead_code_elimination
from ..optimizations.algebraic_simplification import algebraic_simplification
from ..optimizations.strength_reduction import strength_reduction

from .correctness import check_semantic_equivalence
from .diff import tac_diff
from .measure import measure_pass, operation_count, work_remaining


# Strategy map: name -> optimization function
OPTIMIZATION_FUNCTIONS = {
    "constant_folding": constant_folding,
    "constant_propagation": constant_propagation,
    "common_subexpression": common_subexpression_elimination,
    "dead_code_elimination": dead_code_elimination,
    "algebraic_simplification": algebraic_simplification,
    "strength_reduction": strength_reduction,
}


MAX_ITERATIONS = 20


# Documented per-attempt outcomes. Every executed pass lands in exactly one:
#
# * applied_useful      - transformed the program AND removed measured work
#                         (instructions and/or operations).
# * applied_changed     - transformed the program but removed no measured
#                         work: a neutral rewrite that may unlock a later
#                         pass (e.g. `t = c` -> `t = 9` after propagation).
# * executed_no_change  - ran successfully; the program is byte-identical.
# * rejected            - produced a different program that failed the
#                         semantic acceptance check (or would have cycled);
#                         the pre-pass program was kept.
# * failed              - the pass raised; nothing was applied.
# * unsupported         - the selector named a strategy the engine does not
#                         have; nothing was executed.
OUTCOME_APPLIED_USEFUL = "applied_useful"
OUTCOME_APPLIED_CHANGED = "applied_changed"
OUTCOME_NO_CHANGE = "executed_no_change"
OUTCOME_REJECTED = "rejected"
OUTCOME_FAILED = "failed"
OUTCOME_UNSUPPORTED = "unsupported"

PASS_OUTCOMES = (
    OUTCOME_APPLIED_USEFUL,
    OUTCOME_APPLIED_CHANGED,
    OUTCOME_NO_CHANGE,
    OUTCOME_REJECTED,
    OUTCOME_FAILED,
    OUTCOME_UNSUPPORTED,
)


# ---------------------------------------------------------------------------
# Fallback eligibility: does this pass have real work to do in this program?
#
# The selector's own prediction is ALWAYS executed - a trace must be able to
# show whether the prediction helped.  These scans only decide which
# *additional* strategies the fallback may try, so the engine never "runs all
# six passes blindly and claims it selected each one intelligently": every
# fallback candidate has a documented opportunity, detected either by the
# same features the rules use or by mirroring the pass's own trigger.
# ---------------------------------------------------------------------------


def _constant_folding_opportunity(tac) -> bool:
    """True when constant_folding would fold at least one instruction.

    Mirrors optimizations.constant_folding: a binary op with both operands
    literal, excluding the division/modulo-by-zero cases the pass refuses.
    """
    for instr in tac.instructions:
        if instr.opcode not in {
            Instruction.ADD,
            Instruction.SUB,
            Instruction.MUL,
            Instruction.DIV,
            Instruction.MOD,
        }:
            continue
        if instr.arg1 is None or instr.arg2 is None:
            continue
        if not (Instruction._is_constant(instr.arg1) and Instruction._is_constant(instr.arg2)):
            continue
        if instr.opcode in {Instruction.DIV, Instruction.MOD}:
            try:
                if float(instr.arg2) == 0:
                    continue
            except (TypeError, ValueError):
                continue
        return True
    return False


def _constant_propagation_opportunity(tac) -> bool:
    """True when constant_propagation would rewrite at least one instruction.

    Mirrors optimizations.constant_propagation exactly: a forward walk with
    the same must-constant environment (cleared at every label), flagging the
    two rewrites the pass performs - a copy of a known constant, or a binary
    op with at least one known-constant operand.  When no instruction is
    flagged here, the pass leaves the program byte-identical.
    """
    const_env: dict[str, float] = {}
    for instr in tac.instructions:
        if instr.opcode == Instruction.LABEL:
            const_env.clear()
            continue
        if instr.opcode == Instruction.CONST:
            pass
        elif instr.is_assignment() and instr.arg2 is None:
            if instr.arg1 in const_env:
                return True
        elif instr.is_binary_op():
            if instr.arg1 in const_env or (instr.arg2 is not None and instr.arg2 in const_env):
                return True
        # Same environment update as the pass, applied only while nothing has
        # been rewritten yet - so the "no opportunity" verdict is exact.
        var = instr.result
        if var is None:
            continue
        if instr.opcode == Instruction.CONST:
            try:
                const_env[var] = float(instr.arg1)
            except (TypeError, ValueError):
                const_env.pop(var, None)
        elif instr.is_assignment() and instr.arg2 is None:
            if instr.arg1 in const_env:
                const_env[var] = const_env[instr.arg1]
            else:
                const_env.pop(var, None)
        else:
            const_env.pop(var, None)
    return False


def _strength_reduction_opportunity(tac) -> bool:
    """True when strength_reduction's own detection condition fires.

    The pass is a documented conservative no-op: it detects multiplication by
    an integer constant but rewrites nothing without loop-carrying
    information.  Attempting it on such a program therefore records an
    honest ``executed_no_change`` with the pass's limitation in the trace.
    """
    for instr in tac.instructions:
        if not instr.is_binary_op() or instr.opcode != Instruction.MUL:
            continue
        if instr.arg2 is None or not Instruction._is_constant(instr.arg2):
            continue
        try:
            value = float(instr.arg2)
        except (TypeError, ValueError):
            continue
        if value.is_integer() and int(value) >= 2:
            return True
    return False


def eligible_strategies(tac) -> list[str]:
    """Fallback candidates for this program: strategies with real work to do.

    Returns the strategies whose opportunities actually exist in ``tac``,
    ranked by the rule-based scores (ties broken by STRATEGY_TIE_ORDER, the
    same deterministic order ``select_strategy`` uses).  Features come from
    the same extractor the rules and the ML model use; CF/CP/SR additionally
    mirror their passes' trigger conditions, so a candidate can never be a
    blind guess.  Used to gate *fallback* attempts only.
    """
    features = extract_features(tac)
    eligible: set[str] = set()
    if _constant_folding_opportunity(tac):
        eligible.add("constant_folding")
    if _constant_propagation_opportunity(tac):
        eligible.add("constant_propagation")
    if features["num_repeated_expr"] > 0:
        eligible.add("common_subexpression")
    if features["num_dead_stmts"] > 0:
        eligible.add("dead_code_elimination")
    if features["num_algebraic_ops"] > 0:
        eligible.add("algebraic_simplification")
    if _strength_reduction_opportunity(tac):
        eligible.add("strength_reduction")

    scores = rule_based_strategy_scores(tac)
    tie_rank = {name: index for index, name in enumerate(STRATEGY_TIE_ORDER)}
    return sorted(
        eligible,
        key=lambda name: (-scores.get(name, 0.0), tie_rank.get(name, len(tie_rank))),
    )


def _acceptance_verdict(original_tac, tac_before, candidate):
    """Decide whether an accepted transformation may replace the program.

    Policy (bounded evidence, never a claim of formal equivalence):

    1. Compare the ORIGINAL program with the candidate under the semantic
       checker.  ``equivalent`` accepts; ``not_equivalent`` rejects.
    2. When the original cannot be checked (it faults or does not terminate
       on the checker's vectors, so step 1 is inconclusive), fall back to
       comparing the PRE-PASS program with the candidate - isolating whether
       *this* transformation changed observable behavior.  ``equivalent``
       accepts; anything else rejects, because behavior preservation could
       not be established for this pass.

    Returns ``(accepted, status, note)``.
    """
    try:
        verdict = check_semantic_equivalence(original_tac, candidate)
    except Exception as exc:  # noqa: BLE001 - checker bugs must not accept
        return False, "inconclusive", f"Checker raised {type(exc).__name__}: {exc}"
    status = verdict.get("status")
    if status == "equivalent":
        return True, status, verdict.get("detail", "")
    if status == "not_equivalent":
        return False, status, verdict.get("detail", "")

    # Inconclusive against the original: it may fault on every vector.
    try:
        isolated = check_semantic_equivalence(tac_before, candidate)
    except Exception as exc:  # noqa: BLE001 - checker bugs must not accept
        return False, "inconclusive", f"Checker raised {type(exc).__name__}: {exc}"
    isolated_status = isolated.get("status")
    if isolated_status == "equivalent":
        return True, "equivalent", (
            f"{verdict.get('detail', '')} "
            f"(Checked against the pre-pass program instead: {isolated.get('detail', '')})"
        )
    return False, isolated_status, (
        "Correctness could not be established for this transformation: "
        f"original vs candidate: {status} ({verdict.get('detail', '')}); "
        f"pre-pass vs candidate: {isolated_status} ({isolated.get('detail', '')})"
    )


def _exhausted_stop_reason(ever_changed: bool, last_accepted_item) -> str:
    """The convergence reason after every eligible strategy has been tried."""
    if not ever_changed or last_accepted_item is None:
        return "no_further_useful_improvement"
    if last_accepted_item["improvement"] == 0:
        # The last thing that changed only removed operations (or rewrote
        # without shrinking); the instruction count never dropped again.
        return "no_instruction_reduction"
    return "no_further_useful_improvement"


def run_iterative_optimization(
    tac: TAC,
    max_iterations: int = MAX_ITERATIONS,
    selector=None,
    iteration_context=None,
    selector_name: str = "rule_based",
):
    """
    Run the iterative optimization engine.

    One **iteration** = one executed pass attempt (each pass rewrites the
    whole program in one call), so ``total_iterations`` counts attempts and
    every history entry records that attempt's own before/after programs and
    measurements.  A single *selection round* can span several attempts when
    the predicted strategy makes no useful improvement and the controlled
    fallback tries the next eligible strategy.

    Selection vs execution: each entry records ``predicted_strategy`` (what
    the selector chose for this program state) and ``selected_strategy``
    (what the engine actually executed), with ``origin`` = "predicted" for
    the selector's own pick and "fallback" for a recovery attempt.  Every
    attempt's outcome is one of ``PASS_OUTCOMES``.

    Acceptance: a transformation is applied only when the semantic checker
    (backend.optimizer.correctness - bounded evidence over deterministic
    test vectors, not a formal proof) finds the candidate equivalent to the
    original program; when the original itself faults, acceptance falls back
    to the pre-pass program, and anything still unverifiable is rejected and
    the previous program is kept.

    Convergence: the loop does NOT stop because one pass made no change; it
    stops when every eligible strategy has been tried on the current program
    without an accepted improvement (``no_further_useful_improvement`` /
    ``no_instruction_reduction`` - both ``converged``), when ``max_iterations``
    is exhausted (``max_iterations_reached``), or when the selector names a
    strategy the engine does not have (``no_applicable_optimization``).
    A strategy is never retried on an unchanged program state, and a
    transformation that would reproduce a previously seen program state is
    rejected, so cyclic rewrites cannot loop forever.

    Args:
        tac: Initial TAC to optimize
        max_iterations: Maximum number of pass attempts (default: 20)
        selector: Optional ``callable(tac) -> strategy_id`` replacing the
            default rule-based choice.  The ML mode passes a selector backed
            by the trained forest; the engine itself is unchanged and is
            never duplicated.  ``None`` preserves the original rule-based
            behaviour.
        iteration_context: Optional ``callable(tac_before, features, selected) -> dict``
            whose entries are merged into that attempt's record.  It receives
            the TAC the features were extracted from (the input to this
            attempt), so attached explanations describe the decision that was
            actually made.  Used to attach ML probabilities and SHAP
            contributions on request without making them mandatory for
            rule-based runs.
        selector_name: Label recorded on every attempt so the trace states
            which selector drove the run (``rule_based`` / ``ml``).

    Returns:
        dict with keys:
            "history": list of attempt result dicts
            "final_tac": the optimized TAC
            "original_tac": a copy of the input program (unchanged)
            "total_reduction": total instruction reduction
            "total_iterations": number of attempts performed
            "converged": whether optimization converged
            "stop_reason": why the loop ended
            "selector": which selector drove the run
            "passes_rejected": attempts rejected by acceptance (kept out of
                the final program, recorded in the history)
            "fallback_attempts": attempts made by the controlled fallback
    """
    history = []
    current_tac = deepcopy(tac)
    original_tac = deepcopy(tac)
    total_before = len(current_tac)

    choose = selector if selector is not None else select_strategy
    # Program states already visited - an accepted transformation reproducing
    # one of them would cycle, so it is rejected instead.
    seen_states = {tac_to_source(current_tac)}
    # Strategies already attempted on the *current* program state.  Reset
    # whenever an accepted transformation changes the program.
    tried: set[str] = set()
    ever_changed = False
    last_accepted_item = None

    # The loop's *default* end state: it only remains "max_iterations_reached"
    # when the for-loop actually runs out of attempts with the run still
    # willing to continue.  Any explicit ``break`` below clears this flag so
    # the recorded stop reason is the condition that really ended the run.
    stop_reason = "max_iterations_reached"
    hit_iteration_limit = True

    for iteration_num in range(1, max_iterations + 1):
        # 1. Analyze: extract features and rule scores from the CURRENT
        #    program - recalculated after every accepted transformation.
        features = extract_features(current_tac)
        scores = rule_based_strategy_scores(current_tac)
        eligible = eligible_strategies(current_tac)

        # 2. Predict (what the configured selector chooses for this state).
        predicted = choose(current_tac)

        # Snapshot BEFORE the pass runs so the recorded "TAC before" can never
        # reflect a partially-transformed program, even if a future pass
        # mutates its input in place.
        tac_before = deepcopy(current_tac)
        before_count = len(current_tac)
        before_ops = operation_count(current_tac)

        # 3. Decide what to execute: the prediction, unless it already had
        #    its chance on this exact program - then the rule-ranked next
        #    eligible strategy, or convergence when none remain.
        if predicted not in OPTIMIZATION_FUNCTIONS:
            unsupported_reason = (
                f"The selector returned '{predicted}', which is not one of the "
                "six optimization passes; the run stops without guessing "
                "another strategy."
            )
            result = {
                "iteration": iteration_num,
                "selector": selector_name,
                "features": features,
                "scores": scores,
                "selected_strategy": predicted,
                "predicted_strategy": predicted,
                "origin": "predicted",
                "outcome": OUTCOME_UNSUPPORTED,
                "eligible_strategies": eligible,
                "confidence": scores.get(predicted, 0),
                "rationale": unsupported_reason,
                "before_count": before_count,
                "after_count": before_count,
                "improvement": 0,
                "improvement_percentage": 0.0,
                "operation_count_before": before_ops,
                "operation_count_after": before_ops,
                "operation_reduction": 0,
                "useful_improvement": 0,
                "changed": False,
                "accepted": False,
                "rejection_reason": unsupported_reason,
                "will_continue": False,
                "stop_reason": "no_applicable_optimization",
                "diff": None,
                "tac_before": tac_before,
                "tac_after": tac_before,
            }
            history.append(result)
            stop_reason = "no_applicable_optimization"
            hit_iteration_limit = False
            break

        if predicted in tried:
            candidates = [name for name in eligible if name not in tried]
            if not candidates:
                # Every eligible strategy has been tried on this unchanged
                # program: nothing more any selector could gain here.
                stop_reason = _exhausted_stop_reason(ever_changed, last_accepted_item)
                hit_iteration_limit = False
                break
            executed, origin = candidates[0], "fallback"
        else:
            executed, origin = predicted, "predicted"
        tried.add(executed)

        # 4. Execute the chosen pass, reporting crashes instead of raising.
        opt_fn = OPTIMIZATION_FUNCTIONS[executed]
        error_note = None
        new_tac = None
        try:
            new_tac = opt_fn(current_tac)
        except Exception as exc:  # noqa: BLE001 - a crashing pass is reported
            error_note = f"{type(exc).__name__}: {exc}"

        declined = error_note is None and new_tac is None
        candidate = new_tac if error_note is None and new_tac is not None else tac_before
        after_count = len(candidate)
        after_ops = operation_count(candidate)

        # 5. Measure the attempt (statement reduction is the primary metric;
        #    useful_improvement also credits operations that disappeared
        #    without a statement being removed).
        improvement = before_count - after_count
        improvement_percentage = (improvement / before_count * 100) if before_count > 0 else 0.0
        measurements = measure_pass(tac_before, candidate)
        useful_improvement = int(measurements["useful_improvement"])

        changed = (
            error_note is None
            and new_tac is not None
            and tac_to_source(candidate) != tac_to_source(tac_before)
        )

        # 6. Classify and decide acceptance.
        accepted = False
        rejection_reason = None
        correctness_status = None
        correctness_note = None
        state_updated = False

        if error_note is not None:
            outcome = OUTCOME_FAILED
            rejection_reason = error_note
        elif not changed:
            outcome = OUTCOME_NO_CHANGE
            accepted = True  # identity: the current program is kept as-is
            correctness_note = (
                "Pass produced no output for this program." if declined else None
            )
        elif tac_to_source(candidate) in seen_states:
            outcome = OUTCOME_REJECTED
            rejection_reason = (
                "The transformation would reproduce a previously visited "
                "program state (cyclic rewrite); rejected to guarantee "
                "termination."
            )
        else:
            is_acceptable, verdict_status, verdict_note = _acceptance_verdict(
                original_tac, tac_before, candidate
            )
            correctness_status = verdict_status
            correctness_note = verdict_note
            if is_acceptable:
                outcome = (
                    OUTCOME_APPLIED_USEFUL if useful_improvement > 0 else OUTCOME_APPLIED_CHANGED
                )
                accepted = True
                state_updated = True
            else:
                outcome = OUTCOME_REJECTED
                rejection_reason = verdict_note

        # 7. Program update: only accepted transformations move the state.
        if state_updated:
            seen_states.add(tac_to_source(candidate))
            current_tac = candidate
            tried.clear()  # re-evaluate strategy eligibility on the new state
            ever_changed = True

        # 8. Does the run continue?  A changed program reopens every
        #    strategy; an unchanged one continues only while untried
        #    eligible candidates remain (the controlled fallback).
        if state_updated:
            will_continue = True
        else:
            will_continue = any(name not in tried for name in eligible)

        # 9. Record the attempt.
        if origin == "fallback":
            rationale = (
                f"Fallback: '{predicted}' did not produce an accepted "
                f"improvement on this program, so the rule-ranked next "
                f"strategy with a real opportunity is '{executed}'. "
                f"{strategy_reason(tac_before, executed)}"
            )
        else:
            rationale = strategy_reason(tac_before, executed)

        result = {
            "iteration": iteration_num,
            "selector": selector_name,
            "features": features,
            "scores": scores,
            "selected_strategy": executed,
            "predicted_strategy": predicted,
            "origin": origin,
            "outcome": outcome,
            "eligible_strategies": eligible,
            "confidence": scores.get(executed, 0),
            # The rationale must describe the program the decision was made
            # *from* (tac_before), not the already-transformed current_tac -
            # feature counts quoted here (dead assignments, constants, ...)
            # have to match the input that actually drove the selection.
            "rationale": rationale,
            "before_count": before_count,
            "after_count": after_count,
            "improvement": improvement,
            "improvement_percentage": round(improvement_percentage, 2),
            "operation_count_before": before_ops,
            "operation_count_after": after_ops,
            "operation_reduction": int(measurements["operation_reduction"]),
            "useful_improvement": useful_improvement,
            "changed": changed,
            # accepted = this attempt's program became (or stayed) the
            # current program: applied passes and identity no-change runs are
            # accepted; rejected/failed/unsupported attempts are not.
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "correctness_status": correctness_status,
            "correctness_note": correctness_note,
            "will_continue": will_continue,
            "stop_reason": None if will_continue else _exhausted_stop_reason(
                ever_changed, last_accepted_item
            ),
            "diff": tac_diff(tac_before, candidate) if changed else None,
            "tac_before": tac_before,
            "tac_after": candidate,
        }
        if accepted and state_updated:
            last_accepted_item = result
        if iteration_context is not None:
            # The context explains the decision just made, so it receives the
            # TAC the features were extracted from - not the already-transformed
            # result.
            extra = iteration_context(tac_before, features, executed)
            if extra:
                result.update(extra)
        history.append(result)

        if not will_continue:
            stop_reason = result["stop_reason"]
            hit_iteration_limit = False
            break

    total_after = len(current_tac)
    total_reduction = total_before - total_after
    total_iterations = len(history)
    # "Converged" means the engine exhausted every eligible strategy (or
    # never had one to begin with) - not that an error or the iteration
    # limit ended the run.  Reaching exhaustion on the very attempt the cap
    # also happens to allow still counts as convergence: the break above
    # fired on the exhaustion condition first.
    if hit_iteration_limit:
        stop_reason = "max_iterations_reached"
        converged = False
    else:
        converged = stop_reason != "no_applicable_optimization"

    return {
        "history": history,
        "final_tac": current_tac,
        "original_tac": original_tac,
        "total_reduction": total_reduction,
        "total_iterations": total_iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "selector": selector_name,
        "passes_rejected": sum(1 for item in history if item["outcome"] == OUTCOME_REJECTED),
        "fallback_attempts": sum(1 for item in history if item["origin"] == "fallback"),
        "original_instruction_count": total_before,
        "final_instruction_count": total_after,
        "original_operation_count": operation_count(tac),
        "final_operation_count": operation_count(current_tac),
        "useful_improvement": int(work_remaining(tac) - work_remaining(current_tac)),
    }
