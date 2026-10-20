"""Rule-Based vs ML comparison and engine traceability tests.

Two distinct notions are asserted separately, mirroring §10:

* **prediction quality** - does a selector name the stored ``best_strategy``?
* **optimization effectiveness** - how much work does its chosen strategy
  actually remove when executed?

The stored label is never used as either selector's prediction; the tests below
recompute accuracy from the recorded per-program predictions to prove it.
"""

from collections import Counter

import pytest

from backend.features.extractor import extract_features
from backend.ml.compare import (
    FEEDBACK_WINS,
    ML_WINS,
    RULE_WINS,
    TIE,
    apply_strategy_once,
    compare_on_dataset,
    compare_source,
)
from backend.ml.dataset import load_dataset, split_frames
from backend.optimizer.diff import tac_diff
from backend.optimizer.engine import run_iterative_optimization
from backend.optimizer.measure import measure_pass, operation_count, work_remaining
from backend.scoring.rule_based import select_strategy
from backend.tac.parser import parse_tac

from conftest import (
    CSE_TAC,
    DEAD_CODE_TAC,
    FOLDING_TAC,
    NO_OPPORTUNITY_TAC,
    requires_ml,
)

# ---------------------------------------------------------------------------
# Engine traceability (rule-based path - no model required)
# ---------------------------------------------------------------------------


def test_default_engine_run_is_still_rule_based():
    tac = parse_tac(DEAD_CODE_TAC)
    result = run_iterative_optimization(tac)
    assert result["selector"] == "rule_based"
    assert result["history"]
    for item in result["history"]:
        assert item["selector"] == "rule_based"
        if item["origin"] == "predicted":
            # The rule selector's own pick, executed verbatim.
            assert item["selected_strategy"] == select_strategy(item["tac_before"])
            assert item["predicted_strategy"] == item["selected_strategy"]
        else:
            # A controlled fallback: never the prediction, always a strategy
            # that actually has an opportunity on the same program state.
            assert item["origin"] == "fallback"
            assert item["selected_strategy"] != item["predicted_strategy"]
            assert item["selected_strategy"] in item["eligible_strategies"]
            assert "Fallback" in item["rationale"]
    # The first attempt is always the rule selector's own pick on the input.
    first = result["history"][0]
    assert first["origin"] == "predicted"
    assert first["selected_strategy"] == select_strategy(tac)


def test_selector_parameter_replaces_the_choice_without_touching_the_engine():
    tac = parse_tac(DEAD_CODE_TAC)
    always_fold = lambda t: "constant_folding"  # noqa: E731
    result = run_iterative_optimization(tac, selector=always_fold, selector_name="ml")
    assert result["selector"] == "ml"
    assert result["history"]
    # Every attempt records the custom selector's prediction verbatim…
    for item in result["history"]:
        assert item["predicted_strategy"] == "constant_folding"
        assert item["selector"] == "ml"
    # The first executed pass IS that prediction on the input program.
    assert result["history"][0]["selected_strategy"] == "constant_folding"
    assert result["history"][0]["origin"] == "predicted"
    # Every attempt stays internally consistent with the contract: when the
    # executed strategy IS the prediction the origin says "predicted"; any
    # other strategy ran as a labelled fallback, never as the selector's pick.
    for item in result["history"][1:]:
        if item["origin"] == "predicted":
            # A new selection round after an accepted transformation: the
            # selector was asked again on the changed program and its pick
            # (still constant_folding) was executed verbatim.
            assert item["selected_strategy"] == "constant_folding"
        else:
            assert item["origin"] == "fallback"
            assert item["selected_strategy"] != "constant_folding"
            assert item["predicted_strategy"] == "constant_folding"


def test_iteration_context_is_attached_and_receives_the_input_tac():
    """The context must see the input TAC, not the already-transformed result."""
    observed_lengths = []

    def context(tac, features, selected):
        observed_lengths.append(len(tac))
        return {"custom_field": "hello", "features_seen": len(features)}

    tac = parse_tac(FOLDING_TAC)
    result = run_iterative_optimization(tac, iteration_context=context)

    assert result["history"], "expected at least one iteration"
    for item in result["history"]:
        assert item["custom_field"] == "hello"
        assert item["features_seen"] == 8
        # features were extracted from `tac_before`, so the context must have
        # received a TAC of exactly that length.
        assert len(item["tac_before"]) == item["before_count"]

    assert observed_lengths
    assert observed_lengths == [item["before_count"] for item in result["history"]]


def test_tac_before_is_captured_before_the_pass_runs():
    original = "t1 = 5 + 3\nt2 = t1 * 2\nunused = t2 + 1"
    tac = parse_tac(original)
    result = run_iterative_optimization(tac)
    from backend.api.serialize import tac_to_source

    assert tac_to_source(result["history"][0]["tac_before"]) == original
    assert tac_to_source(tac) == original, "the caller's TAC must not be mutated"


def test_every_iteration_records_measurements_and_a_diff():
    tac = parse_tac(DEAD_CODE_TAC)
    result = run_iterative_optimization(tac)

    for item in result["history"]:
        assert item["operation_count_before"] >= 0
        assert item["operation_count_after"] >= 0
        assert isinstance(item["useful_improvement"], int)
        assert item["changed"] is True or item["improvement"] == 0
        assert item["accepted"] is True
        assert item["rationale"]
        if item["changed"]:
            assert item["diff"] is not None
            assert item["diff"]["changed"] is True
        # The final recorded iteration either stopped the loop or was cut off
        # by max_iterations; every earlier one promised to continue.
        if item is result["history"][-1]:
            assert item["will_continue"] is False
            assert item["stop_reason"] in {
                "no_further_useful_improvement",
                "no_instruction_reduction",
            } or result["stop_reason"] == "max_iterations_reached"
        else:
            assert item["will_continue"] is True
            assert item["stop_reason"] is None
        # before/after counts must agree with the recorded TACs.
        assert item["before_count"] == len(item["tac_before"])
        assert item["after_count"] == len(item["tac_after"])


def test_stop_reason_is_one_of_the_documented_conditions():
    known = {
        "no_further_useful_improvement",
        "no_instruction_reduction",
        "no_applicable_optimization",
        "max_iterations_reached",
    }
    for source in (DEAD_CODE_TAC, FOLDING_TAC, CSE_TAC, NO_OPPORTUNITY_TAC):
        result = run_iterative_optimization(parse_tac(source))
        assert result["stop_reason"] in known, result["stop_reason"]


def test_engine_never_exceeds_max_iterations():
    for limit in (1, 2, 3):
        result = run_iterative_optimization(parse_tac(DEAD_CODE_TAC), max_iterations=limit)
        assert result["total_iterations"] <= limit
        assert len(result["history"]) <= limit
        if result["total_iterations"] == limit and result["stop_reason"] == "max_iterations_reached":
            assert result["converged"] is False


def test_engine_totals_match_the_shared_measurement():
    tac = parse_tac(DEAD_CODE_TAC)
    result = run_iterative_optimization(tac)
    measured = measure_pass(tac, result["final_tac"])

    assert result["original_instruction_count"] == measured["original_instruction_count"]
    assert result["final_instruction_count"] == measured["optimized_instruction_count"]
    assert result["total_reduction"] == measured["instruction_reduction"]
    assert result["useful_improvement"] == measured["useful_improvement"]
    assert result["original_operation_count"] == operation_count(tac)
    assert result["final_operation_count"] == operation_count(result["final_tac"])


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_distinguishes_added_removed_modified_and_unchanged():
    """Each status is exercised by a case verified against the real differ."""
    # A rewrite plus a deleted line: modified + removed, one line untouched.
    diff = tac_diff(
        parse_tac("t1 = a + b\nt2 = a + b\nt3 = t1 + t2"),
        parse_tac("t1 = a + b\nt2 = t1"),
    )
    assert diff["changed"] is True
    assert diff["modified"] == 1
    assert diff["removed"] == 1
    assert diff["unchanged"] == 1
    assert diff["added"] == 0

    # Pure insertion: one line becomes two.
    diff = tac_diff(parse_tac("t1 = a + b"), parse_tac("t1 = a + b\nt2 = a + b"))
    assert diff["added"] == 1
    assert diff["removed"] == diff["modified"] == 0
    assert diff["unchanged"] == 1

    # Pure deletion.
    diff = tac_diff(parse_tac("t1 = a + b\nt2 = a + b"), parse_tac("t1 = a + b"))
    assert diff["removed"] == 1
    assert diff["added"] == diff["modified"] == 0
    assert diff["unchanged"] == 1

    # Pure in-place rewrite.
    diff = tac_diff(parse_tac("t1 = 5 + 3\nt2 = t1 * 2"), parse_tac("t1 = 8\nt2 = t1 * 2"))
    assert diff["modified"] == 1
    assert diff["added"] == diff["removed"] == 0
    assert diff["unchanged"] == 1

    # Every entry carries a status and the counts reconcile with them.
    for case in (diff,):
        statuses = [line["status"] for line in case["lines"]]
        assert statuses.count("modified") == case["modified"]
        assert statuses.count("unchanged") == case["unchanged"]


def test_diff_records_a_line_for_every_line_of_both_programs():
    before = parse_tac("t1 = a + b\nt2 = a + b\nt3 = t1 + t2")
    after = parse_tac("t1 = a + b\nt2 = t1")
    diff = tac_diff(before, after)
    assert len(diff["lines"]) >= max(len(before), len(after))
    for entry in diff["lines"]:
        assert entry["status"] in {"added", "removed", "modified", "unchanged"}
        assert entry["before_text"] is None or entry["before_index"] is not None
        assert entry["after_text"] is None or entry["after_index"] is not None


def test_diff_of_identical_tacs_reports_no_change():
    tac = parse_tac(FOLDING_TAC)
    diff = tac_diff(tac, parse_tac(FOLDING_TAC))
    assert diff["changed"] is False
    assert diff["added"] == diff["removed"] == diff["modified"] == 0
    assert diff["unchanged"] == len(tac)


def test_diff_never_claims_a_reduction_that_did_not_happen():
    """strength_reduction is a documented no-op - it must report exactly that."""
    tac = parse_tac("t3 = a * 2\nt4 = t3 + 1\n")
    outcome = apply_strategy_once(tac, "strength_reduction")

    assert outcome["applicable"] is True
    assert outcome["failed"] is False
    assert outcome["changed"] is False
    assert outcome["diff"]["changed"] is False
    assert outcome["diff"]["before_text"] == outcome["diff"]["after_text"]
    assert outcome["measurements"]["instruction_reduction"] == 0
    assert outcome["measurements"]["useful_improvement"] == 0
    assert outcome["after_text"] == outcome["before_text"]


def test_a_pass_that_declines_to_transform_is_not_a_failure():
    """Returning None is a legitimate no-op, not an error."""
    from backend.optimizer.measure import measure_pass

    tac = parse_tac("x = y\nz = x\n")
    outcome = apply_strategy_once(tac, "dead_code_elimination")

    assert outcome["applicable"] is True
    assert outcome["failed"] is False
    # Either it removed the dead store or it left the program alone; both must
    # be measured with the shared implementation rather than asserted here.
    if outcome["changed"]:
        assert outcome["measurements"] == measure_pass(tac, parse_tac(outcome["after_text"]))
        assert outcome["measurements"]["instruction_reduction"] > 0
    else:
        assert outcome["measurements"]["instruction_reduction"] == 0


# ---------------------------------------------------------------------------
# Single-pass execution helper
# ---------------------------------------------------------------------------


def test_apply_strategy_once_runs_on_an_independent_copy():
    tac = parse_tac(DEAD_CODE_TAC)
    before = str(tac)
    outcome = apply_strategy_once(tac, "dead_code_elimination")
    assert outcome["applicable"] is True
    assert outcome["failed"] is False
    assert outcome["changed"] is True
    assert str(tac) == before, "the original TAC must not be mutated"


def test_apply_strategy_once_reports_unknown_strategy_and_crashes_gracefully():
    tac = parse_tac(DEAD_CODE_TAC)
    unknown = apply_strategy_once(tac, "not_a_real_strategy")
    assert unknown["applicable"] is False
    assert unknown["failed"] is True
    assert "Unknown optimization strategy" in unknown["reason"]
    assert unknown["changed"] is False


def test_apply_strategy_once_measures_with_the_shared_implementation():
    tac = parse_tac(DEAD_CODE_TAC)
    outcome = apply_strategy_once(tac, "dead_code_elimination")
    expected = measure_pass(tac, parse_tac(outcome["after_text"]))
    assert outcome["measurements"] == expected


# ---------------------------------------------------------------------------
# Comparison (needs the trained model)
# ---------------------------------------------------------------------------

pytestmark = requires_ml


@pytest.fixture(scope="module")
def comparison(ml_artifacts):
    return compare_on_dataset(load_dataset(), split="test")


def test_rule_selector_is_the_real_rule_based_scorer(comparison):
    """Every recorded rule prediction must equal select_strategy() on that TAC."""
    from backend.tac.parser import parse_tac as _parse

    df = split_frames(load_dataset())["test"]
    by_id = dict(zip(df["sample_id"].astype(str), df["tac_program"].astype(str)))

    for row in comparison["per_program"]:
        expected = select_strategy(_parse(by_id[row["sample_id"]]))
        assert row["rule_strategy"] == expected, row["sample_id"]


def test_ml_selector_is_the_trained_model(comparison, model_payload):
    import numpy as np

    from backend.ml.dataset import FEATURE_KEYS as KEYS
    from backend.ml.dataset import TAC_COLUMN, ID_COLUMN, feature_frame

    df = split_frames(load_dataset())["test"]
    rows = df.set_index(ID_COLUMN)

    classes = [str(c) for c in model_payload["model"].classes_]
    for row in comparison["per_program"]:
        if row["ml_strategy"] is None:
            continue
        frame = feature_frame(
            {k: float(rows.loc[row["sample_id"], k]) for k in KEYS}
        )
        # Stored features were re-extracted from the TAC; use the stored ones
        # only to locate the row, and assert against a fresh model call.
        probabilities = model_payload["model"].predict_proba(frame)[0]
        assert row["ml_strategy"] == classes[int(np.argmax(probabilities))]


def test_both_selectors_start_from_the_same_original_tac(comparison, ml_artifacts):
    df = split_frames(load_dataset())["test"]
    from backend.ml.dataset import TAC_COLUMN, ID_COLUMN
    from backend.tac.parser import parse_tac as _parse

    by_id = dict(zip(df[ID_COLUMN].astype(str), df[TAC_COLUMN].astype(str)))
    for row in comparison["per_program"][:5]:
        original = _parse(by_id[row["sample_id"]])
        # Each run measured against the identical starting program.
        assert row["rule"]["original_instruction_count"] == len(original)
        assert row["ml"]["original_instruction_count"] == len(original)


def test_prediction_accuracy_is_derived_from_real_predictions(comparison):
    """Recompute accuracy from the recorded predictions and labels."""
    rows = comparison["per_program"]
    rule_correct = sum(1 for r in rows if r["rule_strategy"] == r["ground_truth"])
    ml_correct = sum(1 for r in rows if r["ml_strategy"] == r["ground_truth"])

    assert comparison["prediction"]["rule_based"]["evaluated_samples"] == len(rows)
    assert comparison["prediction"]["ml"]["evaluated_samples"] == len(rows)
    # Stored metrics are rounded to six decimals, so the recomputed value may
    # only match to that precision.
    assert comparison["prediction"]["rule_based"]["accuracy"] == pytest.approx(
        rule_correct / len(rows), abs=1e-5
    )
    assert comparison["prediction"]["ml"]["accuracy"] == pytest.approx(
        ml_correct / len(rows), abs=1e-5
    )

    # The stored label was NOT used as a prediction: the predicted distribution
    # differs from the actual one for at least one selector.
    actual = Counter(r["ground_truth"] for r in rows)
    predicted = Counter(r["rule_strategy"] for r in rows)
    assert dict(predicted) != dict(actual) or dict(
        Counter(r["ml_strategy"] for r in rows)
    ) != dict(actual)


def test_prediction_and_optimization_blocks_are_separate(comparison):
    assert set(comparison) >= {"prediction", "optimization", "methodology", "per_program"}
    # Prediction metrics contain no reduction numbers.
    for block in comparison["prediction"].values():
        if isinstance(block, dict) and "accuracy" in block:
            assert "mean_useful_improvement" not in block
            assert "instruction_reduction" not in block
    # Optimization metrics contain no prediction metrics.
    for key in ("rule_based", "ml"):
        assert "accuracy" not in comparison["optimization"][key]
        assert "macro_f1" not in comparison["optimization"][key]
    assert "never combined" in comparison["methodology"]["separation"].lower()


def test_program_winners_follow_the_measured_outcome(comparison):
    wins = Counter()
    for row in comparison["per_program"]:
        rule_gain = row["rule"]["useful_improvement"]
        ml_gain = row["ml"]["useful_improvement"]
        expected = RULE_WINS if rule_gain > ml_gain else (ML_WINS if ml_gain > rule_gain else TIE)
        assert row["program_winner"] == expected, row["sample_id"]
        assert row["rule_correct"] == (row["rule_strategy"] == row["ground_truth"])
        assert row["ml_correct"] == (row["ml_strategy"] == row["ground_truth"])
        wins[expected] += 1

    assert comparison["optimization"]["program_wins"]["rule_based"] == wins[RULE_WINS]
    assert comparison["optimization"]["program_wins"]["ml"] == wins[ML_WINS]
    assert comparison["optimization"]["program_wins"]["tie"] == wins[TIE]


def test_overall_winner_follows_mean_useful_improvement(comparison):
    opt = comparison["optimization"]
    rule_mean = opt["rule_based"]["mean_useful_improvement"]
    ml_mean = opt["ml"]["mean_useful_improvement"]
    if rule_mean > ml_mean:
        assert opt["overall_winner"] == RULE_WINS
    elif ml_mean > rule_mean:
        assert opt["overall_winner"] == ML_WINS
    else:
        assert opt["overall_winner"] == TIE


def test_strength_reduction_runs_are_honest_noops(comparison):
    rows = [r for r in comparison["per_program"] if r["rule_strategy"] == "strength_reduction"]
    for row in rows:
        assert row["rule"]["useful_improvement"] == 0
        assert row["rule"]["changed"] is False


def test_comparison_reports_data_quality(comparison):
    quality = comparison["data_quality"]
    assert quality["parse_failures"] == [], quality["parse_failures"]
    assert quality["feature_mismatches"] == [], quality["feature_mismatches"]


def test_comparison_documents_its_limitations(comparison):
    text = " ".join(comparison["limitations"]).lower()
    assert "held-out" in text or "held out" in text
    assert "strength_reduction" in text
    assert "useful_improvement" in text
    assert comparison["methodology"]["prediction"]
    assert comparison["methodology"]["optimization"]


# ---------------------------------------------------------------------------
# Live comparison of one program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", [DEAD_CODE_TAC, FOLDING_TAC, CSE_TAC, NO_OPPORTUNITY_TAC])
def test_live_compare_runs_both_selectors_independently(source, ml_artifacts):
    result = compare_source(source)
    original = parse_tac(source)

    assert result["available"] is True
    assert result["original_instruction_count"] == len(original)
    # Neither run may mutate the shared original.
    assert str(parse_tac(source)) == str(original)

    for side in ("rule_based", "ml"):
        block = result[side]
        assert block["available"] is True
        assert block["history"], f"{side} produced no iterations"
        assert block["measurements"]["original_instruction_count"] == len(original)
        # useful_improvement is the composite measure: statements removed plus
        # operations removed, using the same shared implementation for both.
        assert block["useful_improvement"] == (
            block["measurements"]["instruction_reduction"]
            + block["measurements"]["operation_reduction"]
        )
        assert block["original_operation_count"] == operation_count(original)


def test_live_compare_winner_is_decided_by_measurement(ml_artifacts):
    result = compare_source(DEAD_CODE_TAC)
    gains = {
        RULE_WINS: result["rule_based"]["useful_improvement"],
        ML_WINS: result["ml"]["useful_improvement"],
        FEEDBACK_WINS: result["feedback"]["useful_improvement"],
    }
    best = max(gains.values())
    leaders = [key for key, gain in gains.items() if gain == best]

    assert result["winner"] in {RULE_WINS, ML_WINS, FEEDBACK_WINS, TIE}
    # Unique highest measured gain wins; anything shared becomes a tie.
    assert result["winner"] == (leaders[0] if len(leaders) == 1 else TIE)
    assert "measured useful_improvement" in result["decided_by"]


def test_live_compare_agreement_covers_the_whole_sequence(ml_artifacts):
    result = compare_source(DEAD_CODE_TAC)
    agreement = result["agreement"]
    rule_seq = [h["selected_strategy"] for h in result["rule_based"]["history"]]
    ml_seq = [h["selected_strategy"] for h in result["ml"]["history"]]

    assert agreement["rule_strategy_sequence"] == rule_seq
    assert agreement["ml_strategy_sequence"] == ml_seq
    assert agreement["strategies_agree"] == (rule_seq == ml_seq)
    assert agreement["initial_strategies_agree"] == (
        agreement["rule_strategy"] == agreement["ml_strategy"]
    )
    assert agreement["explanation"]
    # The explanation must match the facts it describes.
    if agreement["strategies_agree"]:
        assert "identical" in agreement["explanation"]
    elif agreement["diverged_at_iteration"]:
        assert f"iteration {agreement['diverged_at_iteration']}" in agreement["explanation"]


def test_live_compare_records_what_the_other_selector_would_pick(ml_artifacts):
    result = compare_source(DEAD_CODE_TAC)
    for side in ("rule_based", "ml"):
        history = result[side]["history"]
        for item in history:
            assert item["rule_selected_strategy"] in {
                "constant_folding",
                "constant_propagation",
                "common_subexpression",
                "dead_code_elimination",
                "algebraic_simplification",
                "strength_reduction",
            }
            assert item["ml_predicted_strategy"] is not None
            assert item["strategies_agree"] == (
                item["ml_predicted_strategy"] == item["rule_selected_strategy"]
            )
            assert item["shap"]["available"] is True
            assert len(item["shap"]["contributions"]) == 8


def test_live_compare_can_omit_shap(ml_artifacts):
    result = compare_source(DEAD_CODE_TAC, include_shap=False)
    for side in ("rule_based", "ml"):
        assert result[side]["history"][0].get("shap") is None
    # Predictions are still real.
    assert result["rule_based"]["history"][0]["ml_predicted_strategy"]


def test_each_compare_block_returns_the_code_the_diff_panels_render(ml_artifacts):
    """The Comparison screen draws one before/after diff per arm straight from
    these fields, so each one must describe exactly that arm's own run against
    the shared original - never the other arm's output."""
    from backend.api.serialize import tac_to_lines, tac_to_source

    result = compare_source(CSE_TAC)
    original = parse_tac(CSE_TAC)

    # One shared original, line for line, taken from THIS request.
    assert result["original_lines"] == tac_to_lines(original)
    assert result["original_instruction_count"] == len(original)

    for side in ("rule_based", "ml"):
        block = result[side]
        assert block["available"] is True
        final_lines = block["final_lines"]

        # final_lines / final_text are that run's own final TAC.
        assert "\n".join(final_lines) == block["final_text"]
        # The diff is (shared original) -> (this arm's final TAC).
        assert block["diff"]["before_text"] == tac_to_source(original)
        assert block["diff"]["after_text"] == block["final_text"]
        # The counts rendered next to the diff come from those same two TACs.
        assert block["diff"]["before_instruction_count"] == len(original)
        assert block["diff"]["after_instruction_count"] == len(final_lines)
        assert block["original_instruction_count"] == len(original)
        assert block["final_instruction_count"] == len(final_lines)
        assert block["measurements"]["original_instruction_count"] == len(original)
        assert block["measurements"]["optimized_instruction_count"] == len(final_lines)


def test_compare_results_correspond_to_the_request_that_was_just_made(ml_artifacts):
    """Back-to-back comparisons with different inputs must each return their
    own original: no state, cache or stale output may leak between requests."""
    from backend.api.serialize import tac_to_source

    first_source = tac_to_source(parse_tac(DEAD_CODE_TAC))
    second_source = tac_to_source(parse_tac(FOLDING_TAC))
    assert first_source != second_source

    first = compare_source(DEAD_CODE_TAC, include_shap=False)
    second = compare_source(FOLDING_TAC, include_shap=False)

    assert "\n".join(first["original_lines"]) == first_source
    assert "\n".join(second["original_lines"]) == second_source

    for side in ("rule_based", "ml"):
        if not second[side]["available"] or not second[side]["history"]:
            continue
        # The diff and the first iteration both start from this request's TAC.
        assert second[side]["diff"]["before_text"] == second_source
        assert second[side]["history"][0]["tac_before"] == second["original_lines"]


def test_live_compare_reports_unavailable_model_without_a_fake_winner(monkeypatch, ml_artifacts):
    from pathlib import Path

    from backend.ml import model as model_mod

    monkeypatch.setattr(model_mod, "MODEL_FILE", Path("no", "such", "model.joblib"))
    try:
        model_mod.clear_cache()
        result = compare_source(DEAD_CODE_TAC)
    finally:
        model_mod.clear_cache()

    assert result["available"] is True
    assert result["ml"]["available"] is False
    assert result["ml"]["history"] == []
    assert result["ml"]["useful_improvement"] == 0
    # No fabricated comparison result.
    assert result["winner"] is None
    assert "unavailable" in result["winner_note"].lower()
    assert result["agreement"]["strategies_agree"] is None
    assert result["agreement"]["ml_strategy"] is None
    # The Rule-Based side still works.
    assert result["rule_based"]["available"] is True
    assert result["rule_based"]["history"]


# ---------------------------------------------------------------------------
# Before/after diff panels: the two states the faculty review flagged.
# Every number the Comparison screen renders must be the backend's own
# measurement of the same two TACs the panel draws.
# ---------------------------------------------------------------------------

# The Compare screen's default input: nine statements, every one of them dead
# (no result is ever read), so all three runs genuinely end at an empty TAC.
FACULTY9_TAC = """\
t1 = 10
t2 = 20
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
t6 = 100
t7 = t6 + 50
unused = t7
t8 = t3 * 2"""

# Labels, jumps and a conditional over live variables: no pass can change this
# program, so every run must end exactly where it started.
IDENTICAL_TAC = """
i = 0
limit = n
loop:
i = i + 1
if i < limit goto loop
"""


def _assert_side_accounting(block, original_lines):
    """A diff's row counts must tile both sides exactly as rendered."""
    diff = block["diff"]
    final_lines = block["final_lines"]
    assert diff["before_instruction_count"] == len(original_lines)
    assert diff["after_instruction_count"] == len(final_lines)
    # Before side: every original line is removed, modified or unchanged.
    assert diff["removed"] + diff["modified"] + diff["unchanged"] == len(original_lines)
    # After side: every final line is added, modified or unchanged.
    assert diff["added"] + diff["modified"] + diff["unchanged"] == len(final_lines)


def test_identical_final_tac_reports_zero_removed_and_zero_percent(ml_artifacts):
    """When no pass changes the program, the panels must say 0 removed / 0.0% -
    the state the faculty screenshot showed as 'nine removed, 100%'."""
    result = compare_source(IDENTICAL_TAC, include_shap=False)
    original_lines = result["original_lines"]
    assert original_lines, "the loop program must parse to a non-empty TAC"

    for side in ("rule_based", "ml", "feedback"):
        block = result[side]
        assert block["available"] is True, f"{side} did not run"
        # Before and after are the same code, line for line.
        assert block["final_lines"] == original_lines
        assert block["original_instruction_count"] == len(original_lines)
        assert block["final_instruction_count"] == len(original_lines)
        assert block["total_reduction"] == 0
        assert block["useful_improvement"] == 0
        assert block["changed"] is False
        # Nothing removed, added or modified - everything is context.
        assert block["diff"]["removed"] == 0
        assert block["diff"]["added"] == 0
        assert block["diff"]["modified"] == 0
        assert block["diff"]["unchanged"] == len(original_lines)
        assert block["diff"]["changed"] is False
        _assert_side_accounting(block, original_lines)
        # The reduction the frontend computes: (original - final) / original.
        original, final = block["original_instruction_count"], block["final_instruction_count"]
        assert round((original - final) / original * 100, 1) == 0.0


def test_all_dead_program_reports_a_genuinely_empty_final_tac(ml_artifacts):
    """The other flagged state: nine statements highlighted as removed with
    100% reduction. The measurement is real (every result is unused), and the
    counts must match the rendered lines exactly - never a code panel that
    claims removals the final TAC does not show."""
    result = compare_source(FACULTY9_TAC, include_shap=False)
    original_lines = result["original_lines"]
    assert len(original_lines) == 9
    assert result["original_instruction_count"] == 9

    for side in ("rule_based", "ml", "feedback"):
        block = result[side]
        assert block["available"] is True, f"{side} did not run"
        # The actual final TAC of an all-dead program is empty - and the panel
        # receives exactly that, not the original lines recycled.
        assert block["final_lines"] == []
        assert block["final_text"] == ""
        assert block["original_instruction_count"] == 9
        assert block["final_instruction_count"] == 0
        assert block["measurements"]["optimized_instruction_count"] == 0
        assert block["total_reduction"] == 9
        # All nine rows are removals; nothing is invented on the after side.
        assert block["diff"]["removed"] == 9
        assert block["diff"]["added"] == 0
        assert block["diff"]["modified"] == 0
        assert block["diff"]["unchanged"] == 0
        _assert_side_accounting(block, original_lines)
        original, final = block["original_instruction_count"], block["final_instruction_count"]
        assert round((original - final) / original * 100, 1) == 100.0
        # useful_improvement is the real measured value for this empty result.
        assert block["useful_improvement"] == work_remaining(parse_tac(FACULTY9_TAC))


@pytest.mark.parametrize(
    "source",
    [DEAD_CODE_TAC, FOLDING_TAC, CSE_TAC, NO_OPPORTUNITY_TAC, IDENTICAL_TAC, FACULTY9_TAC],
)
def test_diff_counts_always_match_the_rendered_lines(source, ml_artifacts):
    """For identical AND genuinely optimized outputs alike: the counts beside
    the diff come from the same pair of TACs the diff lines are drawn from,
    on both sides of the comparison."""
    result = compare_source(source, include_shap=False)
    original_lines = result["original_lines"]

    for side in ("rule_based", "ml", "feedback"):
        block = result[side]
        assert block["available"] is True, f"{side} did not run"
        assert len(block["final_lines"]) == block["final_instruction_count"]
        assert "\n".join(block["final_lines"]) == block["final_text"]
        _assert_side_accounting(block, original_lines)


def test_empty_tac_is_rejected_by_the_compare_endpoint(client):
    """An empty program is answered with a clear 400, never a crash or a
    fabricated diff."""
    for bad in ("", "   ", "\n\t"):
        response = client.post("/api/compare", json={"tac": bad})
        assert response.status_code == 400, f"tac={bad!r} -> {response.status_code}"


# ---------------------------------------------------------------------------
# Optimization metrics on each run block (§7): pass accounting, measured
# runtime and an explicit non-claim about semantic correctness.
# ---------------------------------------------------------------------------


def test_run_blocks_report_pass_counts_and_a_measured_runtime(ml_artifacts):
    result = compare_source(FOLDING_TAC, include_shap=False)

    for side in ("rule_based", "ml", "feedback"):
        block = result[side]
        assert block["available"] is True, f"{side} did not run"
        history = block["history"]

        # Every count is read from the same history the trace renders.
        assert block["passes_attempted"] == len(history) == block["iterations"]
        assert block["passes_changed"] == sum(1 for item in history if item["changed"])
        assert block["passes_unchanged"] == (
            block["passes_attempted"] - block["passes_changed"]
        )
        assert block["passes_accepted"] == sum(
            1 for item in history if item.get("accepted")
        )
        assert (
            block["passes_attempted"]
            == block["passes_changed"] + block["passes_unchanged"]
        )

        # Wall-clock time actually measured around this run (single request).
        assert isinstance(block["runtime_ms"], float)
        assert block["runtime_ms"] >= 0.0

        # A real semantic check ran: bounded equivalence of the final program
        # against the original, with the observed variables named.
        assert block["correctness_checked"] is True
        correctness = block["correctness"]
        assert correctness["status"] in {"equivalent", "not_equivalent", "inconclusive"}
        assert correctness["note"]


def test_never_executed_run_claims_no_time_and_no_passes(monkeypatch, ml_artifacts):
    from pathlib import Path

    from backend.ml import model as model_mod

    monkeypatch.setattr(model_mod, "MODEL_FILE", Path("no", "such", "model.joblib"))
    try:
        model_mod.clear_cache()
        result = compare_source(DEAD_CODE_TAC, include_shap=False)
    finally:
        model_mod.clear_cache()

    block = result["ml"]
    assert block["available"] is False
    assert block["runtime_ms"] is None
    assert block["passes_attempted"] == 0
    assert block["passes_changed"] == 0
    assert block["passes_unchanged"] == 0
    assert block["passes_accepted"] == 0
    assert block["correctness_checked"] is False
    assert block["correctness"]["status"] == "not_run"


def test_executed_run_reports_real_semantic_verdict(ml_artifacts):
    """The folding demo converges to `t2 = 30`, and with a print line the
    final program still observes that value — so the checker must confirm
    equivalence on the real observable (t2, via its printed output) rather
    than just asserting a flag."""
    result = compare_source(FOLDING_TAC + "print t2\n", include_shap=False)
    block = result["rule_based"]
    assert block["correctness_checked"] is True
    assert block["final_lines"] == ["t2 = 30", "print t2"]
    assert block["converged"] is True
    correctness = block["correctness"]
    assert correctness["status"] == "equivalent", correctness["note"]
    assert "t2" in correctness["observed_vars"]
    assert correctness["outputs_compared"] is True


def test_aggregate_outcome_breaks_results_down_per_strategy():
    from backend.ml.evaluate import aggregate_outcome

    rows = [
        {
            "strategy": "dead_code_elimination",
            "useful_improvement": 2,
            "instruction_reduction": 1,
            "operation_reduction": 1,
        },
        {
            "strategy": "dead_code_elimination",
            "useful_improvement": 0,
            "instruction_reduction": 0,
            "operation_reduction": 0,
        },
        {
            "strategy": "constant_folding",
            "useful_improvement": 3,
            "instruction_reduction": 1,
            "operation_reduction": 2,
        },
    ]
    aggregate = aggregate_outcome(rows)

    assert aggregate["per_strategy"]["dead_code_elimination"] == {
        "programs": 2,
        "mean_useful_improvement": 1.0,
        "improved_programs": 1,
    }
    assert aggregate["per_strategy"]["constant_folding"] == {
        "programs": 1,
        "mean_useful_improvement": 3.0,
        "improved_programs": 1,
    }
    # A strategy the selector never executed is absent, not zero: "never
    # used" must not read as "used and measured 0".
    assert "strength_reduction" not in aggregate["per_strategy"]
