"""
Tests for the measurement helpers and the dataset generator.

Covers: pass measurement, the best_strategy labelling rule, reproducibility,
dataset validation, duplicate detection, split integrity, and the honest
reporting of strength_reduction's measured zero.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS
from backend.optimizer.measure import (
    best_strategy_from,
    measure_pass,
    operation_count,
    work_remaining,
)
from backend.scoring.rule_based import STRATEGY_TIE_ORDER
from backend.tac.parser import parse_tac

from generate_dataset import FEATURE_KEYS, STRATEGIES, generate


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def test_operation_count_and_work_remaining():
    tac = parse_tac("t1 = 5 + 3\nacc = 0\nacc = acc + t1")
    # ADD, ADD are operations; `acc = 0` is a literal definition.
    assert operation_count(tac) == 2
    # 3 statements + 2 operations
    assert work_remaining(tac) == 5


def test_work_remaining_is_zero_for_empty_tac():
    assert work_remaining(parse_tac("")) == 0
    assert operation_count(parse_tac("")) == 0


def test_measure_pass_reports_all_required_metrics():
    original = parse_tac("t1 = 5 + 3\nunused = 999")
    optimized = parse_tac("t1 = 5 + 3")

    stats = measure_pass(original, optimized)
    assert stats["original_instruction_count"] == 2
    assert stats["optimized_instruction_count"] == 1
    assert stats["instruction_reduction"] == 1
    assert stats["reduction_percentage"] == 50.0
    # one statement removed, which carried no operation
    assert stats["operation_reduction"] == 0
    assert stats["useful_improvement"] == 1


def test_measure_pass_never_divides_by_zero():
    stats = measure_pass(parse_tac(""), parse_tac(""))
    assert stats["reduction_percentage"] == 0.0
    assert stats["instruction_reduction"] == 0
    assert stats["useful_improvement"] == 0


def test_measure_pass_credits_operation_removal():
    """A 1:1 rewrite still counts as useful work removed."""
    original = parse_tac("t1 = 5 + 3")
    optimized = parse_tac("t1 = 8")

    stats = measure_pass(original, optimized)
    assert stats["instruction_reduction"] == 0
    assert stats["reduction_percentage"] == 0.0
    assert stats["operation_reduction"] == 1
    assert stats["useful_improvement"] == 1


# ---------------------------------------------------------------------------
# Labelling rule
# ---------------------------------------------------------------------------


def _measurement(improvements: dict) -> dict:
    return {name: {"useful_improvement": value} for name, value in improvements.items()}


def test_best_strategy_takes_the_maximum():
    measurements = _measurement({name: 0 for name in STRATEGIES})
    measurements["common_subexpression"] = {"useful_improvement": 4}
    assert best_strategy_from(measurements, STRATEGY_TIE_ORDER) == "common_subexpression"


def test_best_strategy_breaks_ties_deterministically():
    measurements = _measurement({name: 2 for name in STRATEGIES})
    expected = best_strategy_from(measurements, STRATEGY_TIE_ORDER)
    assert expected == STRATEGY_TIE_ORDER[0]
    # repeated calls must agree - never random
    assert all(best_strategy_from(measurements, STRATEGY_TIE_ORDER) == expected
               for _ in range(5))


def test_best_strategy_rejects_empty_measurements():
    import pytest

    with pytest.raises(ValueError):
        best_strategy_from({}, STRATEGY_TIE_ORDER)


def test_strategy_ids_match_the_backend_engine():
    assert STRATEGIES == list(OPTIMIZATION_FUNCTIONS.keys())
    assert STRATEGIES == STRATEGY_TIE_ORDER
    assert len(FEATURE_KEYS) == 8


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _small(tmp_path, samples=20):
    return generate(target_samples=samples, out_root=tmp_path)


def test_generator_produces_a_valid_dataset(tmp_path):
    result = _small(tmp_path)
    report = result["validation"]

    assert report["passed"] is True, report["errors"]
    assert report["errors"] == []
    assert report["records_checked"] == 20
    assert result["record_count"] == 20


def test_generator_writes_the_expected_outputs(tmp_path):
    _small(tmp_path)
    for relative in (
        "raw/tac_programs.json",
        "processed/optimization_dataset.csv",
        "processed/optimization_dataset.json",
        "metadata/summary.json",
        "metadata/validation_report.json",
        "metadata/splits.json",
    ):
        path = tmp_path / relative
        assert path.is_file(), f"missing output {relative}"
        assert path.stat().st_size > 0, f"empty output {relative}"


def test_every_sample_parses_and_round_trips(tmp_path):
    import csv

    result = _small(tmp_path)
    assert result["validation"]["errors"] == []

    with (tmp_path / "processed" / "optimization_dataset.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 20
    for row in rows:
        source_lines = [
            line for line in row["tac_program"].splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        parsed = parse_tac(row["tac_program"])
        # no line was silently dropped by the parser
        assert len(parsed) == len(source_lines), row["sample_id"]
        assert int(row["num_statements"]) == len(parsed)


def test_labels_are_re_derived_from_stored_measurements(tmp_path):
    """best_strategy must equal argmax of the six stored improvements."""
    result = _small(tmp_path)
    records = result["validation"]["label_counts"]
    assert records  # sanity

    import json

    data = json.loads(
        (tmp_path / "processed" / "optimization_dataset.json").read_text(encoding="utf-8")
    )
    for row in data["records"]:
        improvements = {
            name: {"useful_improvement": row[f"{name}_useful_improvement"]}
            for name in STRATEGIES
        }
        assert best_strategy_from(improvements, STRATEGY_TIE_ORDER) == row["best_strategy"]


def test_dataset_has_no_duplicate_programs(tmp_path):
    import csv

    _small(tmp_path)
    with (tmp_path / "processed" / "optimization_dataset.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    programs = [row["tac_program"] for row in rows]
    assert len(set(programs)) == len(programs)


def test_generator_is_reproducible(tmp_path):
    first = _small(tmp_path / "a")["summary"]
    second = _small(tmp_path / "b")["summary"]
    assert first["label_distribution"] == second["label_distribution"]
    assert first["total_samples"] == second["total_samples"]


def test_split_is_a_disjoint_partition_of_the_records(tmp_path):
    import json

    _small(tmp_path)
    splits = json.loads(
        (tmp_path / "metadata" / "splits.json").read_text(encoding="utf-8")
    )["splits"]

    ids = splits["train"] + splits["validation"] + splits["test"]
    assert len(ids) == 20
    assert len(set(ids)) == 20, "a sample appeared in more than one split"
    # 70/15/15 of 20 -> 14 / 3 / 3
    assert len(splits["train"]) == 14
    assert len(splits["validation"]) == 3
    assert len(splits["test"]) == 3


def test_strength_reduction_is_reported_as_a_honest_zero(tmp_path):
    """SR is a documented no-op; the dataset must record its real 0 and warn,
    never invent a win for it."""
    result = _small(tmp_path)
    report = result["validation"]

    assert report["label_counts"]["strength_reduction"] == 0
    assert any("strength_reduction" in warning for warning in report["warnings"])


def test_features_cover_the_required_eight_keys(tmp_path):
    import json

    _small(tmp_path)
    data = json.loads(
        (tmp_path / "processed" / "optimization_dataset.json").read_text(encoding="utf-8")
    )
    assert data["feature_keys"] == FEATURE_KEYS
    assert data["strategy_ids"] == STRATEGIES
    assert data["label"] == "best_strategy"
