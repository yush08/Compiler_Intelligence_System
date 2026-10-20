"""Optimization-Technique Performance Matrix tests.

The matrix must always be *measured*, never stored or invented:

* every cell equals a fresh independent execution of that one pass over the
  program's original TAC (``apply_strategy_once`` deepcopies internally, so
  no cell can be another pass's output);
* the best strategy of a row is the largest measured value, and ties are
  flagged instead of being given a unique winner;
* pagination and filtering operate on the live dataset;
* a failing pass is a distinct ``failed`` status with no numeric value - it
  is never displayed as zero;
* a missing dataset yields an explicit ``available: false``.
"""

from __future__ import annotations

import math

import pytest

from backend.api import matrix_service
from backend.ml.compare import apply_strategy_once
from backend.ml.dataset import ID_COLUMN, SPLIT_COLUMN, STRATEGIES, load_dataset
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS
from backend.tac.parser import parse_tac

from conftest import requires_ml

pytestmark = requires_ml


def _matrix(client, **params):
    response = client.get("/api/optimization-matrix", params=params)
    assert response.status_code == 200, response.status_code
    return response.json()


def test_matrix_exposes_six_columns_the_metric_and_live_data_source(client):
    payload = _matrix(client, page=1, size=5)

    assert payload["available"] is True
    assert [entry["id"] for entry in payload["strategies"]] == list(STRATEGIES)
    assert len(payload["strategies"]) == 6
    assert all(entry["label"] for entry in payload["strategies"])

    metric = payload["metric"]
    assert metric["key"] == "useful_improvement"
    assert "work_remaining" in metric["definition"]
    assert "Ties keep every tied strategy" in metric["best_selection"]

    assert payload["data_source"]["kind"] == "live"
    assert "on demand" in payload["data_source"]["note"]
    # Pagination fields describe the real dataset.
    assert payload["total"] == len(load_dataset())


def test_every_cell_equals_an_independent_execution_of_that_pass(client):
    payload = _matrix(client, page=1, size=3)
    assert payload["rows"]

    for row in payload["rows"]:
        assert set(row["cells"]) == set(STRATEGIES)
        original = parse_tac("\n".join(row["tac_lines"]))
        for strategy in STRATEGIES:
            # A fresh, independent run of just this pass over the original.
            expected = apply_strategy_once(original, strategy)
            cell = row["cells"][strategy]
            if expected["failed"]:
                assert cell["status"] == "failed"
                assert cell["useful_improvement"] is None
            elif not expected["applicable"]:
                assert cell["status"] == "unsupported"
            else:
                assert cell["status"] == "measured"
                assert cell["useful_improvement"] == int(
                    expected["measurements"]["useful_improvement"]
                )
                assert cell["changed"] == expected["changed"]


def test_best_strategy_of_a_row_is_the_measured_maximum_with_ties_flagged(client):
    payload = _matrix(client, page=1, size=10)

    checked = 0
    for row in payload["rows"]:
        if "error" in row:
            continue
        values = {
            name: cell["useful_improvement"]
            for name, cell in row["cells"].items()
            if cell["status"] == "measured"
        }
        assert values, f"row {row['sample_id']} has no measured cell"
        top = max(values.values())
        expected_tied = [name for name in STRATEGIES if values.get(name) == top]

        best = row["best"]
        assert best is not None
        assert best["value"] == top
        assert best["strategies"] == expected_tied
        assert best["tie"] == (len(expected_tied) > 1)
        checked += 1
    assert checked > 0


def test_matrix_paginates_and_filters_over_the_live_dataset(client):
    dataset = load_dataset()

    first = _matrix(client, page=1, size=5)
    second = _matrix(client, page=2, size=5)
    assert len(first["rows"]) == 5
    assert first["pages"] == math.ceil(len(dataset) / 5)
    assert first["page"] == 1 and second["page"] == 2
    first_ids = [row["sample_id"] for row in first["rows"]]
    second_ids = [row["sample_id"] for row in second["rows"]]
    assert not set(first_ids) & set(second_ids), "pages must not overlap"

    # Search narrows to the exact sample.
    target = first_ids[0]
    found = _matrix(client, q=target)
    assert found["total"] == 1
    assert found["rows"][0]["sample_id"] == target

    # Split filter keeps only rows from that split.
    test_split = _matrix(client, split="test", size=50)
    assert test_split["rows"], "test split must not be empty"
    assert {row["split"] for row in test_split["rows"]} == {"test"}


def test_a_failing_pass_reports_failed_and_never_zero(client, monkeypatch):
    payload = _matrix(client, page=1, size=1)
    sample_id = payload["rows"][0]["sample_id"]

    def boom(_tac):
        raise RuntimeError("pass exploded in the test")

    monkeypatch.setitem(OPTIMIZATION_FUNCTIONS, "constant_folding", boom)
    after = _matrix(client, q=sample_id)
    cell = after["rows"][0]["cells"]["constant_folding"]

    assert cell["status"] == "failed"
    assert cell["useful_improvement"] is None
    assert "pass exploded in the test" in cell["reason"]
    # The other cells are untouched and still measured.
    assert after["rows"][0]["cells"]["dead_code_elimination"]["status"] == "measured"


def test_missing_dataset_reports_available_false_with_a_reason(client, monkeypatch):
    def gone():
        raise FileNotFoundError("Dataset not found. Generate it with: python experiments/generate_dataset.py")

    monkeypatch.setattr(matrix_service.dataset_service, "_load", gone)
    payload = _matrix(client)
    assert payload["available"] is False
    assert "generate_dataset" in payload["reason"].lower()
    assert payload["rows"] == []
    assert payload["strategies"] == []
