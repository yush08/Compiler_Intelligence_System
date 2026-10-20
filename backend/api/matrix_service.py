"""Optimization-Technique Performance Matrix.

A table of *live* per-pass measurements: rows are dataset programs, columns
are the six optimization techniques, and every cell is the result of running
that single technique on an **independent deepcopy of the program's original
TAC** (``ml.compare.apply_strategy_once``).  One technique's output is never
the next technique's input, so the matrix compares individual passes on a
common input rather than an order-dependent chain.

Cell values follow the project's shared measurement conventions:

* ``measured`` - the pass ran; ``useful_improvement`` is the real measured
  value (0 when the pass genuinely changed nothing);
* ``failed``   - the pass raised; the cell carries the error text and
  ``useful_improvement`` stays ``None`` so a crash is never displayed as 0;
* ``unsupported`` - the engine has no function for that strategy (also
  ``None``, never a silent zero).

Rows are paginated and searchable exactly like the dataset sample list, so
the matrix always reflects the current dataset - never a hand-written table.
"""

from __future__ import annotations

from ..ml.compare import apply_strategy_once
from ..ml.dataset import (
    DEFAULT_DATASET_PATH,
    ID_COLUMN,
    SPLIT_COLUMN,
    TAC_COLUMN,
    STRATEGIES,
)
from ..api.serialize import STRATEGY_DISPLAY
from ..tac.parser import parse_tac
from . import dataset_service

__all__ = ["optimization_matrix"]

# Metric contract - shown above the table so every cell is interpretable.
_METRIC = {
    "key": "useful_improvement",
    "label": "Useful improvement",
    "unit": "work units removed (statements + operations)",
    "definition": (
        "work_remaining(original TAC) − work_remaining(TAC after this single "
        "pass), computed by optimizer.measure.measure_pass. One work unit is "
        "one statement or one still-to-evaluate operation."
    ),
    "best_selection": (
        "The best strategy of a row is the measured cell with the largest "
        "useful improvement. Ties keep every tied strategy listed and are "
        "flagged as ties - a tie has no unique winner."
    ),
    "single_pass_note": (
        "Each cell is ONE pass applied once to the original program. The "
        "full iterative optimizer on the Workbench combines passes until "
        "convergence, so its totals can be larger than any single cell."
    ),
}

_LIMITATIONS = [
    "strength_reduction is a documented no-op in this project: it measures 0 "
    "wherever it runs and stays out of the prediction vocabulary until it "
    "genuinely wins somewhere.",
    "A zero means the pass really changed nothing; a failed pass is shown as "
    "an error status, never as zero.",
    "Cells are measured on demand from the current dataset file; regenerating "
    "the dataset changes the matrix with it.",
]


def _filter_sort(df, q: str, split: str | None):
    filtered = df
    if split in {"train", "validation", "test"}:
        filtered = filtered[filtered[SPLIT_COLUMN] == split]

    query = (q or "").strip().lower()
    if query:
        mask = (
            filtered[ID_COLUMN].astype(str).str.lower().str.contains(query, regex=False)
            | filtered["generation_category"].astype(str).str.lower().str.contains(query, regex=False)
            | filtered[TAC_COLUMN].astype(str).str.lower().str.contains(query, regex=False)
        )
        filtered = filtered[mask]

    return filtered.sort_values(ID_COLUMN, kind="mergesort")


def _cell(record: dict) -> dict:
    """Map one apply_strategy_once record onto an honest matrix cell."""
    if not record["applicable"]:
        return {
            "status": "unsupported",
            "useful_improvement": None,
            "instruction_reduction": None,
            "reduction_percentage": None,
            "changed": False,
            "reason": record["reason"] or "This strategy is not implemented in the engine.",
        }
    if record["failed"]:
        return {
            "status": "failed",
            "useful_improvement": None,
            "instruction_reduction": None,
            "reduction_percentage": None,
            "changed": False,
            "reason": record["reason"] or "The pass raised an error.",
        }
    measurements = record["measurements"]
    return {
        "status": "measured",
        "useful_improvement": int(measurements["useful_improvement"]),
        "instruction_reduction": int(measurements["instruction_reduction"]),
        "reduction_percentage": float(measurements["reduction_percentage"]),
        "changed": bool(record["changed"]),
        "reason": record["reason"],
    }


def _best_row(cells: dict) -> dict | None:
    measured = {
        name: cell["useful_improvement"]
        for name, cell in cells.items()
        if cell["status"] == "measured" and cell["useful_improvement"] is not None
    }
    if not measured:
        return None
    top = max(measured.values())
    tied = [name for name in STRATEGIES if measured.get(name) == top]
    return {
        "strategies": tied,
        "value": int(top),
        "tie": len(tied) > 1,
        "label": " · ".join(STRATEGY_DISPLAY.get(name, name) for name in tied),
    }


def optimization_matrix(
    page: int = 1, size: int = 10, q: str = "", split: str | None = None
) -> dict:
    """Paginated matrix of live per-pass measurements over the dataset."""
    try:
        df = dataset_service._load()
    except FileNotFoundError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "rows": [],
            "total": 0,
            "strategies": [],
        }

    page = max(1, int(page))
    size = max(1, min(int(size), 50))

    filtered = _filter_sort(df, q or "", split)
    total = int(len(filtered))
    pages = max(1, -(-total // size))
    page = min(page, pages)
    start = (page - 1) * size
    window = filtered.iloc[start : start + size]

    rows = []
    for _, record in window.iterrows():
        tac_text = str(record[TAC_COLUMN])
        row = {
            "sample_id": str(record[ID_COLUMN]),
            "generation_category": record.get("generation_category"),
            "split": str(record[SPLIT_COLUMN]),
            "instruction_count": int(record.get("num_statements", 0) or 0),
            "tac_lines": tac_text.splitlines(),
            "cells": {},
            "best": None,
        }

        try:
            tac = parse_tac(tac_text)
            if not tac.instructions:
                raise ValueError("Program parsed to zero instructions.")
        except Exception as exc:  # noqa: BLE001 - report the row, keep the page alive
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        cells = {}
        for strategy in STRATEGIES:
            # Independent copy per technique: apply_strategy_once deepcopies
            # the original before running the pass.
            cells[strategy] = _cell(apply_strategy_once(tac, strategy))
        row["cells"] = cells
        row["best"] = _best_row(cells)
        rows.append(row)

    return {
        "available": True,
        "strategies": [
            {
                "id": name,
                "label": STRATEGY_DISPLAY.get(name, name),
                "supported": True,
            }
            for name in STRATEGIES
        ],
        "metric": _METRIC,
        "data_source": {
            "kind": "live",
            "dataset_file": DEFAULT_DATASET_PATH.name,
            "note": (
                "Every cell is executed on demand from the current dataset "
                "file: one pass, one independent copy of the original TAC, "
                "measured with optimizer.measure.measure_pass."
            ),
        },
        "page": page,
        "size": size,
        "pages": pages,
        "total": total,
        "query": q or "",
        "split_filter": split,
        "rows": rows,
        "limitations": _LIMITATIONS,
    }
