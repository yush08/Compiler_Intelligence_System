"""
Reproducible dataset generator for the future ML strategy selector.

    TAC corpus -> feature extraction -> run all 6 optimizations -> measure
    actual improvement -> determine best_strategy -> CSV / JSON dataset

Nothing here is hand-labelled.  Every feature comes from
backend.features.extractor and every label comes from actually running each of
the six passes and measuring what it did (backend.optimizer.measure).

The model is NOT trained here - this phase only produces the data.

Usage
-----
    python experiments/generate_dataset.py               # full dataset
    python experiments/generate_dataset.py --samples 20  # small test dataset

Outputs (relative to the repository root)
-----------------------------------------
    dataset/raw/tac_programs.json             input programs + provenance
    dataset/processed/optimization_dataset.csv
    dataset/processed/optimization_dataset.json
    dataset/metadata/summary.json             distributions and statistics
    dataset/metadata/validation_report.json   quality-check results
    dataset/metadata/splits.json              deterministic 70/15/15 split
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# Reuse the existing corpus rather than rewriting its builders.
from corpus import REPOSITORY_PROGRAMS, SIZES, generate_program  # noqa: E402

from backend.api.serialize import tac_to_lines  # noqa: E402
from backend.features.extractor import extract_features  # noqa: E402
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS  # noqa: E402
from backend.optimizer.measure import (  # noqa: E402
    best_strategy_from,
    measure_pass,
    operation_count,
    work_remaining,
)
from backend.scoring.rule_based import STRATEGY_TIE_ORDER  # noqa: E402
from backend.tac.parser import parse_tac  # noqa: E402

# --- configuration -----------------------------------------------------------

DATASET_SEED = 20261005
TARGET_SIZE = 180

FEATURE_KEYS = [
    "num_statements",
    "num_constants",
    "num_repeated_expr",
    "num_dead_stmts",
    "num_algebraic_ops",
    "num_loop_ops",
    "num_tmp_vars",
    "const_ratio",
]

STRATEGIES = list(OPTIMIZATION_FUNCTIONS.keys())

# The existing corpus covers seven opportunity classes; these builders add the
# categories it does not produce (loops, no-opportunity, temporary-heavy).
EXTRA_CATEGORIES = ["loops", "no_opportunity", "temporaries"]

# Builders that keep every result consumed so that exactly one kind of
# opportunity is present (see _consume below).
LIVE_CATEGORIES = [
    "constant_folding_live",
    "constant_propagation_live",
    "common_subexpression_live",
    "algebraic_simplification_live",
    "dead_code_live",
    "strength_reduction_live",
]

CORPUS_SIZES = [5, 10, 15, 20]
# One replicate per size: the padded corpus programs are realistic but every
# one of them is won by dead code elimination, so over-sampling them would
# swamp the label distribution.
CORPUS_REPLICATES = [0]
LIVE_PER_CATEGORY = 18
EXTRA_PER_CATEGORY = {"loops": 12, "no_opportunity": 10, "temporaries": 10}


# --- program builders for categories the corpus does not cover ---------------


def _loop_program(rng: random.Random) -> str:
    """A counted loop: labels, a conditional branch and a back edge."""
    limit = rng.randint(2, 14)
    body = [
        f"acc = acc + i",
        f"i = i + 1",
    ]
    # Extra body statements vary the statement count and constant ratio.
    for k in range(rng.randint(0, 4)):
        body.insert(rng.randint(0, len(body)), f"acc = acc + w{rng.randint(0, 99)}_{k}")
    lines = [
        f"n = {limit}",
        f"i = {rng.randint(0, 3)}",
        "acc = 0",
        "loop_head:",
        "if i >= n goto loop_end",
        *body,
        "goto loop_head",
        "loop_end:",
        "result = acc",
    ]
    return "\n".join(lines)


def _no_opportunity_program(rng: random.Random) -> str:
    """Straight-line code with no foldable constant, no repeat and no dead code.

    The final self-referential statement keeps every result live, so no pass can
    improve the program at all - these are the baseline examples.
    """
    depth = rng.randint(2, 8)
    inputs = ["a", "b", "c", "d", "e", "f", "g", "h"]
    rng.shuffle(inputs)

    lines = [f"p0 = {inputs[0]} + {inputs[1]}"]
    for i in range(1, depth):
        lines.append(f"p{i} = p{i - 1} + {inputs[(i + 1) % len(inputs)]}")
    # Self reference: the result is read again, so nothing here is dead.
    lines.append(f"p{depth - 1} = p{depth - 1} + {inputs[0]}")
    return "\n".join(lines)


def _temporaries_program(rng: random.Random) -> str:
    """A chain of distinct t1..tn temporaries with a mixed opportunity profile."""
    count = rng.randint(4, 10)
    lines = []
    prev = "t0"
    for i in range(1, count + 1):
        choice = rng.randint(0, 5)
        if choice == 0:
            lines.append(f"t{i} = {rng.randint(1, 50)} + {rng.randint(1, 50)}")
        elif choice == 1:
            lines.append(f"t{i} = {prev} * 1")
        elif choice == 2:
            lines.append(f"t{i} = {prev} + 0")
        elif choice == 3:
            lines.append(f"t{i} = {prev} * {rng.randint(2, 8)}")
        elif choice == 4:
            lines.append(f"t{i} = {prev} + in{i}")
        else:
            lines.append(f"t{i} = {prev} + t{i - 1}")
        prev = f"t{i}"
    # Keep the chain live so the last value is not automatically dead.
    lines.append(f"{prev} = {prev} + {prev}")
    return "\n".join(lines)


EXTRA_BUILDERS = {
    "loops": _loop_program,
    "no_opportunity": _no_opportunity_program,
    "temporaries": _temporaries_program,
}


# --- "live" opportunity builders --------------------------------------------
#
# experiments/corpus.py pads every program with `pad = in + in`, whose result
# is never read. Under this project's liveness rule (a statement is dead when
# nothing reads its result) that makes the padding itself dead code, so dead
# code elimination wins every padded program outright - 93 of 93 corpus
# programs. These builders instead route every produced value into one
# accumulator, leaving exactly one opportunity present so that a strategy can
# only win by actually removing more work than the others.


def _consume(produced: list[str], final: str = "acc") -> list[str]:
    """Append statements that read every produced name and end up live.

    The chain deliberately introduces no opportunity of its own: operands are
    variables (so nothing folds), no operand is 0 or 1 (so no identity), every
    signature differs (so no repeat), and the closing statement reads `final`
    again so the accumulator is never dead.
    """
    if not produced:
        return []

    lines: list[str] = []
    if len(produced) == 1:
        lines.append(f"{final} = {produced[0]} + ztail_input")
    else:
        lines.append(f"{final} = {produced[0]} + {produced[1]}")
        for name in produced[2:]:
            lines.append(f"{final} = {final} + {name}")
    # Self reference: `final` is read here, so it survives dead code removal.
    lines.append(f"{final} = {final} + ztail_input")
    return lines


def _fold_live(rng: random.Random) -> str:
    """Many literal operations - constant folding can resolve each one."""
    k = rng.randint(3, 8)
    lines = [
        f"f{i} = {rng.randint(1, 99)} {rng.choice(['+', '-', '*'])} {rng.randint(1, 99)}"
        for i in range(k)
    ]
    return "\n".join(lines + _consume([f"f{i}" for i in range(k)]))


def _propagate_live(rng: random.Random) -> str:
    """Constants feeding operations - only constant propagation resolves them.

    `c{i}` is a single literal, so constant folding has no operator to fold;
    `p{i} = c{i} + k` has a variable operand, so folding cannot touch it, but
    propagation replaces it and then folds.
    """
    k = rng.randint(3, 8)
    lines = []
    produced = []
    for i in range(k):
        lines.append(f"c{i} = {rng.randint(1, 50)}")
        lines.append(f"p{i} = c{i} + {rng.randint(2, 9)}")
        produced.append(f"p{i}")
    return "\n".join(lines + _consume(produced))


def _cse_live(rng: random.Random) -> str:
    """Repeated expressions - only CSE can collapse the duplicates."""
    k = rng.randint(3, 8)
    operator = rng.choice(["+", "*"])  # both are normalised by the CSE pass
    lines = []
    produced = []
    for i in range(k):
        u = f"zin{i}_{rng.randint(0, 999)}a"
        v = f"zin{i}_{rng.randint(0, 999)}b"
        lines.append(f"r{i}a = {u} {operator} {v}")
        # Half the time write the duplicate with its operands swapped, which is
        # still the same expression for + and * and exercises the pass's
        # commutative normalisation.
        if rng.random() < 0.5:
            lines.append(f"r{i}b = {v} {operator} {u}")
        else:
            lines.append(f"r{i}b = {u} {operator} {v}")
        produced.extend([f"r{i}a", f"r{i}b"])
    return "\n".join(lines + _consume(produced))


def _algebraic_live(rng: random.Random) -> str:
    """Identities - only algebraic simplification collapses them."""
    k = rng.randint(3, 8)
    templates = ["{v} + 0", "{v} * 1", "{v} - 0", "{v} / 1", "{v} * 0"]
    lines = []
    produced = []
    for i in range(k):
        value = f"vin{rng.randint(0, 999)}_{i}"
        lines.append(f"g{i} = " + rng.choice(templates).format(v=value))
        produced.append(f"g{i}")
    return "\n".join(lines + _consume(produced))


def _dead_live(rng: random.Random) -> str:
    """Plain assignments nobody reads - only dead code elimination removes them."""
    k = rng.randint(3, 9)
    lines = [f"unused{i} = {rng.randint(1, 99)}" for i in range(k)]
    lines += [f"stored{i} = in{i}" for i in range(k)]
    return "\n".join(lines)


def _strength_live(rng: random.Random) -> str:
    """Multiply-by-constant patterns. The strength reduction pass is a
    documented conservative no-op, so every measurement for it is 0 and this
    category records that honestly rather than pretending it wins."""
    k = rng.randint(3, 8)
    consts = [2, 4, 8, 16, 32, 3, 5, 6, 7, 10]
    lines = [
        f"s{i} = zsrc{rng.randint(0, 999)}_{i} * {rng.choice(consts)}"
        for i in range(k)
    ]
    return "\n".join(lines + _consume([f"s{i}" for i in range(k)]))


LIVE_BUILDERS = {
    "constant_folding_live": _fold_live,
    "constant_propagation_live": _propagate_live,
    "common_subexpression_live": _cse_live,
    "algebraic_simplification_live": _algebraic_live,
    "dead_code_live": _dead_live,
    "strength_reduction_live": _strength_live,
}


# --- corpus assembly ---------------------------------------------------------


def build_programs() -> list[dict]:
    """Return every program to label, in a fixed, reproducible order."""
    programs: list[dict] = []
    counter = 0

    def add(category: str, tac: str, source: str, size: int, replicate: int):
        nonlocal counter
        counter += 1
        programs.append(
            {
                "sample_id": f"sample-{counter:04d}",
                "generation_category": category,
                "source": source,
                "requested_size": size,
                "replicate": replicate,
                "tac_program": tac,
            }
        )

    # 1. Existing corpus: seven opportunity classes, several sizes each.
    for category in [
        "constant_folding",
        "constant_propagation",
        "common_subexpression",
        "dead_code_elimination",
        "algebraic_simplification",
        "strength_reduction",
        "mixed",
    ]:
        for size in CORPUS_SIZES:
            for replicate in CORPUS_REPLICATES:
                add(
                    category,
                    generate_program(category, size, replicate),
                    "experiments/corpus.generate_program",
                    size,
                    replicate,
                )

    # 2. The repository's own example programs.
    for name, tac in REPOSITORY_PROGRAMS.items():
        add("repository", tac, f"experiments/corpus.REPOSITORY_PROGRAMS[{name!r}]",
            len(tac.splitlines()), 0)

    # 3. Live opportunity builders: every value consumed, so exactly one kind
    #    of opportunity is present and the winner is decided by measurement.
    for category in LIVE_CATEGORIES:
        builder = LIVE_BUILDERS[category]
        for replicate in range(LIVE_PER_CATEGORY):
            rng = random.Random(f"{DATASET_SEED}:{category}:{replicate}")
            add(category, builder(rng), f"experiments/generate_dataset.{category}",
                0, replicate)

    # 4. Categories the corpus does not provide.
    for category in EXTRA_CATEGORIES:
        builder = EXTRA_BUILDERS[category]
        for replicate in range(EXTRA_PER_CATEGORY[category]):
            # Seeded per category+replicate so reruns are byte identical.
            rng = random.Random(f"{DATASET_SEED}:{category}:{replicate}")
            add(category, builder(rng), f"experiments/generate_dataset.{category}",
                0, replicate)

    return programs


def deduplicate(programs: list[dict]) -> tuple[list[dict], int]:
    """Drop repeated programs, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0
    for program in programs:
        key = "\n".join(line.strip() for line in program["tac_program"].splitlines()
                        if line.strip())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(program)
    return unique, duplicates


# --- labelling ---------------------------------------------------------------


def _source_line_count(tac_text: str) -> int:
    return sum(
        1
        for line in tac_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def label_program(program: dict) -> dict | None:
    """Extract features, run all six passes and derive best_strategy.

    Returns None if the program is empty or fails to parse cleanly, which the
    validation step would otherwise have to catch later.
    """
    tac_text = program["tac_program"]
    tac = parse_tac(tac_text)

    if len(tac) == 0:
        return None
    # Malformed TAC: a line that the parser silently dropped.
    if len(tac) != _source_line_count(tac_text):
        return None

    features = extract_features(tac)
    original_text = "\n".join(tac_to_lines(tac))

    measurements: dict[str, dict] = {}
    for name, optimizer in OPTIMIZATION_FUNCTIONS.items():
        optimized = optimizer(deepcopy(tac))
        stats = measure_pass(tac, optimized)
        stats["rewrote_tac"] = "\n".join(tac_to_lines(optimized)) != original_text
        stats["optimized_tac"] = "\n".join(tac_to_lines(optimized))
        measurements[name] = stats

    best = best_strategy_from(measurements, STRATEGY_TIE_ORDER)
    best_stats = measurements[best]

    record = dict(program)
    record.update(
        {
            "parsed_instruction_count": len(tac),
            "operation_instruction_count": operation_count(tac),
            "work_remaining": work_remaining(tac),
            "features": features,
            "strategies": measurements,
            "best_strategy": best,
            # Ranking quantity: useful_improvement of the winning strategy.
            "best_reduction": best_stats["useful_improvement"],
            "best_instruction_reduction": best_stats["instruction_reduction"],
            "best_reduction_percentage": best_stats["reduction_percentage"],
            "has_opportunity": any(
                m["useful_improvement"] > 0 for m in measurements.values()
            ),
        }
    )
    return record


# --- validation --------------------------------------------------------------


def validate(records: list[dict]) -> dict:
    """Run the quality checks required for the dataset.

    Detects missing values, invalid/malformed/empty TAC, duplicates, invalid
    feature values, invalid strategy labels and impossible instruction counts.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_tacs: dict[str, str] = {}

    valid_strategies = set(STRATEGIES)

    if not records:
        errors.append("dataset contains no records")

    for record in records:
        sid = record.get("sample_id", "<missing sample_id>")

        # --- missing values ------------------------------------------------
        for key in ("sample_id", "tac_program", "generation_category",
                    "best_strategy", "features", "strategies"):
            if record.get(key) in (None, ""):
                errors.append(f"{sid}: missing value for {key}")

        # --- TAC validity --------------------------------------------------
        tac_text = record.get("tac_program") or ""
        if not tac_text.strip():
            errors.append(f"{sid}: empty TAC")
            continue
        parsed = parse_tac(tac_text)
        if len(parsed) == 0:
            errors.append(f"{sid}: TAC did not parse")
            continue
        if len(parsed) != _source_line_count(tac_text):
            errors.append(
                f"{sid}: malformed TAC - "
                f"{_source_line_count(tac_text)} source lines but "
                f"{len(parsed)} instructions parsed"
            )

        # --- duplicates ----------------------------------------------------
        key = "\n".join(line.strip() for line in tac_text.splitlines() if line.strip())
        if key in seen_tacs:
            errors.append(f"{sid}: duplicate of {seen_tacs[key]}")
        else:
            seen_tacs[key] = sid

        # --- features ------------------------------------------------------
        features = record.get("features") or {}
        for feature in FEATURE_KEYS:
            if feature not in features:
                errors.append(f"{sid}: missing feature {feature}")
                continue
            value = features[feature]
            if value is None:
                errors.append(f"{sid}: feature {feature} is None")
                continue
            if feature == "const_ratio":
                if not (0.0 <= float(value) <= 1.0):
                    errors.append(f"{sid}: const_ratio out of range: {value}")
            else:
                if float(value) < 0:
                    errors.append(f"{sid}: negative feature {feature}={value}")
                if float(value) != int(value):
                    errors.append(f"{sid}: non-integral feature {feature}={value}")
        # const_ratio consistency: guard against a zero statement count.
        if features.get("num_statements", 0) == 0:
            errors.append(f"{sid}: num_statements is 0")
        elif abs(
            features["const_ratio"]
            - features["num_constants"] / features["num_statements"]
        ) > 1e-9:
            errors.append(
                f"{sid}: const_ratio {features['const_ratio']} inconsistent with "
                f"{features['num_constants']}/{features['num_statements']}"
            )

        # --- strategy measurements ----------------------------------------
        strategies = record.get("strategies") or {}
        if set(strategies) != valid_strategies:
            errors.append(
                f"{sid}: strategy set {sorted(strategies)} != the six valid ids"
            )
        for name, stats in strategies.items():
            if name not in valid_strategies:
                errors.append(f"{sid}: invalid strategy id {name!r}")
                continue
            original = stats.get("original_instruction_count")
            optimized = stats.get("optimized_instruction_count")
            if original is None or optimized is None:
                errors.append(f"{sid}/{name}: missing instruction counts")
                continue
            if original < 0 or optimized < 0:
                errors.append(f"{sid}/{name}: negative instruction count")
            if optimized > original:
                errors.append(
                    f"{sid}/{name}: pass grew the TAC {original} -> {optimized}"
                )
            if stats.get("instruction_reduction") != original - optimized:
                errors.append(f"{sid}/{name}: instruction_reduction inconsistent")
            if not (0.0 <= stats.get("reduction_percentage", -1) <= 100.0):
                errors.append(f"{sid}/{name}: reduction_percentage out of range")

        # --- label ---------------------------------------------------------
        best = record.get("best_strategy")
        if best not in valid_strategies:
            errors.append(f"{sid}: best_strategy {best!r} is not one of the six")
        elif best in strategies:
            expected = best_strategy_from(strategies, STRATEGY_TIE_ORDER)
            if best != expected:
                errors.append(
                    f"{sid}: best_strategy {best} disagrees with re-deriving it "
                    f"({expected})"
                )

    # --- dataset level ----------------------------------------------------
    if len(seen_tacs) != len(records):
        warnings.append(
            f"{len(records) - len(seen_tacs)} duplicate program(s) present"
        )

    label_counts = Counter(r.get("best_strategy") for r in records)
    for strategy in STRATEGIES:
        if label_counts.get(strategy, 0) == 0:
            warnings.append(
                f"strategy {strategy!r} never appears as best_strategy "
                f"(measured reduction is 0 for every sample)"
            )

    return {
        "passed": not errors,
        "records_checked": len(records),
        "distinct_programs": len(seen_tacs),
        "errors": errors,
        "warnings": warnings,
        "label_counts": {k: label_counts.get(k, 0) for k in STRATEGIES},
    }


# --- split -------------------------------------------------------------------

def assign_splits(records: list[dict], seed: int = DATASET_SEED) -> dict:
    """Deterministic 70/15/15 train/validation/test split by sample_id."""
    ids = [r["sample_id"] for r in records]
    random.Random(seed).shuffle(ids)

    n = len(ids)
    train_end = round(n * 0.70)
    val_end = train_end + round(n * 0.15)

    splits = {
        "train": ids[:train_end],
        "validation": ids[train_end:val_end],
        "test": ids[val_end:],
    }

    by_id = {r["sample_id"]: r for r in records}
    for name, split_ids in splits.items():
        for sid in split_ids:
            by_id[sid]["split"] = name

    return splits


# --- output ------------------------------------------------------------------

CSV_COLUMNS = (
    [
        "sample_id",
        "generation_category",
        "source",
        "requested_size",
        "parsed_instruction_count",
        "original_instruction_count",
    ]
    + FEATURE_KEYS
    + [f"{s}_{m}" for s in STRATEGIES for m in
       ("optimized_instruction_count", "instruction_reduction",
        "reduction_percentage", "useful_improvement")]
    + [
        "best_strategy",
        "best_reduction",
        "best_instruction_reduction",
        "best_reduction_percentage",
        "has_opportunity",
        "split",
        "tac_program",
    ]
)


def _flatten(record: dict) -> dict:
    row = {
        "sample_id": record["sample_id"],
        "generation_category": record["generation_category"],
        "source": record["source"],
        "requested_size": record["requested_size"],
        "parsed_instruction_count": record["parsed_instruction_count"],
        "original_instruction_count": record["features"]["num_statements"],
        "best_strategy": record["best_strategy"],
        "best_reduction": record["best_reduction"],
        "best_instruction_reduction": record["best_instruction_reduction"],
        "best_reduction_percentage": record["best_reduction_percentage"],
        "has_opportunity": record["has_opportunity"],
        "split": record.get("split", ""),
        "tac_program": record["tac_program"],
    }
    for feature in FEATURE_KEYS:
        row[feature] = record["features"][feature]
    for strategy in STRATEGIES:
        stats = record["strategies"][strategy]
        row[f"{strategy}_optimized_instruction_count"] = stats["optimized_instruction_count"]
        row[f"{strategy}_instruction_reduction"] = stats["instruction_reduction"]
        row[f"{strategy}_reduction_percentage"] = stats["reduction_percentage"]
        row[f"{strategy}_useful_improvement"] = stats["useful_improvement"]
    return row


def build_summary(records: list[dict], duplicates_dropped: int) -> dict:
    n = len(records)
    label_counts = Counter(r["best_strategy"] for r in records)
    category_counts = Counter(r["generation_category"] for r in records)

    per_strategy_reduces = {
        s: sum(1 for r in records if r["strategies"][s]["instruction_reduction"] > 0)
        for s in STRATEGIES
    }
    per_strategy_rewrites = {
        s: sum(1 for r in records if r["strategies"][s]["rewrote_tac"])
        for s in STRATEGIES
    }
    per_strategy_useful = {
        s: sum(1 for r in records if r["strategies"][s]["useful_improvement"] > 0)
        for s in STRATEGIES
    }

    statement_counts = [r["features"]["num_statements"] for r in records]

    return {
        "generator": "experiments/generate_dataset.py",
        "dataset_seed": DATASET_SEED,
        "total_samples": n,
        "duplicates_dropped": duplicates_dropped,
        "strategy_ids": STRATEGIES,
        "feature_keys": FEATURE_KEYS,
        "label_rule": (
            "best_strategy = argmax useful_improvement across the six passes; "
            "useful_improvement = (statement reduction) + (operation reduction); "
            f"ties broken by {STRATEGY_TIE_ORDER}"
        ),
        "label_distribution": {s: label_counts.get(s, 0) for s in STRATEGIES},
        "label_distribution_pct": {
            s: round(100 * label_counts.get(s, 0) / n, 2) for s in STRATEGIES
        },
        "generation_category_distribution": dict(sorted(category_counts.items())),
        "programs_where_pass_reduces_statements": per_strategy_reduces,
        "programs_where_pass_rewrites_tac": per_strategy_rewrites,
        "programs_where_pass_has_useful_improvement": per_strategy_useful,
        "programs_with_any_opportunity": sum(1 for r in records if r["has_opportunity"]),
        "programs_without_any_opportunity": sum(
            1 for r in records if not r["has_opportunity"]
        ),
        "statement_count": {
            "min": min(statement_counts),
            "max": max(statement_counts),
            "mean": round(sum(statement_counts) / n, 2),
        },
        "const_ratio": {
            "min": round(min(r["features"]["const_ratio"] for r in records), 4),
            "max": round(max(r["features"]["const_ratio"] for r in records), 4),
            "mean": round(
                sum(r["features"]["const_ratio"] for r in records) / n, 4
            ),
        },
        "mean_best_useful_improvement": round(
            sum(r["best_reduction"] for r in records) / n, 4
        ),
        "mean_best_instruction_reduction_percentage": round(
            sum(r["best_reduction_percentage"] for r in records) / n, 4
        ),
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate(target_samples: int | None = None, out_root: Path | None = None) -> dict:
    out_root = out_root or (ROOT / "dataset")

    programs = deduplicate(build_programs())
    programs, duplicates_dropped = programs

    if target_samples and target_samples < len(programs):
        # Spread a small test dataset across the whole corpus instead of
        # taking the first N (which would contain only one opportunity class).
        stride = len(programs) / target_samples
        programs = [programs[int(i * stride)] for i in range(target_samples)]

    # 1. raw programs
    raw = {
        "generator": "experiments/generate_dataset.py",
        "dataset_seed": DATASET_SEED,
        "program_count": len(programs),
        "programs": programs,
    }
    _write(out_root / "raw" / "tac_programs.json", json.dumps(raw, indent=2))

    # 2. label
    records = []
    skipped = []
    for program in programs:
        labelled = label_program(program)
        if labelled is None:
            skipped.append(program["sample_id"])
        else:
            records.append(labelled)

    # 3. split
    splits = assign_splits(records)

    # 4. validate
    report = validate(records)
    report["skipped_programs"] = skipped

    # 5. processed outputs
    rows = [_flatten(r) for r in records]
    processed_json = out_root / "processed" / "optimization_dataset.json"
    processed_json.parent.mkdir(parents=True, exist_ok=True)
    processed_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_ids": STRATEGIES,
                "feature_keys": FEATURE_KEYS,
                "label": "best_strategy",
                "records": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = out_root / "processed" / "optimization_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 6. metadata
    summary = build_summary(records, duplicates_dropped)
    summary["skipped_programs"] = skipped
    _write(out_root / "metadata" / "summary.json", json.dumps(summary, indent=2))
    _write(
        out_root / "metadata" / "validation_report.json",
        json.dumps(report, indent=2),
    )
    _write(
        out_root / "metadata" / "splits.json",
        json.dumps(
            {
                "seed": DATASET_SEED,
                "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "counts": {k: len(v) for k, v in splits.items()},
                "splits": splits,
            },
            indent=2,
        ),
    )

    return {
        "summary": summary,
        "validation": report,
        "splits": {k: len(v) for k, v in splits.items()},
        "out_root": str(out_root),
        "record_count": len(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=None,
                        help="generate a smaller test dataset of N samples")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default: <repo>/dataset)")
    args = parser.parse_args(argv)

    result = generate(
        target_samples=args.samples,
        out_root=Path(args.out) if args.out else None,
    )

    summary = result["summary"]
    validation = result["validation"]

    print("Dataset generation complete.")
    print()
    print(f"Total samples: {summary['total_samples']}")
    if summary["duplicates_dropped"]:
        print(f"Duplicate programs dropped: {summary['duplicates_dropped']}")
    if summary["skipped_programs"]:
        print(f"Programs skipped (empty/malformed): {len(summary['skipped_programs'])}")
    print()
    print("Strategy distribution (best_strategy):")
    display = {
        "constant_folding": "Constant Folding",
        "constant_propagation": "Constant Propagation",
        "common_subexpression": "CSE",
        "dead_code_elimination": "Dead Code Elimination",
        "algebraic_simplification": "Algebraic Simplification",
        "strength_reduction": "Strength Reduction",
    }
    for strategy in STRATEGIES:
        count = summary["label_distribution"][strategy]
        pct = summary["label_distribution_pct"][strategy]
        print(f"  {display[strategy]:26} {count:4}  ({pct:5.2f}%)")
    print()
    print("Generation categories:")
    for category, count in summary["generation_category_distribution"].items():
        print(f"  {category:26} {count:4}")
    print()
    print(
        "Average useful improvement of the selected strategy: "
        f"{summary['mean_best_useful_improvement']}"
    )
    print(
        "Average statement reduction of the selected strategy: "
        f"{summary['mean_best_instruction_reduction_percentage']}%"
    )
    print()
    print(
        "Passes that reduce statement count at least once: "
        + ", ".join(
            f"{s}({n})"
            for s, n in summary["programs_where_pass_reduces_statements"].items()
        )
    )
    print()
    print(
        f"Validation: {'PASSED' if validation['passed'] else 'FAILED'} - "
        f"{validation['records_checked']} records, "
        f"{len(validation['errors'])} error(s), "
        f"{len(validation['warnings'])} warning(s)"
    )
    for error in validation["errors"][:20]:
        print(f"  ERROR {error}")
    for warning in validation["warnings"]:
        print(f"  warn  {warning}")
    print()
    print(f"Split counts: {result['splits']}")
    print(f"Written to {result['out_root']}")

    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
