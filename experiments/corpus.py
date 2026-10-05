"""
Deterministic TAC benchmark corpus for the Compiler Intelligence System evaluation.

The repository ships a handful of illustrative TAC fragments (parser examples,
frontend demo scenarios, test fixtures). That is too small to support the
"Results and Discussion" section of the report, so this module builds a larger,
SYSTEMATIC and REPRODUCIBLE corpus.

Design rules
------------
* Every program is generated from a fixed seed, so re-running the experiment
  produces byte-identical programs.
* Programs are grouped into seven opportunity classes - one per optimization
  the system implements - plus a "mixed" class.
* Each class is instantiated at four program sizes (5, 10, 15, 20 statements)
  with three replicates per size.
* Padding statements are deliberately chosen to introduce NO opportunity of any
  other kind (unique binary expressions over unread input names), so that each
  class stays clean and the measured effect can be attributed.
* The nine TAC programs that already exist in the repository are appended
  unchanged, so the corpus always includes the project's own examples.

Nothing here fabricates RESULTS - results are produced later by running the real
backend over these inputs.
"""

from __future__ import annotations

import random

SEED = 20261005

OPPORTUNITY_CLASSES = [
    "constant_folding",
    "constant_propagation",
    "common_subexpression",
    "dead_code_elimination",
    "algebraic_simplification",
    "strength_reduction",
    "mixed",
]

SIZES = [5, 10, 15, 20]
REPLICATES = 3

# --- Class-specific builders -------------------------------------------------
#
# Each builder returns a list of TAC source lines that concentrates ONE kind of
# optimization opportunity. All names are class-local so nothing leaks between
# builders.


def _fold(rng: random.Random, k: int) -> list[str]:
    out = []
    for i in range(k):
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        op = rng.choice(["+", "-", "*"])
        out.append(f"f{i} = {a} {op} {b}")
    return out


def _propagate(rng: random.Random, k: int) -> list[str]:
    out = []
    for i in range(k):
        c = rng.randint(1, 50)
        out.append(f"c{i} = {c}")
        out.append(f"p{i} = c{i} + {rng.randint(1, 9)}")
    return out


def _cse(rng: random.Random, k: int) -> list[str]:
    out = []
    for i in range(k):
        u, v = f"in{2 * i}", f"in{2 * i + 1}"
        # same expression written twice -> repeated subexpression
        out.append(f"r{i}a = {u} + {v}")
        out.append(f"r{i}b = {u} + {v}")
    return out


def _dead(rng: random.Random, k: int) -> list[str]:
    out = []
    for i in range(k):
        # plain (ASSIGN) statements whose result is never read
        out.append(f"unused{i} = {rng.randint(1, 99)}")
        out.append(f"used{i} = in{i}")
    return out


def _algebraic(rng: random.Random, k: int) -> list[str]:
    identities = [
        lambda n: f"g{n} = in{n} + 0",
        lambda n: f"g{n} = in{n} * 1",
        lambda n: f"g{n} = in{n} - 0",
        lambda n: f"g{n} = in{n} / 1",
        lambda n: f"g{n} = in{n} * 0",
    ]
    return [identities[i % len(identities)](i) for i in range(k)]


def _strength(rng: random.Random, k: int) -> list[str]:
    consts = [2, 4, 8, 16, 3, 5, 10]
    return [f"s{i} = in{i} * {consts[i % len(consts)]}" for i in range(k)]


def _mixed(rng: random.Random, k: int) -> list[str]:
    builders = [_fold, _propagate, _cse, _dead, _algebraic, _strength]
    out: list[str] = []
    i = 0
    while len(out) < k:
        out.extend(builders[i % len(builders)](rng, 1))
        i += 1
    return out


_BUILDERS = {
    "constant_folding": _fold,
    "constant_propagation": _propagate,
    "common_subexpression": _cse,
    "dead_code_elimination": _dead,
    "algebraic_simplification": _algebraic,
    "strength_reduction": _strength,
    "mixed": _mixed,
}


def _pad(rng: random.Random, lines: list[str], target: int) -> list[str]:
    """Pad to exactly `target` statements with opportunity-neutral statements.

    Padding uses a unique binary expression over two unread input names. Because
    the opcode is binary (not ASSIGN) it introduces no dead-assignment
    opportunity, and because every pair is unique it introduces no repeated
    subexpression.
    """
    n = 0
    while len(lines) < target:
        u, v = f"zin{n}a", f"zin{n}b"
        expr = f"{u} + {v}"
        if any(expr in existing for existing in lines):
            n += 1
            continue
        lines.append(f"pad{n} = {expr}")
        n += 1
    return lines[:target]


def generate_program(opportunity_class: str, size: int, replicate: int) -> str:
    """Return one TAC program of exactly `size` statements."""
    rng = random.Random(f"{SEED}:{opportunity_class}:{size}:{replicate}")
    builder = _BUILDERS[opportunity_class]
    # Build roughly half the program from the target opportunity, then pad.
    seed_count = max(1, size // 2)
    lines = builder(rng, seed_count)
    return "\n".join(_pad(rng, lines, size))


# --- Repository programs -----------------------------------------------------

REPOSITORY_PROGRAMS: dict[str, str] = {
    # backend/tac/parser.py example functions
    "repo/parser-constant-folding": "t1 = 5 + 3\nt2 = a + b\nt3 = t1 * 1",
    "repo/parser-dead-code": "t1 = a + b\nx = 10",
    "repo/parser-cse": "t1 = a + b\nt2 = a + b\nt3 = t1 * 1",
    "repo/parser-algebraic": "t1 = 5 + 0\nt2 = 5 * 1\nt3 = 10 - 0",
    "repo/parser-mixed": "t1 = 5 + 3\nt2 = a + b\nt3 = a + b\nt4 = t1 * 1\nt5 = 100\nx = t4",
    # frontend/src/data/demoData.ts scenario programs
    "repo/frontend-constant-folding": "a1 = 4\na2 = 6\na3 = a1 + a2\na4 = a3 * 2\na5 = a4 + 10",
    "repo/frontend-cse": "x1 = input_a\nx2 = input_b\nx3 = x1 + x2\nx4 = x1 + x2\nx5 = x3 * 4",
    "repo/frontend-dead-code": "d1 = 18\nd2 = d1 + 2\nunused_a = 99\nd3 = d2 * 2\nunused_b = d3 + 4\nresult = d3",
    # tests/test_api.py DEFAULT_TAC - NOTE: this is the canonical input that the
    # API pipeline short-circuits; included deliberately so the engine-level and
    # API-level behaviour can be contrasted.
    "repo/test-api-default": (
        "t1 = 10\nt2 = 20\nt3 = t1 + t2\nt4 = t1 + t2\nt5 = t3 * 1\n"
        "t6 = 100\nt7 = t6 + 50\nunused = t7\nt8 = t3 * 2"
    ),
}


def build_corpus() -> list[dict]:
    """Return the full corpus as a list of records."""
    records: list[dict] = []
    uid = 0
    for opportunity_class in OPPORTUNITY_CLASSES:
        for size in SIZES:
            for replicate in range(REPLICATES):
                uid += 1
                records.append(
                    {
                        "id": f"gen-{uid:03d}",
                        "source": "generated",
                        "opportunity_class": opportunity_class,
                        "size": size,
                        "replicate": replicate,
                        "tac": generate_program(opportunity_class, size, replicate),
                    }
                )
    for name, tac in REPOSITORY_PROGRAMS.items():
        uid += 1
        records.append(
            {
                "id": f"repo-{uid:03d}",
                "source": "repository",
                "opportunity_class": "repository",
                "size": len(tac.splitlines()),
                "replicate": 0,
                "tac": tac,
            }
        )
    return records


if __name__ == "__main__":
    corpus = build_corpus()
    generated = [r for r in corpus if r["source"] == "generated"]
    print(f"total programs   : {len(corpus)}")
    print(f"  generated      : {len(generated)}")
    print(f"  repository     : {len(corpus) - len(generated)}")
    for cls in OPPORTUNITY_CLASSES:
        n = len([r for r in generated if r["opportunity_class"] == cls])
        print(f"    {cls:26} {n}")
