"""Line-level TAC diffs for the transformation panels.

Both the optimization history (§11) and the side-by-side comparison screen (§14)
need to show *what actually changed* rather than claim that it did.  This module
diffs the rendered TAC text with :mod:`difflib` and classifies every line as
``unchanged``, ``added``, ``removed`` or ``modified``.

A ``modified`` pair is produced when a statement is rewritten in place (e.g.
``t4 = t1 + t2`` -> ``t4 = t3``) so the UI can render a red/green pair instead
of an unrelated delete + insert.  When nothing changed, ``changed`` is ``False``
and the caller must say "no transformation" rather than inventing an effect.
"""

from __future__ import annotations

import difflib

from ..api.serialize import tac_to_lines

__all__ = ["tac_diff", "tac_diff_text", "render_tac", "source_to_text"]


def render_tac(tac) -> list[str]:
    """Render a TAC as source lines (shared with the API serializer)."""
    return tac_to_lines(tac)


def tac_diff(before, after) -> dict:
    """Diff two TAC objects.

    Returns::

        {
          "changed": bool,
          "lines": [{"op","before_text","after_text","before_index","after_index"}],
          "added": int, "removed": int, "modified": int, "unchanged": int,
          "before_text": str, "after_text": str,
          "before_instruction_count": int, "after_instruction_count": int,
        }

    ``op`` is one of ``context`` / ``insert`` / ``delete`` / ``replace``.
    """
    before_lines = render_tac(before)
    after_lines = render_tac(after)

    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    entries: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                entries.append(
                    {
                        "op": "context",
                        "status": "unchanged",
                        "before_text": before_lines[i1 + offset],
                        "after_text": after_lines[j1 + offset],
                        "before_index": i1 + offset,
                        "after_index": j1 + offset,
                    }
                )
        elif tag == "replace":
            # Pair the overlapping range as modified lines so a rewrite renders
            # as one red/green pair rather than an unrelated delete+insert.
            overlap = min(i2 - i1, j2 - j1)
            for offset in range(overlap):
                entries.append(
                    {
                        "op": "replace",
                        "status": "modified",
                        "before_text": before_lines[i1 + offset],
                        "after_text": after_lines[j1 + offset],
                        "before_index": i1 + offset,
                        "after_index": j1 + offset,
                    }
                )
            for offset in range(overlap, i2 - i1):
                entries.append(
                    {
                        "op": "delete",
                        "status": "removed",
                        "before_text": before_lines[i1 + offset],
                        "after_text": None,
                        "before_index": i1 + offset,
                        "after_index": None,
                    }
                )
            for offset in range(overlap, j2 - j1):
                entries.append(
                    {
                        "op": "insert",
                        "status": "added",
                        "before_text": None,
                        "after_text": after_lines[j1 + offset],
                        "before_index": None,
                        "after_index": j1 + offset,
                    }
                )
        elif tag == "delete":
            for offset in range(i2 - i1):
                entries.append(
                    {
                        "op": "delete",
                        "status": "removed",
                        "before_text": before_lines[i1 + offset],
                        "after_text": None,
                        "before_index": i1 + offset,
                        "after_index": None,
                    }
                )
        elif tag == "insert":
            for offset in range(j2 - j1):
                entries.append(
                    {
                        "op": "insert",
                        "status": "added",
                        "before_text": None,
                        "after_text": after_lines[j1 + offset],
                        "before_index": None,
                        "after_index": j1 + offset,
                    }
                )

    counts = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    for entry in entries:
        counts[entry["status"]] += 1

    return {
        "changed": before_lines != after_lines,
        "lines": entries,
        "added": counts["added"],
        "removed": counts["removed"],
        "modified": counts["modified"],
        "unchanged": counts["unchanged"],
        "before_text": "\n".join(before_lines),
        "after_text": "\n".join(after_lines),
        "before_instruction_count": len(before_lines),
        "after_instruction_count": len(after_lines),
    }


def tac_diff_text(before, after) -> str:
    """Unified-diff style text, useful for terminal output and tests."""
    before_lines = render_tac(before)
    after_lines = render_tac(after)
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    return "\n".join(diff)


def source_to_text(source: str) -> str:
    """Identity helper so callers can pass either a TAC or rendered text."""
    return source if source else ""
