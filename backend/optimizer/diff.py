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
import re

from ..api.serialize import tac_to_lines

__all__ = ["tac_diff", "tac_diff_text", "render_tac", "source_to_text"]


def render_tac(tac) -> list[str]:
    """Render a TAC as source lines (shared with the API serializer)."""
    return tac_to_lines(tac)


def _result_name(line: str) -> str | None:
    """Result target of a rendered statement (``t6 = 42`` -> ``t6``), else ``None``.

    Non-assignment lines (``print``, labels, jumps, conditions) have no result
    target and are paired positionally instead.
    """
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)", line)
    return match.group(1) if match else None


def _replace_pairs(before: list[str], after: list[str]) -> list[tuple[int, int]]:
    """``(before_offset, after_offset)`` modified pairs for one replace block.

    Statements are first matched by result target so a rewritten statement
    pairs with its own after-line no matter where difflib placed the block.
    Positional order inside a replace block is an LCS-alignment artifact: for
    the combined diff of a fold plus earlier deletions it would claim that
    ``unused = 999`` was modified into ``t6 = 42`` while the real fold pair
    (``t6 = 20 + 22`` -> ``t6 = 42``) was reported as a deletion - the counts
    stay correct but the attribution misleads (constant propagation and
    folding shift every following line).  Leftover lines pair in program
    order, preserving the red/green rewrite rendering for statements whose
    result target itself changed.

    The number of pairs is always ``min(len(before), len(after))``: every
    name-matched pair consumes one line from each side, and the leftovers pair
    positionally over what remains (``m + min(nb - m, na - m) == min(nb, na)``),
    so this re-pairing can never change added/removed/modified/unchanged
    counts - only *which* line is shown as modified.
    """
    pairs: list[tuple[int, int]] = []
    used_after: set[int] = set()
    for b, line in enumerate(before):
        name = _result_name(line)
        if name is None:
            continue
        for a, candidate in enumerate(after):
            if a not in used_after and _result_name(candidate) == name:
                used_after.add(a)
                pairs.append((b, a))
                break
    paired_before = {b for b, _ in pairs}
    leftover_before = [b for b in range(len(before)) if b not in paired_before]
    leftover_after = [a for a in range(len(after)) if a not in used_after]
    pairs.extend(zip(leftover_before, leftover_after))
    pairs.sort()
    return pairs


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
            # Pair the block's lines as modified so a rewrite renders as one
            # red/green pair rather than an unrelated delete+insert. Pairing is
            # result-target aware (_replace_pairs) so the pair shown is the
            # statement that actually changed - not a positional LCS artifact
            # that happens to line up because earlier deletions shifted rows.
            # Entries are emitted in original-program order (each before-line
            # appears exactly once, modified or deleted, where it was), so a
            # diff panel reads top to bottom like the input it came from.
            block_before = before_lines[i1:i2]
            block_after = after_lines[j1:j2]
            pairs = _replace_pairs(block_before, block_after)
            after_for_before = {b: a for b, a in pairs}
            paired_after = {a for _, a in pairs}
            for offset in range(len(block_before)):
                paired = after_for_before.get(offset)
                if paired is not None:
                    entries.append(
                        {
                            "op": "replace",
                            "status": "modified",
                            "before_text": block_before[offset],
                            "after_text": block_after[paired],
                            "before_index": i1 + offset,
                            "after_index": j1 + paired,
                        }
                    )
                else:
                    entries.append(
                        {
                            "op": "delete",
                            "status": "removed",
                            "before_text": block_before[offset],
                            "after_text": None,
                            "before_index": i1 + offset,
                            "after_index": None,
                        }
                    )
            for offset in range(len(block_after)):
                if offset not in paired_after:
                    entries.append(
                        {
                            "op": "insert",
                            "status": "added",
                            "before_text": None,
                            "after_text": block_after[offset],
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
