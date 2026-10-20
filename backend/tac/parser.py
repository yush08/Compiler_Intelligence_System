"""
TAC (Three Address Code) Parser.

Accepts TAC source in the format:
    t1 = 5 + 3
    t2 = a + b
    t3 = a + b
    t4 = t1 * 1
    t5 = 100
    x = t4
    L1:
    if x > 0 goto L2
    goto L1
    L2:

Supports:
- Assignment: x = y, x = 5, t = a + b
- Binary operations: +, -, *, /, %
- Constants: integer/float literals
- Labels: identifier followed by colon
- Goto: goto label
- Conditional if: if condition goto label
- Observable output: print x   (and print 42)
"""

import re
from .instruction import Instruction, TAC


def parse_tac(source: str) -> TAC:
    """
    Parse a TAC source string into a TAC object.

    Supported formats:
        # comments
        t1 = 5 + 3          (binary operation with constants)
        t2 = a + b          (binary operation with variables)
        t1 = 5              (constant assignment)
        t = -3              (unary minus)
        L1:                 (label)
        goto L2             (unconditional jump)
        if x > 0 goto L1   (conditional jump)
        print result        (observable output of a variable or literal)
    """
    tac = TAC()
    lines = source.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        instr = _parse_line(line)
        if instr is not None:
            tac.add(instr)
    return tac


def _parse_line(line: str) -> Instruction | None:
    """Parse a single line of TAC."""
    # Label: "L1:" or "L1: "
    label_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
    if label_match:
        label_name = label_match.group(1)
        rest = line[label_match.end():].strip()
        instr = Instruction(Instruction.LABEL, result=label_name)
        return instr

    # Goto: "goto L1"
    goto_match = re.match(r"^goto\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
    if goto_match:
        target = goto_match.group(1)
        return Instruction(Instruction.GOTO, result=target)

    # Conditional: "if x > 0 goto L1"
    if_match = re.match(
        r"^if\s+(.+?)\s+(>=|<=|>|<|==|!=)\s+(.+?)\s+goto\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
        line,
    )
    if if_match:
        arg1 = if_match.group(1)
        op = if_match.group(2)
        arg2 = if_match.group(3)
        label = if_match.group(4)
        return Instruction(
            Instruction.IF,
            result=label,
            arg1=arg1,
            arg2=arg2,
            relop=op,
        )

    # Observable output: "print result" or "print 42".
    # This used to fall through to `return None`, i.e. the line was silently
    # dropped, which is why an optimizer could delete everything a program
    # computed "for" its output: the output statement was never in the TAC.
    print_match = re.match(
        r"^print\s+([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)\s*$",
        line,
    )
    if print_match:
        return Instruction(Instruction.PRINT, arg1=print_match.group(1))

    # Assignment: "result = arg1 op arg2" or "result = value"
    assign_match = re.match(
        r"^(\w+)\s*=\s*(.+)$", line
    )
    if assign_match:
        result = assign_match.group(1)
        rhs = assign_match.group(2).strip()
        return _parse_rhs(rhs, result)

    # Standalone value assignment (e.g., "t = 5")
    # Try: result = value
    simple_assign = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", line)
    if simple_assign:
        result = simple_assign.group(1)
        value = simple_assign.group(2).strip()
        return Instruction(Instruction.ASSIGN, result=result, arg1=value)

    return None


def _parse_rhs(rhs: str, result: str) -> Instruction:
    """Parse the right-hand side of an assignment."""
    # Check for unary minus: "-5" or "-t"
    if rhs.startswith("-") and len(rhs) > 1:
        operand = rhs[1:].strip()
        if Instruction._is_constant(operand) or _is_temp_or_var(operand):
            return Instruction(
                Instruction.UMINUS,
                result=result,
                arg1=operand,
            )

    # Check for constant: "5" or "3.14"
    if Instruction._is_constant(rhs):
        return Instruction(Instruction.CONST, result=result, arg1=rhs)

    # Check for binary operation: "a + b", "x * 5", "t1 / 2"
    # Pattern: arg1 op arg2
    bio_match = re.match(
        r"^(.+?)\s*(==|!=|<=|>=|<|>)\s+(.+)$", rhs
    )
    if bio_match:
        arg1 = bio_match.group(1).strip()
        op = bio_match.group(2)
        arg2 = bio_match.group(3).strip()
        return Instruction(
            Instruction.IF,  # reuse IF opcode for comparison, but treat as binary
            result=result,
            arg1=arg1,
            arg2=arg2,
            relop=op,
        )

    # Binary arithmetic: "a + b", "t1 * 3", "x - y", "a % b"
    # Try to match: arg1 operator arg2
    arith_match = re.match(
        r"^(.+?)\s*([+\-*/%])\s+(.+)$", rhs
    )
    if arith_match:
        arg1 = arith_match.group(1).strip()
        op = arith_match.group(2)
        arg2 = arith_match.group(3).strip()
        # The module docstring promises % support and Instruction.is_binary_op()
        # includes MOD, but the pattern omitted it, so `t1 = a % b` silently
        # degraded to a copy with arg1="a % b": constant folding could not fold
        # it and the feature extractor never saw a constant operand.
        opcode = {
            "+": Instruction.ADD,
            "-": Instruction.SUB,
            "*": Instruction.MUL,
            "/": Instruction.DIV,
            "%": Instruction.MOD,
        }[op]
        return Instruction(opcode, result=result, arg1=arg1, arg2=arg2)

    # Single operand assignment (e.g., "t = a")
    return Instruction(Instruction.ASSIGN, result=result, arg1=rhs.strip())


def _is_temp_or_var(s: str) -> bool:
    """Check if a string looks like a variable or temporary (not a constant)."""
    # A constant is just a number; anything starting with a letter is a var/temp
    return s[0].isalpha() or s[0] == "_" if s else False


# Convenience function for common test cases
def constant_folding_example() -> TAC:
    """Example TAC for constant folding."""
    tac = TAC()
    # t1 = 5 + 3 (binary operation with constants)
    tac.add(Instruction(Instruction.ADD, result="t1", arg1="5", arg2="3"))
    # t2 = a + b (binary operation with variables)
    tac.add(Instruction(Instruction.ADD, result="t2", arg1="a", arg2="b"))
    # t3 = t1 * 1 (multiplication with constant)
    tac.add(Instruction(Instruction.MUL, result="t3", arg1="t1", arg2="1"))
    return tac


def dead_code_example() -> TAC:
    """Example TAC for dead code elimination."""
    tac = TAC()
    # t1 = a + b (dead if t1 never used later)
    tac.add(Instruction(Instruction.ADD, result="t1", arg1="a", arg2="b"))
    # x = 10 (not dead - used as constant)
    tac.add(Instruction(Instruction.ASSIGN, result="x", arg1="10"))
    # y = t1 (uses t1, so t1 is not dead)
    # Actually let's make t1 truly dead - never used
    return tac


def cse_example() -> TAC:
    """Example TAC for common subexpression elimination."""
    tac = TAC()
    # t1 = a + b (first occurrence)
    tac.add(Instruction(Instruction.ADD, result="t1", arg1="a", arg2="b"))
    # t2 = a + b (second occurrence - should be eliminated, replaced with t2 = t1)
    tac.add(Instruction(Instruction.ADD, result="t2", arg1="a", arg2="b"))
    # t3 = t1 * 1 (another instruction)
    tac.add(Instruction(Instruction.MUL, result="t3", arg1="t1", arg2="1"))
    return tac


def algebraic_simplification_example() -> TAC:
    """Example TAC for algebraic simplification."""
    tac = TAC()
    # t1 = 5 + 0 (should simplify to t1 = 5)
    tac.add(Instruction(Instruction.ADD, result="t1", arg1="5", arg2="0"))
    # t2 = 5 * 1 (should simplify to t2 = 5)
    tac.add(Instruction(Instruction.MUL, result="t2", arg1="5", arg2="1"))
    # t3 = 10 - 0 (should simplify to t3 = 10)
    tac.add(Instruction(Instruction.SUB, result="t3", arg1="10", arg2="0"))
    return tac


def mixed_example() -> TAC:
    """Example TAC with multiple optimization opportunities."""
    tac = TAC()
    # t1 = 5 + 3 (constants - opportunity for constant folding)
    tac.add(Instruction(Instruction.ADD, result="t1", arg1="5", arg2="3"))
    # t2 = a + b (repeated expression - opportunity for CSE)
    tac.add(Instruction(Instruction.ADD, result="t2", arg1="a", arg2="b"))
    # t3 = a + b (repeated expression - CSE target)
    tac.add(Instruction(Instruction.ADD, result="t3", arg1="a", arg2="b"))
    # t4 = t1 * 1 (algebraic simplification opportunity)
    tac.add(Instruction(Instruction.MUL, result="t4", arg1="t1", arg2="1"))
    # t5 = 100 (constant)
    tac.add(Instruction(Instruction.CONST, result="t5", arg1="100"))
    # x = t4 (uses t4)
    tac.add(Instruction(Instruction.ASSIGN, result="x", arg1="t4"))
    return tac