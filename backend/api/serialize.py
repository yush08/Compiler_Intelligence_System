"""Serialize TAC objects for JSON API responses without changing optimizer logic."""

from backend.tac.instruction import Instruction, TAC

OP_SYMBOLS = {
    Instruction.ADD: "+",
    Instruction.SUB: "-",
    Instruction.MUL: "*",
    Instruction.DIV: "/",
    Instruction.MOD: "%",
}

STRATEGY_DISPLAY = {
    "constant_folding": "Constant Folding",
    "constant_propagation": "Constant Propagation",
    "common_subexpression": "Common Subexpression Elimination",
    "dead_code_elimination": "Dead Code Elimination",
    "algebraic_simplification": "Algebraic Simplification",
    "strength_reduction": "Strength Reduction",
}


def instruction_to_line(instr: Instruction) -> str:
    """Render one instruction as TAC source text."""
    op = instr.opcode
    if op == Instruction.ASSIGN:
        return f"{instr.result} = {instr.arg1}"
    if op == Instruction.CONST:
        return f"{instr.result} = {instr.arg1}"
    if op == Instruction.LABEL:
        return f"{instr.result}:"
    if op == Instruction.GOTO:
        return f"goto {instr.result}"
    if op == Instruction.IF:
        return f"if {instr.arg1} {instr.arg2} goto {instr.result}"
    if op == Instruction.UMINUS:
        return f"{instr.result} = -{instr.arg1}"
    if op in OP_SYMBOLS:
        return f"{instr.result} = {instr.arg1} {OP_SYMBOLS[op]} {instr.arg2}"
    return str(instr)


def tac_to_lines(tac: TAC | None) -> list[str]:
    if tac is None or not getattr(tac, "instructions", None):
        return []
    return [instruction_to_line(instr) for instr in tac.instructions]


def tac_to_source(tac: TAC | None) -> str:
    return "\n".join(tac_to_lines(tac))
