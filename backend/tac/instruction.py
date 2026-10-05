"""
TAC (Three Address Code) Instruction representation.

Each instruction models a three-address code statement such as:
    t1 = a + b
    goto L1
    if x > 0 goto L2
    t = 5
"""


class Instruction:
    """Represents a single Three Address Code instruction."""

    # Opcodes
    ASSIGN = "assign"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    CONST = "const"
    LABEL = "label"
    GOTO = "goto"
    IF = "if"
    UMINUS = "uminus"  # unary minus

    def __init__(
        self,
        opcode: str,
        result: str | None = None,
        arg1: str | None = None,
        arg2: str | None = None,
        label: str | None = None,
    ):
        self.opcode = opcode
        self.result = result  # target variable/temp
        self.arg1 = arg1  # first operand
        self.arg2 = arg2  # second operand
        self.label = label  # label for jumps/branches

    def __repr__(self):
        if self.opcode == self.ASSIGN:
            if self.arg2 is None:
                return f"{self.result} = {self.arg1}"
            return f"{self.result} = {self.arg1} {self.opcode} {self.arg2}"
        elif self.opcode == self.LABEL:
            return f"{self.result}:"
        elif self.opcode == self.GOTO:
            return f"goto {self.result}"
        elif self.opcode == self.IF:
            return f"if {self.arg1} {self.opcode} {self.arg2} goto {self.result}"
        else:
            return f"{self.result} = {self.arg1} {self.opcode} {self.arg2}"

    def is_assignment(self) -> bool:
        return self.opcode == self.ASSIGN

    def is_binary_op(self) -> bool:
        return self.opcode in {
            self.ADD, self.SUB, self.MUL, self.DIV, self.MOD,
        }

    def is_commutative(self) -> bool:
        """Return True if the operator is commutative (order doesn't matter)."""
        return self.opcode in {self.ADD, self.MUL}

    def has_constant_result(self) -> bool:
        """Check if both arg1 and arg2 are constants (for folding)."""
        if self.arg2 is None:
            return False
        return self._is_constant(self.arg1) and self._is_constant(self.arg2)

    @staticmethod
    def _is_constant(s: str) -> bool:
        """Check if a string looks like a constant number."""
        try:
            float(s)
            return True
        except ValueError:
            return False


class TAC:
    """Container for a list of TAC instructions."""

    def __init__(self, instructions=None):
        self.instructions = instructions if instructions is not None else []

    def add(self, instr: Instruction):
        self.instructions.append(instr)

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return self.instructions[idx]

    def __repr__(self):
        lines = [f"{i+1} {str(instr)}" for i, instr in enumerate(self.instructions)]
        return "\n".join(lines)

    def get_used_vars(self) -> set:
        """Get all variables that are results of instructions (used as operands)."""
        used = set()
        for instr in self.instructions:
            if instr.result is not None:
                used.add(instr.result)
            if instr.arg1 is not None:
                used.add(instr.arg1)
            if instr.arg2 is not None:
                used.add(instr.arg2)
        return used

    def get_assigned_vars(self) -> set:
        """Get all variables that are assigned (results of instructions)."""
        assigned = set()
        for instr in self.instructions:
            if instr.result is not None:
                assigned.add(instr.result)
        return assigned