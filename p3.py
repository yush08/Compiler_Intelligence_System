import sys, inspect
sys.path.insert(0, ".")
from backend.optimizations import constant_folding as cf, constant_propagation as cp, common_subexpression as cs
from backend.tac.instruction import Instruction
print("is_assignment() definition:", inspect.getsource(Instruction.is_assignment).strip().splitlines()[-1])
print("is_binary_op()  definition:", inspect.getsource(Instruction.is_binary_op).strip())
