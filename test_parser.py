import sys
sys.path.insert(0, r'D:\PROJECTS\compiler')
from backend.tac.parser import constant_folding_example
from backend.tac.instruction import Instruction

tac = constant_folding_example()
print('Current instructions:')
for i, instr in enumerate(tac.instructions):
    print(f'  {i}: opcode={instr.opcode!r}, result={instr.result!r}, arg1={instr.arg1!r}, arg2={instr.arg2!r}')
    print(f'     repr: {repr(instr)}')