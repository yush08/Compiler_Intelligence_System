import sys
sys.path.insert(0, r'D:\PROJECTS\compiler')
from backend.tac.parser import constant_folding_example
from backend.features.extractor import extract_features

tac = constant_folding_example()
print('TAC instructions:')
for i, instr in enumerate(tac.instructions):
    print(f'  {i}: opcode={instr.opcode!r}, result={instr.result!r}, arg1={instr.arg1!r}, arg2={instr.arg2!r}')

features = extract_features(tac)
print(f'\nFeatures: {features}')