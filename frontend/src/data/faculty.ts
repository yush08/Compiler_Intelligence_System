export const FACULTY_STEPS = [
  {
    id: "tac-editor",
    title: "STEP 1",
    label: "Enter TAC",
    detail: "Load the canonical Mixed Workload or paste custom Three Address Code with arithmetic, repeated expressions, and dead assignments.",
  },
  {
    id: "code-dna",
    title: "STEP 2",
    label: "Extract Code DNA",
    detail: "Extract 8 measurable features from the TAC: statements, constants, repeated subexpressions, dead assignments, and algebraic opportunities.",
  },
  {
    id: "strategy-matrix",
    title: "STEP 3",
    label: "Score strategies",
    detail: "Rule-based scoring evaluates all 6 optimization passes deterministically from code features. (No black-box machine learning).",
  },
  {
    id: "system-decision",
    title: "STEP 4",
    label: "Select strategy",
    detail: "Select the strategy with the highest opportunity score (e.g. Constant Folding at score 0.91) with human-readable rationale.",
  },
  {
    id: "system-decision",
    title: "STEP 5",
    label: "Apply optimization",
    detail: "Execute the selected optimization pass on the TAC representation to transform the instruction stream.",
  },
  {
    id: "iterations",
    title: "STEP 6",
    label: "Measure gain",
    detail: "Measure reduction in total instruction count and compute delta percentage after pass execution (e.g. 9 → 7 instructions).",
  },
  {
    id: "iterations",
    title: "STEP 7",
    label: "Repeat",
    detail: "Re-extract features from the modified TAC, re-score strategies, and iteratively apply the next best pass (CSE, Algebraic Simplification, Dead Code).",
  },
  {
    id: "analysis",
    title: "STEP 8",
    label: "Converge",
    detail: "Stop when subsequent passes yield zero further instruction reduction. Display final Before/After diff and 55.6% total reduction summary.",
  },
];
