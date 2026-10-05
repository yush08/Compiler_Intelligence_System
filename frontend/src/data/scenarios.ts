import type { Scenario } from "../types";

export const DEFAULT_TAC = `t1 = 10
t2 = 20
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
t6 = 100
t7 = t6 + 50
unused = t7
t8 = t3 * 2`;

export const SCENARIOS: Scenario[] = [
  {
    id: "constant_folding",
    title: "Constant Folding",
    description: "Arithmetic-heavy TAC with constant expressions.",
    tac: `t1 = 10 + 20
t2 = 5 * 4
t3 = 100 - 25
t4 = 8 / 2
t5 = t1 + t2`,
  },
  {
    id: "cse",
    title: "Common Subexpression",
    description: "Repeated expressions suitable for CSE.",
    tac: `t1 = a + b
t2 = a + b
t3 = c + d
t4 = a + b
t5 = t1 * 2`,
  },
  {
    id: "dead_code",
    title: "Dead Code",
    description: "Unused assignments and temporary values.",
    tac: `t1 = a + b
t2 = 10
unused = t2
x = t1
dead = 99
y = x`,
  },
  {
    id: "mixed",
    title: "Mixed Workload",
    description: "Multiple optimization opportunities.",
    tac: DEFAULT_TAC,
    recommended: true,
  },
];
