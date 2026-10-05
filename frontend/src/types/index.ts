export interface OptimizationFeatures {
  num_statements: number;
  num_constants: number;
  num_repeated_expr: number;
  num_dead_stmts: number;
  num_algebraic_ops: number;
  num_loop_ops: number;
  num_tmp_vars: number;
  const_ratio: number;
}

export interface StrategyScore {
  id: string;
  name: string;
  score: number;
  reason: string;
  opportunity: boolean;
  selected: boolean;
}

export interface SelectedStrategy {
  id: string;
  name: string;
  score: number;
  reason: string;
}

export type IterationStatus = "applied" | "converged" | "no_change";

export interface OptimizationIteration {
  iteration: number;
  strategy: string;
  strategy_name: string;
  score: number;
  before_count: number;
  after_count: number;
  reduction: number;
  improvement_percentage: number;
  tac_before: string[];
  tac_after: string[];
  status: IterationStatus;
}

export interface FinalResult {
  original_count: number;
  final_count: number;
  total_reduction: number;
  reduction_percentage: number;
  iterations: number;
  status: string;
  original_tac: string[];
  final_tac: string[];
}

export interface OptimizationResponse {
  tac: string[];
  features: OptimizationFeatures;
  strategies: StrategyScore[];
  selected_strategy: SelectedStrategy;
  iterations: OptimizationIteration[];
  final_result: FinalResult | Record<string, never>;
}

export interface HealthResponse {
  status: string;
  system: string;
  mode: string;
  ready: boolean;
}

export interface Scenario {
  id: string;
  title: string;
  description: string;
  tac: string;
  recommended?: boolean;
}

export function isFinalResult(
  value: OptimizationResponse["final_result"] | undefined,
): value is FinalResult {
  return Boolean(value && "original_count" in value && "final_tac" in value);
}
