import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { DEMO_ANALYZE, DEMO_RESPONSE } from "../data/demo";
import { DEFAULT_TAC } from "../data/scenarios";
import {
  analyzeTac,
  ApiError,
  getHealth,
  optimizeTac,
  simulateTac,
} from "../services/api";
import { isFinalResult, type OptimizationResponse, type Scenario } from "../types";

export function useOptimization() {
  const [tac, setTac] = useState(DEFAULT_TAC);
  const [demoMode, setDemoMode] = useState(false);
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<OptimizationResponse | null>(null);
  const [simulation, setSimulation] = useState<OptimizationResponse | null>(null);
  const [pipelineStage, setPipelineStage] = useState(0);
  const [facultyDemo, setFacultyDemo] = useState(false);
  const [facultyStep, setFacultyStep] = useState(0);

  const probeHealth = useCallback(async () => {
    try {
      const health = await getHealth();
      const ready = health.ready === true;
      setLive(ready);
      setDemoMode(!ready);
    } catch {
      setLive(false);
      setDemoMode(true);
    }
  }, []);

  useEffect(() => {
    void probeHealth();
  }, [probeHealth]);

  const runAnalyze = useCallback(async () => {
    if (!tac.trim()) {
      toast.error("TAC source is empty.");
      return;
    }
    setLoading(true);
    setPipelineStage(3);
    try {
      const result = await analyzeTac(tac);
      setAnalysis(result);
      setSimulation(null);
      setDemoMode(false);
      setLive(true);
      setPipelineStage(4);
      toast.success("TAC analyzed with rule-based scoring.");
    } catch (error) {
      if (error instanceof ApiError && error.code === "invalid") {
        toast.error(error.message);
        setPipelineStage(0);
      } else {
        setAnalysis(DEMO_ANALYZE);
        setSimulation(null);
        setDemoMode(true);
        setLive(false);
        setPipelineStage(4);
        toast.message("Backend unavailable. Showing deterministic demo analysis.");
      }
    } finally {
      setLoading(false);
    }
  }, [tac]);

  const runOptimize = useCallback(async () => {
    if (!tac.trim()) {
      toast.error("TAC source is empty.");
      return;
    }
    setLoading(true);
    setPipelineStage(5);
    try {
      const result = await optimizeTac(tac, analysis?.selected_strategy.id);
      setSimulation(result);
      setAnalysis((prev) =>
        prev
          ? {
              ...prev,
              strategies: result.strategies.length ? result.strategies : prev.strategies,
              selected_strategy: result.selected_strategy.id
                ? result.selected_strategy
                : prev.selected_strategy,
            }
          : result,
      );
      setTac(result.tac.join("\n"));
      setDemoMode(false);
      setLive(true);
      setPipelineStage(6);
      toast.success("Selected strategy applied.");
    } catch (error) {
      if (error instanceof ApiError && error.code === "invalid") {
        toast.error(error.message);
      } else {
        setSimulation(DEMO_RESPONSE);
        setAnalysis(DEMO_RESPONSE);
        setTac(
          isFinalResult(DEMO_RESPONSE.final_result)
            ? DEMO_RESPONSE.final_result.final_tac.join("\n")
            : DEMO_RESPONSE.tac.join("\n"),
        );
        setDemoMode(true);
        setPipelineStage(8);
        toast.message("Backend unavailable. Showing deterministic demo optimization.");
      }
    } finally {
      setLoading(false);
    }
  }, [analysis?.selected_strategy.id, tac]);

  const runSimulate = useCallback(async () => {
    if (!tac.trim()) {
      toast.error("TAC source is empty.");
      return;
    }
    setLoading(true);
    setPipelineStage(5);
    try {
      const result = await simulateTac(tac);
      setSimulation(result);
      setAnalysis({
        ...result,
        iterations: result.iterations,
      });
      if (isFinalResult(result.final_result) && result.final_result.final_tac.length) {
        setTac(result.final_result.final_tac.join("\n"));
      } else if (result.tac.length) {
        setTac(result.tac.join("\n"));
      }
      setDemoMode(false);
      setLive(true);
      const status = isFinalResult(result.final_result) ? result.final_result.status : "";
      setPipelineStage(status.toLowerCase().includes("converged") ? 8 : 7);
      toast.success("Iterative optimization complete.");
    } catch (error) {
      if (error instanceof ApiError && error.code === "invalid") {
        toast.error(error.message);
      } else {
        setSimulation(DEMO_RESPONSE);
        setAnalysis(DEMO_RESPONSE);
        setTac(
          isFinalResult(DEMO_RESPONSE.final_result)
            ? DEMO_RESPONSE.final_result.final_tac.join("\n")
            : DEMO_RESPONSE.tac.join("\n"),
        );
        setDemoMode(true);
        setPipelineStage(8);
        toast.message("Backend unavailable. Showing deterministic demo simulation.");
      }
    } finally {
      setLoading(false);
    }
  }, [tac]);

  const reset = useCallback(() => {
    setTac(DEFAULT_TAC);
    setAnalysis(null);
    setSimulation(null);
    setPipelineStage(0);
    toast.success("Editor reset to default TAC.");
  }, []);

  const loadScenario = useCallback((scenario: Scenario) => {
    setTac(scenario.tac);
    setAnalysis(null);
    setSimulation(null);
    setPipelineStage(1);
    toast.success(`${scenario.title} loaded into TAC editor.`);
  }, []);

  const display = useMemo(() => {
    if (simulation) return simulation;
    if (analysis) return analysis;
    return demoMode ? DEMO_ANALYZE : null;
  }, [analysis, demoMode, simulation]);

  return {
    tac,
    setTac,
    demoMode,
    live,
    loading,
    analysis,
    simulation,
    display,
    pipelineStage,
    setPipelineStage,
    facultyDemo,
    setFacultyDemo,
    facultyStep,
    setFacultyStep,
    runAnalyze,
    runOptimize,
    runSimulate,
    reset,
    loadScenario,
    probeHealth,
  };
}
