import { useState } from "react";
import { AdaptiveComparison } from "../components/analysis/AdaptiveComparison";
import { ArchitectureDiagram } from "../components/analysis/ArchitectureDiagram";
import { BeforeAfterTac } from "../components/analysis/BeforeAfterTac";
import { CodeDna } from "../components/analysis/CodeDna";
import { FinalResults } from "../components/analysis/FinalResults";
import { HistoryTable } from "../components/analysis/HistoryTable";
import { IterationTimeline } from "../components/analysis/IterationTimeline";
import { OptimizationStepper } from "../components/analysis/OptimizationStepper";
import { StrategyMatrix } from "../components/analysis/StrategyMatrix";
import { SystemDecision } from "../components/analysis/SystemDecision";
import { FacultyDemo } from "../components/faculty/FacultyDemo";
import { Hero } from "../components/hero/Hero";
import { Navbar } from "../components/layout/Navbar";
import { ScenarioCards } from "../components/simulation/ScenarioCards";
import { TacEditor } from "../components/simulation/TacEditor";
import { useOptimization } from "../hooks/useOptimization";
import { isFinalResult } from "../types";

export function Dashboard() {
  const opt = useOptimization();
  const [scenarioId, setScenarioId] = useState("mixed");
  const display = opt.display;
  const final = display && isFinalResult(display.final_result) ? display.final_result : undefined;
  const iterations = display?.iterations ?? [];

  return (
    <div className="min-h-screen pb-24">
      <Navbar
        demoMode={opt.demoMode}
        live={opt.live}
        facultyDemo={opt.facultyDemo}
        onFacultyDemo={() => {
          const next = !opt.facultyDemo;
          opt.setFacultyDemo(next);
          if (next) {
            opt.setFacultyStep(0);
            document.getElementById("tac-editor")?.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }}
      />
      <Hero
        live={opt.live}
        pipelineStage={opt.pipelineStage}
        onStart={() => document.getElementById("simulation")?.scrollIntoView({ behavior: "smooth" })}
        onExplore={() => document.getElementById("architecture")?.scrollIntoView({ behavior: "smooth" })}
      />
      <ScenarioCards
        selectedId={scenarioId}
        onSelect={(scenario) => {
          setScenarioId(scenario.id);
          opt.loadScenario(scenario);
        }}
      />
      <TacEditor
        value={opt.tac}
        onChange={opt.setTac}
        onAnalyze={() => void opt.runAnalyze()}
        onReset={opt.reset}
        loading={opt.loading}
      />
      <CodeDna features={display?.features} />
      <StrategyMatrix strategies={display?.strategies ?? []} />
      <SystemDecision
        selected={display?.selected_strategy}
        onApply={() => void opt.runOptimize()}
        onSimulate={() => void opt.runSimulate()}
        loading={opt.loading}
      />
      <OptimizationStepper current={opt.pipelineStage} />
      <IterationTimeline iterations={iterations} />
      <BeforeAfterTac result={final} />
      <FinalResults result={final} />
      <HistoryTable iterations={iterations} />
      <AdaptiveComparison />
      <ArchitectureDiagram />
      <FacultyDemo
        open={opt.facultyDemo}
        step={opt.facultyStep}
        onStep={opt.setFacultyStep}
        onClose={() => opt.setFacultyDemo(false)}
      />
    </div>
  );
}
