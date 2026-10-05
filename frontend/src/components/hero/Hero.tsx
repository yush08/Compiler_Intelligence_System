import { motion } from "framer-motion";
import { ArrowRight, Cpu } from "lucide-react";
import { Button } from "../ui/button";
import { PipelineStrip } from "./PipelineStrip";

export function Hero({
  live,
  onStart,
  onExplore,
  pipelineStage,
}: {
  live: boolean;
  onStart: () => void;
  onExplore: () => void;
  pipelineStage: number;
}) {
  return (
    <section id="overview" className="relative overflow-hidden px-6 pb-10 pt-14">
      <div className="pointer-events-none absolute left-1/2 top-0 h-64 w-[48rem] -translate-x-1/2 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="mx-auto max-w-7xl">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">
          <div className="section-kicker">COMPILER INTELLIGENCE SYSTEM</div>
          <h1 className="mt-4 text-5xl font-semibold leading-[1.05] tracking-tight text-white md:text-6xl">
            Automated Optimization
            <br />
            Strategy Selection
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400">
            An adaptive compiler optimization engine that analyzes Three Address Code, extracts code
            characteristics, evaluates six optimization strategies, and dynamically selects the most
            suitable optimization at each iteration.
          </p>
          <div className="mt-6 flex items-center gap-2 text-sm font-mono tracking-wider">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                live
                  ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.9)]"
                  : "bg-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.9)]"
              }`}
            />
            <span className={live ? "text-emerald-300 font-semibold" : "text-amber-300 font-semibold"}>
              {live ? "SYSTEM READY" : "DEMO MODE (STANDALONE FALLBACK)"}
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-xs text-slate-400">Rule-Based Strategy Engine</span>
          </div>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button variant="brutal" size="lg" onClick={onStart}>
              Start Simulation
              <ArrowRight className="h-4 w-4" />
            </Button>
            <Button variant="secondary" size="lg" onClick={onExplore}>
              <Cpu className="h-4 w-4" />
              Explore Architecture
            </Button>
          </div>
        </motion.div>
        <div className="mt-12">
          <PipelineStrip active={Math.min(7, Math.max(0, pipelineStage))} />
        </div>
      </div>
    </section>
  );
}
