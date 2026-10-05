import { motion } from "framer-motion";

export const PIPELINE_STEPS = [
  "TAC",
  "FEATURES",
  "6 STRATEGIES",
  "DECISION",
  "OPTIMIZE",
  "MEASURE",
  "RE-ANALYZE",
  "CONVERGED",
];

export function PipelineStrip({ active = 0 }: { active?: number }) {
  return (
    <div className="w-full">
      <div className="mb-2 flex items-center justify-between text-[11px] font-mono tracking-[0.2em] text-slate-500 uppercase">
        <span>Execution Control Pipeline</span>
        <span className="text-cyan-400">Step {Math.min(8, Math.max(1, active + 1))} of 8</span>
      </div>
      <div className="overflow-x-auto pb-2">
        <div className="flex min-w-[980px] items-center gap-1.5">
          {PIPELINE_STEPS.map((step, index) => {
            const isCompleted = index < active;
            const isCurrent = index === active;

            return (
              <div key={step} className="flex flex-1 items-center gap-1.5">
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                  className={`relative flex h-14 flex-1 flex-col items-center justify-center rounded-xl px-2.5 transition-all duration-300 ${
                    isCurrent
                      ? "aurora border border-cyan-400/60 bg-cyan-950/40 text-cyan-200 shadow-[0_0_24px_rgba(34,211,238,0.25)]"
                      : isCompleted
                        ? "border border-emerald-500/40 bg-emerald-950/20 text-emerald-300"
                        : "border border-white/6 bg-white/[0.02] text-slate-500 hover:border-white/12"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-mono text-[9px] tracking-widest uppercase opacity-70">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    {isCurrent && (
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" />
                    )}
                    {isCompleted && <span className="text-emerald-400">✓</span>}
                  </div>
                  <div className="mt-0.5 text-center font-mono text-[11px] font-semibold tracking-[0.12em] whitespace-nowrap">
                    {step}
                  </div>
                </motion.div>
                {index < PIPELINE_STEPS.length - 1 && (
                  <span
                    className={`font-mono text-xs transition-colors duration-200 ${
                      index < active ? "text-cyan-400 font-bold" : "text-slate-700"
                    }`}
                  >
                    →
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
