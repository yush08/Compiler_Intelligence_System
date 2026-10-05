import type { OptimizationIteration } from "../../types";
import { formatScore } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { SectionHeader } from "../ui/SectionHeader";

export function IterationTimeline({ iterations }: { iterations: OptimizationIteration[] }) {
  return (
    <section id="iterations" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="ITERATION TIMELINE"
          title="Optimization Pipeline History"
          description="Sequential optimization passes selected dynamically. Optimization converges when instruction delta reaches zero."
        />
        <div className="relative ml-4 border-l-2 border-white/10 pl-8 space-y-6">
          {iterations.length === 0 ? (
            <div className="glass rounded-2xl p-6 text-sm text-slate-400">
              Run a simulation to populate the iterative optimization history from the backend engine.
            </div>
          ) : (
            iterations.map((item) => {
              const isConverged = item.status === "converged" || item.reduction === 0;

              return (
                <div key={item.iteration} className="relative group">
                  {/* Node Dot on Timeline */}
                  <span
                    className={`absolute -left-[41px] top-6 flex h-5 w-5 items-center justify-center rounded-full border-2 bg-[#03050a] ${
                      isConverged
                        ? "border-emerald-400 text-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.8)]"
                        : "border-cyan-400 text-cyan-400 shadow-[0_0_16px_rgba(34,211,238,0.8)]"
                    }`}
                  >
                    <span
                      className={`h-2 w-2 rounded-full ${
                        isConverged ? "bg-emerald-400" : "bg-cyan-400"
                      }`}
                    />
                  </span>

                  <div
                    className={`glass relative rounded-2xl p-6 transition-all duration-200 ${
                      isConverged
                        ? "border-emerald-500/30 bg-emerald-950/10 shadow-[0_0_24px_rgba(52,211,153,0.1)]"
                        : "border-white/8 hover:border-white/20"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <div className="font-mono text-xs font-semibold tracking-[0.2em] text-cyan-400">
                          ITERATION {item.iteration}
                        </div>
                        <div className="mt-1 text-xl font-bold tracking-tight text-white">
                          {item.strategy_name}
                        </div>
                      </div>

                      {/* Prominent Transition Pill: e.g. 9 → 7 or CONVERGED */}
                      <div className="flex items-center gap-3">
                        {isConverged ? (
                          <div className="rounded-xl border border-emerald-400/50 bg-emerald-500/10 px-4 py-1.5 font-mono text-base font-bold text-emerald-300 shadow-[0_0_14px_rgba(52,211,153,0.3)]">
                            CONVERGED
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-cyan-500/10 px-4 py-1.5 font-mono text-base font-bold text-cyan-200">
                            <span>{item.before_count}</span>
                            <span className="text-cyan-400">→</span>
                            <span className="text-white">{item.after_count}</span>
                          </div>
                        )}
                        <Badge tone={isConverged ? "emerald" : "cyan"}>
                          {isConverged ? "Finished" : `Reduction: -${item.reduction}`}
                        </Badge>
                      </div>
                    </div>

                    <div className="mt-4 grid gap-3 font-mono text-xs text-slate-300 sm:grid-cols-4 border-t border-white/6 pt-3">
                      <div>
                        <span className="text-slate-500 block">Score:</span>
                        <span className="font-semibold text-slate-200">
                          {item.score ? formatScore(item.score) : "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Before:</span>
                        <span className="font-semibold text-slate-200">
                          {item.before_count} instructions
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">After:</span>
                        <span className="font-semibold text-slate-200">
                          {item.after_count} instructions
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Gain:</span>
                        <span
                          className={`font-semibold ${
                            item.reduction > 0 ? "text-emerald-400" : "text-slate-400"
                          }`}
                        >
                          {item.reduction > 0
                            ? `-${item.reduction} (${item.improvement_percentage}%)`
                            : "0 (converged)"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
}
