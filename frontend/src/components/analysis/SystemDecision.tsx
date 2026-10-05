import { Play, Zap } from "lucide-react";
import type { SelectedStrategy } from "../../types";
import { formatScore } from "../../lib/utils";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";

export function SystemDecision({
  selected,
  onApply,
  onSimulate,
  loading,
}: {
  selected?: SelectedStrategy;
  onApply: () => void;
  onSimulate: () => void;
  loading: boolean;
}) {
  return (
    <section id="system-decision" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <div className="glass aurora relative overflow-hidden rounded-3xl border-2 border-cyan-500/40 p-8 shadow-[0_0_50px_rgba(34,211,238,0.15)] md:p-10">
          <div className="pointer-events-none absolute right-0 top-0 h-96 w-96 rounded-full bg-gradient-to-bl from-cyan-500/10 to-violet-500/5 blur-3xl" />

          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
            <div>
              <div className="flex items-center gap-2 font-mono text-xs font-semibold tracking-[0.25em] text-cyan-400 uppercase">
                <Zap className="h-3.5 w-3.5 fill-current" />
                SYSTEM DECISION
              </div>
              <div className="mt-1 font-mono text-sm tracking-widest text-slate-400 uppercase">
                BEST NEXT MOVE
              </div>
            </div>
            <Badge
              tone="cyan"
              className="px-3 py-1 font-mono text-xs font-bold tracking-widest uppercase border-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.4)]"
            >
              DYNAMIC SELECTION
            </Badge>
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <div className="font-mono text-4xl font-extrabold tracking-tight text-white uppercase sm:text-5xl">
                {selected?.name ?? "CONSTANT FOLDING"}
              </div>

              <div className="mt-6 flex items-center gap-4">
                <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 font-mono">
                  <span className="text-xs text-slate-400 uppercase tracking-wider block">Score:</span>
                  <span className="text-2xl font-bold text-cyan-300">
                    {selected ? formatScore(selected.score) : "0.91"}
                  </span>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 font-mono">
                  <span className="text-xs text-slate-400 uppercase tracking-wider block">Decision Mode:</span>
                  <span className="text-sm font-semibold text-emerald-400">
                    Rule-Based Highest Rank
                  </span>
                </div>
              </div>

              <div className="mt-6">
                <span className="text-xs font-mono uppercase tracking-widest text-slate-400">
                  Reason:
                </span>
                <p className="mt-2 text-base leading-7 text-slate-200 font-sans italic border-l-2 border-cyan-400 pl-4 py-1 bg-cyan-950/20 rounded-r-lg">
                  "{selected?.reason ?? "Multiple constant expressions detected."}"
                </p>
              </div>

              <p className="mt-4 text-xs font-mono text-slate-500">
                Rule-based strategy selection evaluates features at this iteration and prioritizes maximum instruction reduction.
              </p>
            </div>

            <div className="flex flex-col justify-center gap-3 border-t border-white/8 pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
              <Button
                variant="brutal"
                size="lg"
                onClick={onApply}
                disabled={loading || !selected}
                className="w-full justify-center shadow-[0_0_24px_rgba(255,75,75,0.4)] font-mono tracking-wider"
              >
                <Zap className="h-4 w-4" />
                APPLY SELECTED STRATEGY
              </Button>

              <Button
                variant="secondary"
                size="lg"
                onClick={onSimulate}
                disabled={loading}
                className="w-full justify-center font-mono tracking-wider hover:border-cyan-400"
              >
                <Play className="h-4 w-4 fill-current text-cyan-400" />
                RUN FULL SIMULATION
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
