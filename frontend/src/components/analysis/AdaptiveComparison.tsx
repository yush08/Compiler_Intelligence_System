import { ArrowRight, CheckCircle2, Shuffle, X } from "lucide-react";
import { SectionHeader } from "../ui/SectionHeader";

const TRADITIONAL = [
  { label: "Apply Constant Folding", detail: "Pass 1 of 6 — always" },
  { label: "Apply Dead Code Elim.", detail: "Pass 2 of 6 — always" },
  { label: "Apply Copy Propagation", detail: "Pass 3 of 6 — always" },
  { label: "Apply CSE", detail: "Pass 4 of 6 — always" },
  { label: "Apply Loop Opt.", detail: "Pass 5 of 6 — always" },
  { label: "Apply Strength Red.", detail: "Pass 6 of 6 — always" },
];

const ADAPTIVE = [
  { label: "Parse & Analyze TAC", detail: "Structural understanding" },
  { label: "Extract 8 Code Metrics", detail: "Const ratio, branches, loops…" },
  { label: "Score All 6 Strategies", detail: "Rule-based heuristic engine" },
  { label: "Select Best Strategy", detail: "Highest-scoring wins" },
  { label: "Apply One Pass", detail: "Targeted transformation" },
  { label: "Re-analyze & Loop", detail: "Repeat until convergence" },
];

const WASTE = [
  "Wastes cycles on irrelevant passes",
  "Cannot adapt to code structure",
  "Fixed ordering misses synergies",
  "No feedback between passes",
];

const GAINS = [
  "Every pass is justified by data",
  "Adapts to actual code patterns",
  "Dynamic ordering finds synergies",
  "Feedback-driven convergence",
];

function TraditionalColumn() {
  return (
    <div className="relative flex flex-col rounded-2xl border border-red-400/10 bg-red-900/5 p-6">
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-full border border-red-400/30 bg-red-400/10">
          <Shuffle className="h-3.5 w-3.5 text-red-400" />
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-red-400/70">
            Traditional
          </div>
          <div className="text-sm font-semibold text-red-200">
            Fixed Sequence
          </div>
        </div>
      </div>
      <div className="mb-5 text-xs text-slate-500">
        Applies the same 6 passes in the same order — regardless of code
        structure.
      </div>

      {/* Steps */}
      <div className="mb-6 space-y-2">
        {TRADITIONAL.map((step, i) => (
          <div key={step.label} className="flex items-start gap-3">
            <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border border-red-400/20 bg-red-900/20 font-mono text-[10px] text-red-400/70">
              {i + 1}
            </div>
            <div>
              <div className="text-sm text-slate-300">{step.label}</div>
              <div className="text-[10px] text-slate-600">{step.detail}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Problems */}
      <div className="mt-auto space-y-2 rounded-xl border border-red-400/10 bg-red-900/10 p-3">
        <div className="font-mono text-[10px] uppercase tracking-widest text-red-400/60">
          Problems
        </div>
        {WASTE.map((w) => (
          <div key={w} className="flex items-start gap-2">
            <X className="mt-0.5 h-3 w-3 shrink-0 text-red-400/60" />
            <span className="text-xs text-slate-400">{w}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AdaptiveColumn() {
  return (
    <div className="relative flex flex-col rounded-2xl border border-emerald-400/20 bg-emerald-900/5 p-6 shadow-[0_0_40px_rgba(52,211,153,0.06)]">
      {/* Active glow border */}
      <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-400/5 via-transparent to-emerald-400/5" />

      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-400/10">
          <ArrowRight className="h-3.5 w-3.5 text-emerald-400" />
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-emerald-400/70">
            Adaptive
          </div>
          <div className="text-sm font-semibold text-emerald-200">
            Strategy Selection
          </div>
        </div>
        <div className="ml-auto rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-widest text-emerald-400">
          Active
        </div>
      </div>
      <div className="mb-5 text-xs text-slate-500">
        Selects the best strategy on each iteration based on live code metrics
        and rule-based scoring.
      </div>

      {/* Steps */}
      <div className="mb-6 space-y-2">
        {ADAPTIVE.map((step, i) => {
          const colors = [
            "cyan",
            "cyan",
            "violet",
            "violet",
            "amber",
            "emerald",
          ];
          const c = colors[i];
          const borderCls = {
            cyan: "border-cyan-400/30 bg-cyan-900/20 text-cyan-400",
            violet: "border-violet-400/30 bg-violet-900/20 text-violet-400",
            amber: "border-amber-400/30 bg-amber-900/20 text-amber-400",
            emerald: "border-emerald-400/30 bg-emerald-900/20 text-emerald-400",
          }[c]!;

          return (
            <div key={step.label} className="flex items-start gap-3">
              <div
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border font-mono text-[10px] ${borderCls}`}
              >
                {i + 1}
              </div>
              <div>
                <div className="text-sm text-slate-200">{step.label}</div>
                <div className="text-[10px] text-slate-500">{step.detail}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Benefits */}
      <div className="mt-auto space-y-2 rounded-xl border border-emerald-400/10 bg-emerald-900/10 p-3">
        <div className="font-mono text-[10px] uppercase tracking-widest text-emerald-400/60">
          Benefits
        </div>
        {GAINS.map((g) => (
          <div key={g} className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-emerald-400/70" />
            <span className="text-xs text-slate-400">{g}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function AdaptiveComparison() {
  return (
    <section className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="Why adaptive optimization"
          title="Fixed sequence vs. data-driven selection"
          description="The proposed system selects the next strategy based on live TAC characteristics — not a predetermined pass order."
        />

        {/* VS divider */}
        <div className="relative mb-6 flex items-center">
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-red-400/30 to-transparent" />
          <div className="mx-4 flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-slate-900 font-mono text-xs font-bold text-slate-400">
            VS
          </div>
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-emerald-400/30 to-transparent" />
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <TraditionalColumn />
          <AdaptiveColumn />
        </div>

        {/* Bottom insight bar */}
        <div className="mt-6 rounded-2xl border border-cyan-400/10 bg-cyan-900/5 px-6 py-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="font-mono text-xs text-cyan-400/60">INSIGHT</div>
            <div className="h-3 w-px bg-white/10" />
            <p className="text-sm text-slate-400">
              In workloads heavy with constants, the adaptive system immediately
              prioritizes{" "}
              <span className="text-cyan-300">Constant Folding</span>. In
              loop-heavy code it leads with{" "}
              <span className="text-violet-300">Loop Optimization</span> —
              saving multiple redundant passes.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
