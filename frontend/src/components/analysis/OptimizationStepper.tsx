import { Check, Cpu, GitBranch, Layers, Play, RefreshCw, Search, Target, Zap } from "lucide-react";
import { SectionHeader } from "../ui/SectionHeader";

const STAGES = [
  {
    label: "Analyze TAC",
    icon: Search,
    desc: "Parse three-address code into IR",
    color: "cyan",
  },
  {
    label: "Extract Features",
    icon: Cpu,
    desc: "Compute 8 structural code metrics",
    color: "cyan",
  },
  {
    label: "Score Strategies",
    icon: Layers,
    desc: "Rule-based heuristic ranking",
    color: "violet",
  },
  {
    label: "Select Best",
    icon: Target,
    desc: "Pick highest-scoring strategy",
    color: "violet",
  },
  {
    label: "Optimize",
    icon: Zap,
    desc: "Apply transformation pass",
    color: "amber",
  },
  {
    label: "Measure Gain",
    icon: GitBranch,
    desc: "Count instruction reduction",
    color: "amber",
  },
  {
    label: "Re-analyze",
    icon: RefreshCw,
    desc: "Loop back with updated TAC",
    color: "amber",
  },
  {
    label: "Converged",
    icon: Play,
    desc: "No further improvement possible",
    color: "emerald",
  },
];

const COLOR = {
  cyan: {
    ring: "border-cyan-400/50 bg-cyan-400/10 text-cyan-300",
    glow: "shadow-[0_0_18px_rgba(34,211,238,0.35)]",
    done: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300",
    text: "text-cyan-400",
    bar: "bg-cyan-400/70",
    card: "border-cyan-400/20 bg-cyan-900/10",
  },
  violet: {
    ring: "border-violet-400/50 bg-violet-400/10 text-violet-300",
    glow: "shadow-[0_0_18px_rgba(167,139,250,0.35)]",
    done: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300",
    text: "text-violet-400",
    bar: "bg-violet-400/70",
    card: "border-violet-400/20 bg-violet-900/10",
  },
  amber: {
    ring: "border-amber-400/50 bg-amber-400/10 text-amber-300",
    glow: "shadow-[0_0_18px_rgba(251,191,36,0.35)]",
    done: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300",
    text: "text-amber-400",
    bar: "bg-amber-400/70",
    card: "border-amber-400/20 bg-amber-900/10",
  },
  emerald: {
    ring: "border-emerald-400/50 bg-emerald-400/10 text-emerald-300",
    glow: "shadow-[0_0_18px_rgba(52,211,153,0.35)]",
    done: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300",
    text: "text-emerald-400",
    bar: "bg-emerald-400/70",
    card: "border-emerald-400/20 bg-emerald-900/10",
  },
};

export function OptimizationStepper({ current }: { current: number }) {
  const progress = Math.min(8, Math.max(0, current));

  return (
    <section id="pipeline" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="Optimization pipeline"
          title="Control loop"
          description="8-stage iterative engine — loops until no further instruction reduction is achievable."
        />

        {/* Progress bar */}
        <div className="mb-8 flex items-center gap-3">
          <span className="font-mono text-xs text-slate-500">STAGE</span>
          <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-violet-400 to-emerald-400 transition-all duration-700"
              style={{ width: `${(progress / 8) * 100}%` }}
            />
          </div>
          <span className="font-mono text-xs text-slate-400">
            {progress}/8
          </span>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {STAGES.map((stage, index) => {
            const step = index + 1;
            const complete = current > step;
            const active =
              current === step || (current >= 8 && step === 8);
            const c = COLOR[stage.color as keyof typeof COLOR];
            const Icon = stage.icon;

            return (
              <div
                key={stage.label}
                className={`group relative flex flex-col gap-3 rounded-2xl border p-4 transition-all duration-300 ${
                  active
                    ? `${c.card} ${c.glow}`
                    : complete
                      ? "border-emerald-400/10 bg-emerald-900/5"
                      : "border-white/5 bg-white/[0.02]"
                }`}
              >
                {/* Step number connector */}
                {index < STAGES.length - 1 && (
                  <div className="absolute -right-1.5 top-1/2 z-10 hidden h-0.5 w-3 -translate-y-1/2 xl:block">
                    <div
                      className={`h-full w-full transition-all duration-500 ${
                        complete ? "bg-emerald-400/50" : "bg-white/10"
                      }`}
                    />
                  </div>
                )}

                <div className="flex items-center gap-3">
                  {/* Icon circle */}
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-xs ${
                      complete
                        ? c.done
                        : active
                          ? `${c.ring} animate-pulse`
                          : "border-white/10 text-slate-600"
                    }`}
                  >
                    {complete ? (
                      <Check className="h-4 w-4 text-emerald-400" />
                    ) : (
                      <Icon className="h-4 w-4" />
                    )}
                  </div>

                  {/* Label + step number */}
                  <div className="min-w-0">
                    <div
                      className={`font-mono text-xs font-semibold tracking-wide ${
                        complete
                          ? "text-emerald-400"
                          : active
                            ? c.text
                            : "text-slate-500"
                      }`}
                    >
                      {String(step).padStart(2, "0")}
                    </div>
                    <div
                      className={`truncate text-sm font-medium ${
                        active || complete ? "text-white" : "text-slate-400"
                      }`}
                    >
                      {stage.label}
                    </div>
                  </div>
                </div>

                {/* Description */}
                <div className="text-xs leading-relaxed text-slate-500">
                  {stage.desc}
                </div>

                {/* Active pulse bar */}
                {active && (
                  <div className="absolute inset-x-0 bottom-0 h-0.5 rounded-b-2xl overflow-hidden">
                    <div className={`h-full w-full animate-pulse ${c.bar}`} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
