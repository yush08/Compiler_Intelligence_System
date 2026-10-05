import { Activity, Binary, Cpu, Layers, Repeat, Skull, Sparkles, Variable } from "lucide-react";
import type { OptimizationFeatures } from "../../types";
import { formatPercent } from "../../lib/utils";
import { SectionHeader } from "../ui/SectionHeader";
import { Badge } from "../ui/badge";

export function CodeDna({ features }: { features?: OptimizationFeatures }) {
  const stmts = features?.num_statements ?? 0;
  const constRatio = features?.const_ratio ?? 0;

  return (
    <section id="code-dna" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="CODE DNA"
          title="Extracted TAC Features"
          description="Measurable code characteristics extracted by the backend AST/TAC analyzer before scoring strategies."
        />

        {/* Bento Grid */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3 lg:grid-cols-4">
          {/* Bento Card 1: Total Statements (Featured 2-col) */}
          <div className="glass group relative overflow-hidden rounded-2xl p-6 md:col-span-2 border border-white/10 hover:border-cyan-500/40 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-cyan-400" />
                <span className="text-xs font-mono uppercase tracking-[0.2em]">num_statements</span>
              </div>
              <Badge tone="cyan">Primary Metric</Badge>
            </div>
            <div className="mt-4 flex items-baseline gap-3">
              <span className="font-mono text-5xl font-bold tracking-tight text-white">
                {features?.num_statements ?? "—"}
              </span>
              <span className="text-sm text-slate-400 font-mono">TAC instructions</span>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-400">
              Baseline instruction density analyzed for optimization opportunity and convergence metrics.
            </p>
            <div className="mt-4 flex gap-1">
              {Array.from({ length: Math.min(16, stmts) }).map((_, i) => (
                <div
                  key={i}
                  className="h-1.5 flex-1 rounded-full bg-cyan-400/40"
                />
              ))}
            </div>
          </div>

          {/* Bento Card 2: Constant Ratio (Featured 2-col) */}
          <div className="glass group relative overflow-hidden rounded-2xl p-6 md:col-span-2 border border-white/10 hover:border-violet-500/40 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-violet-400" />
                <span className="text-xs font-mono uppercase tracking-[0.2em]">const_ratio</span>
              </div>
              <Badge tone="cyan">{formatPercent(constRatio)}</Badge>
            </div>
            <div className="mt-4 flex items-baseline gap-3">
              <span className="font-mono text-5xl font-bold tracking-tight text-violet-200">
                {features ? formatPercent(features.const_ratio) : "—"}
              </span>
              <span className="text-sm text-slate-400 font-mono">constant density</span>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-400">
              Ratio of constant operations influencing folding and propagation priority scores.
            </p>
            <div className="mt-4 h-2 w-full rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400 transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, constRatio * 100))}%` }}
              />
            </div>
          </div>

          {/* Bento Card 3: Constants */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_constants</span>
              <Binary className="h-4 w-4 text-cyan-300" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_constants ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">Literal values detected</div>
          </div>

          {/* Bento Card 4: Repeated Expressions */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_repeated_expr</span>
              <Repeat className="h-4 w-4 text-amber-300" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_repeated_expr ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">CSE targets available</div>
          </div>

          {/* Bento Card 5: Dead Statements */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_dead_stmts</span>
              <Skull className="h-4 w-4 text-rose-400" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_dead_stmts ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">Unused register assigns</div>
          </div>

          {/* Bento Card 6: Algebraic Ops */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_algebraic_ops</span>
              <Cpu className="h-4 w-4 text-emerald-300" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_algebraic_ops ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">Identity simplification</div>
          </div>

          {/* Bento Card 7: Loop Ops */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_loop_ops</span>
              <Activity className="h-4 w-4 text-blue-400" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_loop_ops ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">Loop strength patterns</div>
          </div>

          {/* Bento Card 8: Temp Vars */}
          <div className="glass rounded-2xl p-5 border border-white/8 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em]">num_tmp_vars</span>
              <Variable className="h-4 w-4 text-indigo-300" />
            </div>
            <div className="mt-3 font-mono text-3xl font-bold text-white">
              {features?.num_tmp_vars ?? "—"}
            </div>
            <div className="mt-2 text-xs text-slate-500">Live variables & temps</div>
          </div>
        </div>
      </div>
    </section>
  );
}
