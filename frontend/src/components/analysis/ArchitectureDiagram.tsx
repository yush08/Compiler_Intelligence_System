import { Cpu, Database, Eye, Globe, Layers, Repeat, Server, Terminal, User } from "lucide-react";
import { SectionHeader } from "../ui/SectionHeader";
import { Badge } from "../ui/badge";

const SIX_OPTIMIZERS = [
  "Constant Folding",
  "Constant Propagation",
  "Common Subexpression Elimination",
  "Dead Code Elimination",
  "Algebraic Simplification",
  "Strength Reduction",
];

export function ArchitectureDiagram() {
  return (
    <section id="architecture" className="px-6 py-12">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          kicker="ARCHITECTURE"
          title="System Topology & Pipeline Dataflow"
          description="End-to-end execution flow from client interaction down to iterative bytecode/TAC passes."
        />

        <div className="glass relative overflow-hidden rounded-3xl border border-white/10 p-8 md:p-12 shadow-[0_20px_60px_rgba(0,0,0,0.5)]">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-cyan-900/10 via-transparent to-transparent" />

          <div className="relative flex flex-col items-center gap-3">
            {/* 1. USER */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <User className="h-4 w-4 text-cyan-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  USER
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">Interaction & TAC Input</span>
            </div>

            <div className="font-mono text-sm text-cyan-400 font-bold">↓</div>

            {/* 2. REACT FRONTEND */}
            <div className="aurora flex w-full max-w-md items-center justify-between rounded-xl border border-cyan-400/60 bg-cyan-950/30 px-5 py-3.5 shadow-[0_0_20px_rgba(34,211,238,0.2)]">
              <div className="flex items-center gap-3">
                <Globe className="h-4 w-4 text-cyan-300" />
                <span className="font-mono text-xs font-bold tracking-widest text-cyan-200 uppercase">
                  REACT FRONTEND
                </span>
              </div>
              <span className="text-[11px] font-mono text-cyan-300/80">React + TS + Vite + Tailwind</span>
            </div>

            <div className="font-mono text-sm text-cyan-400 font-bold">↓</div>

            {/* 3. REST API */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <Server className="h-4 w-4 text-violet-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  REST API
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">FastAPI HTTP Endpoints</span>
            </div>

            <div className="font-mono text-sm text-violet-400 font-bold">↓</div>

            {/* 4. PYTHON BACKEND */}
            <div className="aurora flex w-full max-w-md items-center justify-between rounded-xl border border-violet-400/60 bg-violet-950/30 px-5 py-3.5 shadow-[0_0_20px_rgba(139,92,246,0.2)]">
              <div className="flex items-center gap-3">
                <Terminal className="h-4 w-4 text-violet-300" />
                <span className="font-mono text-xs font-bold tracking-widest text-violet-200 uppercase">
                  PYTHON BACKEND
                </span>
              </div>
              <span className="text-[11px] font-mono text-violet-300/80">Core Optimization Subsystem</span>
            </div>

            <div className="font-mono text-sm text-violet-400 font-bold">↓</div>

            {/* 5. TAC PARSER */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <Database className="h-4 w-4 text-cyan-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  TAC PARSER
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">Grammar & IR Tokenizer</span>
            </div>

            <div className="font-mono text-sm text-cyan-400 font-bold">↓</div>

            {/* 6. FEATURE EXTRACTION */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <Layers className="h-4 w-4 text-emerald-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  FEATURE EXTRACTION
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">8 Code DNA Metrics</span>
            </div>

            <div className="font-mono text-sm text-emerald-400 font-bold">↓</div>

            {/* 7. RULE-BASED SCORING */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <Cpu className="h-4 w-4 text-amber-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  RULE-BASED SCORING
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">Deterministic Opportunity Matrix</span>
            </div>

            <div className="font-mono text-sm text-amber-400 font-bold">↓</div>

            {/* 8. OPTIMIZATION ENGINE */}
            <div className="aurora flex w-full max-w-md items-center justify-between rounded-xl border border-amber-400/50 bg-amber-950/20 px-5 py-3.5">
              <div className="flex items-center gap-3">
                <Repeat className="h-4 w-4 text-amber-300" />
                <span className="font-mono text-xs font-bold tracking-widest text-amber-200 uppercase">
                  OPTIMIZATION ENGINE
                </span>
              </div>
              <span className="text-[11px] font-mono text-amber-300/80">Iterative Control Loop</span>
            </div>

            <div className="font-mono text-sm text-amber-400 font-bold">↓</div>

            {/* 9. 6 OPTIMIZERS */}
            <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-white/[0.02] p-5 backdrop-blur">
              <div className="mb-3 text-center font-mono text-xs font-bold tracking-widest text-cyan-300 uppercase">
                6 OPTIMIZERS
              </div>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                {SIX_OPTIMIZERS.map((opt) => (
                  <div
                    key={opt}
                    className="rounded-lg border border-white/6 bg-white/[0.02] px-3 py-2 text-center font-mono text-xs text-slate-300 hover:border-cyan-400/40 hover:text-white transition"
                  >
                    {opt}
                  </div>
                ))}
              </div>
            </div>

            <div className="font-mono text-sm text-cyan-400 font-bold">↓</div>

            {/* 10. RE-ANALYSIS */}
            <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 backdrop-blur">
              <div className="flex items-center gap-3">
                <Eye className="h-4 w-4 text-sky-400" />
                <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
                  RE-ANALYSIS
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">Re-evaluate TAC State</span>
            </div>

            <div className="font-mono text-sm text-emerald-400 font-bold">↓</div>

            {/* 11. CONVERGED */}
            <div className="aurora flex w-full max-w-md items-center justify-between rounded-xl border-2 border-emerald-400/80 bg-emerald-950/40 px-5 py-3.5 shadow-[0_0_30px_rgba(52,211,153,0.3)]">
              <div className="flex items-center gap-3">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
                <span className="font-mono text-xs font-extrabold tracking-widest text-emerald-200 uppercase">
                  CONVERGED
                </span>
              </div>
              <Badge tone="emerald">Delta = 0 (Complete)</Badge>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
