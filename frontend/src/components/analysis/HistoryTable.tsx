import { DEMO_RESPONSE } from "../../data/demo";
import type { OptimizationIteration } from "../../types";
import { formatScore } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { SectionHeader } from "../ui/SectionHeader";

export function HistoryTable({ iterations }: { iterations: OptimizationIteration[] }) {
  const displayRows = iterations.length > 0 ? iterations : DEMO_RESPONSE.iterations;

  return (
    <section id="history" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="HISTORY"
          title="Optimization Pass Log"
          description="Detailed step-by-step audit of each strategy score, input size, output size, and instruction delta."
        />
        <div className="glass overflow-x-auto rounded-2xl border border-white/10 shadow-[0_12px_40px_rgba(0,0,0,0.4)]">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-white/8 bg-[#070b14] text-[11px] font-mono uppercase tracking-[0.18em] text-slate-400">
              <tr>
                <th className="px-5 py-3.5">Iteration</th>
                <th className="px-5 py-3.5">Strategy</th>
                <th className="px-5 py-3.5">Score</th>
                <th className="px-5 py-3.5">Before</th>
                <th className="px-5 py-3.5">After</th>
                <th className="px-5 py-3.5">Reduction</th>
                <th className="px-5 py-3.5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/6 font-mono text-xs">
              {displayRows.map((row) => (
                <tr
                  key={row.iteration}
                  className="hover:bg-white/[0.02] transition-colors"
                >
                  <td className="px-5 py-3.5 font-bold text-cyan-300">
                    #{String(row.iteration).padStart(2, "0")}
                  </td>
                  <td className="px-5 py-3.5 font-sans font-semibold text-white">
                    {row.strategy_name}
                  </td>
                  <td className="px-5 py-3.5 text-slate-300">
                    {row.score ? formatScore(row.score) : "—"}
                  </td>
                  <td className="px-5 py-3.5 text-slate-300">{row.before_count}</td>
                  <td className="px-5 py-3.5 text-slate-300">{row.after_count}</td>
                  <td className="px-5 py-3.5">
                    <span
                      className={`font-bold ${
                        row.reduction > 0 ? "text-emerald-400" : "text-slate-500"
                      }`}
                    >
                      {row.reduction > 0 ? `-${row.reduction}` : "0"}
                    </span>
                  </td>
                  <td className="px-5 py-3.5">
                    <Badge tone={row.status === "converged" ? "emerald" : "cyan"}>
                      {row.status === "converged" ? "CONVERGED" : "APPLIED"}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
