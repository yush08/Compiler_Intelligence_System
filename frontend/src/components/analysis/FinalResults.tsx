import type { FinalResult } from "../../types";
import { SectionHeader } from "../ui/SectionHeader";

export function FinalResults({ result }: { result?: FinalResult }) {
  const orig = result?.original_count ?? 9;
  const final = result?.final_count ?? 4;
  const reduction = result?.total_reduction ?? 5;
  const pct = result?.reduction_percentage ? `${result.reduction_percentage.toFixed(1)}%` : "55.6%";
  const iters = result?.iterations ?? 4;
  const status = result?.status ? result.status.toUpperCase() : "CONVERGED";

  const cards = [
    { label: "ORIGINAL INSTRUCTIONS", value: String(orig), tone: "slate" },
    { label: "FINAL INSTRUCTIONS", value: String(final), tone: "cyan" },
    { label: "TOTAL REDUCTION", value: String(reduction), tone: "emerald" },
    { label: "REDUCTION", value: pct, tone: "emerald" },
    { label: "ITERATIONS", value: String(iters), tone: "cyan" },
    { label: "STATUS", value: status, tone: "amber" },
  ];

  return (
    <section className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="RESULTS"
          title="Optimization Metrics Summary"
          description="High-level performance outcome and instruction reduction efficiency."
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {cards.map((card) => (
            <div
              key={card.label}
              className={`glass relative overflow-hidden rounded-2xl p-5 border transition-all hover:-translate-y-0.5 ${
                card.label === "STATUS"
                  ? "border-emerald-500/40 bg-emerald-950/20 shadow-[0_0_20px_rgba(52,211,153,0.15)]"
                  : card.label === "REDUCTION"
                    ? "border-cyan-500/40 bg-cyan-950/20 shadow-[0_0_20px_rgba(34,211,238,0.15)]"
                    : "border-white/8 hover:border-white/20"
              }`}
            >
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-slate-400">
                {card.label}
              </div>
              <div className="mt-3 font-mono text-3xl font-extrabold tracking-tight text-white">
                {card.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
