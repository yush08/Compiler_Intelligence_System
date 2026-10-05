import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { StrategyScore } from "../../types";
import { formatScore } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";
import { SectionHeader } from "../ui/SectionHeader";

export function StrategyMatrix({ strategies }: { strategies: StrategyScore[] }) {
  const ordered = [...strategies].sort((a, b) => b.score - a.score);
  return (
    <section id="strategy-matrix" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="STRATEGY MATRIX"
          title="Rule-Based Strategy Scoring"
          description="Six compiler optimization passes evaluated against current TAC code characteristics."
        />

        {/* Rule-Based Transparency Notice */}
        <div className="mb-6 flex items-center gap-3 rounded-xl border border-cyan-500/20 bg-cyan-950/20 px-4 py-2.5 text-xs text-cyan-200">
          <span className="flex h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>
            <strong className="font-semibold uppercase tracking-wider text-white">Rule-Based Scoring Architecture:</strong> Strategy scores are computed deterministically from the 8 extracted TAC features. No black-box machine learning models are used.
          </span>
        </div>

        {ordered.length === 0 ? (
          <div className="glass rounded-2xl p-6 text-sm text-slate-400">
            Analyze TAC to score all six rule-based optimization strategies.
          </div>
        ) : (
          <>
            <div className="mb-6 h-48 rounded-2xl p-4 glass border border-white/8">
              <div className="mb-2 font-mono text-[11px] uppercase tracking-wider text-slate-400">
                Opportunity Score Distribution
              </div>
              <ResponsiveContainer width="100%" height="85%">
                <BarChart data={ordered} layout="vertical" margin={{ left: 16, right: 16 }}>
                  <XAxis type="number" hide domain={[0, 1]} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={220}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    contentStyle={{
                      background: "#0b1220",
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="score" fill="#22d3ee" radius={4} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {ordered.map((strategy, idx) => (
                <div
                  key={strategy.id}
                  className={`glass relative rounded-2xl p-5 transition-all duration-200 ${
                    strategy.selected
                      ? "aurora border-2 border-cyan-400 bg-cyan-950/30 shadow-[0_0_32px_rgba(34,211,238,0.25)]"
                      : "border border-white/8 hover:border-white/16"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-slate-500">
                          0{idx + 1}
                        </span>
                        <div className="text-sm font-semibold text-white tracking-wide">
                          {strategy.name}
                        </div>
                      </div>
                      <div className="mt-2 flex items-baseline gap-2 font-mono">
                        <span className="text-3xl font-bold text-cyan-200">
                          {formatScore(strategy.score)}
                        </span>
                        <span className="text-xs text-slate-500">score</span>
                      </div>
                    </div>
                    {strategy.selected ? (
                      <Badge
                        tone="cyan"
                        className="font-bold tracking-widest uppercase border-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.6)]"
                      >
                        SELECTED
                      </Badge>
                    ) : null}
                  </div>

                  <Progress value={strategy.score * 100} className="mt-4 h-2" />

                  <p className="mt-3 text-xs leading-5 text-slate-300 min-h-[40px]">
                    {strategy.reason}
                  </p>

                  <div className="mt-4 flex items-center justify-between border-t border-white/6 pt-3 text-[11px] font-mono">
                    <span className="text-slate-500 uppercase tracking-wider">Opportunity:</span>
                    <span
                      className={`font-semibold ${
                        strategy.opportunity ? "text-emerald-400" : "text-slate-500"
                      }`}
                    >
                      {strategy.opportunity ? "● OPPORTUNITY DETECTED" : "○ LOW OPPORTUNITY"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
