import { motion } from "framer-motion";
import { SCENARIOS } from "../../data/scenarios";
import type { Scenario } from "../../types";
import { Badge } from "../ui/badge";
import { SectionHeader } from "../ui/SectionHeader";

export function ScenarioCards({
  selectedId,
  onSelect,
}: {
  selectedId?: string;
  onSelect: (scenario: Scenario) => void;
}) {
  return (
    <section id="simulation" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="Start a simulation"
          title="Choose a TAC workload"
          description="Selecting a scenario populates the editor. Mixed Workload is the recommended faculty demonstration."
        />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {SCENARIOS.map((scenario, index) => (
            <motion.button
              key={scenario.id}
              type="button"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onSelect(scenario)}
              className={`glass group relative rounded-2xl p-5 text-left transition-all duration-200 hover:-translate-y-1 ${
                selectedId === scenario.id
                  ? "aurora border-cyan-400/80 bg-cyan-950/20 shadow-[0_0_24px_rgba(34,211,238,0.2)]"
                  : "border-white/8 hover:border-white/20"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="text-sm font-semibold text-white tracking-wide">{scenario.title}</div>
                {scenario.recommended ? (
                  <Badge tone="cyan" className="animate-pulse shadow-[0_0_12px_rgba(34,211,238,0.4)]">
                    RECOMMENDED DEMO
                  </Badge>
                ) : null}
              </div>
              <p className="mt-3 text-xs leading-5 text-slate-400">{scenario.description}</p>
              <div className="mt-4 flex items-center justify-between border-t border-white/6 pt-3 text-[11px] font-mono text-slate-500">
                <span>{scenario.tac.split("\n").length} instructions</span>
                <span className="text-cyan-400 opacity-0 transition-opacity group-hover:opacity-100">
                  Load TAC →
                </span>
              </div>
            </motion.button>
          ))}
        </div>
      </div>
    </section>
  );
}
