import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import type { FinalResult } from "../../types";
import { SectionHeader } from "../ui/SectionHeader";
import { Badge } from "../ui/badge";

const DEFAULT_BEFORE = [
  "t1 = 10",
  "t2 = 20",
  "t3 = t1 + t2",
  "t4 = t1 + t2",
  "t5 = t3 * 1",
  "t6 = 100",
  "t7 = t6 + 50",
  "unused = t7",
  "t8 = t3 * 2",
];

const DEFAULT_AFTER = [
  "t3 = 30",
  "t4 = 30",
  "t5 = t3",
  "t7 = 150",
  "t8 = t3 * 2",
];

function Panel({
  title,
  lines,
  badge,
  highlight = false,
}: {
  title: string;
  lines: string[];
  badge: string;
  highlight?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(lines.join("\n"));
    setCopied(true);
    toast.success(`${title} TAC copied to clipboard.`);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`glass overflow-hidden rounded-2xl border transition-all ${
        highlight
          ? "border-emerald-500/40 bg-emerald-950/10 shadow-[0_0_30px_rgba(52,211,153,0.15)]"
          : "border-white/10"
      }`}
    >
      <div className="flex items-center justify-between border-b border-white/8 bg-[#070b14] px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs font-bold tracking-[0.2em] text-white">
            {title}
          </span>
          <Badge tone={highlight ? "emerald" : "cyan"}>{badge}</Badge>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded-lg border border-white/8 px-2.5 py-1 text-xs font-mono text-slate-400 hover:text-white hover:border-white/20 transition"
        >
          {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <pre className="min-h-[240px] bg-[#050811] p-4 font-mono text-[13px] leading-7 overflow-x-auto">
        {lines.map((line, index) => (
          <div
            key={`${title}-${index}`}
            className={`flex items-center gap-3 rounded px-2 ${
              highlight
                ? "bg-emerald-500/10 text-emerald-200"
                : "text-slate-200 hover:bg-white/[0.02]"
            }`}
          >
            <span className="select-none text-right font-mono text-xs text-slate-600 w-6">
              {index + 1}
            </span>
            <span className="font-mono">{line}</span>
          </div>
        ))}
      </pre>
    </div>
  );
}

export function BeforeAfterTac({ result }: { result?: FinalResult }) {
  const before =
    result?.original_tac && result.original_tac.length > 0
      ? result.original_tac
      : DEFAULT_BEFORE;
  const after =
    result?.final_tac && result.final_tac.length > 0
      ? result.final_tac
      : DEFAULT_AFTER;

  return (
    <section id="analysis" className="px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          kicker="BEFORE / AFTER"
          title="Side-by-Side Instruction Comparison"
          description="Visual diff between the unoptimized TAC and the final converged instruction stream."
        />
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="BEFORE"
            lines={before}
            badge={`${before.length} INSTRUCTIONS`}
          />
          <Panel
            title="AFTER"
            lines={after}
            badge={`${after.length} INSTRUCTIONS (OPTIMIZED)`}
            highlight
          />
        </div>
      </div>
    </section>
  );
}
