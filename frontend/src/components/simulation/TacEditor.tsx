import { useState } from "react";
import { Code2, Copy, Play, RotateCcw, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";

function highlightTacLine(raw: string): string {
  const escaped = raw
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

  if (escaped.trim().startsWith("#")) {
    return `<span class="text-slate-500 italic">${escaped}</span>`;
  }

  return escaped
    .replace(/^([A-Za-z_][A-Za-z0-9_]*):/g, '<span class="text-sky-400 font-bold">$1:</span>')
    .replace(/\b(goto|if)\b/g, '<span class="text-amber-400 font-semibold">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="text-emerald-300 font-semibold">$1</span>')
    .replace(/([=+\-*/%])/g, '<span class="text-rose-400 font-bold">$1</span>')
    .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?![^<]*>)/g, '<span class="text-cyan-200">$1</span>');
}

export function TacEditor({
  value,
  onChange,
  onAnalyze,
  onReset,
  loading,
}: {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onReset: () => void;
  loading: boolean;
}) {
  const [showPreview, setShowPreview] = useState(false);
  const lines = value.split("\n");

  return (
    <section id="tac-editor" className="px-6 py-4">
      <div className="mx-auto max-w-7xl">
        <div className="glass overflow-hidden rounded-2xl border border-white/10 shadow-[0_12px_40px_rgba(0,0,0,0.5)]">
          {/* Top Bar */}
          <div className="flex flex-wrap items-center justify-between border-b border-white/8 bg-[#070b14] px-4 py-3">
            <div className="flex items-center gap-3">
              <span className="flex h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
              <span className="mono text-xs font-semibold tracking-[0.2em] text-slate-200">
                TAC_INPUT_STREAM
              </span>
              <Badge tone="cyan">{lines.length} STATEMENTS</Badge>
              <span className="hidden font-mono text-[11px] text-slate-500 md:inline">
                Three Address Code IR
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowPreview(!showPreview)}
                className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-mono transition ${
                  showPreview
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200 border border-white/6"
                }`}
              >
                <Code2 className="h-3 w-3" />
                {showPreview ? "Edit Raw" : "Syntax View"}
              </button>
              <Button
                variant="ghost"
                size="sm"
                onClick={async () => {
                  await navigator.clipboard.writeText(value);
                  toast.success("TAC copied to clipboard.");
                }}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy
              </Button>
              <Button variant="ghost" size="sm" onClick={onReset}>
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </Button>
            </div>
          </div>

          {/* Editor Area with Line Numbers */}
          <div className="grid grid-cols-[3.2rem_1fr] bg-[#050811]">
            {/* Gutter */}
            <div className="mono select-none border-r border-white/6 bg-[#04060c] py-3.5 text-right text-[12px] leading-7 text-slate-600">
              {lines.map((_, index) => (
                <div key={index} className="px-2.5">
                  {index + 1}
                </div>
              ))}
            </div>

            {/* Input or Syntax View */}
            {showPreview ? (
              <div className="mono min-h-[240px] overflow-x-auto px-4 py-3.5 text-[13px] leading-7">
                {lines.map((line, idx) => (
                  <div
                    key={idx}
                    dangerouslySetInnerHTML={{ __html: highlightTacLine(line) }}
                  />
                ))}
              </div>
            ) : (
              <textarea
                value={value}
                onChange={(event) => onChange(event.target.value)}
                spellCheck={false}
                rows={Math.max(9, lines.length)}
                className="mono min-h-[240px] w-full resize-y bg-transparent px-4 py-3.5 text-[13px] leading-7 text-slate-100 outline-none placeholder:text-slate-600 focus:ring-0"
                placeholder="Enter Three Address Code here..."
              />
            )}
          </div>

          {/* Bottom Action Footer */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/8 bg-[#070b14]/90 px-4 py-3">
            <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Valid TAC Grammar
              </span>
              <span className="hidden sm:inline text-slate-600">|</span>
              <span className="hidden sm:inline">
                {new Blob([value]).size} bytes
              </span>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={onReset}>
                Reset
              </Button>
              <Button
                variant="brutal"
                size="md"
                onClick={onAnalyze}
                disabled={loading}
                className="shadow-[0_0_18px_rgba(255,75,75,0.35)]"
              >
                {loading ? (
                  <>
                    <Sparkles className="h-4 w-4 animate-spin" />
                    ANALYZING TAC...
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5 fill-current" />
                    ANALYZE TAC
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
