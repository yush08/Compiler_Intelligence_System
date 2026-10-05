import { Check, ChevronLeft, ChevronRight, GraduationCap, X } from "lucide-react";
import { FACULTY_STEPS } from "../../data/faculty";
import { Button } from "../ui/button";

export function FacultyDemo({
  open,
  step,
  onStep,
  onClose,
}: {
  open: boolean;
  step: number;
  onStep: (step: number) => void;
  onClose: () => void;
}) {
  if (!open) return null;
  const current = FACULTY_STEPS[step];

  function go(next: number) {
    const index = Math.max(0, Math.min(FACULTY_STEPS.length - 1, next));
    onStep(index);
    document
      .getElementById(FACULTY_STEPS[index].id)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <div className="fixed inset-x-0 bottom-6 z-50 mx-auto max-w-3xl px-4 animate-in fade-in slide-in-from-bottom-5 duration-300">
      <div className="glass aurora relative overflow-hidden rounded-2xl border-2 border-cyan-400/80 bg-[#070c18]/95 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.8)] backdrop-blur-2xl">
        {/* Progress bar line at top */}
        <div className="absolute inset-x-0 top-0 h-1 bg-white/10">
          <div
            className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-300"
            style={{ width: `${((step + 1) / FACULTY_STEPS.length) * 100}%` }}
          />
        </div>

        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="flex h-5 items-center gap-1.5 rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-300 uppercase tracking-widest border border-cyan-500/40">
                <GraduationCap className="h-3 w-3" />
                FACULTY PRESENTATION DEMO
              </span>
              <span className="font-mono text-xs text-slate-500">
                {current.title} of 8
              </span>
            </div>

            <div className="mt-2 text-xl font-bold tracking-tight text-white">
              {current.label}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              {current.detail}
            </p>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-slate-400 hover:text-white"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Step indicator pills */}
        <div className="mt-5 flex items-center justify-between border-t border-white/8 pt-4">
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {FACULTY_STEPS.map((s, idx) => {
              const isDone = idx < step;
              const isCurr = idx === step;
              return (
                <button
                  key={s.id + idx}
                  type="button"
                  onClick={() => go(idx)}
                  className={`flex h-7 w-7 items-center justify-center rounded-lg font-mono text-xs font-semibold transition ${
                    isCurr
                      ? "bg-cyan-400 text-slate-950 shadow-[0_0_12px_rgba(34,211,238,0.6)] font-bold"
                      : isDone
                        ? "bg-emerald-950/60 text-emerald-400 border border-emerald-500/30"
                        : "bg-white/5 text-slate-500 hover:text-slate-300"
                  }`}
                  title={s.label}
                >
                  {isDone ? <Check className="h-3 w-3" /> : idx + 1}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => go(step - 1)}
              disabled={step === 0}
              className="font-mono text-xs"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Back
            </Button>

            {step < FACULTY_STEPS.length - 1 ? (
              <Button
                variant="brutal"
                size="sm"
                onClick={() => go(step + 1)}
                className="font-mono text-xs shadow-[0_0_14px_rgba(255,75,75,0.4)]"
              >
                Next Step
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button
                variant="primary"
                size="sm"
                onClick={onClose}
                className="font-mono text-xs bg-emerald-500 text-slate-950 hover:bg-emerald-400"
              >
                Complete Demo ✓
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
