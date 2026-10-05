import { motion } from "framer-motion";
import { GraduationCap } from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { cn } from "../../lib/utils";

const LINKS = [
  { href: "#overview", label: "Overview" },
  { href: "#simulation", label: "Simulation" },
  { href: "#analysis", label: "Analysis" },
  { href: "#history", label: "History" },
];

export function Navbar({
  demoMode,
  live,
  facultyDemo,
  onFacultyDemo,
}: {
  demoMode: boolean;
  live: boolean;
  facultyDemo: boolean;
  onFacultyDemo: () => void;
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/8 bg-[#03050a]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-6 py-3">
        <div>
          <div className="text-[11px] font-semibold tracking-[0.24em] text-cyan-300">
            COMPILER INTELLIGENCE SYSTEM
          </div>
          <div className="text-xs text-slate-500">Automated Optimization Strategy Selection</div>
        </div>
        <nav className="hidden items-center gap-1 md:flex">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="rounded-md px-3 py-1.5 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white"
            >
              {link.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          {demoMode ? (
            <Badge tone="amber" className="font-mono text-[10px] tracking-wider uppercase">
              DEMO MODE
            </Badge>
          ) : (
            <Badge tone="emerald" className="font-mono text-[10px] tracking-wider uppercase">
              API CONNECTED
            </Badge>
          )}
          <div
            className={`flex items-center gap-2 text-xs font-mono font-medium ${
              live ? "text-emerald-300" : "text-amber-300"
            }`}
          >
            <motion.span
              className={`h-2 w-2 rounded-full ${
                live ? "bg-emerald-400" : "bg-amber-400"
              }`}
              animate={{ opacity: [1, 0.35, 1] }}
              transition={{ duration: 1.8, repeat: Infinity }}
            />
            {live ? "● SYSTEM READY" : "● DEMO MODE"}
          </div>
          <Button
            variant={facultyDemo ? "primary" : "secondary"}
            size="sm"
            onClick={onFacultyDemo}
            className={cn(facultyDemo && "aurora")}
          >
            <GraduationCap className="h-3.5 w-3.5" />
            Faculty Demo
          </Button>
        </div>
      </div>
    </header>
  );
}
