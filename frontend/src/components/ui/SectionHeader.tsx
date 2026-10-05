import { cn } from "../../lib/utils";

export function SectionHeader({
  kicker,
  title,
  description,
  className,
}: {
  kicker: string;
  title: string;
  description?: string;
  className?: string;
}) {
  return (
    <div className={cn("mb-5", className)}>
      <div className="section-kicker">{kicker}</div>
      <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">{title}</h2>
      {description ? <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">{description}</p> : null}
    </div>
  );
}
