import { cn } from "../../lib/utils.js";

export default function ProgressBar({ value = 0, max = 100, className, barClassName }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className={cn("h-2.5 w-full overflow-hidden rounded-full bg-slate-200", className)}>
      <div
        className={cn("h-full rounded-full bg-slate-900 transition-all", barClassName)}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
