import { cn } from "../lib/utils.js";

/**
 * Amazon's review-summary breakdown: one row per star level with a bar showing
 * what share of reviews sit there. `histogram` maps 5..1 -> percentage.
 */
export default function RatingHistogram({ histogram, className }) {
  return (
    <ul className={cn("space-y-1.5", className)}>
      {[5, 4, 3, 2, 1].map((star) => {
        const pct = histogram[star] ?? 0;
        return (
          <li key={star} className="group flex items-center gap-3">
            <span className="w-12 shrink-0 text-xs font-medium text-sky-700 transition-colors group-hover:text-orange-600 group-hover:underline">
              {star} star
            </span>
            <span className="h-4 flex-1 overflow-hidden rounded-sm border border-slate-300 bg-slate-100">
              <span
                className="block h-full rounded-l-sm bg-gradient-to-r from-amber-400 to-amber-500 transition-[width] duration-500 ease-out motion-reduce:transition-none"
                style={{ width: `${pct}%` }}
              />
            </span>
            <span className="w-9 shrink-0 text-right text-xs font-medium tabular-nums text-slate-600">
              {pct}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}
