import { cn } from "../lib/utils.js";

/**
 * Amazon's review-summary breakdown: one row per star level with a bar showing
 * what share of reviews sit there. `histogram` maps 5..1 -> percentage.
 */
export default function RatingHistogram({ histogram, className }) {
  return (
    <ul className={cn("space-y-2", className)}>
      {[5, 4, 3, 2, 1].map((star) => {
        const pct = histogram[star] ?? 0;
        return (
          <li key={star} className="flex items-center gap-3">
            <span className="w-12 shrink-0 text-xs font-medium text-slate-600">{star} star</span>
            <span
              className="h-3 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-100"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${star} star`}
            >
              <span
                className="block h-full rounded-full bg-amber-400 transition-[width] duration-500 ease-out motion-reduce:transition-none"
                style={{ width: `${pct}%` }}
              />
            </span>
            <span className="w-10 shrink-0 text-right text-xs font-medium tabular-nums text-slate-600">
              {pct}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}
