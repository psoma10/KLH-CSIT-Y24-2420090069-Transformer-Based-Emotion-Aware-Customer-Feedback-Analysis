import Badge from "./ui/Badge.jsx";
import { getBucketColor } from "../lib/emotions.js";
import { cn } from "../lib/utils.js";

/**
 * Per-sentence emotion breakdown for a review that carries more than one
 * verdict.
 *
 * A single score over the whole text averages opposing sentiment away — praise
 * in one sentence cancels a complaint in the next, and the review reports as
 * satisfied. Showing each part separately is what makes the mixed case legible,
 * and marks which segment the headline verdict came from.
 */
export default function SegmentBreakdown({ segments, overallIndex, isMixed }) {
  // A single-segment review is already fully described by the headline result.
  if (!segments || segments.length < 2) return null;

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Breakdown by sentence
        </h3>
        {isMixed && (
          <span className="rounded-full bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-800">
            Mixed signals
          </span>
        )}
      </div>

      {isMixed && (
        <p className="mt-2 text-sm text-slate-500">
          This review says different things in different places. The overall verdict follows the
          strongest negative part, since that is the actionable half.
        </p>
      )}

      <ol className="mt-4 space-y-2.5">
        {segments.map((segment, index) => {
          const isOverall = index === overallIndex;
          return (
            <li
              key={`${index}-${segment.text.slice(0, 24)}`}
              className={cn(
                "rounded-xl border p-3 transition-colors duration-200",
                isOverall ? "border-slate-300 bg-slate-50" : "border-slate-200 bg-white"
              )}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge color={getBucketColor(segment.business_bucket ?? segment.top_label)}>
                  {segment.top_label}
                </Badge>
                <span className="text-xs tabular-nums text-slate-500">
                  {(segment.top_score * 100).toFixed(1)}%
                </span>
                {isOverall && (
                  <span className="text-xs font-medium text-slate-600">drives the verdict</span>
                )}
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-700">{segment.text}</p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
