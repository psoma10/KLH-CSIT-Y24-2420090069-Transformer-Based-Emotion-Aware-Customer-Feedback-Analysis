import EmotionBarChart from "./EmotionBarChart.jsx";
import SegmentBreakdown from "./SegmentBreakdown.jsx";
import TokenHeatmap from "./TokenHeatmap.jsx";
import Badge from "./ui/Badge.jsx";
import ProgressBar from "./ui/ProgressBar.jsx";
import { getBucketColor } from "../lib/emotions.js";
import { MIN_ANALYZE_CHARS } from "../hooks/useLiveAnalysis.js";
import { cn } from "../lib/utils.js";

function PanelShell({ title, aside, children, className }) {
  return (
    <section
      className={cn(
        "rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6",
        className
      )}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          {title}
        </h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

/** Shown before the user has typed enough — keeps the column from being empty. */
function EmptyState() {
  return (
    <PanelShell title="Live emotion analysis">
      <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
        <span aria-hidden="true" className="text-4xl opacity-60">
          ✍️
        </span>
        <p className="max-w-xs text-sm leading-relaxed text-slate-500">
          Start writing your review and it will be analyzed for emotion in real
          time — at least {MIN_ANALYZE_CHARS} characters.
        </p>
      </div>
    </PanelShell>
  );
}

function ModelUnavailable() {
  return (
    <PanelShell title="Live emotion analysis" className="border-amber-300 bg-amber-50">
      <p className="text-sm font-semibold text-amber-900">Model not loaded yet</p>
      <p className="mt-1.5 text-sm leading-relaxed text-amber-800">
        The emotion model hasn&apos;t been trained or loaded on the backend. Run{" "}
        <code className="rounded bg-amber-100 px-1 py-0.5 font-mono text-xs">
          ml/train_roberta.py
        </code>{" "}
        and restart the API — the review composer still works in the meantime.
      </p>
    </PanelShell>
  );
}

function ErrorState({ error }) {
  return (
    <PanelShell title="Live emotion analysis" className="border-red-300 bg-red-50">
      <p className="text-sm font-semibold text-red-900">Analysis failed</p>
      <p className="mt-1.5 text-sm leading-relaxed text-red-800">{error.message}</p>
    </PanelShell>
  );
}

function TopResult({ result, isStale }) {
  const bucketColor = getBucketColor(result.business_bucket ?? result.top_label);
  const confidence = result.top_score * 100;

  return (
    <PanelShell
      title="Live emotion analysis"
      aside={
        result.business_bucket ? (
          <Badge color={bucketColor} className="px-3 py-1 text-xs capitalize">
            {result.business_bucket}
          </Badge>
        ) : (
          // No bucket means no emotion cleared its tuned threshold. The label
          // below is still the model's best guess, but it is a guess — saying
          // so is more honest than showing a confident-looking bare result.
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
            Low confidence
          </span>
        )
      }
      className={cn(
        "transition-opacity duration-300 motion-reduce:transition-none",
        isStale && "opacity-60"
      )}
    >
      <div className="space-y-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Top emotion
            </p>
            <p
              key={result.top_label}
              className="animate-rise-in text-3xl font-bold capitalize tracking-tight text-slate-900 motion-reduce:animate-none sm:text-4xl"
              style={{ color: bucketColor }}
            >
              {result.top_label}
            </p>
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Confidence
            </p>
            <p className="text-3xl font-bold tabular-nums tracking-tight text-slate-900 sm:text-4xl">
              {confidence.toFixed(1)}%
            </p>
          </div>
        </div>

        <ProgressBar
          value={confidence}
          max={100}
          className="h-2"
          barClassName="transition-[width] duration-500 ease-out motion-reduce:transition-none"
        />

        {(() => {
          // Computed once so the heading and the badges agree: rendering the
          // "Also detected" label whenever predicted_labels is non-empty — even
          // if top_label is the only entry — leaves an orphaned heading with
          // nothing under it.
          const secondaryLabels = (result.predicted_labels ?? []).filter(
            (l) => l !== result.top_label
          );
          if (secondaryLabels.length === 0) return null;

          return (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-xs font-medium uppercase tracking-wide text-slate-500">
                Also detected
              </span>
              {secondaryLabels.slice(0, 6).map((label) => (
                <span
                  key={label}
                  className="rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize transition-transform duration-200 hover:scale-105 motion-reduce:transition-none"
                  style={{
                    color: getBucketColor(label),
                    borderColor: `${getBucketColor(label)}55`,
                    backgroundColor: `${getBucketColor(label)}0f`,
                  }}
                >
                  {label}
                </span>
              ))}
            </div>
          );
        })()}
      </div>
    </PanelShell>
  );
}

function Explanation({ explanation, isExplaining, isModelUnavailable }) {
  return (
    <PanelShell
      title="Why — token attribution"
      aside={
        explanation.data ? (
          <span className="text-xs font-medium capitalize text-slate-500">
            explaining “{explanation.data.explained_label}”
          </span>
        ) : null
      }
    >
      {/* min-height reserves the slot so swapping states never shifts the page. */}
      <div className="min-h-[7rem]">
        {isModelUnavailable ? (
          <p className="text-sm text-amber-700">
            SHAP explainer isn&apos;t loaded on the backend yet.
          </p>
        ) : explanation.isError ? (
          <p className="text-sm text-red-700">{explanation.error.message}</p>
        ) : explanation.data ? (
          <div
            className={cn(
              "space-y-3 transition-opacity duration-300 motion-reduce:transition-none",
              isExplaining && "opacity-50"
            )}
          >
            <p className="text-xs leading-relaxed text-slate-500">
              Warm tokens push{" "}
              <span className="font-medium capitalize text-slate-700">
                {explanation.data.explained_label}
              </span>{" "}
              up, cool tokens push it down (score{" "}
              {explanation.data.label_score.toFixed(4)}).
            </p>
            <TokenHeatmap
              tokens={explanation.data.tokens}
              attributions={explanation.data.attributions}
            />
          </div>
        ) : (
          <div className="space-y-2" aria-hidden="true">
            {[90, 75, 60].map((w) => (
              <div
                key={w}
                className="relative h-5 overflow-hidden rounded bg-slate-100"
                style={{ width: `${w}%` }}
              >
                <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/70 to-transparent motion-reduce:animate-none" />
              </div>
            ))}
          </div>
        )}
      </div>
    </PanelShell>
  );
}

export default function LiveAnalysisPanel({
  enabled,
  result,
  isStale,
  prediction,
  explanation,
  isExplaining,
  isModelUnavailable,
  segments,
  overallSegmentIndex,
  isMixed,
  children,
}) {
  if (isModelUnavailable) return <ModelUnavailable />;
  if (prediction.isError) return <ErrorState error={prediction.error} />;
  if (!enabled || !result) return <EmptyState />;

  return (
    <div className="space-y-5">
      <TopResult result={result} isStale={isStale} />
      {children}
      <SegmentBreakdown
        segments={segments}
        overallIndex={overallSegmentIndex}
        isMixed={isMixed}
      />
      <Explanation
        explanation={explanation}
        isExplaining={isExplaining}
        isModelUnavailable={isModelUnavailable}
      />
      {/* Collapsed by default: the full 28-bar chart is taller than the
          viewport and would otherwise defeat the sticky column. */}
      <details className="group rounded-2xl border border-slate-200 bg-white shadow-sm">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-2xl p-5 transition-colors duration-200 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 sm:p-6">
          <span className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            All 28 emotion scores
          </span>
          <span className="flex items-center gap-2">
            <span className="text-xs text-slate-400">{result.model_version}</span>
            <svg
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
              className="h-4 w-4 text-slate-400 transition-transform duration-300 group-open:rotate-180 motion-reduce:transition-none"
            >
              <path
                fillRule="evenodd"
                d="M5.23 7.21a.75.75 0 011.06.02L10 11.17l3.71-3.94a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
                clipRule="evenodd"
              />
            </svg>
          </span>
        </summary>
        <div className="px-2 pb-5 sm:px-3">
          <EmotionBarChart
            scores={result.scores}
            domain={[0, 1]}
            height={780}
            labelInterval={0}
            yAxisWidth={96}
            valueFormatter={(v) => `${(v * 100).toFixed(1)}%`}
          />
        </div>
      </details>
    </div>
  );
}
