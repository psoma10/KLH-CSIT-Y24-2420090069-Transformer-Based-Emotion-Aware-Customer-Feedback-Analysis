import { useQuery } from "@tanstack/react-query";
import { analyzeReview, explainReview } from "../lib/api.js";
import useDebouncedValue from "./useDebouncedValue.js";

export const MIN_ANALYZE_CHARS = 3;
export const DEBOUNCE_MS = 500;

/** Normalise so trailing whitespace edits don't spawn a redundant request. */
function normalise(text) {
  return text.trim().replace(/\s+/g, " ");
}

/**
 * Real-time emotion analysis for a piece of text.
 *
 * Race safety: both queries are keyed on the *debounced, normalised* text.
 * TanStack Query stores each key's result in its own cache entry and this hook
 * only ever returns the entry for the key currently being rendered. A slow
 * response for "great produ" therefore cannot overwrite a fast response for
 * "great product" — it resolves into a different cache slot that nothing reads.
 * That is a structural guarantee, not a timing heuristic, so no sequence
 * counter or AbortController bookkeeping is needed.
 *
 * `placeholderData: keepPrevious` keeps the last good result on screen while a
 * new key resolves, which is what prevents the results panel from collapsing
 * to empty (and the page from jumping) on every edit.
 */
export default function useLiveAnalysis(rawText) {
  const debouncedRaw = useDebouncedValue(rawText, DEBOUNCE_MS);
  const text = normalise(debouncedRaw);
  const enabled = text.length >= MIN_ANALYZE_CHARS;

  // True while the user has typed something new that hasn't settled yet.
  const isSettling = normalise(rawText) !== text;

  const prediction = useQuery({
    queryKey: ["analyze", text],
    queryFn: () => analyzeReview(text),
    enabled,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    retry: (failureCount, error) => error?.status !== 503 && failureCount < 1,
    placeholderData: (previous) => previous,
  });

  const overall = prediction.data?.overall ?? null;
  const topLabel = overall?.top_label ?? null;
  // Explain the segment that decided the verdict, not the whole review — the
  // attribution is meaningless if it spans text that didn't drive the result.
  const explainedText = overall?.text ?? null;

  const explanation = useQuery({
    queryKey: ["explain", explainedText, topLabel],
    queryFn: () => explainReview(explainedText, topLabel),
    // Only run once we know which label to explain, and only for the text that
    // produced that label — never for a stale pairing.
    enabled: enabled && Boolean(topLabel) && Boolean(explainedText) && !prediction.isPlaceholderData,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    retry: (failureCount, error) => error?.status !== 503 && failureCount < 1,
    placeholderData: (previous) => previous,
  });

  const isModelUnavailable =
    prediction.error?.status === 503 || explanation.error?.status === 503;

  return {
    analyzedText: text,
    enabled,
    prediction,
    explanation,
    isModelUnavailable,
    // "Analyzing" covers both the debounce window and the in-flight request so
    // the indicator never flickers off between the two phases.
    isAnalyzing:
      enabled && (isSettling || prediction.isFetching || prediction.isPlaceholderData),
    isExplaining: explanation.isFetching,
    // Prediction we're safe to render as "current" (not left over from an
    // earlier key while a new one loads). This is the chosen segment's result
    // rather than a score over the whole review, so the panel keeps its
    // existing flat shape while reporting the segment that actually decided
    // the verdict.
    result: overall ? { ...overall, model_version: prediction.data.model_version } : null,
    // Per-segment breakdown, so a mixed review can show both halves instead of
    // averaging them into one misleading label.
    segments: prediction.data?.segments ?? [],
    overallSegmentIndex: prediction.data?.overall_segment_index ?? 0,
    isMixed: Boolean(prediction.data?.is_mixed),
    explainedText,
    isStale: prediction.isPlaceholderData,
  };
}
