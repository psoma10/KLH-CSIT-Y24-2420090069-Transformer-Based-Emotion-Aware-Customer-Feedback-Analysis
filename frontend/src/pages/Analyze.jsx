import { useCallback, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import ProductCard from "../components/ProductCard.jsx";
import ReviewComposer from "../components/ReviewComposer.jsx";
import LiveAnalysisPanel from "../components/LiveAnalysisPanel.jsx";
import SentimentMismatchHint, { getMismatch } from "../components/SentimentMismatchHint.jsx";
import useLiveAnalysis from "../hooks/useLiveAnalysis.js";
import { analyzeReview } from "../lib/api.js";
import { DEMO_PRODUCTS } from "../lib/demoProducts.js";

/**
 * Amazon-style product page where the visitor writes a review and sees the
 * emotion model's read on their text update live as they type.
 *
 * The composed text sent for analysis is "headline. body" so the headline —
 * often the most emotionally loaded part of a review — is not ignored.
 */
export default function Analyze() {
  const [productId, setProductId] = useState(DEMO_PRODUCTS[0].id);
  const [rating, setRating] = useState(0);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");

  const product = useMemo(
    () => DEMO_PRODUCTS.find((p) => p.id === productId) ?? DEMO_PRODUCTS[0],
    [productId]
  );

  const composedText = useMemo(() => {
    const head = title.trim();
    const rest = body.trim();
    if (head && rest) return `${head}. ${rest}`;
    return head || rest;
  }, [title, body]);

  const analysis = useLiveAnalysis(composedText);

  // The real Submit action — separate from useLiveAnalysis's preview query,
  // which never persists. mutate/reset are stable across renders even though
  // the mutation object itself is not, so depending on them below (rather
  // than on submitMutation as a whole) keeps handleSelectProduct and
  // handleBodyChange stably identified for ProductCard's memo.
  const submitMutation = useMutation({
    mutationFn: () =>
      analyzeReview(composedText, { persist: true, starRating: rating, productId }),
  });
  const { mutate: submitReview, reset: resetSubmit } = submitMutation;

  const mismatch = useMemo(() => {
    if (!analysis.result || analysis.isStale) return null;
    return getMismatch({
      rating,
      bucket: analysis.result.business_bucket,
      topLabel: analysis.result.top_label,
      score: analysis.result.top_score,
    });
  }, [analysis.result, analysis.isStale, rating]);

  // Immutable resets — every handler builds new state rather than mutating.
  // Stable identities matter here: this component re-renders on every
  // keystroke, and a fresh function each time would defeat the memo() on
  // ProductCard and force it to re-render along with it.
  const handleSelectProduct = useCallback(
    (id) => {
      setProductId(id);
      resetSubmit();
    },
    [resetSubmit]
  );

  const handleSubmit = useCallback(() => submitReview(), [submitReview]);

  const handleBodyChange = useCallback(
    (value) => {
      setBody(value);
      resetSubmit();
    },
    [resetSubmit]
  );

  return (
    <div className="space-y-6">
      <nav aria-label="Breadcrumb" className="text-xs text-slate-500">
        <ol className="flex flex-wrap items-center gap-1.5">
          {["Electronics", product.brand, "Customer reviews"].map((crumb, i, all) => (
            <li key={crumb} className="flex items-center gap-1.5">
              <span
                className={
                  i === all.length - 1
                    ? "font-medium text-slate-700"
                    : "cursor-default transition-colors hover:text-orange-600"
                }
              >
                {crumb}
              </span>
              {i < all.length - 1 && (
                <span aria-hidden="true" className="text-slate-300">
                  ›
                </span>
              )}
            </li>
          ))}
        </ol>
      </nav>

      <ProductCard
        products={DEMO_PRODUCTS}
        product={product}
        onSelect={handleSelectProduct}
      />

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <ReviewComposer
          product={product}
          rating={rating}
          onRatingChange={setRating}
          title={title}
          onTitleChange={setTitle}
          body={body}
          onBodyChange={handleBodyChange}
          isAnalyzing={analysis.isAnalyzing}
          onSubmit={handleSubmit}
          isSubmitting={submitMutation.isPending}
          submitSuccess={submitMutation.isSuccess}
          submitError={submitMutation.isError ? submitMutation.error.message : null}
        />

        {/* Sticky on desktop so the live result stays in view while typing. */}
        <div className="lg:sticky lg:top-6">
          <LiveAnalysisPanel
            enabled={analysis.enabled}
            result={analysis.result}
            isStale={analysis.isStale}
            prediction={analysis.prediction}
            explanation={analysis.explanation}
            isExplaining={analysis.isExplaining}
            isModelUnavailable={analysis.isModelUnavailable}
            segments={analysis.segments}
            overallSegmentIndex={analysis.overallSegmentIndex}
            isMixed={analysis.isMixed}
          >
            <SentimentMismatchHint mismatch={mismatch} />
          </LiveAnalysisPanel>
        </div>
      </div>
    </div>
  );
}
