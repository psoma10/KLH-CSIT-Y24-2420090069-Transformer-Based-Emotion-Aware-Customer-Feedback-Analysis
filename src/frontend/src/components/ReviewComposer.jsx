import { StarInput } from "./StarRating.jsx";
import Input from "./ui/Input.jsx";
import Textarea from "./ui/Textarea.jsx";
import Button from "./ui/Button.jsx";
import { MIN_ANALYZE_CHARS } from "../hooks/useLiveAnalysis.js";
import { cn } from "../lib/utils.js";

const MAX_BODY = 5000;

/**
 * Amazon's "Write a customer review" panel. Fully controlled — all state lives
 * in the Analyze page so the live analysis can read the body text.
 */
export default function ReviewComposer({
  product,
  rating,
  onRatingChange,
  title,
  onTitleChange,
  body,
  onBodyChange,
  isAnalyzing,
  onSubmit,
  isSubmitting,
  submitSuccess,
  submitError,
}) {
  const tooShort = body.trim().length > 0 && body.trim().length < MIN_ANALYZE_CHARS;

  return (
    <section
      aria-labelledby="composer-heading"
      className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
    >
      <header className="border-b border-slate-200 bg-gradient-to-b from-slate-50 to-white px-5 py-4 sm:px-7">
        <h2 id="composer-heading" className="text-lg font-semibold text-slate-900">
          Write a customer review
        </h2>
        <p className="mt-0.5 text-sm text-slate-500">
          Share your thoughts on the{" "}
          <span className="font-medium text-slate-700">{product.brand}</span> with other
          customers
        </p>
      </header>

      <div className="space-y-6 px-5 py-6 sm:px-7">
        <fieldset className="space-y-2.5">
          <legend className="text-sm font-semibold text-slate-900">
            Overall rating
          </legend>
          <StarInput value={rating} onChange={onRatingChange} />
        </fieldset>

        <div className="space-y-2">
          <label
            htmlFor="review-title"
            className="block text-sm font-semibold text-slate-900"
          >
            Add a headline
          </label>
          <Input
            id="review-title"
            value={title}
            onChange={(e) => onTitleChange(e.target.value)}
            placeholder="What's most important to know?"
            maxLength={150}
            className="w-full transition-shadow duration-200 focus:shadow-md"
          />
        </div>

        <div className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <label
              htmlFor="review-body"
              className="block text-sm font-semibold text-slate-900"
            >
              Add a written review
            </label>
            {/* Reserved-height row: the analyzing pill swaps in without shifting layout. */}
            <span className="flex h-5 items-center gap-2 text-xs">
              <span
                aria-hidden={!isAnalyzing}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full bg-sky-50 px-2 py-0.5 font-medium text-sky-700 ring-1 ring-sky-200",
                  "transition-opacity duration-300 motion-reduce:transition-none",
                  isAnalyzing ? "opacity-100" : "opacity-0"
                )}
              >
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-500" />
                Analyzing…
              </span>
              <span className="tabular-nums text-slate-400">
                {body.length}/{MAX_BODY}
              </span>
            </span>
          </div>

          <Textarea
            id="review-body"
            rows={8}
            maxLength={MAX_BODY}
            value={body}
            onChange={(e) => onBodyChange(e.target.value)}
            placeholder="What did you like or dislike? What did you use this product for? Start typing — your review is analyzed for emotion as you write."
            aria-describedby="review-body-help"
            className="resize-y leading-relaxed transition-shadow duration-200 focus:shadow-md"
          />

          <p id="review-body-help" className="text-xs leading-relaxed text-slate-500">
            {tooShort
              ? `Keep going — at least ${MIN_ANALYZE_CHARS} characters are needed to analyze.`
              : "Emotion analysis runs automatically about half a second after you stop typing."}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 border-t border-slate-200 pt-5">
          <Button
            onClick={onSubmit}
            disabled={!rating || !body.trim() || isSubmitting}
            className="rounded-full bg-gradient-to-b from-amber-300 to-amber-400 px-8 text-slate-900 shadow-sm transition-all duration-200 hover:-translate-y-px hover:from-amber-200 hover:to-amber-300 hover:shadow-md active:translate-y-0 disabled:from-slate-200 disabled:to-slate-200 disabled:text-slate-400 disabled:shadow-none motion-reduce:transition-none motion-reduce:hover:translate-y-0"
          >
            {isSubmitting ? "Submitting…" : "Submit"}
          </Button>
          {/* The live region stays mounted so the confirmation is announced when
              it appears, but the message itself is only rendered once there is
              something to confirm — fading it with opacity alone would leave it
              in the accessibility tree, where screen readers announce a
              submission that never happened. */}
          <span aria-live="polite" className="text-sm font-medium">
            {submitSuccess && (
              <span className="animate-fade-in text-green-700">
                Thanks — your review was recorded for this demo.
              </span>
            )}
            {submitError && (
              <span className="animate-fade-in text-red-700">
                Couldn't save your review: {submitError}
              </span>
            )}
          </span>
        </div>
      </div>
    </section>
  );
}
