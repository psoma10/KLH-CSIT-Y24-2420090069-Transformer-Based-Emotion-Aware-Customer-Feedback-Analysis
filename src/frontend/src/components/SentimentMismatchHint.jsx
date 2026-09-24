import { cn } from "../lib/utils.js";

// Confidence floor: below this the model isn't sure enough to second-guess
// what the user explicitly selected.
const MIN_CONFIDENCE = 0.35;

/**
 * Compares the star rating the user picked against the emotion detected in
 * their text and returns a gentle nudge when the two disagree. Returns null
 * when they agree, when confidence is low, or when nothing is rated yet.
 */
export function getMismatch({ rating, bucket, topLabel, score }) {
  if (!rating || !bucket || !topLabel) return null;
  if (score < MIN_CONFIDENCE) return null;

  const positiveStars = rating >= 4;
  const negativeStars = rating <= 2;

  if (positiveStars && bucket === "frustration") {
    return {
      tone: "warning",
      title: "Your words read more frustrated than your rating",
      body: `You selected ${rating} stars, but the strongest emotion detected is “${topLabel}”. If something went wrong, a lower rating may reflect your experience more accurately.`,
    };
  }

  if (negativeStars && bucket === "satisfaction") {
    return {
      tone: "info",
      title: "Your words read more positive than your rating",
      body: `You selected ${rating} star${rating === 1 ? "" : "s"}, but the strongest emotion detected is “${topLabel}”. Did you mean to rate this higher?`,
    };
  }

  if (positiveStars && bucket === "confusion") {
    return {
      tone: "info",
      title: "Sounds like something was unclear",
      body: `Your ${rating}-star review reads as “${topLabel}”. Adding what confused you helps other shoppers most.`,
    };
  }

  if (rating === 3 && bucket === "frustration" && score > 0.5) {
    return {
      tone: "warning",
      title: "This reads stronger than a neutral rating",
      body: `The dominant emotion detected is “${topLabel}”, which is more negative than 3 stars usually signals.`,
    };
  }

  return null;
}

const TONES = {
  warning: {
    wrap: "border-amber-300 bg-amber-50",
    title: "text-amber-900",
    body: "text-amber-800",
    icon: "text-amber-500",
  },
  info: {
    wrap: "border-sky-300 bg-sky-50",
    title: "text-sky-900",
    body: "text-sky-800",
    icon: "text-sky-500",
  },
};

export default function SentimentMismatchHint({ mismatch }) {
  if (!mismatch) return null;
  const tone = TONES[mismatch.tone] ?? TONES.info;

  return (
    <div
      role="status"
      className={cn(
        "flex gap-3 rounded-xl border p-4 shadow-sm",
        "animate-rise-in motion-reduce:animate-none",
        tone.wrap
      )}
    >
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        aria-hidden="true"
        className={cn("mt-0.5 h-5 w-5 shrink-0", tone.icon)}
      >
        <path
          fillRule="evenodd"
          d="M18 10A8 8 0 112 10a8 8 0 0116 0zm-9-4a1 1 0 112 0v4a1 1 0 11-2 0V6zm1 8.5a1.25 1.25 0 100-2.5 1.25 1.25 0 000 2.5z"
          clipRule="evenodd"
        />
      </svg>
      <div className="min-w-0 space-y-1">
        <p className={cn("text-sm font-semibold", tone.title)}>{mismatch.title}</p>
        <p className={cn("text-sm leading-relaxed", tone.body)}>{mismatch.body}</p>
      </div>
    </div>
  );
}
