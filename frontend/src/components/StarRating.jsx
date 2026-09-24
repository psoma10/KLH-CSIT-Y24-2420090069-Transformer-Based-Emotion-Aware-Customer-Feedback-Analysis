import { useId, useState } from "react";
import { cn } from "../lib/utils.js";

const SIZES = {
  sm: "h-3.5 w-3.5",
  md: "h-5 w-5",
  lg: "h-8 w-8",
};

/** Single star glyph. `fill` is 0–1 so we can render half/partial stars. */
function Star({ fill, sizeClass, className }) {
  // useId keeps the gradient id unique per rendered star; two stars with the
  // same fill would otherwise emit duplicate DOM ids.
  const clipId = `star-${useId()}`;
  return (
    <svg
      viewBox="0 0 20 20"
      aria-hidden="true"
      className={cn(sizeClass, "shrink-0", className)}
    >
      <defs>
        <linearGradient id={clipId}>
          <stop offset={`${fill * 100}%`} stopColor="currentColor" />
          <stop offset={`${fill * 100}%`} stopColor="transparent" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d="M10 1.6l2.6 5.28 5.83.85-4.22 4.11.996 5.81L10 14.92l-5.21 2.74.996-5.81L1.57 7.73l5.83-.85L10 1.6z"
        fill={`url(#${clipId})`}
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinejoin="round"
        className="opacity-95"
      />
    </svg>
  );
}

/** Read-only star row, supports fractional averages like 4.3. */
export function StarDisplay({ value, size = "sm", className, label }) {
  const sizeClass = SIZES[size] ?? SIZES.sm;
  return (
    <span
      className={cn("inline-flex items-center gap-0.5 text-amber-500", className)}
      role="img"
      aria-label={label ?? `${value} out of 5 stars`}
    >
      {[0, 1, 2, 3, 4].map((i) => (
        <Star key={i} fill={Math.min(1, Math.max(0, value - i))} sizeClass={sizeClass} />
      ))}
    </span>
  );
}

const RATING_WORDS = {
  0: "Select a rating",
  1: "I hate it",
  2: "I don't like it",
  3: "It's okay",
  4: "I like it",
  5: "I love it",
};

/**
 * Interactive rating input. Implemented as a radiogroup so arrow keys and
 * Tab work natively and screen readers announce the selection.
 */
export function StarInput({ value, onChange, id = "star-input" }) {
  const [hovered, setHovered] = useState(0);
  const shown = hovered || value;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div
        role="radiogroup"
        aria-label="Overall product rating"
        id={id}
        className="flex items-center gap-1"
        onMouseLeave={() => setHovered(0)}
      >
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            role="radio"
            aria-checked={value === star}
            aria-label={`${star} star${star === 1 ? "" : "s"} — ${RATING_WORDS[star]}`}
            onClick={() => onChange(star)}
            onMouseEnter={() => setHovered(star)}
            onFocus={() => setHovered(star)}
            onBlur={() => setHovered(0)}
            className={cn(
              "rounded-md p-0.5 transition-all duration-200 ease-out",
              "hover:scale-110 active:scale-95",
              "focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-2",
              star <= shown ? "text-amber-500" : "text-slate-300 hover:text-amber-300"
            )}
          >
            <Star fill={star <= shown ? 1 : 0} sizeClass={SIZES.lg} />
          </button>
        ))}
      </div>
      <span
        aria-live="polite"
        className={cn(
          "text-sm font-semibold transition-colors duration-200",
          shown ? "text-slate-900" : "text-slate-400"
        )}
      >
        {RATING_WORDS[shown]}
      </span>
    </div>
  );
}

export default StarDisplay;
