// Mirrors ml/emotion_labels.py exactly — label order matters for model I/O,
// and the bucket mapping is duplicated here because the backend does not
// expose it as an endpoint (it's a static, code-level constant on that side too).

export const GOEMOTIONS_LABELS = [
  "admiration", "amusement", "anger", "annoyance", "approval", "caring",
  "confusion", "curiosity", "desire", "disappointment", "disapproval",
  "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
  "joy", "love", "nervousness", "optimism", "pride", "realization",
  "relief", "remorse", "sadness", "surprise", "neutral",
];

export const BUSINESS_BUCKETS = {
  satisfaction: [
    "admiration", "amusement", "approval", "caring", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
  ],
  frustration: [
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "remorse", "sadness", "grief", "embarrassment",
  ],
  confusion: [
    "confusion", "curiosity", "realization", "surprise", "nervousness",
  ],
  concern: [
    "desire", "fear",
  ],
};

const BUCKET_COLORS = {
  satisfaction: "#16a34a", // green-600
  frustration: "#dc2626", // red-600
  confusion: "#d97706", // amber-600
  concern: "#7c3aed", // violet-600
  neutral: "#64748b", // slate-500
};

export function labelToBucket(label) {
  for (const [bucket, labels] of Object.entries(BUSINESS_BUCKETS)) {
    if (labels.includes(label)) return bucket;
  }
  return label === "neutral" ? "neutral" : null;
}

/** Consistent chart color for a bucket name (or a label, resolved to its bucket first). */
export function getBucketColor(bucketOrLabel) {
  if (BUCKET_COLORS[bucketOrLabel]) return BUCKET_COLORS[bucketOrLabel];
  const bucket = labelToBucket(bucketOrLabel);
  return bucket ? BUCKET_COLORS[bucket] : "#94a3b8"; // slate-400 fallback
}

export const BUCKET_NAMES = ["satisfaction", "frustration", "confusion", "concern"];
