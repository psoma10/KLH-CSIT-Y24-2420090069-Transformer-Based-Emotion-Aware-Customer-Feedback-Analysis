"""Splits a review into independently-classifiable segments.

A single emotion score for a whole review averages away exactly the cases
worth surfacing: "The fit is too good. What the fuck is this bullshit" scores
`admiration` overall, because a strongly positive clause and a strongly
negative one cancel. Scored apart, the same text is `admiration` 0.92 and
`anger` 0.82 — a mixed review, which is the truth.

Kept free of torch and FastAPI imports so it can be tested as a pure function.
"""
import re

# Split on sentence-ending punctuation followed by whitespace. Deliberately
# simple: a full NLP sentence tokenizer would pull in another dependency to
# handle abbreviations that barely occur in product reviews, and over-splitting
# is harmless here — each fragment is still scored on its own.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Reviews pivot mid-sentence far more often than they pivot between sentences:
# "the sound is great but the battery is garbage" is one sentence carrying two
# opposite verdicts, and scoring it whole reports only the first. Splitting
# before a contrastive connective keeps each verdict separately visible. The
# connective stays with the clause it introduces, since it belongs to it.
_CONTRAST_BOUNDARY = re.compile(
    r"\s+(?=(?:but|however|although|though|except|unfortunately|whereas)\b)",
    re.IGNORECASE,
)

# Below this, a fragment carries too little signal to classify on its own
# ("Ok.", "Nope.") and would just add noise to the breakdown.
MIN_SEGMENT_CHARS = 12

# Guards against a pathological input ("a. " * 1000) turning one request into
# a thousand-row batch.
MAX_SEGMENTS = 12


def split_segments(text: str) -> list[str]:
    """Returns the parts of `text` worth scoring separately.

    Always returns at least one segment for non-blank input. Short trailing
    fragments are merged into the previous segment rather than dropped, so no
    part of the review goes unscored.
    """
    stripped = text.strip()
    if not stripped:
        return []

    parts = [
        clause.strip()
        for sentence in _SENTENCE_BOUNDARY.split(stripped)
        for clause in _CONTRAST_BOUNDARY.split(sentence)
        if clause.strip()
    ]
    if not parts:
        return []

    merged: list[str] = []
    for part in parts:
        if merged and len(part) < MIN_SEGMENT_CHARS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    # A leading fragment has no predecessor to merge into, so fold it forward.
    if len(merged) > 1 and len(merged[0]) < MIN_SEGMENT_CHARS:
        merged = [f"{merged[0]} {merged[1]}", *merged[2:]]

    if len(merged) > MAX_SEGMENTS:
        head = merged[: MAX_SEGMENTS - 1]
        return [*head, " ".join(merged[MAX_SEGMENTS - 1 :])]

    return merged


# Buckets that represent a problem the business would want surfaced. When a
# review mixes praise and complaint, the complaint is the actionable half —
# reporting "satisfaction" because the praise scored marginally higher is how
# a support queue misses an angry customer.
NEGATIVE_BUCKETS = frozenset({"frustration", "concern", "confusion"})


def choose_headline_segment(results: list[dict]) -> int:
    """Picks which segment's emotion should represent the whole review.

    Prefers the most confident negative segment; falls back to the most
    confident segment overall when nothing negative was found.
    """
    if not results:
        raise ValueError("cannot choose a headline segment from an empty list")

    negatives = [
        (i, r) for i, r in enumerate(results) if r["business_bucket"] in NEGATIVE_BUCKETS
    ]
    pool = negatives or list(enumerate(results))

    return max(pool, key=lambda pair: pair[1]["top_score"])[0]
