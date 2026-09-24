"""Tests for review segmentation.

split_segments is a pure function, so these need no model and run instantly.
The endpoint tests in test_api.py cover the scoring side.
"""
from app.services.segmentation import (
    MAX_SEGMENTS,
    choose_headline_segment,
    split_segments,
)


def test_splits_on_sentence_boundaries():
    assert split_segments("The fit is too good. What is this bullshit") == [
        "The fit is too good.",
        "What is this bullshit",
    ]


def test_splits_on_contrastive_connectives():
    """Reviews pivot mid-sentence more often than between sentences.

    "great but garbage" scored as one segment reports only the praise.
    """
    assert split_segments("The sound is great but the battery is garbage") == [
        "The sound is great",
        "but the battery is garbage",
    ]


def test_does_not_split_a_single_coherent_sentence():
    text = "I absolutely love these headphones and use them daily"

    assert split_segments(text) == [text]


def test_short_fragments_merge_rather_than_becoming_their_own_segment():
    """"Ok." on its own carries no signal and would only add noise."""
    segments = split_segments("These are great. Ok. Really happy with them.")

    assert "Ok." not in segments
    assert len(segments) == 2


def test_leading_short_fragment_folds_forward():
    """A short first fragment has no predecessor to merge into."""
    segments = split_segments("Wow. These headphones changed my commute entirely.")

    assert len(segments) == 1
    assert segments[0].startswith("Wow.")


def test_blank_text_yields_no_segments():
    assert split_segments("   \n ") == []


def test_segment_count_is_bounded():
    """Guards against "a. " * 1000 turning one request into a huge batch."""
    segments = split_segments("This is a sentence. " * 200)

    assert len(segments) == MAX_SEGMENTS


def test_no_text_is_dropped_when_capping():
    text = "Sentence number one here. " * 40
    joined = " ".join(split_segments(text))

    # Every word survives, even though the segment count is capped.
    assert joined.count("Sentence") == 40


def _segment(bucket, score):
    return {"business_bucket": bucket, "top_score": score}


def test_headline_prefers_a_negative_segment_over_a_more_confident_positive_one():
    """The actionable half of a mixed review is the complaint.

    Reporting "satisfaction" because the praise scored higher is how a support
    queue loses an angry customer.
    """
    results = [_segment("satisfaction", 0.92), _segment("frustration", 0.55)]

    assert choose_headline_segment(results) == 1


def test_headline_picks_the_strongest_negative_when_several_exist():
    results = [
        _segment("satisfaction", 0.99),
        _segment("concern", 0.40),
        _segment("frustration", 0.71),
    ]

    assert choose_headline_segment(results) == 2


def test_headline_falls_back_to_the_most_confident_segment_when_all_positive():
    results = [_segment("satisfaction", 0.60), _segment("satisfaction", 0.88)]

    assert choose_headline_segment(results) == 1
