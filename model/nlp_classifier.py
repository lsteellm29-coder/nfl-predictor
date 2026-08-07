# Hugging Face NLP layer: headline classification, player extraction, injury-language nuance
"""MCP Integration spec, Section 3 (Hugging Face). Firecrawl/Playwright get
raw text; this is where a real model interprets it -- turning scraped
headlines and press-conference summaries into structured signal instead
of hand-written parsing rules. Runs entirely locally (transformers, no
API key/account needed) via two off-the-shelf pretrained models, neither
fine-tuned:

- facebook/bart-large-mnli (zero-shot classification) for both headline
  categorization and injury-language nuance extraction -- the same
  mechanism, different candidate labels, rather than two separate models.
- dslim/bert-base-NER (token classification) for player-name extraction
  from free text.

Same rule as data/fetch_news.py and every other supplementary signal in
this codebase: nothing here feeds model/train.py's FEATURE_COLS or
model/td_model.py's projections. classify_headline() is available as a
higher-quality alternative to data/fetch_news.py's hand-written keyword
classifier (CATEGORY_PATTERNS) -- both still exist; the keyword version
stays the fast, dependency-free default (it's what actually ran all
session, catching real cases like the Saints suspension before the
official injury report), this is opt-in for callers that want the
extra nuance and can afford the model-load cost.
"""

HEADLINE_LABELS = ["injury", "suspension", "lineup change", "trade or roster move", "coaching news", "general commentary"]
INJURY_SENTIMENT_LABELS = [
    "expected to play", "questionable or uncertain status", "trending toward being out",
    "ruled out or will not play", "no injury-related content",
]

_zero_shot = None
_ner = None


def _get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        from transformers import pipeline
        _zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _zero_shot


def _get_ner():
    global _ner
    if _ner is None:
        from transformers import pipeline
        _ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return _ner


def classify_headline(text: str, labels: list[str] = HEADLINE_LABELS) -> dict:
    """{"label": top category, "score": confidence, "all_scores": {label: score, ...}}
    -- a model-based alternative to data/fetch_news.py's keyword matcher,
    for callers willing to pay the model-load/inference cost for better
    nuance (e.g. distinguishing "trade rumor" from "trade confirmed"
    language a fixed keyword list can't)."""
    result = _get_zero_shot()(text, candidate_labels=labels)
    return {
        "label": result["labels"][0], "score": result["scores"][0],
        "all_scores": dict(zip(result["labels"], result["scores"])),
    }


def extract_player_names(text: str) -> list[str]:
    """Every PERSON entity mentioned in `text`, re-sliced from the
    original string by character offset rather than concatenated from
    subword tokens -- the token-level output frequently fragments a
    name like "Jalen Hurts" into "J" / "##ale" / "##n Hurts" for names
    the model's tokenizer doesn't have as whole vocabulary tokens (common
    for player names), which re-slicing the source text avoids entirely.
    Returns raw extracted strings, not resolved to a specific roster
    player or GSIS id -- that resolution (and the name-collision risk
    that comes with it) is left to the caller, same principle as
    model/player_stats.py's own name-matching, which never guesses past
    an unambiguous match."""
    entities = _get_ner()(text)
    merged = []
    for e in entities:
        if e["entity_group"] != "PER":
            continue
        if merged and merged[-1]["end"] >= e["start"] - 1:
            merged[-1]["end"] = e["end"]
        else:
            merged.append({"start": e["start"], "end": e["end"]})
    return [text[m["start"]:m["end"]] for m in merged]


def extract_injury_sentiment(text: str) -> dict:
    """Same shape as classify_headline, using INJURY_SENTIMENT_LABELS --
    "expected to play," "trending toward out," "not optimistic" carry
    real signal a binary injury-report tag doesn't capture (MCP
    Integration spec, Section 3). Supplementary only: this is meant to
    sit alongside data/fetch_injuries.py's structured injury report for
    a human/downstream review to weigh, never to overwrite or feed it
    directly -- the same "never a direct replacement for structured
    injury/inactive data" rule the spec states explicitly."""
    return classify_headline(text, INJURY_SENTIMENT_LABELS)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--mode", choices=["headline", "entities", "injury"], default="headline")
    args = parser.parse_args()

    if args.mode == "headline":
        print(classify_headline(args.text))
    elif args.mode == "entities":
        print(extract_player_names(args.text))
    else:
        print(extract_injury_sentiment(args.text))


if __name__ == "__main__":
    main()
