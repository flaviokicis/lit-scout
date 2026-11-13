# The quote search (normalize_text, best_window, numbers_grounded, authentic_alignment, find_quote and the number
# tokenizer) is adapted from co-researcher, skills/literature-review/scripts/check_claims.py,
# https://github.com/poemswe/co-researcher. MIT License, Copyright (c) 2025-2026 poemswe: permission is granted free
# of charge to use, copy, modify and distribute it, provided this notice is kept. THE SOFTWARE IS PROVIDED "AS IS",
# WITHOUT WARRANTY OF ANY KIND.
"""Check the reader's line for each kept paper against the text the reader was given.

Usage:
  check_quotes.py < claims.json

Reads {"context": "<the conversation's text>", "papers": [{"source", "title", "year", "sentence", "quote"}]}, where
`source` is the paper's text as the reader saw it (the trimmed full text, or the abstract), and prints one JSON object
per paper, in order: {"verified": bool, "method": "exact" | "normalized" | "fuzzy" | "per_sentence" | null,
"coverage": float, "problems": [str]}.

A paper is verified when both hold:
- the quote is in the source: as written, else after normalizing (ligatures, hyphen breaks, whitespace, quote marks,
  case), else as a window of the source that covers at least QUOTE_MATCH_THRESHOLD of it, where every difference is a
  PDF artifact (punctuation, a running header, a page number) and every number equals the source number it aligns to;
- every number in the sentence is in the source, its title or year, or the conversation (a sentence may cite the
  researcher's own setup, such as a model size).

Stdlib only.
"""

import difflib
import json
import re
import sys
import unicodedata
from collections import Counter

QUOTE_MATCH_THRESHOLD = 0.90
MIN_QUOTE_WORDS = 8
MAX_QUOTE_WORDS = 80
FURNITURE_REPEATS = 3
FURNITURE_MAX_CHARS = 120

CHAR_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "−": "-", "­": ""})
MEANINGFUL_SYMBOLS = frozenset("%<>=≤≥±")
GAP_POLARITY = re.compile(
    r"\b(?:not|no|never|neither|without|increase[ds]?|decrease[ds]?|reduce[ds]?|raise[ds]?|lower(?:ed|s)?|improve[ds]?|worsen(?:ed|s)?)\b"
    r"|[%<>=≤≥±]"
)
PAGE_GAP = re.compile(r"(?:(?:running\s+header\s+)?page\s+|p\.\s*)\d+(?:\s+of\s+\d+)?")
AUTHOR_HEADER_GAP = re.compile(r".{1,60}?\bet\s+al\.?\s+(?:19|20)\d{2}(?:\s+\d+)?")
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")
CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
NON_PROSE_TOKEN = re.compile(r"`[^`]*`|\S*[a-z]\S*/\S*|\S*/\S*[a-z]\S*")
BARE_NUMBER_LINE = re.compile(r"\n[ \t]*\n[ \t]*\d{1,4}[ \t]*\n[ \t]*\n")


def normalize_text(text):
    """NFKC (ligatures), a word hyphenated across a line joined, one kind of quote mark and dash, single spaces, lower case.

    Only a hyphen between letters is a line-end hyphenation: a dash between numbers, as in a range 0.43-0.61 broken
    over two lines, is kept.
    """
    text = re.sub(r"([^\W\d_])-\s+([^\W\d_])", r"\1\2", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text.translate(CHAR_MAP).lower()).strip()


def without_page_furniture(text):
    """The text without running headers (short lines repeated on many pages) and page numbers alone between blank lines."""
    lines = text.split("\n")
    counts = Counter(line.strip() for line in lines)
    furniture = {line for line, count in counts.items() if count >= FURNITURE_REPEATS and len(line) <= FURNITURE_MAX_CHARS and re.search(r"[A-Za-z]", line)}
    kept = "\n".join(line for line in lines if line.strip() not in furniture)
    return BARE_NUMBER_LINE.sub("\n\n", kept)


def number_spans(text):
    return [(found.start(), found.end(), found.group().replace(",", "")) for found in NUMBER.finditer(text)]


def numbers_in(text):
    return [number for _, _, number in number_spans(text)]


def source_number_at(source, position):
    """The whole source number token spanning `position`, so 18 never matches 118, -18 or the 8 of 0.8."""
    low = max(0, position - 24)
    for found in NUMBER.finditer(source[low : position + 24]):
        if found.start() <= position - low < found.end():
            return found.group().replace(",", "")
    return None


def numbers_grounded(quote, source, locate):
    """Every number in the quote equals the whole source number it aligns to."""
    for start, end, number in number_spans(quote):
        position = locate(start, end)
        if position is None or source_number_at(source, position) != number:
            return False
    return True


def window_locator(quote, window, offset):
    blocks = difflib.SequenceMatcher(None, quote, window, autojunk=False).get_matching_blocks()

    def locate(start, end):
        for block in blocks:
            if block.a <= start and end <= block.a + block.size:
                return offset + block.b + (start - block.a)
        return None

    return locate


def best_window(quote, source):
    """The source window, twice the quote's length, whose matching blocks cover the most of the quote."""
    length = len(quote)
    width = length * 2
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(quote)
    prefilter = 2 * QUOTE_MATCH_THRESHOLD * length / (width + length)
    best = (0.0, source[:width], 0)
    for start in range(0, max(1, len(source) - length + 1), max(1, length // 2)):
        window = source[start : start + width]
        matcher.set_seq1(window)
        if matcher.real_quick_ratio() < prefilter or matcher.quick_ratio() < prefilter:
            continue
        coverage = sum(block.size for block in matcher.get_matching_blocks()) / length
        if coverage > best[0]:
            best = (coverage, window, start)
    return best


def has_meaningful_text(text):
    return any(char.isalnum() or char in MEANINGFUL_SYMBOLS for char in text)


def is_artifact_gap(text):
    """A source-side gap a PDF puts inside a sentence: punctuation, a page marker or an author running header."""
    gap = text.strip(" \t\n.,;:()[]{}-")
    if not has_meaningful_text(gap):
        return True
    if "|" in gap or len(gap) > 180 or GAP_POLARITY.search(gap):
        return False
    return bool(PAGE_GAP.fullmatch(gap) or AUTHOR_HEADER_GAP.fullmatch(gap))


def aligned_span(quote, window, offset):
    blocks = [block for block in difflib.SequenceMatcher(None, quote, window, autojunk=False).get_matching_blocks() if block.size]
    return offset + blocks[0].b, offset + blocks[-1].b + blocks[-1].size


def authentic_alignment(quote, window):
    """No meaningful edit on the quote's side, and nothing but PDF artifacts inserted inside the matched source passage.

    Coverage alone is one-sided: a quote that drops a source "not" or "%" still covers itself fully.
    """
    matcher = difflib.SequenceMatcher(None, quote, window, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return False
    source_start, source_end = blocks[0].b, blocks[-1].b + blocks[-1].size
    for tag, quote_from, quote_to, source_from, source_to in matcher.get_opcodes():
        if tag == "equal" or (tag == "insert" and (source_to <= source_start or source_from >= source_end)):
            continue
        if has_meaningful_text(quote[quote_from:quote_to]):
            return False
        if source_start < source_to and source_from < source_end and not is_artifact_gap(window[source_from:source_to]):
            return False
    return True


def find_quote(quote, source):
    """Where the normalized quote is in the normalized source: exact, a fuzzy window, or sentence by sentence, in order.

    Sentence by sentence accepts a quote whose sentences sit in source order within three times its length, with only
    PDF artifacts between them: room for one caption or header, not for sentences stitched from distant sections.
    """
    at = source.find(quote)
    if at != -1:
        return {"method": "normalized", "coverage": 1.0, "start": at, "end": at + len(quote)}
    coverage, window, start = best_window(quote, source)
    if coverage >= QUOTE_MATCH_THRESHOLD and authentic_alignment(quote, window) and numbers_grounded(quote, source, window_locator(quote, window, start)):
        match_start, match_end = aligned_span(quote, window, start)
        return {"method": "fuzzy", "coverage": coverage, "start": match_start, "end": match_end}
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", quote) if sentence]
    if len(sentences) >= 2 and all(len(sentence) >= 20 for sentence in sentences):
        parts = [find_quote(sentence, source) for sentence in sentences]
        if all(part["method"] is not None for part in parts):
            gaps = [source[before["end"] : after["start"]] for before, after in zip(parts, parts[1:])]
            in_order = all(before["end"] <= after["start"] for before, after in zip(parts, parts[1:]))
            span = parts[-1]["end"] - parts[0]["start"]
            if in_order and span <= 3 * len(quote) and all(is_artifact_gap(gap) for gap in gaps):
                return {"method": "per_sentence", "coverage": min(part["coverage"] for part in parts), "start": parts[0]["start"], "end": parts[-1]["end"]}
    return {"method": None, "coverage": coverage, "start": start, "end": None}


def quote_finding(quote, source):
    """(method, coverage, problem): how the quote was found in the source, or why it was not."""
    if quote.strip() == "":
        return None, 0.0, "no quote was given"
    words = len(quote.split())
    if words < MIN_QUOTE_WORDS:
        return None, 0.0, f"the quote is too short to check ({words} words; give at least {MIN_QUOTE_WORDS})"
    if words > MAX_QUOTE_WORDS:
        return None, 0.0, f"the quote is too long ({words} words; give at most {MAX_QUOTE_WORDS})"
    at = source.find(quote)
    if at != -1 and numbers_grounded(quote, source, lambda start, _end: at + start):
        return "exact", 1.0, None
    found = find_quote(normalize_text(quote), normalize_text(without_page_furniture(source)))
    if found["method"] is None:
        return None, found["coverage"], f"the quote is not in the paper text (the closest passage matches {found['coverage']:.0%} of it, and any changed word or number fails)"
    return found["method"], found["coverage"], None


def prose_numbers(sentence):
    """The numbers a sentence states, leaving out citations such as [2, 5], code spans, paths and links."""
    prose = NON_PROSE_TOKEN.sub(" ", CITATION.sub(" ", normalize_text(sentence)))
    return list(dict.fromkeys(numbers_in(prose)))


def unbacked_numbers(sentence, paper, context):
    known = set(numbers_in(normalize_text(" ".join([paper["source"], paper.get("title") or "", str(paper.get("year") or ""), context]))))
    return [number for number in prose_numbers(sentence) if number not in known]


def check_paper(paper, context):
    method, coverage, problem = quote_finding(paper.get("quote") or "", paper["source"])
    problems = [] if problem is None else [problem]
    missing = unbacked_numbers(paper.get("sentence") or "", paper, context)
    if missing:
        problems.append(f"the number{'s' if len(missing) > 1 else ''} {', '.join(missing)} in the sentence {'are' if len(missing) > 1 else 'is'} not in the paper text")
    return {"verified": not problems, "method": method, "coverage": round(coverage, 3), "problems": problems}


def main():
    claims = json.load(sys.stdin)
    context = claims.get("context") or ""
    json.dump([check_paper(paper, context) for paper in claims["papers"]], sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
