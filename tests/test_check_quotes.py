"""Tests of scripts/check_quotes.py: genuine quotes pass through modeled PDF noise, and edited quotes and numbers fail.

The fixture eval follows co-researcher's tests/eval_check_claims.py (MIT, https://github.com/poemswe/co-researcher):
positives are verbatim sentences checked against the same text run through modeled pdftotext noise; negatives are
paraphrases, one-number changes, negation flips and sentences taken from the other paper. Run:
python3 -B -m unittest discover -s tests
"""

import importlib.util
import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_quotes.py"
spec = importlib.util.spec_from_file_location("check_quotes", SCRIPT)
check_quotes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_quotes)

PAPERS = {
    "steering": (
        "We study how the strength of an activation steering vector should be chosen when the steered model is not the "
        "model the vector was fitted on. Across 36 open-weight models, the best coefficient varied by more than a factor "
        "of 4, and no architectural feature predicted it. Tuning the coefficient on a held-out split of 25% of the "
        "prompts recovered 91.5% of the oracle gain, while a fixed coefficient of 1.0 recovered only 62% of it. "
        "Steering did not improve refusal accuracy on Mistral-7B at any strength we tried, and strengths above 8 "
        "degraded fluency on every model. A norm-matched random direction with the same coefficient explained most of "
        "the effect on the smaller models, so matched controls must share the steering gain. We release code that "
        "sweeps 12 strengths per model and reports the full gain curve together with the rate of degenerate answers. "
        "Per-token scaling of the coefficient, learned from a small classifier, raised the toxicity-utility trade-off "
        "but required labeled data from the target model."
    ),
    "dedup": (
        "Exact hashing of normalized sentences misses near-duplicates that differ only by a name, a number or a place. "
        "We find that 13.6% of the examples in a widely used corpus are near-duplicates under a MinHash threshold of "
        "0.8, and that models trained on the deduplicated corpus emit memorized text ten times less often. Suffix "
        "array matching removes repeated substrings of at least 50 tokens and finds 3.04% more duplicates than "
        "MinHash alone. Deduplication did not reduce perplexity on held-out data, but it cut the train-test overlap "
        "that inflates reported accuracy. We recommend running both filters before any synthetic data is mixed in, "
        "because generated examples repeat templates far more often than human-written ones do. Filtering took under "
        "two hours on a single machine with 96 cores for a corpus of 350 billion tokens."
    ),
}

RUNNING_HEADER = "Published as a conference paper at ICLR 2026"


def hyphenate_line_breaks(text, width=70):
    """Wraps the text into lines of about `width` characters, breaking a long word at a line end with a hyphen."""
    lines, line = [], ""
    for word in text.split(" "):
        if len(line) + len(word) + 1 > width and len(word) > 7:
            middle = len(word) // 2
            lines.append(f"{line} {word[:middle]}-".strip())
            line = word[middle:]
        elif len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join([*lines, line])


def ligate(text):
    return text.replace("fi", "ﬁ").replace("fl", "ﬂ")


def add_page_breaks(text, lines_per_page=4):
    """A running header and a page number, alone between blank lines, every few lines, as pdftotext prints page breaks."""
    lines = text.split("\n")
    pages = [lines[start : start + lines_per_page] for start in range(0, len(lines), lines_per_page)]
    return "\n\n".join(f"{RUNNING_HEADER}\n" + "\n".join(page) + f"\n\n{number}" for number, page in enumerate(pages, start=1))


def pdf_noise(text):
    return add_page_breaks(ligate(hyphenate_line_breaks(text)))


def sentences(text):
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.split()) >= check_quotes.MIN_QUOTE_WORDS]


def quote_verified(quote, paper):
    method, _coverage, _problem = check_quotes.quote_finding(quote, pdf_noise(PAPERS[paper]))
    return method is not None


POSITIVES = [(sentence, paper) for paper, text in PAPERS.items() for sentence in sentences(text)]

PARAPHRASES = [
    ("Picking the gain on a quarter of the prompts held out got back most of the best possible improvement.", "steering"),
    ("Strong steering hurt how fluent every model was once the strength went past eight.", "steering"),
    ("Near duplicates that change only a name or a number slip past exact sentence hashing.", "dedup"),
]

ONE_NUMBER_CHANGED = [
    ("Tuning the coefficient on a held-out split of 25% of the prompts recovered 81.5% of the oracle gain", "steering"),
    ("Across 37 open-weight models, the best coefficient varied by more than a factor of 4", "steering"),
    ("We find that 13.6% of the examples in a widely used corpus are near-duplicates under a MinHash threshold of 0.9", "dedup"),
    ("Suffix array matching removes repeated substrings of at least 150 tokens", "dedup"),
]

NEGATION_FLIPPED = [
    ("Steering improved refusal accuracy on Mistral-7B at any strength we tried", "steering"),
    ("Deduplication did reduce perplexity on held-out data, but it cut the train-test overlap", "dedup"),
    ("no architectural feature failed to predict it, and tuning the coefficient on a held-out split", "steering"),
]

FROM_THE_OTHER_PAPER = [(sentences(PAPERS["dedup"])[0], "steering"), (sentences(PAPERS["steering"])[1], "dedup")]

NEGATIVES = PARAPHRASES + ONE_NUMBER_CHANGED + NEGATION_FLIPPED + FROM_THE_OTHER_PAPER


class QuoteSearch(unittest.TestCase):
    def test_the_noise_model_puts_in_every_artifact(self):
        noised = pdf_noise(PAPERS["steering"])
        self.assertIn("-\n", noised)
        self.assertIn("ﬁ", noised)
        self.assertGreaterEqual(noised.count(RUNNING_HEADER), check_quotes.FURNITURE_REPEATS)
        self.assertRegex(noised, r"\n\n3\n\n")

    def test_every_genuine_sentence_is_found_through_pdf_noise(self):
        rejected = [quote for quote, paper in POSITIVES if not quote_verified(quote, paper)]
        self.assertGreaterEqual(len(POSITIVES), 12)
        self.assertEqual(rejected, [], "false rejects of verbatim quotes")

    def test_every_edited_or_borrowed_quote_is_caught(self):
        passed = [quote for quote, paper in NEGATIVES if quote_verified(quote, paper)]
        self.assertEqual(passed, [], "edited quotes that passed")

    def test_a_quote_dropping_a_source_negation_is_caught(self):
        self.assertFalse(quote_verified("Steering did improve refusal accuracy on Mistral-7B at any strength we tried", "steering"))
        self.assertFalse(quote_verified("Deduplication reduced perplexity on held-out data, but it cut the train-test overlap", "dedup"))

    def test_an_author_running_header_inside_a_quote_is_skipped(self):
        source = "the coefficient was tuned on a held-out split Rimsky et al. 2024 7 and reported with the full gain curve"
        method, _coverage, problem = check_quotes.quote_finding("the coefficient was tuned on a held-out split and reported with the full gain curve", source)
        self.assertEqual((method, problem), ("fuzzy", None))

    def test_a_number_range_broken_over_a_line_is_found(self):
        source = "for Indonesian, our prompt scores 0.72\u20130.87 against 0.43\u2013\n0.61 for the baselines, indicating that the word candidates broaden"
        quote = "our prompt scores 0.72\u20130.87 against 0.43\u20130.61 for the baselines, indicating that the word candidates broaden"
        self.assertEqual(check_quotes.quote_finding(quote, source)[2], None)

    def test_too_short_and_missing_quotes_say_why(self):
        self.assertIn("too short", check_quotes.quote_finding("tuned on a held-out split", PAPERS["steering"])[2])
        self.assertEqual(check_quotes.quote_finding("", PAPERS["steering"])[2], "no quote was given")


class SentenceNumbers(unittest.TestCase):
    QUOTE = "Tuning the coefficient on a held-out split of 25% of the prompts recovered 91.5% of the oracle gain"

    def check(self, sentence, context=""):
        paper = {"source": pdf_noise(PAPERS["steering"]), "title": "Steering Strength Across Models", "year": 2025, "sentence": sentence, "quote": self.QUOTE}
        return check_quotes.check_paper(paper, context)

    def test_a_sentence_whose_numbers_are_in_the_paper_or_the_conversation_is_verified(self):
        result = self.check("Like your plan [2, 5], they tune the gain on 25% held out and keep 91.5% of the oracle gain (2025); your Qwen3-14B run can do the same.", "We transfer to Qwen3-14B.")
        self.assertEqual(result, {"verified": True, "method": "normalized", "coverage": 1.0, "problems": []})

    def test_a_sentence_with_one_number_changed_is_caught(self):
        result = self.check("They tune the gain on 25% held out and keep 92.5% of the oracle gain.")
        self.assertFalse(result["verified"])
        self.assertEqual(result["problems"], ["the number 92.5 in the sentence is not in the paper text"])

    def test_both_findings_are_reported(self):
        paper = {"source": PAPERS["dedup"], "title": "", "year": None, "sentence": "They drop 14% of the corpus.", "quote": "We find that 14% of the examples in a widely used corpus are near-duplicates"}
        problems = check_quotes.check_paper(paper, "")["problems"]
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("the quote is not in the paper text"))


if __name__ == "__main__":
    unittest.main()
