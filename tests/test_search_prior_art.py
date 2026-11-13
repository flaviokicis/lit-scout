"""Tests of scripts/search_prior_art.py: the multi-query merge, the search deadline, the sources and the full-text step. Run: python3 -B -m unittest discover -s tests"""

import email.message
import http.server
import importlib.util
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
SCRIPT = TESTS.parent / "scripts" / "search_prior_art.py"
HUGGINGFACE_SAMPLE = TESTS / "huggingface_papers_search.json"
spec = importlib.util.spec_from_file_location("search_prior_art", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class HtmlPage(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html>a landing page, not a PDF</html>")

    def log_message(self, *args):
        pass


def paper(title, arxiv_id=None, abstract=""):
    return script.result(title, 2026, f"https://example.org/{title}", abstract, arxiv_id)


class MultiQuerySearch(unittest.TestCase):
    def search(self, sources, queries, gaps=None):
        with mock.patch.dict(script.SOURCES, sources, clear=True), mock.patch.dict(script.QUERY_GAP_S, gaps or {}, clear=True):
            return script.search(queries)

    def test_a_paper_found_by_several_queries_and_sources_is_one_result_credited_to_each(self):
        def openalex(query, deadline):
            if query == "jargon":
                raise RuntimeError("HTTP 503")
            return [paper("Draft Trees", "2609.22098"), paper(f"Only {query}")]

        def arxiv(query, deadline):
            return [paper("Draft trees!", "2609.22098v2", "the abstract"), paper("only in arxiv")] if query == "jargon" else []

        output = self.search({"openalex": openalex, "arxiv": arxiv}, ["plain", "jargon"])

        top = output["results"][0]
        self.assertEqual((top["title"], top["abstract"]), ("Draft Trees", "the abstract"))
        self.assertEqual(sorted((hit["query"], hit["source"], hit["rank"]) for hit in top["foundBy"]), [(0, "openalex", 1), (1, "arxiv", 1)])
        self.assertEqual(len(output["results"]), 3)
        self.assertEqual(output["sources"]["openalex"], [{"status": "ok", "count": 2}, {"status": "error", "detail": "RuntimeError: HTTP 503"}])

    def test_arxiv_queries_start_the_gap_apart_and_one_past_the_deadline_is_timed_out(self):
        starts = []

        def arxiv(query, deadline):
            starts.append(time.monotonic())
            deadline.remaining()
            return [paper(query)]

        with mock.patch.object(script, "SEARCH_DEADLINE_S", 0.5):
            output = self.search({"arxiv": arxiv}, ["one", "two", "three"], gaps={"arxiv": 0.3})

        self.assertAlmostEqual(starts[1] - starts[0], 0.3, delta=0.1)
        self.assertEqual([status["status"] for status in output["sources"]["arxiv"]], ["ok", "ok", "timed-out"])

    def test_a_source_still_running_at_the_deadline_is_reported_as_timed_out_without_holding_the_rest(self):
        def hangs(query, deadline):
            time.sleep(5)
            return [paper("too late")]

        started = time.monotonic()
        with mock.patch.object(script, "SEARCH_DEADLINE_S", 0.2):
            output = self.search({"slow": hangs, "openalex": lambda query, deadline: [paper("in time")]}, ["q"])

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(output["sources"]["slow"], [{"status": "timed-out", "detail": "no answer within the 0.2 s search deadline"}])
        self.assertEqual([result["title"] for result in output["results"]], ["in time"])

    def test_each_source_gets_its_own_deadline_and_an_early_answer_adds_no_wait(self):
        def late_but_in_time(query, deadline):
            time.sleep(0.5)
            return [paper("slow openalex")]

        def hangs(query, deadline):
            time.sleep(5)
            return []

        with mock.patch.object(script, "SEARCH_DEADLINE_S", 0.2), mock.patch.dict(script.SOURCE_DEADLINE_S, {"openalex": 2.0}, clear=True):
            started = time.monotonic()
            output = self.search({"openalex": late_but_in_time, "other": hangs}, ["q"])
            slow_run = time.monotonic() - started
            started = time.monotonic()
            self.search({"openalex": lambda query, deadline: [paper("fast")], "other": lambda query, deadline: []}, ["q"])
            fast_run = time.monotonic() - started

        self.assertEqual([result["title"] for result in output["results"]], ["slow openalex"])
        self.assertEqual(output["sources"]["other"], [{"status": "timed-out", "detail": "no answer within the 0.2 s search deadline"}])
        self.assertLess(slow_run, 3.0)
        self.assertLess(fast_run, 0.3)

    def test_a_request_past_the_deadline_is_timed_out_not_an_error(self):
        status, found = script.run_source(lambda query, deadline: script.fetch("https://example.org", deadline), "q", script.Deadline(-1))

        self.assertEqual((status["status"], found), ("timed-out", []))

    def test_sources_that_ran_without_an_api_key_are_named(self):
        with mock.patch.dict(script.os.environ, {"OPENALEX_API_KEY": "k"}, clear=True):
            self.assertEqual(self.search({"arxiv": lambda queries, deadline: []}, ["q"])["keyless"], ["semantic_scholar"])

    def test_the_merged_list_is_capped_by_fused_rank(self):
        def many(query, deadline):
            return [paper(f"{query} {rank}") for rank in range(20)]

        output = self.search({"a": many, "b": many}, ["one", "two"])

        self.assertEqual(len(output["results"]), script.MERGED_CAP)
        self.assertEqual({hit["rank"] for result in output["results"] for hit in result["foundBy"]}, set(range(1, 16)))


class HuggingFaceSource(unittest.TestCase):
    def test_a_recorded_response_maps_to_arxiv_papers(self):
        with mock.patch.object(script, "fetch", return_value=HUGGINGFACE_SAMPLE.read_text()) as fetched:
            found = script.search_huggingface("speculative decoding drafter", script.Deadline(5))

        self.assertIn("api/papers/search?q=speculative+decoding+drafter&limit=8", fetched.call_args.args[0])
        self.assertEqual(
            found[1],
            {
                "title": "P-EAGLE: Parallel-Drafting EAGLE with Scalable Training",
                "year": 2026,
                "url": "https://arxiv.org/abs/2602.01469",
                "abstract": json.loads(HUGGINGFACE_SAMPLE.read_text())[1]["paper"]["summary"],
                "arxivId": "2602.01469",
                "pdfUrl": None,
            },
        )

    def test_a_rate_limit_from_any_source_is_reported_with_its_wait(self):
        headers = email.message.Message()
        headers["RateLimit"] = '"search";r=0;t=120'
        refused = urllib.error.HTTPError("https://huggingface.co", 429, "Too Many Requests", headers, None)

        with mock.patch.object(script, "fetch", side_effect=refused):
            status, found = script.run_source(script.search_huggingface, "anything", script.Deadline(5))

        self.assertEqual(found, [])
        self.assertEqual(status, {"status": "rate-limited", "detail": 'HTTP 429 (RateLimit: "search";r=0;t=120)'})

    def test_semantic_scholar_without_a_key_retries_once_then_reports_the_rate_limit(self):
        refused = urllib.error.HTTPError("https://api.semanticscholar.org", 429, "Too Many Requests", email.message.Message(), None)

        with mock.patch.dict(script.os.environ, {}, clear=True), mock.patch.object(script, "fetch", side_effect=refused) as fetched, mock.patch.object(script.time, "sleep"):
            status, _ = script.run_source(script.search_semantic_scholar, "anything", script.Deadline(5))

        self.assertEqual(fetched.call_count, 2)
        self.assertEqual(status, {"status": "rate-limited", "detail": "HTTP 429 after 1 retries"})


class ComparisonText(unittest.TestCase):
    def test_references_and_appendix_go_first(self):
        body = "1 Introduction\n" + "method text " * 400
        text = f"{body}\nReferences\n[1] Someone. 2020.\nAppendix A Proofs\nlong proof"

        kept, truncated = script.comparison_text(text, 60_000)

        self.assertEqual(kept, body.rstrip())
        self.assertTrue(truncated)

    def test_related_work_and_acknowledgements_are_cut_and_lettered_appendix_sections_end_the_text(self):
        text = "\n".join(
            [
                "1\n\nIntroduction\nwe study induction heads.",
                "2\n\nRelated Work\nmany others studied this. " * 1,
                "2.1 Older work\nstill related work.",
                "3\n\nMethod\nwe train transcoders.",
                "4 Results\nthey form at step 1000." + " filler" * 300,
                "Acknowledgements\nthanks to all.",
                "5 Conclusion\nit works.",
                "A.1 Proof details\nlong proof.",
            ]
        )

        kept, truncated = script.comparison_text(text, 60_000)

        self.assertTrue(truncated)
        for wanted in ("we study induction heads.", "we train transcoders.", "they form at step 1000.", "it works."):
            self.assertIn(wanted, kept)
        for unwanted in ("many others", "still related work", "thanks to all", "long proof"):
            self.assertNotIn(unwanted, kept)

    def test_a_long_paper_keeps_the_start_of_every_section_within_the_cap(self):
        sections = [f"{number} {name}\n{name} starts here. " + "x" * 30_000 for number, name in enumerate(["Introduction", "Method", "Results", "Conclusion"], start=1)]

        kept, truncated = script.comparison_text("Abstract: the idea.\n" + "\n".join(sections), 25_000)

        self.assertLessEqual(len(kept), 25_000)
        self.assertTrue(truncated)
        self.assertTrue(kept.startswith("Abstract: the idea."))
        for name in ("Introduction", "Method", "Results", "Conclusion"):
            self.assertIn(f"{name} starts here.", kept)

    def test_a_long_paper_with_no_section_headings_keeps_its_head_and_conclusion_within_the_cap(self):
        text = "Abstract and introduction. " + "x" * 200_000 + "\n5 Conclusion\nWe conclude that it works.\n"

        kept, truncated = script.comparison_text(text, 60_000)

        self.assertLessEqual(len(kept), 60_000)
        self.assertTrue(truncated)
        self.assertTrue(kept.startswith("Abstract and introduction."))
        self.assertIn("We conclude that it works.", kept)
        self.assertIn("characters cut by lit-scout", kept)

    def test_a_short_paper_is_whole(self):
        self.assertEqual(script.comparison_text("short paper", 60_000), ("short paper", False))


class AbstractOnlyFallback(unittest.TestCase):
    def fetch(self, paper):
        with tempfile.TemporaryDirectory() as folder:
            return script.fetch_one({"url": "https://example.org/paper", **paper}, folder, 60_000)

    def test_a_paper_with_no_link_is_abstract_only_with_the_reason(self):
        fetched = self.fetch({"arxivId": None, "pdfUrl": None})

        self.assertEqual((fetched["access"], fetched["charsKept"]), ("abstract only", 0))
        self.assertIn("no open-access link", fetched["reason"])

    def test_an_open_access_link_that_is_no_pdf_is_abstract_only(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), HtmlPage)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            fetched = self.fetch({"arxivId": None, "pdfUrl": f"http://127.0.0.1:{server.server_port}/paper"})
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(fetched["access"], "abstract only")
        self.assertIn("not a PDF (text/html)", fetched["reason"])


if __name__ == "__main__":
    unittest.main()
