#!/usr/bin/env python3
"""Search OpenAlex, arXiv, Semantic Scholar, Hugging Face papers and GitHub for prior art, and fetch the full text of chosen papers.

Usage:
  search_prior_art.py search "<query 1>" ["<query 2>" ...]
  search_prior_art.py fetch --max-chars N < papers.json

`search` runs every query against every source at once and prints one JSON object:
{"queries", "sources": {name: [{"status", "detail"?, "count"?}, ...]}, "sourceMs": {name: ms}, "keyless", "results"}.
Each source gets one status per query. Every request starts at once, except that a source in QUERY_GAP_S starts its
queries that many seconds apart (arXiv asks for one request per 3 s). A request with no answer by its source's deadline
(SOURCE_DEADLINE_S, else SEARCH_DEADLINE_S) is reported as timed out, so a slow source holds the rest only until its own deadline. Results found more than once (the same arXiv id, else the same title) are merged; the
list is ranked by reciprocal rank fusion over every (query, source) list and cut to the best MERGED_CAP. Each result has
title, year, url, abstract ("" when no source gave one), arxivId and pdfUrl (null when unknown), and foundBy:
[{"query": index, "source", "rank"}], every list that returned it.

`fetch` reads a JSON list of {"url", "arxivId", "pdfUrl"} and prints one JSON object per paper, in order:
{"url", "access": "full text" | "abstract only", "reason"?, "text", "charsKept", "truncated", "ms"}. Papers are fetched
FETCH_WORKERS at a time, arXiv papers from arxiv.org/pdf, others from their open-access PDF link; each PDF is turned into
text with pdftotext in a temporary directory that is deleted afterwards. The text keeps what a comparison needs: the
references, appendices and the sections in DROPPED_SECTION are cut, then the longest remaining sections are cut to one
common length, keeping each one's start, until the text fits --max-chars.

Only the query and the paper URLs leave this machine. Stdlib only, plus pdftotext for `fetch`.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor

PER_SOURCE = 8
MERGED_CAP = 30
RRF_K = 60
TIMEOUT_S = 20
SEARCH_DEADLINE_S = 8.0
SOURCE_DEADLINE_S = {"openalex": 20.0}
ABSTRACT_CHARS = 1500
USER_AGENT = "lit-scout/0.3 (prior-art search)"
S2_RETRIES = {"keyed": 4, "keyless": 1}
S2_FIRST_WAIT_S = 1.5
ATOM = "{http://www.w3.org/2005/Atom}"
# OpenAlex fields: Computer Science, Mathematics (statistics), Engineering (robotics, signals), Neuroscience.
OPENALEX_FIELDS = ("fields/17", "fields/26", "fields/22", "fields/28")

PDF_TIMEOUT_S = 60
PDF_MAX_BYTES = 60 * 1024 * 1024
PDFTOTEXT_TIMEOUT_S = 60
FETCH_WORKERS = 4
CONCLUSION_CHARS = 10_000
CUT_MARK = "\n\n[... {} characters cut by lit-scout ...]\n\n"
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.IGNORECASE)
# A section number (1 to 15, with subsections; or roman), then the title on the same line or after a line break.
SECTION_NUMBER = r"(?:(?:1[0-5]|[1-9])(?:\.\d{1,2})*\.?|[IVX]{1,4}\.)"
TOP_SECTION_NUMBER = r"(?:(?:1[0-5]|[1-9])\.?|[IVX]{1,4}\.)"
TITLE_AFTER_NUMBER = r"(?:[ \t]+|[ \t]*\n[ \t]*\n?[ \t]*)"
REFERENCES_HEADING = re.compile(r"^[ \t]*(?:\d+\.?[ \t]*)?(?:references|bibliography|literature cited)[ \t]*$", re.IGNORECASE | re.MULTILINE)
APPENDIX_HEADING = re.compile(
    r"^[ \t]*(?:[A-Z]\.?[ \t]+)?(?:APPENDIX|Appendix|Appendices|Supplementary Material)(?:[ \t]+[A-Z0-9][.0-9]*)?[.:]?(?:[ \t]+[A-Z][a-z][^\n]{0,60})?[ \t]*$",
    re.MULTILINE,
)
CONCLUSION_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(?:conclusions?|concluding remarks|discussion and conclusions?|summary and conclusions?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
DROPPED_SECTION = re.compile(
    rf"^[ \t]*(?:{SECTION_NUMBER}{TITLE_AFTER_NUMBER})?(?:related work|background and related work|prior work|acknowledge?ments?|ethics statement|"
    r"broader impacts?(?: statement)?|impact statement|reproducibility statement|author contributions)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
TOP_SECTION_HEADING = re.compile(rf"^[ \t]*{TOP_SECTION_NUMBER}{TITLE_AFTER_NUMBER}[A-Z][A-Za-z][^\n]{{0,70}}$", re.MULTILINE)
# An appendix subsection such as "B.2 Setup": appendices often carry no "Appendix" heading.
LETTERED_SUBSECTION = re.compile(rf"^[ \t]*[A-Z]\.[1-9]\d?{TITLE_AFTER_NUMBER}[A-Z][A-Za-z][^\n]{{0,70}}$", re.MULTILINE)


class RateLimited(Exception):
    pass


class Skipped(Exception):
    pass


class TimedOut(Exception):
    pass


class NoFullText(Exception):
    """Why a paper is read from its abstract only."""


class Deadline:
    def __init__(self, seconds):
        self.seconds = seconds
        self.end = time.monotonic() + seconds

    def remaining(self):
        """The seconds left; raises TimedOut when none are."""
        left = self.end - time.monotonic()
        if left <= 0:
            raise TimedOut(f"no answer within the {self.seconds:g} s search deadline")
        return left


def request(url, headers=None):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def fetch(url, deadline, headers=None):
    with urllib.request.urlopen(request(url, headers), timeout=min(TIMEOUT_S, deadline.remaining())) as response:
        return response.read().decode("utf-8")


def arxiv_id_in(text):
    """The arXiv id in an arxiv.org URL or an arXiv DOI, else None."""
    if not text:
        return None
    lowered = text.lower()
    for marker in ("arxiv.org/abs/", "arxiv.org/pdf/", "10.48550/arxiv."):
        if marker in lowered:
            found = ARXIV_ID.match(text[lowered.index(marker) + len(marker) :])
            if found:
                return found.group(1).removesuffix(".pdf")
    return None


def result(title, year, url, abstract, arxiv_id=None, pdf_url=None):
    return {
        "title": " ".join((title or "").split()),
        "year": year,
        "url": url or "",
        "abstract": " ".join((abstract or "").split())[:ABSTRACT_CHARS],
        "arxivId": arxiv_id,
        "pdfUrl": pdf_url or None,
    }


def rebuild_abstract(inverted_index):
    if not inverted_index:
        return ""
    placed = [(position, word) for word, positions in inverted_index.items() for position in positions]
    return " ".join(word for _, word in sorted(placed))


def openalex_links(work):
    """The arXiv id and the open-access PDF link OpenAlex gives for a work."""
    best = work.get("best_oa_location") or {}
    oa_url = (work.get("open_access") or {}).get("oa_url")
    candidates = [work.get("doi"), best.get("landing_page_url"), best.get("pdf_url"), oa_url]
    arxiv_id = next((found for found in map(arxiv_id_in, candidates) if found), None)
    return arxiv_id, best.get("pdf_url") or oa_url


def search_openalex(query, deadline):
    """Works in OPENALEX_FIELDS only: `search=` also matches full texts, which brings in off-topic works from other fields."""
    params = {
        "search": query,
        "filter": "primary_topic.field.id:" + "|".join(OPENALEX_FIELDS),
        "per-page": PER_SOURCE,
        "select": "id,doi,title,publication_year,abstract_inverted_index,best_oa_location,open_access",
    }
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    data = json.loads(fetch("https://api.openalex.org/works?" + urllib.parse.urlencode(params), deadline))
    return [
        result(
            work.get("title"),
            work.get("publication_year"),
            work.get("doi") or work.get("id"),
            rebuild_abstract(work.get("abstract_inverted_index")),
            *openalex_links(work),
        )
        for work in data.get("results", [])
    ]


def search_arxiv(query, deadline):
    """Each word searched in every field: a bare `all:` covers only the first word, the rest go to a default field."""
    params = {"search_query": " ".join(f"all:{word}" for word in query.split()), "max_results": PER_SOURCE, "sortBy": "relevance"}
    feed = ElementTree.fromstring(fetch("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params), deadline))
    return [
        result(
            entry.findtext(f"{ATOM}title"),
            int(entry.findtext(f"{ATOM}published", "0")[:4]) or None,
            entry.findtext(f"{ATOM}id"),
            entry.findtext(f"{ATOM}summary"),
            arxiv_id_in(entry.findtext(f"{ATOM}id")),
        )
        for entry in feed.findall(f"{ATOM}entry")
    ]


def search_semantic_scholar(query, deadline):
    """Without a key, requests share one rate pool that is often full: one retry, then the 429 is reported."""
    params = {"query": query, "limit": PER_SOURCE, "fields": "title,year,url,abstract,externalIds,openAccessPdf"}
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    api_key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    retries = S2_RETRIES["keyed" if api_key else "keyless"]
    wait_s = S2_FIRST_WAIT_S
    for attempt in range(retries + 1):
        try:
            data = json.loads(fetch(url, deadline, headers))
            break
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            if attempt == retries or deadline.remaining() <= wait_s:
                raise RateLimited(f"HTTP 429 after {attempt} retries") from error
            time.sleep(wait_s)
            wait_s *= 2
    return [
        result(
            paper.get("title"),
            paper.get("year"),
            paper.get("url"),
            paper.get("abstract"),
            (paper.get("externalIds") or {}).get("ArXiv"),
            (paper.get("openAccessPdf") or {}).get("url"),
        )
        for paper in data.get("data") or []
    ]


def search_huggingface(query, deadline):
    url = "https://huggingface.co/api/papers/search?" + urllib.parse.urlencode({"q": query, "limit": PER_SOURCE})
    return [huggingface_result(entry["paper"]) for entry in json.loads(fetch(url, deadline))]


def huggingface_result(paper):
    """Hugging Face papers are arXiv papers: `id` is the arXiv id."""
    return result(
        paper.get("title"),
        int(paper["publishedAt"][:4]) if paper.get("publishedAt") else None,
        f"https://arxiv.org/abs/{paper['id']}",
        paper.get("summary"),
        paper["id"],
    )


def search_github(query, deadline):
    if shutil.which("gh") is None:
        raise Skipped("gh is not installed")
    fields = "fullName,description,url,createdAt"
    try:
        ran = subprocess.run(
            ["gh", "search", "repos", query, "--limit", str(PER_SOURCE), "--json", fields],
            capture_output=True,
            text=True,
            timeout=deadline.remaining(),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimedOut(f"no answer within the {deadline.seconds:g} s search deadline") from error
    if ran.returncode != 0:
        raise RuntimeError(f"gh exited {ran.returncode}: {ran.stderr.strip()[:200]}")
    return [
        result(repo.get("fullName"), int(repo.get("createdAt", "0")[:4]) or None, repo.get("url"), repo.get("description"))
        for repo in json.loads(ran.stdout)
    ]


SOURCES = {
    "openalex": search_openalex,
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "huggingface": search_huggingface,
    "github": search_github,
}
QUERY_GAP_S = {"arxiv": 3.0}
KEY_VARIABLES = {"openalex": "OPENALEX_API_KEY", "semantic_scholar": "S2_API_KEY"}


def run_source(search, query, deadline):
    try:
        found = search(query, deadline)
        return {"status": "ok", "count": len(found)}, found
    except RateLimited as error:
        return {"status": "rate-limited", "detail": str(error)}, []
    except urllib.error.HTTPError as error:
        if error.code == 429:
            return {"status": "rate-limited", "detail": rate_limit_detail(error)}, []
        return {"status": "error", "detail": f"HTTPError: {error}"[:300]}, []
    except Skipped as error:
        return {"status": "skipped", "detail": str(error)}, []
    except (TimedOut, TimeoutError) as error:
        return timed_out(error, deadline), []
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            return timed_out(error.reason, deadline), []
        return {"status": "error", "detail": f"URLError: {error}"[:300]}, []
    except Exception as error:  # noqa: BLE001 - every failure is reported per source, never dropped
        return {"status": "error", "detail": f"{type(error).__name__}: {error}"[:300]}, []


def timed_out(error, deadline):
    detail = str(error) if isinstance(error, TimedOut) else f"no answer within the {deadline.seconds:g} s search deadline ({error or 'timed out'})"
    return {"status": "timed-out", "detail": detail}


def rate_limit_detail(error):
    """'HTTP 429' with the wait the source asks for, from whichever of its rate-limit headers it sends."""
    waits = [f"{name}: {error.headers[name]}" for name in ("Retry-After", "RateLimit") if error.headers and error.headers.get(name)]
    return f"HTTP 429 ({'; '.join(waits)})" if waits else "HTTP 429"


def timed(work, *args):
    """What `work(*args)` returns, and how long it took in milliseconds."""
    start = time.monotonic()
    value = work(*args)
    return value, round((time.monotonic() - start) * 1000)


def run_jobs(jobs):
    """Runs every (source, query index, query) job in a daemon thread, each after its source's gap; a job still running
    at its source's deadline is reported as timed out and left behind. Each outcome's time counts from the start of the search."""
    deadlines = {name: Deadline(SOURCE_DEADLINE_S.get(name, SEARCH_DEADLINE_S)) for name, _, _ in jobs}
    outcomes = {}
    started = time.monotonic()

    def run(key, name, index, query):
        time.sleep(index * QUERY_GAP_S.get(name, 0.0))
        found = run_source(SOURCES[name], query, deadlines[name])
        outcomes[key] = found, round((time.monotonic() - started) * 1000)

    threads = [threading.Thread(target=run, args=(key, *job), daemon=True) for key, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for (name, _, _), thread in zip(jobs, threads):
        thread.join(max(0.0, deadlines[name].end - time.monotonic()) + 1.0)
    return [outcomes.get(key) or late_outcome(deadlines[name]) for key, (name, _, _) in enumerate(jobs)]


def late_outcome(deadline):
    return (timed_out(TimedOut(f"no answer within the {deadline.seconds:g} s search deadline"), deadline), []), round(deadline.seconds * 1000)


def search(queries):
    jobs = [(name, index, query) for name in SOURCES for index, query in enumerate(queries)]
    outcomes = run_jobs(jobs)
    sources = {name: [] for name in SOURCES}
    source_ms = dict.fromkeys(SOURCES, 0)
    ranked_lists = []
    for (name, query, _), ((status, found), ms) in zip(jobs, outcomes):
        sources[name].append(status)
        source_ms[name] = max(source_ms[name], ms)
        ranked_lists.append((query, name, found))
    keyless = [name for name, variable in KEY_VARIABLES.items() if not os.environ.get(variable)]
    return {"queries": queries, "sources": sources, "sourceMs": source_ms, "keyless": keyless, "results": merged(ranked_lists)}


def identity_keys(item):
    """What makes two hits the same paper: the arXiv id without its version, and the title in letters and digits."""
    keys = ["title:" + "".join(char for char in item["title"].lower() if char.isalnum())]
    if item["arxivId"]:
        keys.insert(0, "arxiv:" + re.sub(r"v\d+$", "", item["arxivId"].lower()))
    return keys


def merged(ranked_lists):
    """Merges [(query index, source, results)] into one list: each paper once, with every list that found it, by fused rank."""
    papers = []
    by_key = {}
    for query, source, found in ranked_lists:
        for rank, item in enumerate((item for item in found if item["title"]), start=1):
            keys = identity_keys(item)
            paper = next((by_key[key] for key in keys if key in by_key), None)
            if paper is None:
                paper = {**item, "foundBy": [], "score": 0.0}
                papers.append(paper)
            elif any(hit["query"] == query and hit["source"] == source for hit in paper["foundBy"]):
                continue
            for field in ("abstract", "arxivId", "pdfUrl", "year"):
                paper[field] = paper[field] or item[field]
            paper["foundBy"].append({"query": query, "source": source, "rank": rank})
            paper["score"] += 1.0 / (RRF_K + rank)
            by_key.update(dict.fromkeys(keys + identity_keys(paper), paper))
    papers.sort(key=lambda paper: -paper["score"])
    return [{key: value for key, value in paper.items() if key != "score"} for paper in papers[:MERGED_CAP]]


def download_pdf(url, folder):
    """Saves the PDF at `url` into `folder`; raises NoFullText when it is no PDF or cannot be had."""
    try:
        with urllib.request.urlopen(request(url, {"Accept": "application/pdf"}), timeout=PDF_TIMEOUT_S) as response:
            body = response.read(PDF_MAX_BYTES + 1)
            content_type = response.headers.get("Content-Type", "unknown type")
    except urllib.error.HTTPError as error:
        raise NoFullText(f"fetch failed: HTTP {error.code} from {url}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise NoFullText(f"fetch failed: {type(error).__name__}: {error} ({url})") from error
    if len(body) > PDF_MAX_BYTES:
        raise NoFullText(f"PDF over {PDF_MAX_BYTES // (1024 * 1024)} MB: {url}")
    if not body.startswith(b"%PDF"):
        raise NoFullText(f"the open-access link is not a PDF ({content_type}): {url}")
    handle, name = tempfile.mkstemp(suffix=".pdf", dir=folder)
    with os.fdopen(handle, "wb") as file:
        file.write(body)
    return name


def pdf_text(path):
    ran = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True, timeout=PDFTOTEXT_TIMEOUT_S, check=False)
    if ran.returncode != 0:
        raise NoFullText(f"pdftotext exited {ran.returncode}: {ran.stderr.decode('utf-8', 'replace').strip()[:200]}")
    text = re.sub(r"\n{3,}", "\n\n", ran.stdout.decode("utf-8", "replace").replace("\f", "\n"))
    if len(text.strip()) < 500:
        raise NoFullText("the PDF holds no extractable text (a scan?)")
    return text.strip()


def heading_after(pattern, text, start):
    """The offset of the first heading matching `pattern` at or after `start`, else None."""
    found = pattern.search(text, start)
    return found.start() if found else None


def without_back_matter(text):
    """The text before the references and the appendix (each searched only past the first part of the paper)."""
    body = text
    for pattern, from_share in ((REFERENCES_HEADING, 0.3), (APPENDIX_HEADING, 0.5), (LETTERED_SUBSECTION, 0.4)):
        cut_at = heading_after(pattern, body, int(len(text) * from_share))
        if cut_at is not None:
            body = body[:cut_at].rstrip()
    return body


def without_dropped_sections(text):
    """Cuts each DROPPED_SECTION up to the next numbered top-level heading; one with no such heading after it is kept."""
    kept = []
    position = 0
    for dropped in DROPPED_SECTION.finditer(text):
        if dropped.start() < position:
            continue
        next_heading = TOP_SECTION_HEADING.search(text, dropped.end())
        if next_heading is None:
            continue
        kept.append(text[position : dropped.start()])
        position = next_heading.start()
    return "".join(kept) + text[position:]


def shortened_sections(text, max_chars):
    """Cuts the longest top-level sections to one common length, keeping each one's start; None with no such headings."""
    starts = [0] + [found.start() for found in TOP_SECTION_HEADING.finditer(text) if found.start() > 0]
    if len(starts) < 3:
        return None
    sections = [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]
    budgets = section_budgets([len(section) for section in sections], max_chars - len(sections) * len(CUT_MARK.format(len(text))))
    return "".join(section if keep >= len(section) else section[:keep] + CUT_MARK.format(len(section) - keep) for section, keep in zip(sections, budgets))


def section_budgets(lengths, room):
    """Each section's share of `room`: the short sections whole, the rest cut to one common length."""
    budgets = [0] * len(lengths)
    left = max(0, room)
    by_length = sorted(range(len(lengths)), key=lambda index: lengths[index])
    for position, index in enumerate(by_length):
        budgets[index] = min(lengths[index], left // (len(lengths) - position))
        left -= budgets[index]
    return budgets


def head_and_conclusion(body, max_chars):
    """The head and the conclusion, the middle cut: for a text with no section headings to shorten by."""
    tail_chars = min(CONCLUSION_CHARS, max_chars // 4)
    conclusion_at = heading_after(CONCLUSION_HEADING, body, len(body) // 2)
    tail_start = conclusion_at if conclusion_at is not None else len(body) - tail_chars
    tail = body[tail_start : tail_start + tail_chars]
    head = body[: min(tail_start, max_chars - len(tail) - len(CUT_MARK.format(len(body))))]
    return head + CUT_MARK.format(len(body) - len(head) - len(tail)) + tail


def comparison_text(text, max_chars):
    """What a comparison with a plan needs, within `max_chars`; and whether anything was cut."""
    body = without_dropped_sections(without_back_matter(text))
    if len(body) > max_chars:
        body = shortened_sections(body, max_chars) or head_and_conclusion(body, max_chars)
    return body, len(body) < len(text)


def full_text(paper, folder):
    """The paper's text from arXiv, else from its open-access PDF; raises NoFullText with the reason."""
    if paper.get("arxivId"):
        return pdf_text(download_pdf(f"https://arxiv.org/pdf/{paper['arxivId']}", folder))
    if paper.get("pdfUrl"):
        return pdf_text(download_pdf(paper["pdfUrl"], folder))
    raise NoFullText("no arXiv id and no open-access link (paywalled or not a paper)")


def fetch_one(paper, folder, max_chars):
    fetched, ms = timed(fetch_text, paper, folder, max_chars)
    return {**fetched, "ms": ms}


def fetch_text(paper, folder, max_chars):
    try:
        text, truncated = comparison_text(full_text(paper, folder), max_chars)
        return {"url": paper["url"], "access": "full text", "text": text, "charsKept": len(text), "truncated": truncated}
    except NoFullText as reason:
        return {"url": paper["url"], "access": "abstract only", "reason": str(reason), "text": "", "charsKept": 0, "truncated": False}


def fetch_papers(papers, max_chars):
    if shutil.which("pdftotext") is None:
        sys.exit("pdftotext is not installed; lit-scout needs it to read papers (macOS: brew install poppler; Debian: apt install poppler-utils)")
    with tempfile.TemporaryDirectory(prefix="lit-scout-") as folder, ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        return list(pool.map(lambda paper: fetch_one(paper, folder, max_chars), papers))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("search").add_argument("queries", nargs="+")
    fetch_command = commands.add_parser("fetch")
    fetch_command.add_argument("--max-chars", type=int, required=True)
    args = parser.parse_args()
    if args.command == "search":
        queries = [query.strip() for query in args.queries]
        if not all(queries):
            parser.error("a query is empty")
        output = search(queries)
    else:
        output = fetch_papers(json.load(sys.stdin), args.max_chars)
    json.dump(output, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
