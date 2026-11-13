#!/usr/bin/env python3
"""Eval of the lit-scout trigger: Haiku labels each prompt LIT or OTHER, exactly as the mod asks it.

The prompt, the context builder, the reply checks and the request settings (model, effort, max tokens)
are the mod's own code: the runner loads hooks/prompts.ts with bun and calls its exports, so the eval
sends what the mod sends. An item may list `checked`: the ideas the session already checked, which the
mod passes to the classifier. Needs ANTHROPIC_API_KEY.

  python3 eval/run_eval.py --dry-run               # no model calls: renders and parses 3 items
  python3 eval/run_eval.py --limit 10              # smoke run against the model
  python3 eval/run_eval.py --split holdout_v2      # one split only
  python3 eval/run_eval.py --model claude-sonnet-5-5  # another model, same prompt, effort and max tokens
  python3 eval/run_eval.py                         # full run, metrics per split
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
MOD_DIR = EVAL_DIR.parent
PROMPTS_TS = MOD_DIR / "hooks" / "prompts.ts"
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
# The mod names the model by its Claude Code alias; the API needs the model id behind it.
MODEL_IDS = {"haiku": "claude-haiku-5-5"}
# USD per million tokens (Claude Haiku 5.5: prompts up to 100K tokens).
PRICES_PER_MTOK = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
}
MAX_RETRIES = 8
RETRY_STATUSES = {429, 500, 502, 503, 529}
DEFAULT_WORKERS = 64

# Runs inside bun. Reads {"op", "items"} on stdin and answers with one JSON result per item.
TS_BRIDGE = r"""
const { op, modulePath, items } = JSON.parse(await Bun.stdin.text())
const mod = await import(modulePath)
const missing = [
  ...['classifierSystem', 'classifierInput', 'parseClassifier', 'replyFault'].filter(name => typeof mod[name] !== 'function'),
  ...(typeof mod.CLASSIFIER_CALL === 'object' ? [] : ['CLASSIFIER_CALL']),
]
if (missing.length > 0) {
  console.error(`prompts module lacks the exports: ${missing.join(', ')}`)
  process.exit(2)
}
// The same order as the mod's classify(): an unusable reply first, then the parser.
const interpret = ({ text, outputTokens }) => {
  const fault = mod.replyFault(text, outputTokens, mod.CLASSIFIER_CALL.maxTokens)
  if (fault !== null) return { label: 'ERROR', error: fault }
  try { return mod.parseClassifier(text) } catch (error) { return { label: 'ERROR', error: String(error?.message ?? error) } }
}
const ops = {
  call: () => mod.CLASSIFIER_CALL,
  render: item => ({ system: mod.classifierSystem(''), user: mod.classifierInput(item.context, item.prompt, item.checked) }),
  interpret,
}
process.stdout.write(JSON.stringify(items.map(ops[op])))
"""


def main() -> None:
    args = parse_args()
    items = load_dataset(args.dataset, args.split, 3 if args.dry_run else args.limit)
    bridge = TsBridge(PROMPTS_TS)

    if args.dry_run:
        dry_run(bridge, items)
        return

    classifier = Classifier(bridge, ApiCaller.from_environment(), args.model)
    print(f"{len(items)} items, request: {classifier.request_settings()}, workers: {args.workers}", file=sys.stderr)
    started = utc_stamp()
    records = classifier.run(items, args.workers)
    metrics = compute_metrics(records, classifier.model_id)
    out_dir = write_results(started, records, metrics, args, classifier)
    print_table(metrics)
    print(f"\nresults: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default=None, help="evaluate only this split (v1, holdout_v2, postcutoff_v3, longmixed_v4, followup_tune, followup_v5); default all")
    parser.add_argument("--model", default=None, choices=sorted(PRICES_PER_MTOK),
                        help="API model id to send instead of the mod's model; effort and max tokens stay the mod's")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel API calls; 429s back off")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items (smoke run)")
    parser.add_argument("--dry-run", action="store_true", help="render and parse 3 items; call no model")
    return parser.parse_args()


def load_dataset(path: Path, split: str | None, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if split is not None:
        items = [item for item in items if item["split"] == split]
        if not items:
            raise SystemExit(f"no items in split {split!r}")
    return items if limit is None else items[:limit]


class TsBridge:
    """Calls the exports of the mod's prompts module through bun."""

    def __init__(self, prompts_ts: Path) -> None:
        if not prompts_ts.is_file():
            raise SystemExit(f"prompts module not found: {prompts_ts}")
        self.prompts_ts = prompts_ts.resolve()

    def call(self, op: str, items: list) -> list:
        payload = {"op": op, "modulePath": str(self.prompts_ts), "items": items}
        done = subprocess.run(
            ["bun", "-e", TS_BRIDGE], input=json.dumps(payload), capture_output=True, text=True, cwd=EVAL_DIR
        )
        if done.returncode != 0:
            raise RuntimeError(f"bun {op} failed ({done.returncode}): {done.stderr.strip()}")
        return json.loads(done.stdout)

    def classifier_call(self) -> dict:
        return self.call("call", [None])[0]

    def render(self, items: list[dict]) -> list[dict]:
        return self.call("render", [{"context": item["context"], "prompt": item["prompt"], "checked": item.get("checked", [])}
                                    for item in items])

    def interpret(self, replies: list[dict]) -> list[dict]:
        return self.call("interpret", replies)


class ApiCaller:
    """The Messages API, with exponential backoff on 429 and overload."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_environment(cls) -> "ApiCaller":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        return cls(key)

    def complete(self, body: dict) -> dict:
        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        for attempt in range(MAX_RETRIES + 1):
            request = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}")
                time.sleep(backoff_seconds(attempt, error.headers.get("retry-after")))
        raise AssertionError("unreachable")


def backoff_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None and retry_after.replace(".", "", 1).isdigit():
        return float(retry_after) + random.uniform(0, 1)
    return min(60.0, 2.0 ** attempt) + random.uniform(0, 1)


class Classifier:
    """Sends the mod's classifier request for each item and reads the reply as the mod does."""

    def __init__(self, bridge: TsBridge, caller: ApiCaller, model_override: str | None = None) -> None:
        self.bridge = bridge
        self.caller = caller
        self.call = bridge.classifier_call()
        if self.call["model"] not in MODEL_IDS:
            raise SystemExit(f"no API model id known for the mod's model alias {self.call['model']!r}")
        self.model_id = model_override or MODEL_IDS[self.call["model"]]

    def request_settings(self) -> dict:
        return {"model": self.model_id, "effort": self.call["effort"], "max_tokens": self.call["maxTokens"]}

    def run(self, items: list[dict], workers: int) -> list[dict]:
        """One record per item: the item, what was sent, and the outcome {label, query, raw, usage, error}."""
        rendered = self.bridge.render(items)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            replies = list(pool.map(self._send, rendered))
        interpreted = self.bridge.interpret(
            [{"text": reply["raw"] or "", "outputTokens": reply["usage"]["output_tokens"]} for reply in replies if reply["error"] is None]
        )
        answered = iter(interpreted)
        records = []
        for item, render, reply in zip(items, rendered, replies):
            outcome = {"label": "ERROR", "query": None, **reply}
            if reply["error"] is None:
                verdict = next(answered)
                outcome.update(label=verdict["label"], query=verdict.get("query"), error=verdict.get("error"))
            records.append({"item": item, "rendered": render, "haiku": outcome})
        return records

    def _send(self, render: dict) -> dict:
        body = {
            "model": self.model_id,
            "max_tokens": self.call["maxTokens"],
            "output_config": {"effort": self.call["effort"]},
            "system": render["system"],
            "messages": [{"role": "user", "content": render["user"]}],
        }
        try:
            reply = self.caller.complete(body)
        except Exception as error:
            print(f"call failed: {error}", file=sys.stderr)
            return {"raw": None, "usage": zero_usage(), "error": f"call failed: {error}"}
        text = "".join(block.get("text", "") for block in reply["content"] if block["type"] == "text")
        return {"raw": text, "usage": usage_of(reply), "error": None}


def classify_item(item: dict) -> dict:
    """Runs Haiku on one dataset item exactly as the mod does: {label, query, raw, usage, error}."""
    record = Classifier(TsBridge(PROMPTS_TS), ApiCaller.from_environment()).run([item], workers=1)[0]
    return record["haiku"]


def zero_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def usage_of(reply: dict) -> dict:
    usage = reply.get("usage", {})
    return {key: usage.get(key) or 0 for key in zero_usage()}


def compute_metrics(records: list[dict], model_id: str) -> dict:
    groupings = {
        "overall": lambda item: "all",
        "split": lambda item: item["split"],
        "domain": lambda item: item["domain"],
        "case_type": lambda item: item["case_type"],
    }
    metrics = {}
    for grouping, key_of in groupings.items():
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(key_of(record["item"]), []).append(record)
        metrics[grouping] = {key: group_metrics(groups[key]) for key in sorted(groups)}
    errors = [record["haiku"]["error"] for record in records if record["haiku"]["label"] == "ERROR"]
    metrics["errors"] = {
        "total": len(errors),
        "empty_reply": sum(1 for error in errors if error == "the reply was empty"),
        "token_cap": sum(1 for error in errors if "token cap" in error),
    }
    metrics["usage"] = usage_totals(records, model_id)
    return metrics


def group_metrics(records: list[dict]) -> dict:
    gold = [record["item"]["label"] == "LIT" for record in records]
    flags = [record["haiku"]["label"] == "LIT" for record in records]
    # Key name kept for eval/ui.py, which reads it from every run.
    return {"prompts": len(records), "gold_lit": sum(gold), "haiku_alone": scores(gold, flags)}


def scores(gold: list[bool], flags: list[bool]) -> dict:
    tp = sum(g and f for g, f in zip(gold, flags))
    fp = sum(f and not g for g, f in zip(gold, flags))
    fn = sum(g and not f for g, f in zip(gold, flags))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None if precision is None or recall is None else 2 * tp / (2 * tp + fp + fn)
    flagged_pct = 100 * sum(flags) / len(flags) if flags else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": len(gold) - tp - fp - fn,
            "precision": precision, "recall": recall, "f1": f1, "flagged_pct": flagged_pct}


def usage_totals(records: list[dict], model_id: str) -> dict:
    totals = zero_usage()
    for record in records:
        for key in totals:
            totals[key] += record["haiku"]["usage"][key]
    price = PRICES_PER_MTOK[model_id]
    cost = (totals["input_tokens"] * price["input"] + totals["output_tokens"] * price["output"]
            + totals["cache_creation_input_tokens"] * price["cache_write"]
            + totals["cache_read_input_tokens"] * price["cache_read"]) / 1e6
    return {"calls": len(records), **totals, "cost_usd": round(cost, 6)}


def write_results(started: str, records: list[dict], metrics: dict, args: argparse.Namespace, classifier: Classifier) -> Path:
    out_dir = RESULTS_DIR / (started if args.split is None else f"{started}-{args.split}")
    out_dir.mkdir(parents=True)
    with (out_dir / "items.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    config = {
        "started_utc": started,
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": git_commit(),
        "request": classifier.request_settings(),
        "workers": args.workers,
        "split": args.split,
        "limit": args.limit,
        "items": len(records),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset.read_bytes()),
        "prompts_ts_sha256": sha256(PROMPTS_TS.read_bytes()),
        "classifier_system_sha256": sha256(records[0]["rendered"]["system"].encode()) if records else None,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


def git_commit() -> str | None:
    done = subprocess.run(["git", "-C", str(MOD_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def print_table(metrics: dict) -> None:
    headers = ["Group", "Prompts", "Gold LIT prompts", "Precision (LIT)", "Recall (LIT)", "F1 (LIT)",
               "Prompts flagged as LIT (%)"]
    rows = []
    for grouping in ("overall", "split", "domain", "case_type"):
        for key, group in metrics[grouping].items():
            score = group["haiku_alone"]
            rows.append([f"{grouping}: {key}" if grouping != "overall" else "overall", str(group["prompts"]),
                         str(group["gold_lit"]), fmt(score["precision"]), fmt(score["recall"]), fmt(score["f1"]),
                         fmt(score["flagged_pct"], digits=1)])
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, ["-" * width for width in widths], *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    print(f"\nHaiku calls that ended in error (counted as not flagged, as the mod logs them): {metrics['errors']}")
    print(f"Token use and cost: {metrics['usage']}")


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def dry_run(bridge: TsBridge, items: list[dict]) -> None:
    """Renders 3 items and interprets canned replies, with no model call and no file written."""
    call = bridge.classifier_call()
    assert call["model"] in MODEL_IDS, call
    rendered = bridge.render(items)
    for item, render in zip(items, rendered):
        user = render["user"]
        assert "RECENT TURNS, oldest first:" in user and item["prompt"].strip()[:40] in user, item["id"]
        print(f"--- {item['id']} ({item['label']}, {item['case_type']}, {item['split']}): context {len(user)} chars, "
              f"{len(item['context'])} prior turns")
        print(user[:600] + ("\n[...]" if len(user) > 600 else ""))

    cap = call["maxTokens"]
    cases = [{"text": "LIT: contrastive pairs instead of reward model", "outputTokens": 12},
             {"text": "OTHER", "outputTokens": 3}, {"text": "maybe", "outputTokens": 3},
             {"text": "", "outputTokens": 900}, {"text": "LIT: cut", "outputTokens": cap}]
    got = bridge.interpret(cases)
    assert got[0] == {"label": "LIT", "query": "contrastive pairs instead of reward model"}, got[0]
    assert got[1] == {"label": "OTHER"} and got[2]["label"] == "ERROR", got[1:3]
    assert got[3] == {"label": "ERROR", "error": "the reply was empty"}, got[3]
    assert got[4]["label"] == "ERROR" and "token cap" in got[4]["error"], got[4]
    print(f"\nrequest {call}; system prompt {len(rendered[0]['system'])} chars; reply checks OK; dry run passed")


if __name__ == "__main__":
    main()
