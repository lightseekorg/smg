#!/usr/bin/env python3
"""Streaming load generator for the no-GPU SMG simulation (realistic mock fleet).

Drives the gateway's OpenAI chat endpoint with an open-loop (Poisson) arrival
process and a workload designed to exercise the routing policies:

- a *shared* system/few-shot prefix prepended to a tunable fraction of requests,
  so prefix caching (and cache-aware routing) can pay off;
- a varied unique body, so prompt length (hence prefill/TTFT) varies.

It measures what those policies actually move: time-to-first-token (TTFT),
inter-token latency (ITL), end-to-end latency, output throughput, and errors —
and prints percentiles plus a machine-readable summary line. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Result:
    ok: bool
    ttft: float | None = None  # seconds to first token
    e2e: float | None = None  # seconds to final token
    itls: list[float] = field(default_factory=list)  # inter-token gaps (s)
    output_tokens: int = 0


def make_prefix_pool(n: int, words: int) -> list[str]:
    """A pool of `n` distinct but internally-fixed prefixes (think: sessions or
    few-shot templates). Each is identical every time it is reused, so it caches;
    distinct prefixes spread across workers, which is where cache-aware routing
    pays off (route each prefix to the worker that holds it) and load-balancing
    does not (it scatters reuse, forcing recompute)."""
    return [" ".join(f"sys{i}_{j}" for j in range(words)) for i in range(max(1, n))]


def build_prompt(
    rng: random.Random, args: argparse.Namespace, prefixes: list[str], idx: int
) -> str:
    """Construct one prompt: a reused shared prefix (cache hit) plus a unique
    body (varied length → varied prefill)."""
    parts: list[str] = []
    if rng.random() < args.shared_prefix_frac:
        parts.append(rng.choice(prefixes))
    body_words = rng.randint(args.body_words_min, args.body_words_max)
    parts.append(f"request {idx} topic {rng.randint(0, 100_000)}")
    parts.append(" ".join(f"w{rng.randint(0, 5000)}" for _ in range(body_words)))
    return " ".join(parts)


def _request_body(endpoint: str, model: str, prompt: str, max_tokens: int) -> bytes:
    """Build the request body for the chat or completions endpoint."""
    if endpoint == "completions":
        payload = {"model": model, "prompt": prompt}
    else:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    payload.update({"stream": True, "max_tokens": max_tokens})
    return json.dumps(payload).encode()


def _chunk_text(endpoint: str, obj: dict) -> str | None:
    """Extract the incremental text from a streamed chunk for either endpoint."""
    choice = obj.get("choices", [{}])[0]
    if endpoint == "completions":
        return choice.get("text") or None
    return choice.get("delta", {}).get("content") or None


def send_streaming(url: str, endpoint: str, model: str, prompt: str, max_tokens: int) -> Result:
    """Send one streaming request, timing first/last token and inter-token gaps."""
    body = _request_body(endpoint, model, prompt, max_tokens)
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    start = time.perf_counter()
    last = start
    res = Result(ok=False)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if _chunk_text(endpoint, obj):
                    now = time.perf_counter()
                    if res.ttft is None:
                        res.ttft = now - start
                    else:
                        res.itls.append(now - last)
                    last = now
                    res.output_tokens += 1
        res.e2e = time.perf_counter() - start
        res.ok = res.output_tokens > 0
        return res
    except (urllib.error.URLError, TimeoutError, OSError):
        return Result(ok=False)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="gateway chat/completions URL")
    ap.add_argument(
        "--endpoint",
        choices=["chat", "completions"],
        default="chat",
        help="request/response format; use 'completions' for tokenizers without a chat template",
    )
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--rps", type=float, default=50.0)
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--shared-prefix-words", type=int, default=200)
    ap.add_argument("--shared-prefix-frac", type=float, default=0.7)
    ap.add_argument(
        "--num-prefixes",
        type=int,
        default=16,
        help="size of the reused-prefix pool (distinct sessions/templates)",
    )
    ap.add_argument("--body-words-min", type=int, default=16)
    ap.add_argument("--body-words-max", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", default="", help="optional path to write JSON summary")
    ap.add_argument("--label", default="", help="label echoed in the summary")
    args = ap.parse_args()
    if args.rps <= 0:
        # A non-positive rate makes the arrival loop spin with no sleep,
        # submitting tasks as fast as possible until the process OOMs.
        ap.error("--rps must be greater than 0")

    rng = random.Random(args.seed)
    prefixes = make_prefix_pool(args.num_prefixes, args.shared_prefix_words)
    results: list[Result] = []
    results_lock = threading.Lock()
    submitted = 0
    deadline = time.monotonic() + args.duration

    def task(prompt: str) -> None:
        r = send_streaming(args.url, args.endpoint, args.model, prompt, args.output_tokens)
        with results_lock:
            results.append(r)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Open-loop Poisson arrivals: exponential inter-arrival gaps at `rps`.
        while time.monotonic() < deadline:
            prompt = build_prompt(rng, args, prefixes, submitted)
            pool.submit(task, prompt)
            submitted += 1
            if args.rps > 0:
                time.sleep(rng.expovariate(args.rps))
        # Drain in-flight requests after the arrival window closes.
    elapsed = time.monotonic() - t0

    ok = [r for r in results if r.ok]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok if r.e2e is not None]
    all_itls = [g for r in ok for g in r.itls]
    total_out = sum(r.output_tokens for r in ok)
    achieved_rps = len(ok) / elapsed if elapsed > 0 else 0.0
    out_tps = total_out / elapsed if elapsed > 0 else 0.0

    summary = {
        "label": args.label,
        "sent": submitted,
        "ok": len(ok),
        "err": len(results) - len(ok),
        "elapsed_s": round(elapsed, 2),
        "achieved_rps": round(achieved_rps, 1),
        "output_tps": round(out_tps, 1),
        "ttft_ms_p50": round(pct(ttfts, 0.50) * 1000, 1),
        "ttft_ms_p90": round(pct(ttfts, 0.90) * 1000, 1),
        "ttft_ms_p99": round(pct(ttfts, 0.99) * 1000, 1),
        "itl_ms_p50": round(pct(all_itls, 0.50) * 1000, 2),
        "itl_ms_p99": round(pct(all_itls, 0.99) * 1000, 2),
        "e2e_ms_p50": round(pct(e2es, 0.50) * 1000, 1),
        "e2e_ms_p99": round(pct(e2es, 0.99) * 1000, 1),
        "ttft_ms_mean": round(statistics.fmean(ttfts) * 1000, 1) if ttfts else 0.0,
    }

    print(
        f"[{args.label or 'load'}] ok={summary['ok']} err={summary['err']} "
        f"rps={summary['achieved_rps']} out_tps={summary['output_tps']} "
        f"TTFT(ms) p50={summary['ttft_ms_p50']} p90={summary['ttft_ms_p90']} "
        f"p99={summary['ttft_ms_p99']} | ITL(ms) p50={summary['itl_ms_p50']} "
        f"p99={summary['itl_ms_p99']} | E2E(ms) p50={summary['e2e_ms_p50']} "
        f"p99={summary['e2e_ms_p99']}"
    )
    print("SUMMARY_JSON " + json.dumps(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
