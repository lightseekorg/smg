"""Prefix-heavy cache-aware routing benchmark for KV-event-driven routing.

Generates N prefix groups, each a long shared prefix followed by M short unique
suffixes, and replays them through SMG's OpenAI-compatible endpoint while
measuring time-to-first-token (TTFT) and end-to-end latency. Run the same
workload under different SMG --policy settings to compare.

Usage:
  pip install aiohttp
  python scripts/bench_kv_events_cache_hit.py \
    --base-url http://localhost:30000 --model meta-llama/Llama-3.1-8B-Instruct \
    --prefix-groups 16 --suffixes-per-group 16 --concurrency 8 --label event-driven
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


def build_requests(prefix_groups: int, suffixes_per_group: int, prefix_tokens: int):
    """Return a shuffled-by-construction list of (group_id, messages) requests."""
    requests = []
    for g in range(prefix_groups):
        # A long, group-specific shared prefix (~prefix_tokens words).
        shared = f"System context for group {g}: " + " ".join(
            f"fact{g}_{i}" for i in range(prefix_tokens)
        )
        for s in range(suffixes_per_group):
            messages = [
                {"role": "system", "content": shared},
                {"role": "user", "content": f"Question {s}: summarize in one word."},
            ]
            requests.append((g, messages))
    # Interleave groups so consecutive requests don't trivially hit the same worker.
    requests.sort(key=lambda gm: (gm[1][1]["content"], gm[0]))
    return requests


async def one_request(session, base_url, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 8,
        "stream": True,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    ttft = None
    async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            if not line or not line.startswith(b"data:"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - start
            if line.strip() == b"data: [DONE]":
                break
    total = time.perf_counter() - start
    return ttft if ttft is not None else total, total


async def run(args):
    requests = build_requests(args.prefix_groups, args.suffixes_per_group, args.prefix_tokens)
    sem = asyncio.Semaphore(args.concurrency)
    ttfts, totals = [], []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:

        async def worker(messages):
            async with sem:
                ttft, total = await one_request(session, args.base_url, args.model, messages)
                ttfts.append(ttft)
                totals.append(total)

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker(m) for _, m in requests))
        wall = time.perf_counter() - wall_start

    def pct(xs, p):
        return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]

    print(
        json.dumps(
            {
                "label": args.label,
                "requests": len(requests),
                "concurrency": args.concurrency,
                "wall_seconds": round(wall, 3),
                "throughput_rps": round(len(requests) / wall, 2),
                "ttft_p50_ms": round(pct(ttfts, 50) * 1000, 1),
                "ttft_p99_ms": round(pct(ttfts, 99) * 1000, 1),
                "total_p50_ms": round(pct(totals, 50) * 1000, 1),
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-groups", type=int, default=16)
    ap.add_argument("--suffixes-per-group", type=int, default=16)
    ap.add_argument("--prefix-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default="run")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
