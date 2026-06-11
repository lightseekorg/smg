#!/usr/bin/env python3
"""Minimal stdlib load generator for the SMG scale-test rig.

Sends chat-completion requests at a target rate with varied prompts (so
cache-aware routing branches its radix tree), and reports achieved RPS plus
latency percentiles. No third-party deps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


def send_one(url: str, model: str, prompt: str) -> float | None:
    """Send one request; return latency in seconds, or None on error."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 8,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return time.perf_counter() - start
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--rps", type=int, default=200)
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=0)
    args = ap.parse_args()

    concurrency = args.concurrency or min(512, max(8, args.rps))
    deadline = time.monotonic() + args.duration
    interval = 1.0 / args.rps if args.rps > 0 else 0.0

    lat: list[float] = []
    sent = ok = 0
    lock = threading.Lock()
    next_slot = time.monotonic()
    slot_lock = threading.Lock()

    def worker(wid: int) -> None:
        nonlocal sent, ok, next_slot
        i = 0
        while time.monotonic() < deadline:
            # Global pacing to approximate the target RPS.
            if interval:
                with slot_lock:
                    now = time.monotonic()
                    if next_slot < now:
                        next_slot = now
                    wait = next_slot - now
                    next_slot += interval
                if wait > 0:
                    time.sleep(wait)
            prompt = f"worker {wid} request {i}: summarize topic {(wid * 7 + i) % 997}"
            i += 1
            dt = send_one(args.url, args.model, prompt)
            with lock:
                sent += 1
                if dt is not None:
                    ok += 1
                    lat.append(dt)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    print(f"load: sent={sent} ok={ok} err={sent - ok} elapsed={elapsed:.1f}s")
    if lat:
        lat.sort()
        p50 = statistics.median(lat)
        p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
        print(
            f"load: achieved_rps={ok / elapsed:.0f} "
            f"p50={p50 * 1000:.0f}ms p99={p99 * 1000:.0f}ms"
        )


if __name__ == "__main__":
    main()
