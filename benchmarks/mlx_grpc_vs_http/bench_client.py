"""MLX bench client — mimics vllm bench serve for OpenAI-compatible endpoints.

Replaces genai-bench because:
- vllm bench requires installing the full vllm package (CUDA-only on PyPI).
- genai-bench's traffic-scenario syntax (D(in,out)) doesn't load real
  datasets — only synthetic length distributions.

This client loads real prompt distributions:
  - chat:  ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered) —
           single-turn user prompts from the standard ShareGPT corpus.
           Represents typical chat traffic.
  - agent: vdaita/edit_10k_char — real coding-agent prompts with
           ~10k character (~2.5k token) code context. Represents
           the canonical Mac local-agent workload (Cursor/Continue
           style code editing).

Output JSON shape mirrors the metrics that vllm bench --save-result
produces, so downstream tooling can swap to vllm bench later if mac
support arrives.

Per (scenario, concurrency) cell:
  {
    "scenario": "agent",
    "concurrency": 16,
    "num_prompts_requested": 128,
    "completed_requests": 124,
    "error_count": 4,
    "duration_s": 38.2,
    "request_throughput": 3.24,
    "output_throughput": 832.1,
    "total_input_tokens_est": 312000,
    "total_output_tokens": 31750,
    "ttft_ms": {"mean": ..., "p50": ..., "p95": ..., "p99": ...},
    "tpot_ms": {"mean": ..., "p50": ..., "p95": ..., "p99": ...}
  }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset


def _load_chat_prompts(num_samples: int) -> list[str]:
    """Load ShareGPT prompts — single-turn user messages."""
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    prompts: list[str] = []
    # ShareGPT row: {"id": str, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}
    for row in ds.shuffle(seed=42).select(range(min(num_samples * 4, len(ds)))):
        convs = row.get("conversations") or []
        if not convs or convs[0].get("from") != "human":
            continue
        text = (convs[0].get("value") or "").strip()
        # Skip very-short and pathological entries.
        if len(text) < 20 or len(text) > 8000:
            continue
        prompts.append(text)
        if len(prompts) >= num_samples:
            break
    if not prompts:
        raise RuntimeError("ShareGPT loaded zero usable prompts")
    return prompts


def _load_agent_prompts(num_samples: int) -> list[str]:
    """Load vdaita/edit_10k_char — code-agent prompts with long file context.

    The dataset has ~10k character source files paired with edit
    instructions. We concatenate instruction + code to form a realistic
    local-agent prompt (the kind Cursor/Continue would send).
    """
    ds = load_dataset("vdaita/edit_10k_char", split="train")
    prompts: list[str] = []
    for row in ds.shuffle(seed=42).select(range(min(num_samples * 4, len(ds)))):
        # The schema isn't perfectly consistent across forks; try common keys.
        instruction = row.get("instruction") or row.get("prompt") or ""
        context = (
            row.get("input")
            or row.get("code")
            or row.get("file")
            or row.get("source")
            or ""
        )
        if instruction and context:
            prompt = f"{instruction.strip()}\n\n{context.strip()}"
        elif instruction or context:
            prompt = (instruction or context).strip()
        else:
            continue
        if len(prompt) < 100:
            continue
        prompts.append(prompt)
        if len(prompts) >= num_samples:
            break
    if not prompts:
        raise RuntimeError("vdaita/edit_10k_char loaded zero usable prompts")
    return prompts


SCENARIOS: dict[str, dict[str, Any]] = {
    "chat": {
        "loader": _load_chat_prompts,
        "max_tokens": 256,
    },
    "agent": {
        "loader": _load_agent_prompts,
        "max_tokens": 256,
    },
}


async def _one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": True,
    }
    start = time.monotonic()
    first_token_t: float | None = None
    tokens = 0
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(connect=10, read=600, write=10, pool=10),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_t is None:
                        first_token_t = time.monotonic()
                    tokens += 1
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if first_token_t is None:
        return {"error": "no tokens generated"}
    end = time.monotonic()
    return {
        "ttft_ms": (first_token_t - start) * 1000,
        "total_s": end - start,
        "tokens": tokens,
        "tpot_ms": ((end - first_token_t) / (tokens - 1)) * 1000 if tokens > 1 else None,
    }


async def _run_concurrent(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    num_prompts: int,
) -> tuple[list[dict[str, Any]], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency * 2,
    )

    async with httpx.AsyncClient(http2=True, limits=limits) as client:

        async def worker(prompt: str) -> None:
            async with sem:
                r = await _one_request(client, base_url, model, prompt, max_tokens)
                results.append(r)

        start = time.monotonic()
        tasks = [
            asyncio.create_task(worker(prompts[i % len(prompts)]))
            for i in range(num_prompts)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

    return results, elapsed


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    return {
        "mean": statistics.fmean(xs),
        "p50": xs[len(xs) // 2],
        "p95": xs[min(int(len(xs) * 0.95), len(xs) - 1)],
        "p99": xs[min(int(len(xs) * 0.99), len(xs) - 1)],
    }


def _summarize(
    results: list[dict[str, Any]],
    elapsed: float,
    scenario: str,
    concurrency: int,
    num_prompts: int,
    max_tokens: int,
    label: str,
    model: str,
) -> dict[str, Any]:
    completed = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    ttfts = [r["ttft_ms"] for r in completed]
    tpots = [r["tpot_ms"] for r in completed if r.get("tpot_ms") is not None]
    total_output = sum(r.get("tokens", 0) for r in completed)
    return {
        "label": label,
        "model": model,
        "scenario": scenario,
        "concurrency": concurrency,
        "num_prompts_requested": num_prompts,
        "max_tokens": max_tokens,
        "duration_s": elapsed,
        "completed_requests": len(completed),
        "error_count": len(errors),
        "first_few_errors": [r["error"] for r in errors[:3]],
        "request_throughput": (len(completed) / elapsed) if elapsed else 0.0,
        "output_throughput": (total_output / elapsed) if elapsed else 0.0,
        "total_output_tokens": total_output,
        "ttft_ms": _percentiles(ttfts),
        "tpot_ms": _percentiles(tpots),
    }


async def _main(args: argparse.Namespace) -> int:
    cfg = SCENARIOS[args.scenario]
    print(f"[bench] loading prompts for scenario={args.scenario}", flush=True)
    prompts = cfg["loader"](args.num_prompts)
    print(f"[bench] loaded {len(prompts)} prompts", flush=True)

    print(
        f"[bench] running label={args.label} scenario={args.scenario} "
        f"concurrency={args.concurrency} num_prompts={args.num_prompts}",
        flush=True,
    )
    results, elapsed = await _run_concurrent(
        args.base_url,
        args.model,
        prompts,
        cfg["max_tokens"],
        args.concurrency,
        args.num_prompts,
    )

    summary = _summarize(
        results,
        elapsed,
        args.scenario,
        args.concurrency,
        args.num_prompts,
        cfg["max_tokens"],
        args.label,
        args.model,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"[bench]   completed={summary['completed_requests']} "
        f"errors={summary['error_count']} "
        f"rps={summary['request_throughput']:.2f} "
        f"ttft_p50={summary['ttft_ms'].get('p50', 0):.0f}ms "
        f"tpot_p50={summary['tpot_ms'].get('p50', 0):.0f}ms",
        flush=True,
    )
    # Non-zero exit if every request failed — surfaces server overload.
    if summary["completed_requests"] == 0:
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="MLX OpenAI-compatible bench client")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="http or grpc — used in JSON")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
