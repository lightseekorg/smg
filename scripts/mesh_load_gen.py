"""
Load generator for mesh profiling.

Sends chat completion requests with varied prompts to build up the
radix tree in cache-aware routing. Uses diverse prompt prefixes to
create realistic tree branching.

Usage: python3 scripts/mesh_load_gen.py [--rps 200] [--duration 60] [--gateway-ports 30000]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import aiohttp

# Diverse prompt prefixes to create tree branching
TOPICS = [
    "Explain the concept of",
    "Write a Python function that",
    "What is the difference between",
    "How do I implement",
    "Describe the architecture of",
    "Give me a summary of",
    "Can you help me debug",
    "What are the best practices for",
    "Tell me about the history of",
    "Compare and contrast",
]

SUBJECTS = [
    "machine learning",
    "neural networks",
    "transformers",
    "attention mechanisms",
    "gradient descent",
    "backpropagation",
    "convolutional networks",
    "recurrent networks",
    "reinforcement learning",
    "generative models",
    "diffusion models",
    "tokenization",
    "embedding layers",
    "fine-tuning",
    "transfer learning",
    "data augmentation",
    "batch normalization",
    "dropout regularization",
    "learning rate scheduling",
    "distributed training",
    "model parallelism",
    "data parallelism",
    "pipeline parallelism",
    "quantization",
    "pruning",
    "knowledge distillation",
    "mixture of experts",
    "retrieval augmented generation",
    "chain of thought",
    "prompt engineering",
    "RLHF",
    "DPO",
    "constitutional AI",
    "safety alignment",
    "red teaming",
    "kubernetes deployment",
    "docker containers",
    "microservices",
    "load balancing",
    "circuit breakers",
    "rate limiting",
    "health checks",
    "graceful shutdown",
    "cache-aware routing",
    "prefix caching",
    "KV cache management",
    "radix trees",
]

EXTRAS = [
    "in production systems",
    "for large-scale applications",
    "with Python and Rust",
    "using modern best practices",
    "step by step",
    "with code examples",
    "in a distributed environment",
    "for beginners",
    "from scratch",
    "with performance optimization",
]


def make_prompt(pad_to: int = 0) -> str:
    topic = random.choice(TOPICS)
    subject = random.choice(SUBJECTS)
    extra = random.choice(EXTRAS)
    # Add some randomness to avoid exact duplicates
    suffix = f" (variant {random.randint(1, 10000)})"
    prompt = f"{topic} {subject} {extra}{suffix}"
    # Optionally pad to simulate longer prompts (production-like)
    if pad_to > len(prompt):
        # Add realistic-looking filler text
        filler_words = [
            "furthermore",
            "additionally",
            "specifically",
            "considering",
            "implementation",
            "architecture",
            "optimization",
            "performance",
            "distributed",
            "scalability",
            "reliability",
            "deployment",
        ]
        while len(prompt) < pad_to:
            prompt += " " + random.choice(filler_words)
    return prompt[:pad_to] if pad_to > 0 else prompt


_prompt_pad_size = 0


_retry_timeout = 0.0  # seconds; 0 = no retry
_max_retries = 0


async def send_request(
    session: aiohttp.ClientSession, url: str, stats: dict, urls: list | None = None
):
    prompt = make_prompt(pad_to=_prompt_pad_size)
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    retries = 0
    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=_retry_timeout if _retry_timeout > 0 else None)
            start = time.monotonic()
            async with session.post(url, json=payload, timeout=timeout) as resp:
                async for _ in resp.content:
                    pass
                elapsed = time.monotonic() - start
                stats["success"] += 1
                stats["total_latency"] += elapsed
                return
        except TimeoutError:
            if retries >= _max_retries:
                stats["errors"] += 1
                return
            retries += 1
            stats["retries"] += 1
            # Retry on a DIFFERENT gateway (simulates real retry storm)
            if urls:
                url = random.choice(urls)
            continue
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  Request error: {e}")
            return


async def run_load(
    gateway_ports: list[int],
    rps: int,
    duration: int,
):
    urls = [f"http://127.0.0.1:{p}/v1/chat/completions" for p in gateway_ports]

    stats = {"success": 0, "errors": 0, "retries": 0, "total_latency": 0.0}
    interval = 1.0 / rps if rps > 0 else 0.01
    end_time = time.monotonic() + duration

    connector = aiohttp.TCPConnector(limit=500)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = set()
        report_interval = 5
        last_report = time.monotonic()
        last_success = 0

        print(f"Sending ~{rps} req/s across {len(urls)} gateways for {duration}s...")

        while time.monotonic() < end_time:
            # Round-robin across gateways
            url = random.choice(urls)
            task = asyncio.create_task(send_request(session, url, stats, urls=urls))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

            # Periodic report
            now = time.monotonic()
            if now - last_report >= report_interval:
                recent_rps = (stats["success"] - last_success) / (now - last_report)
                avg_latency = (
                    stats["total_latency"] / stats["success"] if stats["success"] > 0 else 0
                )
                print(
                    f"  [{int(now - end_time + duration)}s] "
                    f"rps={recent_rps:.0f} "
                    f"total={stats['success']} "
                    f"errors={stats['errors']} "
                    f"retries={stats['retries']} "
                    f"avg_latency={avg_latency * 1000:.0f}ms"
                )
                last_report = now
                last_success = stats["success"]

            await asyncio.sleep(interval)

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    print(
        f"\nDone. {stats['success']} successful, {stats['errors']} errors, {stats['retries']} retries"
    )
    if stats["success"] > 0:
        print(f"Avg latency: {stats['total_latency'] / stats['success'] * 1000:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Mesh profiling load generator")
    parser.add_argument("--rps", type=int, default=200, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument(
        "--gateway-ports",
        type=str,
        default="30000,30001,30002",
        help="Comma-separated gateway ports",
    )
    parser.add_argument(
        "--prompt-size",
        type=int,
        default=0,
        help="Pad prompts to this many chars (0=no padding, 500=realistic, 2000=large)",
    )
    parser.add_argument(
        "--retry-timeout",
        type=float,
        default=0,
        help="Client timeout in seconds. Timed-out requests retry (simulates retry storm). 0=no timeout.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per request when --retry-timeout is set (default: 3)",
    )
    args = parser.parse_args()

    ports = [int(p) for p in args.gateway_ports.split(",")]
    global _prompt_pad_size, _retry_timeout, _max_retries
    _prompt_pad_size = args.prompt_size
    _retry_timeout = args.retry_timeout
    _max_retries = args.max_retries
    asyncio.run(run_load(ports, args.rps, args.duration))


if __name__ == "__main__":
    main()
