#!/usr/bin/env python3
"""Benchmark the real, hosted Jev (TypeSafe's System One model) via OpenRouter's decisions endpoint,
for a speed comparison against our local kev-4b/kev-9b (scripts/benchmark.py). Same request shape,
same sample states/questions, same timing methodology - only the transport differs (a network round
trip to OpenRouter/TypeSafe's servers, not a local process).

Note openrouter.ai's regular /api/v1/chat/completions does NOT serve Jev itself - the only chat-style
listing (typesafe/jev-router) is a general-purpose model router that dispatches to arbitrary underlying
LLMs (confirmed: routed to openai/gpt-6-luna in testing), not Jev. The actual Jev decision model is
"typesafe/jev-1.13", served only via the dedicated /api/alpha/decisions endpoint (chat/completions
rejects it with "is a decisions model, use /api/alpha/decisions instead").

This hits a real, metered API - each request costs a small amount (~$0.0000165 seen here); -n 100 is a
fraction of a cent, but it is real spend against OPENROUTER_API_KEY, unlike the free local benchmarks.

Usage: OPENROUTER_API_KEY=... benchmark_openrouter.py [-n 100] [--warmup 5] [--model typesafe/jev-1.13]
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

from benchmark import QUESTIONS, SAMPLE_STATES

URL = "https://openrouter.ai/api/alpha/decisions"


def one_request(api_key, model, i):
    body = json.dumps({
        "state": SAMPLE_STATES[i % len(SAMPLE_STATES)],
        "model": model,
        "questions": QUESTIONS,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    })
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    dt = time.perf_counter() - t0
    usage = data.get("usage", {})
    return dt, usage.get("input_tokens", 0), usage.get("output_tokens", 0), data.get("model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-requests", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model", default="typesafe/jev-1.13")
    ap.add_argument("--json", action="store_true", help="print raw JSON result instead of a table")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    for i in range(args.warmup):
        one_request(api_key, args.model, i)

    latencies_ms, in_toks, out_toks, resolved_model = [], [], [], None
    t_start = time.perf_counter()
    for i in range(args.num_requests):
        dt, itok, otok, resolved_model = one_request(api_key, args.model, i)
        latencies_ms.append(dt * 1000)
        in_toks.append(itok)
        out_toks.append(otok)
    total_wall = time.perf_counter() - t_start

    latencies_ms.sort()
    n = len(latencies_ms)

    def pct(p):
        return latencies_ms[min(n - 1, int(p / 100 * n))]

    result = {
        "model": args.model,
        "resolved_model": resolved_model,
        "requests": n,
        "warmup_requests": args.warmup,
        "total_wall_s": round(total_wall, 3),
        "req_per_s": round(n / total_wall, 3),
        "mean_ms": round(statistics.mean(latencies_ms), 1),
        "p50_ms": round(pct(50), 1),
        "p90_ms": round(pct(90), 1),
        "p99_ms": round(pct(99), 1),
        "min_ms": round(min(latencies_ms), 1),
        "max_ms": round(max(latencies_ms), 1),
        "mean_input_tokens": round(statistics.mean(in_toks), 1),
        "mean_output_tokens": round(statistics.mean(out_toks), 1),
        "output_tokens_per_s": round(sum(out_toks) / total_wall, 1),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
