#!/usr/bin/env python3
"""Benchmark a running kev.serve instance: fire N /v1/systemone requests sequentially and
report latency (ms/req), throughput (req/s), and token throughput.

Usage: benchmark.py <port> [-n 100] [--warmup 5]

Cycles through a small pool of distinct sample states (not one repeated request) so the
server's prefix cache doesn't make results unrealistically fast.
"""
import argparse
import json
import statistics
import time
import urllib.request

SAMPLE_STATES = [
    "I was charged twice for the same order last week and support has not replied in 3 days. This is unacceptable, fix it now.",
    "Thanks so much for the quick shipping! The product arrived in perfect condition and works great.",
    "My subscription renewed but I cancelled it last month over the phone, please refund me immediately.",
    "The app crashes every time I try to upload a photo larger than 5MB, this has been happening for a week.",
    "Just wanted to say the new update is fantastic, the dashboard loads so much faster now.",
    "I've emailed support three times about my broken order and nobody has responded, I want a manager.",
    "Can you tell me what your return policy is for items purchased more than 30 days ago?",
    "The delivery driver left my package in the rain and it's completely ruined, I need a replacement.",
    "Loving the new feature set, especially the dark mode and keyboard shortcuts.",
    "Why was my account suspended without any warning or explanation? This is affecting my business.",
]

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this about a billing problem?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the customer tone?",
        "criteria": {"calm": None, "frustrated": None, "angry": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "very urgent", "critical"],
    },
}


def one_request(url, model, i):
    body = json.dumps({
        "state": SAMPLE_STATES[i % len(SAMPLE_STATES)],
        "model": model,
        "questions": QUESTIONS,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    dt = time.perf_counter() - t0
    usage = data.get("usage", {})
    return dt, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("-n", "--num-requests", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model", default="kev-latest")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--json", action="store_true", help="print raw JSON result instead of a table")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/systemone"

    for i in range(args.warmup):
        one_request(url, args.model, i)

    latencies_ms, in_toks, out_toks = [], [], []
    t_start = time.perf_counter()
    for i in range(args.num_requests):
        dt, itok, otok = one_request(url, args.model, i)
        latencies_ms.append(dt * 1000)
        in_toks.append(itok)
        out_toks.append(otok)
    total_wall = time.perf_counter() - t_start

    latencies_ms.sort()
    n = len(latencies_ms)

    def pct(p):
        return latencies_ms[min(n - 1, int(p / 100 * n))]

    result = {
        "port": args.port,
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
