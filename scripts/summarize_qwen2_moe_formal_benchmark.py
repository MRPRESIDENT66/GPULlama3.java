#!/usr/bin/env python3
"""Turn Qwen2-MoE formal-benchmark CSV rows into a Markdown result table."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = (
    ("Prefill latency", "prefill_seconds", "s"),
    ("Prefill throughput", "prefill_tok_s", "tok/s"),
    ("Decode throughput", "decode_tok_s", "tok/s"),
    ("End-to-end throughput", "total_tok_s", "tok/s"),
    ("Peak GPU memory", "peak_gpu_memory_mib", "MiB"),
)


def mean_and_sd(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def summarize(rows, column, unit):
    values = [float(row[column]) for row in rows if row[column]]
    if not values:
        return "—"
    mean, sd = mean_and_sd(values)
    if unit == "MiB":
        return f"{mean:.0f} {unit}"
    return f"{mean:.2f} ± {sd:.2f} {unit}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} RESULTS.csv")

    groups = defaultdict(list)
    with Path(sys.argv[1]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["run"] != "warmup":
                groups[row["mode"]].append(row)

    order = ("Single", "B4", "B8", "B16")
    print("# Qwen1.5-MoE Q8_0 Formal Batch Benchmark — Generated Summary\n")
    print("The table contains measured runs only; the separate warm-up is excluded.\n")
    print("| Mode | Prefill latency | Prefill throughput | Decode throughput | End-to-end throughput | Peak GPU memory |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode in order:
        rows = groups.get(mode)
        if not rows:
            continue
        formatted = [summarize(rows, column, unit) for _, column, unit in METRICS]
        print(f"| {mode} | " + " | ".join(formatted) + " |")

    print("\n## Per-run data\n")
    print("| Mode | Run | Prefill tokens | Decode tokens | Prefill tok/s | Decode tok/s | Total tok/s | Peak GPU memory |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in order:
        for row in groups.get(mode, []):
            prefill_rate = f"{float(row['prefill_tok_s']):.2f}" if row["prefill_tok_s"] else "—"
            decode_rate = f"{float(row['decode_tok_s']):.2f}" if row["decode_tok_s"] else "—"
            prefill_tokens = row["prefill_tokens"] or "—"
            decode_tokens = row["decode_tokens"] or "—"
            print(
                f"| {mode} | {row['run']} | {prefill_tokens} | {decode_tokens} | "
                f"{prefill_rate} | {decode_rate} | {float(row['total_tok_s']):.2f} | "
                f"{float(row['peak_gpu_memory_mib']):.0f} MiB |"
            )


if __name__ == "__main__":
    main()
