#!/usr/bin/env python3
"""Summarize concatenated TornadoVM profiler JSON objects by MoE task category."""

import argparse
import collections
import json
from pathlib import Path


ATTENTION_TASKS = {
    "attn_rms_reduce",
    "attn_rms_finalize",
    "attn_rms_qkv_projection",
    "fused_qkv_bias",
    "rope_and_kv_cache",
    "attention",
    "attn_output_proj",
}


def load_objects(path: Path):
    """Read TornadoVM's stream of adjacent top-level JSON objects."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    objects = []
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        objects.append(value)
    return objects


def task_category(full_name: str):
    task = full_name.rsplit(".", 1)[-1]
    if task.startswith("routed_expert_gate_up_"):
        return "Routed Gate/Up"
    if task.startswith("routed_expert_down_"):
        return "Routed Down"
    if task.startswith("shared_expert_"):
        return "Shared expert"
    if task.startswith("router_"):
        return "Router + Top-K"
    if task in ATTENTION_TASKS:
        return "Attention"
    if task.startswith("ffn_rms_"):
        return "FFN RMSNorm"
    return "Other"


def collect(objects, warmup_per_graph):
    occurrences = collections.Counter()
    categories = collections.Counter()
    tasks = collections.Counter()
    totals = collections.Counter()
    graph_counts = collections.Counter()
    task_count = 0

    for record in objects:
        graph_name = next(iter(record))
        graph = record[graph_name]
        occurrence = occurrences[graph_name]
        occurrences[graph_name] += 1
        if occurrence < warmup_per_graph:
            continue

        graph_counts[graph_name] += 1
        totals["kernel"] += int(graph.get("TOTAL_KERNEL_TIME", 0))
        totals["copy_in"] += int(graph.get("COPY_IN_TIME", 0))
        totals["task_graph"] += int(graph.get("TOTAL_TASK_GRAPH_TIME", 0))
        totals["copy_bytes"] += int(graph.get("TOTAL_COPY_IN_SIZE_BYTES", 0))

        for task_name, task in graph.items():
            if isinstance(task, dict):
                task_time = int(task.get("TASK_KERNEL_TIME", 0))
                categories[task_category(task_name)] += task_time
                tasks[task_name.rsplit(".", 1)[-1]] += task_time
                task_count += 1

    return occurrences, graph_counts, categories, tasks, totals, task_count


def milliseconds(nanoseconds):
    return nanoseconds / 1_000_000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--warmup-per-graph",
        type=int,
        default=1,
        help="ignore this many initial executions of every TaskGraph (default: 1)",
    )
    args = parser.parse_args()

    objects = load_objects(args.profile)
    occurrences, graphs, categories, tasks, totals, task_count = collect(
        objects, args.warmup_per_graph
    )
    retained_iterations = min(graphs.values(), default=0)
    kernel_time = totals["kernel"]
    residual = totals["task_graph"] - kernel_time - totals["copy_in"]

    print(f"profile objects:              {len(objects)}")
    print(f"TaskGraphs:                   {len(occurrences)}")
    print(f"executions per graph:         {sorted(set(occurrences.values()))}")
    print(f"warmups skipped per graph:    {args.warmup_per_graph}")
    print(f"steady iterations retained:   {retained_iterations}")
    print(f"task executions retained:     {task_count}")
    print()

    print("GPU kernel breakdown")
    for name, time_ns in categories.most_common():
        percentage = 100.0 * time_ns / kernel_time if kernel_time else 0.0
        per_iteration = (
            milliseconds(time_ns) / retained_iterations
            if retained_iterations
            else 0.0
        )
        print(
            f"{name:20s} {milliseconds(time_ns):9.3f} ms total"
            f"  {per_iteration:8.3f} ms/iter  {percentage:6.2f}%"
        )

    print()
    print("Top individual tasks")
    for name, time_ns in tasks.most_common(15):
        percentage = 100.0 * time_ns / kernel_time if kernel_time else 0.0
        per_iteration = (
            milliseconds(time_ns) / retained_iterations
            if retained_iterations
            else 0.0
        )
        print(
            f"{name:38s} {per_iteration:8.3f} ms/iter  {percentage:6.2f}%"
        )

    print()
    print("Runtime-level totals")
    for name, value in (
        ("GPU kernels", kernel_time),
        ("copy-in", totals["copy_in"]),
        ("runtime/profiler residual", residual),
        ("TaskGraph total", totals["task_graph"]),
    ):
        per_iteration = (
            milliseconds(value) / retained_iterations
            if retained_iterations
            else 0.0
        )
        print(
            f"{name:26s} {milliseconds(value):9.3f} ms total"
            f"  {per_iteration:8.3f} ms/iter"
        )

    print()
    print(
        "Note: runtime/profiler residual is an upper bound containing host-side "
        "dispatch, synchronization, event handling, and profiler overhead; it is "
        "not a pure kernel-launch measurement."
    )


if __name__ == "__main__":
    main()
