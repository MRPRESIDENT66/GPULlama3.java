#!/usr/bin/env python3
"""Compare Qwen2-MoE CPU and GPU JSONL correctness traces."""

import argparse
import json
import math
from pathlib import Path


def load_trace(path: Path):
    records = {"router": {}, "logits": {}, "token": {}}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                print(f"warning: ignoring incomplete final line in {path}")
                break
            raise
        kind = record["type"]
        if kind == "router":
            key = (record["position"], record["layer"])
        elif kind == "logits":
            key = record["position"]
        elif kind == "token":
            key = record["index"]
        else:
            raise ValueError(f"{path}:{line_number}: unknown record type {kind}")
        if key in records[kind]:
            raise ValueError(f"{path}:{line_number}: duplicate {kind} key {key}")
        records[kind][key] = record
    return records


def errors(left, right):
    if len(left) != len(right):
        raise ValueError(f"array lengths differ: {len(left)} != {len(right)}")
    differences = [abs(a - b) for a, b in zip(left, right)]
    if not all(math.isfinite(value) for value in differences):
        return math.inf, math.inf
    return max(differences, default=0.0), sum(differences) / max(1, len(differences))


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def matching_keys(cpu, gpu, kind, common_only):
    cpu_keys = set(cpu[kind])
    gpu_keys = set(gpu[kind])
    if cpu_keys != gpu_keys and not common_only:
        missing_gpu = sorted(cpu_keys - gpu_keys)
        missing_cpu = sorted(gpu_keys - cpu_keys)
        raise ValueError(
            f"{kind} keys differ; missing on GPU={missing_gpu[:10]}, "
            f"missing on CPU={missing_cpu[:10]}"
        )
    return sorted(cpu_keys & gpu_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_trace", type=Path)
    parser.add_argument("gpu_trace", type=Path)
    parser.add_argument(
        "--common-only", action="store_true",
        help="compare only records completed by both traces",
    )
    args = parser.parse_args()

    cpu = load_trace(args.cpu_trace)
    gpu = load_trace(args.gpu_trace)

    token_keys = matching_keys(cpu, gpu, "token", args.common_only)
    token_mismatches = [
        key for key in token_keys
        if cpu["token"][key]["id"] != gpu["token"][key]["id"]
    ]

    router_keys = matching_keys(cpu, gpu, "router", args.common_only)
    expert_order_mismatches = []
    expert_set_mismatches = []
    router_max = router_mean_sum = routing_max = routing_mean_sum = 0.0
    for key in router_keys:
        cpu_record = cpu["router"][key]
        gpu_record = gpu["router"][key]
        if cpu_record["experts"] != gpu_record["experts"]:
            expert_order_mismatches.append(key)
        if set(cpu_record["experts"]) != set(gpu_record["experts"]):
            expert_set_mismatches.append(key)
        maximum, mean = errors(cpu_record["logits"], gpu_record["logits"])
        router_max = max(router_max, maximum)
        router_mean_sum += mean
        maximum, mean = errors(cpu_record["weights"], gpu_record["weights"])
        routing_max = max(routing_max, maximum)
        routing_mean_sum += mean

    logits_keys = matching_keys(cpu, gpu, "logits", args.common_only)
    top1_mismatches = []
    logits_max = logits_mean_sum = 0.0
    for key in logits_keys:
        cpu_values = cpu["logits"][key]["values"]
        gpu_values = gpu["logits"][key]["values"]
        if argmax(cpu_values) != argmax(gpu_values):
            top1_mismatches.append(key)
        maximum, mean = errors(cpu_values, gpu_values)
        logits_max = max(logits_max, maximum)
        logits_mean_sum += mean

    print(f"tokens compared:             {len(token_keys)}")
    print(f"token ID mismatches:         {len(token_mismatches)} {token_mismatches[:10]}")
    print(f"router layers compared:      {len(router_keys)}")
    print(f"Top-K order mismatches:      {len(expert_order_mismatches)} {expert_order_mismatches[:10]}")
    print(f"Top-K set mismatches:        {len(expert_set_mismatches)} {expert_set_mismatches[:10]}")
    print(f"router logits max abs error: {router_max:.8g}")
    print(f"router logits mean abs err:  {router_mean_sum / max(1, len(router_keys)):.8g}")
    print(f"routing weight max abs err:  {routing_max:.8g}")
    print(f"routing weight mean abs err: {routing_mean_sum / max(1, len(router_keys)):.8g}")
    print(f"logit vectors compared:      {len(logits_keys)}")
    print(f"final Top-1 mismatches:      {len(top1_mismatches)} {top1_mismatches[:10]}")
    print(f"final logits max abs error:  {logits_max:.8g}")
    print(f"final logits mean abs error: {logits_mean_sum / max(1, len(logits_keys)):.8g}")

    if expert_set_mismatches:
        first = expert_set_mismatches[0]
        cpu_record = cpu["router"][first]
        gpu_record = gpu["router"][first]
        print(f"first Top-K set mismatch {first}:")
        print(f"  CPU experts={cpu_record['experts']} weights={cpu_record['weights']}")
        print(f"  GPU experts={gpu_record['experts']} weights={gpu_record['weights']}")
        for expert in sorted(set(cpu_record["experts"]) | set(gpu_record["experts"])):
            cpu_logit = cpu_record["logits"][expert]
            gpu_logit = gpu_record["logits"][expert]
            print(
                f"  expert {expert:2d}: CPU raw={cpu_logit: .8f}, "
                f"GPU raw={gpu_logit: .8f}, abs err={abs(cpu_logit - gpu_logit):.3g}"
            )

    passed = not token_mismatches and not expert_set_mismatches and not top1_mismatches
    print("RESULT: " + ("PASS" if passed else "FAIL"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
