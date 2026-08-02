# Qwen2-MoE Q8_0 GPU Baseline

This document records the working single-token GPU baseline before further
kernel and scheduling optimizations. The baseline is preserved on branch
`qwen2-moe-gpu-baseline`.

## Scope

- Model: `Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf`
- GGUF file size: approximately 15 GB
- Weight format: Q8_0 (32 int8 values and one FP16 scale per block)
- Execution mode: single-token inference
- GPU backend: TornadoVM PTX
- Unsupported in this baseline: sequential prefill/decode and batch prefill/decode

The GPU path includes Qwen2 attention, router projection, softmax and Top-K,
four routed experts, the shared expert, residual accumulation, and final logits.

## Test Environment

- Server: `storm`
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- TornadoVM SDK: 5.1.0, JDK 21, PTX and OpenCL backends installed
- Model path: `/home/mingyi/models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf`
- GPU memory limit: 20 GB

Representative command:

```bash
./llama-tornado \
  --gpu --ptx \
  --gpu-memory 20GB \
  --heap-min 2g --heap-max 8g \
  --model /home/mingyi/models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf \
  --prompt "Hi" \
  --temperature 0 \
  --seed 42 \
  --max-tokens 128
```

## Correctness Results

The repository contains an optional JSONL trace and comparison script that
compare generated token IDs, per-layer router logits, Top-K expert IDs,
routing weights, and final logits between CPU and GPU executions.

When CPU activation quantization was disabled so that the CPU arithmetic more
closely matched the current GPU kernels, the common trace prefix produced:

- 3 compared generated token IDs with no mismatch
- 20 final Top-1 predictions with no mismatch
- 4 Top-K expert-set mismatches out of 495 layer comparisons
- mean absolute router-logit error: 0.001323
- mean absolute final-logit error: 0.01094

The first expert-set mismatch occurred around an almost tied routing decision.
This indicates that the main remaining differences are numerical rather than a
large structural error in the MoE pipeline. Longer correctness traces are still
required before claiming full numerical equivalence.

## Throughput Baseline

Three interleaved 128-token PTX runs measured:

| Run | Throughput |
|---:|---:|
| 1 | 17.75 tokens/s |
| 2 | 17.77 tokens/s |
| 3 | 17.58 tokens/s |
| **Mean** | **17.70 tokens/s** |

A later 19-token smoke test reached 24.63 tokens/s. This short result is kept as
a health check, not as the main baseline, because short generations are more
sensitive to prompt length, warm-up, and measurement variance.

## Kernel Profiling Results

The TornadoVM profiler was run with 16 generated tokens. The first execution of
each TaskGraph was excluded because it includes initialization and initial
weight transfer, leaving 15 steady-state iterations.

| Component | Time per token | Share of GPU kernel time |
|---|---:|---:|
| Attention | 5.222 ms | 28.62% |
| Shared expert | 4.796 ms | 26.29% |
| Routed Gate/Up | 3.539 ms | 19.40% |
| Routed Down | 2.536 ms | 13.90% |
| Router and Top-K | 1.179 ms | 6.46% |
| FFN RMSNorm | 0.536 ms | 2.94% |
| Other kernels | 0.435 ms | 2.39% |

Routed and shared expert computation accounts for approximately 59.6% of GPU
kernel time. The most expensive individual tasks were:

| Task | Time per token | Share of GPU kernel time |
|---|---:|---:|
| Attention kernel | 3.420 ms | 18.75% |
| Four routed Gate/Up kernels | 3.539 ms | 19.40% |
| Four routed Down kernels | 2.536 ms | 13.90% |
| Shared expert Gate/Up | 2.105 ms | 11.54% |
| Shared expert Down | 1.824 ms | 10.00% |
| Shared gate and accumulation | 0.867 ms | 4.75% |
| Router projection | 0.671 ms | 3.68% |
| Softmax and Top-K | 0.508 ms | 2.78% |

Runtime-level profiler totals per token were:

- GPU kernels: 18.244 ms
- copy-in: 4.962 ms
- runtime/profiler residual: 24.118 ms
- total TaskGraph time: 47.323 ms

The residual is only an upper bound. It combines host dispatch,
synchronization, event handling, and profiler overhead, so it must not be
reported as pure kernel-launch time.

## Q8_0 Activation Experiments

Several experimental kernels quantized activations to Q8_0 and reused the
quantized values for integer dot products. These experiments were rolled back
from the working baseline:

| Experiment | Result |
|---|---:|
| Routed Gate/Up only | approximately 2.5% faster |
| Routed and shared Gate/Up | approximately 3.4% faster |
| All tested Q8_0 matrix-vector paths | approximately 3.7% slower |

The full conversion also increased numerical error. The current baseline
therefore keeps FP32 activations and reads Q8_0 weights by applying each block's
FP16 scale during the dot product.

## Next Optimization Target

Profiling shows that Top-K selection is too small to be the first target. The
next branch should investigate expert-task fusion and scheduling overhead:

- reduce the four routed Gate/Up tasks to one task per layer where practical;
- reduce the four routed Down tasks to one task per layer where practical;
- preserve the current correctness trace as the regression oracle;
- compare steady-state throughput and profiler results against 17.70 tokens/s;
- keep any optimization only if it improves performance without unacceptable
  correctness loss.
