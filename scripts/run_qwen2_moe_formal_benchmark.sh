#!/usr/bin/env bash
# Reproducible RTX 4090 benchmark for Qwen1.5-MoE Q8_0 batch prefill.
#
# Run this script on storm after sourcing the PTX TornadoVM environment:
#   source ~/opt/TornadoVM-5.1-ptx/setvars.sh
#   cd ~/workspace/GPULlama3.java
#   bash scripts/run_qwen2_moe_formal_benchmark.sh
#
# It writes raw logs and a CSV summary under benchmark-results/.  The inference
# metrics exclude model loading, compilation, warmup and the first weight copy.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/mingyi/models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf}"
RESULT_DIR="${RESULT_DIR:-benchmark-results/qwen2-moe-formal-$(date +%Y%m%d-%H%M%S)}"
RUNS="${RUNS:-3}"
GPU_MEMORY="${GPU_MEMORY:-20GB}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT="${PROMPT:-Write exactly 200 words about the history of computing. Do not stop before completing all 200 words.}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
printf 'mode,run,total_tokens,total_seconds,total_tok_s,prefill_tokens,prefill_seconds,prefill_tok_s,decode_tokens,decode_seconds,decode_tok_s,peak_gpu_memory_mib,log_file\n' \
  > "$RESULT_DIR/results.csv"

run_once() {
  local mode="$1"
  local run="$2"
  local batch_size="$3"
  local log_file="$RESULT_DIR/${mode}-run-${run}.log"
  local memory_file="$RESULT_DIR/${mode}-run-${run}-gpu-memory.csv"

  echo "Running $mode measurement $run/$RUNS"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=100 > "$memory_file" &
  local monitor_pid=$!

  set +e
  if [[ "$mode" == "Single" ]]; then
    ./llama-tornado \
      --gpu --ptx \
      --gpu-memory "$GPU_MEMORY" --heap-min 2g --heap-max 8g \
      --model "$MODEL_PATH" --prompt "$PROMPT" \
      --temperature 0 --seed 42 --max-tokens "$MAX_TOKENS" \
      > "$log_file" 2>&1
  else
    ./llama-tornado \
      --gpu --ptx \
      --with-prefill-decode --batch-prefill-size "$batch_size" \
      --gpu-memory "$GPU_MEMORY" --heap-min 2g --heap-max 8g \
      --model "$MODEL_PATH" --prompt "$PROMPT" \
      --temperature 0 --seed 42 --max-tokens "$MAX_TOKENS" \
      > "$log_file" 2>&1
  fi
  local status=$?
  set -e

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "Benchmark failed; see $log_file" >&2
    exit "$status"
  fi

  local total_line prefill_line decode_line peak
  total_line=$(grep -m1 -E '(Total )?achieved tok/s:' "$log_file")
  peak=$(awk 'BEGIN { max = 0 } /^[0-9]+$/ && $1 > max { max = $1 } END { print max }' "$memory_file")

  local total_rate total_tokens total_seconds prefill_rate prefill_tokens prefill_seconds decode_rate decode_tokens decode_seconds
  read -r total_rate total_tokens total_seconds < <(sed -E 's/.*tok\/s: ([0-9.]+)\. Tokens: ([0-9]+), seconds: ([0-9.]+).*/\1 \2 \3/' <<< "$total_line")
  if [[ "$mode" == "Single" ]]; then
    prefill_rate=""
    prefill_tokens=""
    prefill_seconds=""
    decode_rate=""
    decode_tokens=""
    decode_seconds=""
  else
    prefill_line=$(grep -m1 'Prefill achieved tok/s:' "$log_file")
    decode_line=$(grep -m1 'Decode achieved tok/s:' "$log_file")
    read -r prefill_rate prefill_tokens prefill_seconds < <(sed -E 's/.*tok\/s: ([0-9.]+)\. Tokens: ([0-9]+), seconds: ([0-9.]+).*/\1 \2 \3/' <<< "$prefill_line")
    read -r decode_rate decode_tokens decode_seconds < <(sed -E 's/.*tok\/s: ([0-9.]+)\. Tokens: ([0-9]+), seconds: ([0-9.]+).*/\1 \2 \3/' <<< "$decode_line")
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$mode" "$run" "$total_tokens" "$total_seconds" "$total_rate" \
    "$prefill_tokens" "$prefill_seconds" "$prefill_rate" \
    "$decode_tokens" "$decode_seconds" "$decode_rate" "$peak" "$log_file" \
    >> "$RESULT_DIR/results.csv"
}

for mode_and_batch in "Single 0" "B4 4" "B8 8" "B16 16"; do
  read -r mode batch_size <<< "$mode_and_batch"
  echo "Warm-up: $mode"
  run_once "$mode" warmup "$batch_size"
  for run in $(seq 1 "$RUNS"); do
    run_once "$mode" "$run" "$batch_size"
  done
done

python3 scripts/summarize_qwen2_moe_formal_benchmark.py \
  "$RESULT_DIR/results.csv" > "$RESULT_DIR/summary.md"
echo "Finished. Results: $RESULT_DIR/summary.md"
