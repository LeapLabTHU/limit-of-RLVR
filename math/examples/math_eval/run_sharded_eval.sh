#!/bin/bash
# Data-parallel pass@k generation across all visible GPUs, for a single
# HF-format model directory and a single benchmark. Shards the question set
# by --start/--end (one math_eval.py process per GPU); outputs land in the
# same dir and are keyed by question idx, so shards merge automatically.
#
# Usage:
#   ./run_sharded_eval.sh <model_dir> <output_dir> <benchmark> <n_sampling> \
#       [temperature=0.6] [top_p=0.95] [max_tokens=2048] [prompt_type=qwen-boxed] [seed=1] [num_gpus=auto]
set -euo pipefail

MODEL_DIR=$1
OUTPUT_DIR=$2
BENCHMARK=$3
N_SAMPLING=$4
TEMPERATURE=${5:-0.6}
TOP_P=${6:-0.95}
MAX_TOKENS=${7:-2048}
PROMPT_TYPE=${8:-qwen-boxed}
SEED=${9:-1}
NUM_GPUS=${10:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}

DATA_FILE="data/${BENCHMARK}/test.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "No such benchmark data file: $DATA_FILE"
    exit 1
fi
TOTAL=$(wc -l < "$DATA_FILE")

mkdir -p "$OUTPUT_DIR"

echo "Sharding $TOTAL questions from $BENCHMARK across $NUM_GPUS GPUs (n_sampling=$N_SAMPLING each)"

# math_eval.py does `rm -rf $model_name_or_path` right after loading the
# model (see setup() in math_eval.py) — it's written for a scratch/converted
# checkpoint that's meant to be deleted post-eval. Running N shards against
# the same MODEL_DIR would let the first shard to finish delete it out from
# under the rest, and would destroy the real checkpoint. Give each shard its
# own hardlinked copy (no extra disk — same inodes) so each `rm -rf` only
# removes that shard's private copy.
SHARD_ROOT=$(mktemp -d "${OUTPUT_DIR%/}/model_shards.XXXXXX")
trap 'rm -rf "$SHARD_ROOT"' EXIT

per_shard=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
pids=()
for ((g=0; g<NUM_GPUS; g++)); do
    start=$((g * per_shard))
    end=$(((g + 1) * per_shard))
    if [ "$start" -ge "$TOTAL" ]; then
        break
    fi
    if [ "$end" -gt "$TOTAL" ]; then
        end=-1
    fi
    shard_model_dir="$SHARD_ROOT/gpu${g}"
    cp -al "$MODEL_DIR" "$shard_model_dir"
    echo "GPU $g: questions [$start, $end), private model copy at $shard_model_dir"
    CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false python -u math_eval.py \
        --model_name_or_path "$shard_model_dir" \
        --data_name "$BENCHMARK" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample -1 \
        --max_tokens_per_call "$MAX_TOKENS" \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --n_sampling "$N_SAMPLING" \
        --top_p "$TOP_P" \
        --start "$start" \
        --end "$end" \
        --use_vllm \
        --save_outputs \
        > "$OUTPUT_DIR/shard_${g}.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
    echo "One or more shards failed — check $OUTPUT_DIR/shard_*.log"
    exit 1
fi

echo "All shards complete. Outputs in $OUTPUT_DIR/$BENCHMARK/"
