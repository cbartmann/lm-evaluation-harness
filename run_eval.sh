#!/bin/bash

# LM Evaluation Harness runner script
# Disables torch compilation to avoid Blackwell GPU Triton issues

# Default values
GPU="${GPU:-7}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TASKS="${TASKS:-hellaswag}"
#  BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-auto}"
GPU_MEM="${GPU_MEM:-0.8}"

# Use conda CUDA 13.1 ptxas which supports Blackwell (sm_103a)
export TRITON_PTXAS_PATH=/system/apps/userenv/bartmann/lm-eval/bin/ptxas
export PATH=/system/apps/userenv/bartmann/lm-eval/bin:$PATH
export LD_LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Add stubs directory for compile-time linking (libcuda.so for JIT compilation)
export LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib/stubs:$LIBRARY_PATH

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Running lm_eval with:"
echo "  GPU: $GPU"
echo "  Model: $MODEL"
echo "  Tasks: $TASKS"
# echo "  Limit: $LIMIT"
echo "  Batch size: $BATCH_SIZE"
echo ""

lm_eval --model vllm \
    --model_args pretrained="$MODEL",dtype="$DTYPE",gpu_memory_utilization="$GPU_MEM" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    # --limit "$LIMIT" \
    "$@"
