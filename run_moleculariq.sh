#!/bin/bash
# MolecularIQ evaluation runner
# Usage:
#   ./run_moleculariq.sh <model_path>                    # full eval
#   ./run_moleculariq.sh <model_path> --limit 10         # quick test
#   ./run_moleculariq.sh <model_path> --dry-run          # print command only
#
# Examples:
#   ./run_moleculariq.sh /path/to/qwen3-4b-chemistry-sft-merged
#   ./run_moleculariq.sh Qwen/Qwen3-8B --limit 10 --dry-run

set -e

# Blackwell GPU support (sm_103a) - CUDA 13.1 + Triton
export TRITON_PTXAS_PATH=/system/apps/userenv/bartmann/lm-eval/bin/ptxas
export PATH=/system/apps/userenv/bartmann/lm-eval/bin:$PATH
export LD_LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib/stubs:$LIBRARY_PATH

if [ $# -lt 1 ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 <model_path> [options]"
    echo ""
    echo "Options:"
    echo "  --limit N      Limit number of samples"
    echo "  --tp N         Tensor parallel size (default: 1)"
    echo "  --reasoning X  Reasoning effort: low/medium/high (passed to chat template)"
    echo "  --dry-run      Print command without executing"
    exit 1
fi

MODEL_PATH="$1"
shift

# Defaults
LIMIT=""
TP=1
REASONING=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)     LIMIT="$2"; shift 2 ;;
        --tp)        TP="$2"; shift 2 ;;
        --reasoning) REASONING="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_NAME="$(basename "$MODEL_PATH")"
if [ -n "$REASONING" ]; then
    MODEL_NAME="${MODEL_NAME}-reasoning-${REASONING}"
    # When chat_template_args is needed, --model_args must be full JSON
    MODEL_ARGS='{"pretrained":"'"${MODEL_PATH}"'","dtype":"bfloat16","gpu_memory_utilization":0.9,"tensor_parallel_size":'"${TP}"',"chat_template_args":{"reasoning_effort":"'"${REASONING}"'"}}'
else
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,gpu_memory_utilization=0.9,tensor_parallel_size=${TP}"
fi
OUTPUT_DIR="results/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)"

CMD=(lm_eval
  --model vllm
  --model_args "$MODEL_ARGS"
  --tasks moleculariq_pass_at_k
  --apply_chat_template
  --batch_size auto
  --log_samples
  --output_path "${OUTPUT_DIR}"
)
if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

echo "=== MolecularIQ Evaluation ==="
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "${CMD[@]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] Not executing."
    exit 0
fi

"${CMD[@]}"



#   lm_eval --model vllm --model_args pretrained=Qwen/Qwen3.5-4B,dtype=bfloat16,gpu_memory_utilization=0.9 --tasks moleculariq_pass_at_k --apply_chat_template --batch_size auto --log_samples --output_path results/Qwen3.5-4B --gen_kwargs "temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0"