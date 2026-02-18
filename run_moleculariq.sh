#!/bin/bash
# MolecularIQ evaluation runner
# Usage: ./run_moleculariq.sh <config_name> [--limit N] [--dry-run]
#
# Examples:
#   ./run_moleculariq.sh opt-125m --limit 5      # Quick test
#   ./run_moleculariq.sh qwen3-8b                # Full evaluation
#   ./run_moleculariq.sh ether0 --dry-run        # Show command without running

set -e

# Use conda CUDA 13.1 ptxas which supports Blackwell (sm_103a)
export TRITON_PTXAS_PATH=/system/apps/userenv/bartmann/lm-eval/bin/ptxas
export PATH=/system/apps/userenv/bartmann/lm-eval/bin:$PATH
export LD_LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/system/apps/userenv/bartmann/lm-eval/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Add stubs directory for compile-time linking (libcuda.so for JIT compilation)
export LIBRARY_PATH=/system/apps/userenv/bartmann/lm-eval/lib/stubs:$LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <config_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  config_name    Name of config file in configs/ (without .yaml)"
    echo ""
    echo "Options:"
    echo "  --limit N      Limit number of samples (overrides config)"
    echo "  --dry-run      Print command without executing"
    echo "  --output DIR   Output directory (default: ./results)"
    echo "  --help         Show this help message"
    echo ""
    echo "Available configs:"
    ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
}

# Check arguments
if [ $# -lt 1 ] || [ "$1" == "--help" ]; then
    usage
fi

CONFIG_NAME="$1"
shift

CONFIG_FILE="${CONFIG_DIR}/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
    echo ""
    echo "Available configs:"
    ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

# Parse additional arguments
LIMIT_OVERRIDE=""
DRY_RUN="no"
OUTPUT_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT_OVERRIDE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="yes"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

echo -e "${GREEN}Loading config: ${CONFIG_FILE}${NC}"

# Run lm_eval using Python to properly handle all arguments
python3 << EOF
import yaml
import subprocess
import sys
import os
from datetime import datetime

with open("${CONFIG_FILE}", 'r') as f:
    config = yaml.safe_load(f)

# Required fields
model_path = config.get('model_path', '')
backend = config.get('backend', 'vllm')
task = config.get('task', 'moleculariq_pass_at_k')

# Model args (tensor_parallel_size, dtype, gpu_memory_utilization, etc.)
model_args = config.get('model_args', {})

# Generation kwargs - pass through ALL kwargs from config
gen_kwargs = config.get('gen_kwargs', {})

# Optional fields
system_prompt = config.get('system_prompt', '').strip()
limit = config.get('limit', '')
batch_size = config.get('batch_size', 'auto')
apply_chat_template = config.get('apply_chat_template', False)
fewshot_as_multiturn = config.get('fewshot_as_multiturn', False)
chat_template_kwargs = config.get('chat_template_kwargs', {})

# If chat_template_kwargs specified, add to model_args as chat_template_args
if chat_template_kwargs:
    model_args['chat_template_args'] = chat_template_kwargs

# Override limit if provided
limit_override = "${LIMIT_OVERRIDE}"
if limit_override:
    limit = limit_override

output_dir = "${OUTPUT_DIR}"
config_name = "${CONFIG_NAME}"
dry_run = "${DRY_RUN}" == "yes"

# Build model_args as JSON object (required for nested dicts like chat_template_args)
import json
model_args_dict = {'pretrained': model_path}
model_args_dict.update(model_args)
model_args_str = json.dumps(model_args_dict)

# Build command as list (avoids shell escaping issues)
cmd = [
    'lm_eval',
    '--model', backend,
    '--model_args', model_args_str,
    '--tasks', task,
]

# Add gen_kwargs if any are specified
if gen_kwargs:
    # Translate lm-eval param names to vLLM SamplingParams names
    param_translations = {
        'max_gen_toks': 'max_tokens',  # lm-eval -> vLLM
    }
    translated_kwargs = {}
    for k, v in gen_kwargs.items():
        translated_key = param_translations.get(k, k)
        translated_kwargs[translated_key] = v

    gen_kwargs_str = ','.join(f'{k}={v}' for k, v in translated_kwargs.items())
    cmd.extend(['--gen_kwargs', gen_kwargs_str])

if apply_chat_template:
    cmd.append('--apply_chat_template')

if fewshot_as_multiturn:
    cmd.append('--fewshot_as_multiturn')

if system_prompt:
    cmd.extend(['--system_instruction', system_prompt])

if limit:
    cmd.extend(['--limit', str(limit)])

cmd.extend(['--batch_size', str(batch_size)])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'{output_dir}/{task}/{config_name}/{timestamp}'
os.makedirs(output_path, exist_ok=True)
cmd.extend(['--output_path', output_path])
cmd.extend(['--log_samples'])

# Print command for visibility
print()
print('\033[1;33mCommand:\033[0m')
# Show a readable version
readable_parts = []
for i, arg in enumerate(cmd):
    if arg == '--system_instruction':
        readable_parts.append(arg)
        readable_parts.append(f'"<system_prompt ({len(system_prompt)} chars)>"')
    elif i > 0 and cmd[i-1] == '--system_instruction':
        continue  # Skip, already handled
    else:
        readable_parts.append(arg)
print(' '.join(readable_parts))
print()

if dry_run:
    print('\033[1;33mDry run - not executing\033[0m')
    print()
    if system_prompt:
        print('Full system prompt:')
        print('-' * 40)
        print(system_prompt)
        print('-' * 40)
    sys.exit(0)

print('\033[0;32mRunning evaluation...\033[0m')
print()

# Run the command
result = subprocess.run(cmd)
if result.returncode == 0:
    print()
    print(f'\033[0;32mDone! Results saved to: {output_path}\033[0m')
sys.exit(result.returncode)
EOF
