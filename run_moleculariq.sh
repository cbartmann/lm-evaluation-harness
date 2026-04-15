#!/bin/bash
# MolecularIQ evaluation runner — reads a YAML config and launches lm_eval.
#
# Usage:
#   ./run_moleculariq.sh <config_name> [--limit N] [--dry-run] [--output DIR]
#
# Examples:
#   ./run_moleculariq.sh qwen3-0.6b --limit 5     # Quick test
#   ./run_moleculariq.sh qwen3-8b                  # Full evaluation
#   ./run_moleculariq.sh ether0 --dry-run           # Show command without running
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

if [ $# -lt 1 ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 <config_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  config_name    Name of config file in configs/ (without .yaml)"
    echo ""
    echo "Options:"
    echo "  --limit N      Limit number of samples"
    echo "  --dry-run      Print command without executing"
    echo "  --output DIR   Output directory (default: ./results)"
    echo ""
    echo "Available configs:"
    ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

CONFIG_NAME="$1"
shift

CONFIG_FILE="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: ${CONFIG_FILE}"
    echo ""
    echo "Available configs:"
    ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

LIMIT_OVERRIDE=""
DRY_RUN="no"
OUTPUT_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)   LIMIT_OVERRIDE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="yes"; shift ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

python3 - "${CONFIG_FILE}" "${CONFIG_NAME}" "${LIMIT_OVERRIDE}" "${DRY_RUN}" "${OUTPUT_DIR}" << 'PYTHON'
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml

config_file, config_name, limit_override, dry_run_flag, output_dir = sys.argv[1:6]
dry_run = dry_run_flag == "yes"

with open(config_file) as f:
    config = yaml.safe_load(f)

model_path = config.get("model_path", "")
backend = config.get("backend", "vllm")
task = config.get("task", "moleculariq_pass_at_k")
model_args = config.get("model_args", {})
gen_kwargs = config.get("gen_kwargs", {})
system_prompt = config.get("system_prompt", "").strip()
limit = limit_override or config.get("limit", "")
batch_size = config.get("batch_size", "auto")
apply_chat_template = config.get("apply_chat_template", False)
chat_template_kwargs = config.get("chat_template_kwargs", {})

if chat_template_kwargs:
    model_args["chat_template_args"] = chat_template_kwargs

model_args_dict = {"pretrained": model_path, **model_args}

cmd = ["lm_eval", "--model", backend, "--model_args", json.dumps(model_args_dict), "--tasks", task]

if gen_kwargs:
    cmd.extend(["--gen_kwargs", ",".join(f"{k}={v}" for k, v in gen_kwargs.items())])

if apply_chat_template:
    cmd.append("--apply_chat_template")

if system_prompt:
    cmd.extend(["--system_instruction", system_prompt])

if limit:
    cmd.extend(["--limit", str(limit)])

cmd.extend(["--batch_size", str(batch_size)])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{output_dir}/{task}/{config_name}/{timestamp}"
os.makedirs(output_path, exist_ok=True)
cmd.extend(["--output_path", output_path, "--log_samples"])

# Print readable command (truncate system prompt for display)
display_cmd = []
skip_next = False
for i, arg in enumerate(cmd):
    if skip_next:
        skip_next = False
        continue
    if arg == "--system_instruction":
        display_cmd.append(f'--system_instruction "<{len(system_prompt)} chars>"')
        skip_next = True
    else:
        display_cmd.append(arg)

print(f"\nConfig:  {config_name}")
print(f"Model:   {model_path}")
print(f"Task:    {task}")
print(f"Output:  {output_path}")
print(f"\n{' '.join(display_cmd)}\n")

if dry_run:
    if system_prompt:
        print(f"System prompt:\n{'-'*40}\n{system_prompt}\n{'-'*40}")
    print("[dry-run] Not executing.")
    sys.exit(0)

result = subprocess.run(cmd)
if result.returncode == 0:
    print(f"\nResults saved to: {output_path}")
sys.exit(result.returncode)
PYTHON
