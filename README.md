# MolecularIQ Evaluation

[![Paper](https://img.shields.io/badge/arXiv-2601.15279-b31b1b.svg)](https://arxiv.org/abs/2601.15279)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=RqwEzZqMFv)
[![Leaderboard](https://img.shields.io/badge/HF-Leaderboard-yellow.svg)](https://huggingface.co/spaces/ml-jku/molecularIQ_leaderboard)
[![Dataset](https://img.shields.io/badge/HF-Dataset-green.svg)](https://huggingface.co/datasets/ml-jku/moleculariq-v0.0)
[![Project](https://img.shields.io/badge/GitHub-MolecularIQ-black.svg)](https://github.com/ml-jku/moleculariq)

Evaluation harness for the **MolecularIQ** benchmark — a fully symbolically verifiable benchmark for assessing chemical reasoning capabilities of LLMs over molecular graphs.

This repository is a fork of [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the MolecularIQ task integrated as a custom benchmark.

## Overview

MolecularIQ evaluates LLMs on three reasoning task types across six categories of molecular features:

| Task Type | Description | Examples |
|-----------|-------------|----------|
| **Feature Counting** | Count molecular properties | "How many rotatable bonds?" |
| **Index-based Attribution** | Identify atom indices for features | "Which atoms are in the Murcko scaffold?" |
| **Constrained Generation** | Generate molecules satisfying constraints | "Build a molecule with exactly 5 saturated rings" |

The benchmark comprises **5,111 questions** from 849 unique molecules, with symbolic verification ensuring exact evaluation. See the [paper](https://arxiv.org/abs/2601.15279) for details.

## Installation

```bash
git clone https://github.com/ml-jku/moleculariq-eval.git
cd moleculariq-eval
pip install -e ".[vllm]"
pip install moleculariq-core rdkit
```

### Branches

| Branch | Description |
|--------|-------------|
| `main` | Minimal setup — run evaluations directly via `lm_eval` CLI |
| `with_config` | Includes pre-built configs for all 34 models from the paper + runner script |

## Quick Start

```bash
lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen3-0.6B,dtype=bfloat16 \
    --tasks moleculariq_pass_at_k \
    --apply_chat_template \
    --batch_size auto \
    --limit 10
```

## Evaluation

### Option 1: Direct CLI (`main` branch)

Run `lm_eval` directly, passing the system instruction on the command line:

```bash
lm_eval --model vllm \
    --model_args pretrained=YOUR_MODEL,dtype=bfloat16 \
    --tasks moleculariq_pass_at_k \
    --apply_chat_template \
    --system_instruction "You are an expert chemist. Provide answers in <answer>JSON</answer> format." \
    --batch_size auto \
    --log_samples \
    --output_path results/
```

For models that need instructions embedded in the prompt (non-chat models):

```bash
lm_eval --model vllm \
    --model_args pretrained=YOUR_MODEL,dtype=bfloat16 \
    --tasks moleculariq_inline \
    --batch_size auto
```

### Option 2: Config-based (`with_config` branch)

The `with_config` branch includes ready-to-use configs for all models evaluated in the paper:

```bash
git checkout with_config

# List available configs
./run_moleculariq.sh --help

# Dry-run (shows the command without executing)
./run_moleculariq.sh qwen3-8b --dry-run

# Run evaluation
./run_moleculariq.sh qwen3-8b

# Quick test with limited samples
./run_moleculariq.sh qwen3-8b --limit 10
```

Each config specifies model path, generation parameters, system prompt, and chat template settings. See `configs/` for all available configurations.

### Available Tasks

| Task | Repeats | Metrics | Use case |
|------|---------|---------|----------|
| `moleculariq_pass_at_k` | 3 | pass@1, pass@3, avg_accuracy | Standard evaluation with system instruction |
| `moleculariq_inline` | 3 | pass@1, pass@3, avg_accuracy | Models needing instructions in the prompt |

## Leaderboard

Submit your results to the [MolecularIQ Leaderboard](https://huggingface.co/spaces/ml-jku/molecularIQ_leaderboard).

To submit, provide:
- **Model**: name and weights (open weights or free API access required)
- **System prompt**: the instruction used during evaluation
- **Sampling parameters**: temperature, top_p, top_k, etc.
- **Extraction method**: how answers are parsed from model output

A config file is optional but encouraged for reproducibility.

## Citation

```bibtex
@inproceedings{bartmann2026moleculariq,
  title={Molecular{IQ}: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs},
  author={Christoph Bartmann and Johannes Schimunek and Mykyta Ielanskyi and Philipp Seidl and G{\"u}nter Klambauer and Sohvi Luukkonen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=RqwEzZqMFv}
}
```

## Links

- **Project**: [github.com/ml-jku/moleculariq](https://github.com/ml-jku/moleculariq)
- **Paper**: [arxiv.org/abs/2601.15279](https://arxiv.org/abs/2601.15279)
- **Dataset**: [huggingface.co/datasets/ml-jku/moleculariq-v0.0](https://huggingface.co/datasets/ml-jku/moleculariq-v0.0)
- **Leaderboard**: [huggingface.co/spaces/ml-jku/molecularIQ_leaderboard](https://huggingface.co/spaces/ml-jku/molecularIQ_leaderboard)
- **Base framework**: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
