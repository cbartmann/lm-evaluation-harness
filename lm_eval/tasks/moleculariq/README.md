# MolecularIQ Benchmark Task

MolecularIQ is a comprehensive chemistry benchmark for evaluating language models on molecular understanding tasks. This implementation provides Pass@k metrics.

## Dataset

- **Dataset**: `tschouis/moleculariq_arxiv`
- **Split**: test

## Task Types

The benchmark includes the following task categories:

1. **Count Tasks**: Counting molecular features (atoms, rings, functional groups)
   - `single_count`: Single property counting
   - `multi_count`: Multiple property counting in one question

2. **Index Tasks**: Identifying atom indices for specific features
   - `single_index`: Single index identification
   - `multi_index`: Multiple index identification

3. **Constraint Generation**: Generating molecules that satisfy given constraints

## Metrics

- **Pass@1**: Accuracy on first attempt
- **Pass@3**: Any correct answer in first 3 attempts
- **Pass@5**: Any correct answer in first 5 attempts
- **Pass@8**: Any correct answer in first 8 attempts
- **avg_accuracy**: Average accuracy across all attempts

## Available Tasks

| Task | Description |
|------|-------------|
| `moleculariq_pass_at_k` | Raw question only - use with `--system_instruction` or chat models |
| `moleculariq_inline` | Question with inline prompt (instructions + answer format) |

## Usage

### Basic Usage (Raw Question)

For chat models or when using `--system_instruction`:

```bash
lm_eval --model vllm \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_pass_at_k \
    --system_instruction "You are a chemistry expert. Provide answers in <answer>JSON</answer> format." \
    --batch_size auto
```

### With Inline Prompt

For models that need explicit instructions in the prompt:

```bash
lm_eval --model vllm \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_inline \
    --batch_size auto
```

### Quick Test (Limited Samples)

```bash
lm_eval --model vllm \
    --model_args pretrained=facebook/opt-125m \
    --tasks moleculariq_pass_at_k \
    --limit 10 \
    --batch_size auto
```

## Directory Structure

```
moleculariq/
├── __init__.py                    # Package init
├── moleculariq_pass_at_k.yaml     # Task with raw question only
├── moleculariq_inline.yaml        # Task with inline prompt
├── task_processor.py              # Processing hooks (uses moleculariq_core)
├── extractors.py                  # Answer extraction functions
└── README.md
```

## Dependencies

- `moleculariq_core`: Core library for molecular reasoning and reward computation
- `rdkit`: Chemistry toolkit (dependency of moleculariq_core)
- `datasets`: For loading the HuggingFace dataset

Install dependencies:
```bash
pip install moleculariq-core rdkit
```

## Answer Format

Models should return answers in JSON format within answer tags:

```
<answer>{"property_name": value}</answer>
```

For count tasks:
```
<answer>{"ring_count": 2}</answer>
```

For index tasks:
```
<answer>{"carbon_indices": [0, 1, 3]}</answer>
```

For constraint generation:
```
<answer>{"smiles": "CCO"}</answer>
```

The extraction function also supports the ether0 format:
```
<|answer_start|>{"smiles": "CCO"}<|answer_end|>
```

## Atom Indexing Convention

Atoms are indexed from 0 to N-1, reading the SMILES string left to right, counting only heavy atoms (non-hydrogen). Examples:

- `"CCO"`: C(0), C(1), O(2)
- `"CC(C)O"`: C(0), C(1), C(2), O(3)
- `"CC(=O)N"`: C(0), C(1), O(2), N(3)

## Customization

### Custom Extraction via Filters

The lm-eval `filter_list` mechanism allows users to customize output extraction. Users can create a task variant YAML:

```yaml
# my_model_moleculariq.yaml
include: moleculariq_pass_at_k.yaml
task: my_model_moleculariq

filter_list:
  - name: my_extraction
    filter:
      - function: custom
        filter_fn: !function my_extractors.extract_my_format
```

### Custom Preprocessing via `doc_to_text`

Users can customize prompt preprocessing by overriding `doc_to_text`:

```yaml
# my_model_moleculariq.yaml
include: moleculariq_pass_at_k.yaml
task: my_model_moleculariq

doc_to_text: !function my_utils.custom_doc_to_text
```

### Using Regex Filters

For simple pattern extraction, use the built-in regex filter:

```yaml
filter_list:
  - name: extract_answer
    filter:
      - function: regex
        regex_pattern: r"<my_answer>(.*?)</my_answer>"
        group_select: 0
        fallback: ""
```

## Citation

If you use this benchmark, please cite the MolecularIQ paper:

```bibtex
@article{moleculariq,
  title={MolecularIQ: A Comprehensive Benchmark for Molecular Intelligence},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```
