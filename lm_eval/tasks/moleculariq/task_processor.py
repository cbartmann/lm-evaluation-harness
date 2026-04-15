"""
Task processor for MolecularIQ benchmark.
Contains the specific hooks: process_docs, doc_to_text, and process_results.
"""
import statistics
from collections import defaultdict
from typing import Any, Dict, List

import datasets
from moleculariq_core import evaluate_answer
from tqdm import tqdm

from lm_eval.tasks.moleculariq.extractors import extract_moleculariq_answer


# Common system prompt for MolecularIQ tasks
SYSTEM_PROMPT = """You are an expert chemist. Answer molecular property, understanding, structural analysis and molecular generation questions precisely and accurately.

CRITICAL: Only content within <answer></answer> tags will be extracted. ALWAYS return JSON format.

KEY REQUIREMENT: Use EXACT key names from the question. Never modify or invent keys.

INDEXING: Atoms are indexed from 0 to the end of the SMILES string from left to right. Only heavy atoms (skip [H], include [2H]/[3H]).
Examples:
    - "CCO": C(0), C(1), O(2)
    - "CC(C)O": C(0), C(1), C(2), O(3)
    - "CC(=O)N": C(0), C(1), O(2), N(3)

ABSENT FEATURES: Use 0 for counts, [] for indices. Never null or omit.

ALWAYS USE JSON with EXACT keys from the question:

Single count (key from question: "alcohol_count"):
<answer>{"alcohol_count": 2}</answer>
<answer>{"alcohol_count": 0}</answer>  (if absent)

Single index (key from question: "ketone_indices"):
<answer>{"ketone_indices": [5]}</answer>
<answer>{"ketone_indices": []}</answer>  (if absent)

Multiple properties (keys from question: "ring_count", "halogen_indices"):
<answer>{"ring_count": 2, "halogen_indices": [3, 7]}</answer>
<answer>{"ring_count": 0, "halogen_indices": []}</answer>  (if all absent)

Constraint generation:
<answer>{"smiles": "CC(O)C"}</answer>

Include ALL requested properties. Never null or omit."""


def get_system_prompt() -> str:
    """Return the system prompt for MolecularIQ tasks."""
    return SYSTEM_PROMPT


class MolecularIQProcessor:
    """Processor for the MolecularIQ benchmark."""

    def __init__(self):
        """Initialize the MolecularIQ processor."""
        self.extraction_function = extract_moleculariq_answer

    def doc_to_text(self, doc: dict) -> str:
        """Convert document to input text for the model."""
        return doc.get('question', '')

    def process_docs(self, docs):
        """Process documents for MolecularIQ task."""
        if hasattr(docs, '__len__') and len(docs) > 10:
            print(f"Processing {len(docs)} MolecularIQ documents...")
        return docs


# Create processor instance for use by the task
processor = MolecularIQProcessor()


# Inline prompt for models that need explicit instructions
INLINE_PROMPT = """You are an expert chemist. Answer the following chemistry question.
Provide your final answer inside <answer></answer> tags in JSON format.

Question: {question}

Answer:"""


# Export the hook functions for the YAML configuration
def doc_to_text(doc: dict) -> str:
    return processor.doc_to_text(doc)


def doc_to_text_inline(doc: dict) -> str:
    """Convert document to input text with inline prompt for models needing explicit instructions."""
    question = doc.get('question', '')
    return INLINE_PROMPT.format(question=question)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return processor.process_docs(dataset)


def process_results(doc: dict, results: List[str]) -> Dict[str, Any]:
    """Process results using direct evaluation with MolecularIQ rewards."""
    metrics = defaultdict(list)

    # Handle nested list results
    if isinstance(results[0], list):
        results = results[0]

    # Validate results format
    if not (isinstance(results, list) and isinstance(results[0], str)):
        raise ValueError(f"Results must be a list of strings, got {type(results[0])}!")

    per_category_rewards = defaultdict(list)

    # Progress bar for processing results
    result_iterator = tqdm(enumerate(results, start=1),
                          total=len(results),
                          desc="Evaluating MolecularIQ responses",
                          unit="response",
                          leave=False)

    for i, result in result_iterator:
        if not isinstance(result, str):
            raise ValueError(f"Result must be string, got {type(result)}!")

        # Update progress bar description with current response number
        result_iterator.set_postfix({"Response": f"{i}/{len(results)}"})

        # Get reward using direct evaluation
        eval_result = moleculariq_bencheval(doc=doc, answer=result)

        # Extract category and reward
        category = eval_result.get('task_type', 'unknown')
        reward = eval_result.get('reward', 0)

        per_category_rewards[category].append(reward)
        metrics['extracted_answers'].append(eval_result.get('extracted_answer', ''))

    # Calculate category averages
    for category, rewards in sorted(per_category_rewards.items()):
        if rewards:  # Avoid division by zero
            avg_reward = statistics.mean(rewards)
            metrics[category] = avg_reward

    return dict(metrics)


def process_results_pass_at_k(doc: dict, results: List[str]) -> Dict[str, Any]:
    """
    Process results for Pass@k evaluation.

    Pass@k means: does the model get the correct answer in at least one of k attempts?
    For k attempts, if any attempt is correct, the problem gets a score of 1, otherwise 0.
    """
    metrics = defaultdict(list)

    # Handle nested list results
    if isinstance(results[0], list):
        results = results[0]

    # Validate results format
    if not (isinstance(results, list) and isinstance(results[0], str)):
        raise ValueError(f"Results must be a list of strings, got {type(results[0])}!")

    # Evaluate all results for this document
    eval_results = []
    all_rewards = []

    # Progress bar for Pass@k evaluation
    result_iterator = tqdm(results,
                          desc="Pass@k MolecularIQ evaluation",
                          unit="attempt",
                          leave=False)

    for i, result in enumerate(result_iterator, 1):
        if not isinstance(result, str):
            raise ValueError(f"Result must be string, got {type(result)}!")

        # Update progress bar with current attempt info
        result_iterator.set_postfix({"Attempt": f"{i}/{len(results)}"})

        # Get evaluation using direct evaluation
        eval_result = moleculariq_bencheval(doc=doc, answer=result)
        eval_results.append(eval_result)
        all_rewards.append(eval_result.get('reward', 0))

        # Store extracted answers for debugging
        metrics['extracted_answers'].append(eval_result.get('extracted_answer', ''))

    # Calculate Pass@k metrics
    num_attempts = len(all_rewards)

    # Pass@1: first attempt correct
    metrics['pass_at_1'] = float(all_rewards[0] > 0) if num_attempts > 0 else 0.0

    # Pass@3: any of first 3 attempts correct
    if num_attempts >= 3:
        metrics['pass_at_3'] = float(any(reward > 0 for reward in all_rewards[:3]))
    elif num_attempts > 0:
        # If fewer than 3 attempts, use all available
        metrics['pass_at_3'] = float(any(reward > 0 for reward in all_rewards))
    else:
        metrics['pass_at_3'] = 0.0

    # Average accuracy across all attempts (traditional metric for comparison)
    metrics['avg_accuracy'] = statistics.mean(all_rewards) if all_rewards else 0.0

    # Category-specific metrics
    task_type = doc.get('task_type', 'unknown')
    supercategory = doc.get('supercategory', 'unknown')

    # Task type specific metrics
    metrics[f'{task_type}_avg_accuracy'] = metrics['avg_accuracy']

    # Supercategory specific metrics
    if supercategory and supercategory != 'unknown':
        metrics[f'{supercategory}_avg_accuracy'] = metrics['avg_accuracy']

    return dict(metrics)


def moleculariq_bencheval(doc: dict, answer: str) -> Dict[str, Any]:
    """
    Evaluate a MolecularIQ answer using the appropriate reward function.

    Args:
        doc: Document containing task data including task_type and target
        answer: Model's text answer

    Returns:
        Dict with evaluation results including reward and extracted answer
    """
    # Extract task information
    task_type = doc.get('task_type', '')
    supercategory = doc.get('supercategory', '')

    extracted_answer = processor.extraction_function(answer)

    # Normalize task type for constraint tasks
    task_type_norm = task_type
    if 'single_constraint_generation' in task_type:
        task_type_norm = 'constraint_generation'
    elif 'multi_constraint_generation' in task_type:
        task_type_norm = 'constraint_generation'

    # For constraint tasks, use the constraints field
    if 'constraint' in task_type.lower() or 'generation' in task_type.lower():
        constraints = doc.get('constraints') or doc.get('constraint')
        reward = evaluate_answer(
            task_type=task_type_norm,
            predicted=extracted_answer,
            constraints=constraints,
            return_details=False
        )
    else:
        # For other tasks, pass the target directly
        target = doc.get('target')
        reward = evaluate_answer(
            task_type=task_type_norm,
            predicted=extracted_answer,
            target=target,
            return_details=False
        )

    # Ensure reward is a float
    if isinstance(reward, dict):
        reward = reward.get('reward', 0.0)
    reward = float(reward)

    return {
        'task_type': task_type,
        'reward': reward,
        'extracted_answer': extracted_answer,
        'supercategory': supercategory
    }
