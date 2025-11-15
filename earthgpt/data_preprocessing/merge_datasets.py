"""
Merge multiple preprocessed datasets into unified train/val splits.

This script combines DOTA (OBB), RSVQA (VQA), and RSICD (Captioning)
into single unified JSONL files with task mixture control.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def merge_datasets(
    dataset_paths: Dict[str, str],
    output_path: str,
    task_ratios: Dict[str, float] = None,
    max_samples_per_task: Dict[str, int] = None,
    shuffle: bool = True
):
    """
    Merge multiple datasets with task-based sampling.

    Args:
        dataset_paths: Dict mapping task name to JSONL path
        output_path: Path to save merged JSONL
        task_ratios: Desired ratio for each task (will be normalized)
        max_samples_per_task: Maximum samples to use from each task
        shuffle: Whether to shuffle the final dataset
    """

    if task_ratios is None:
        task_ratios = {
            'obb': 0.4,
            'vqa': 0.3,
            'captioning': 0.3
        }

    # Load all datasets
    all_data = defaultdict(list)

    for task, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping...")
            continue

        data = load_jsonl(path)
        print(f"Loaded {len(data)} samples from {task}: {path}")

        # Apply max samples limit
        if max_samples_per_task and task in max_samples_per_task:
            data = data[:max_samples_per_task[task]]

        all_data[task] = data

    # Calculate total samples and per-task counts based on ratios
    total_samples = sum(len(samples) for samples in all_data.values())

    # Normalize ratios
    ratio_sum = sum(task_ratios.values())
    normalized_ratios = {k: v / ratio_sum for k, v in task_ratios.items()}

    # Sample from each task according to ratios
    merged_data = []

    for task, samples in all_data.items():
        if task not in normalized_ratios:
            # Include all samples if no ratio specified
            merged_data.extend(samples)
        else:
            # Sample according to ratio
            desired_count = int(len(samples) * normalized_ratios[task] /
                               (len(samples) / total_samples))
            desired_count = min(desired_count, len(samples))

            sampled = random.sample(samples, desired_count) if desired_count < len(samples) else samples
            merged_data.extend(sampled)

            print(f"Sampled {len(sampled)} samples from {task} (ratio: {normalized_ratios[task]:.2f})")

    # Shuffle
    if shuffle:
        random.shuffle(merged_data)

    # Save
    save_jsonl(merged_data, output_path)

    print(f"\nMerged dataset saved to {output_path}")
    print(f"Total samples: {len(merged_data)}")

    # Print task distribution
    task_counts = defaultdict(int)
    for item in merged_data:
        task_counts[item.get('task', 'unknown')] += 1

    print("\nTask distribution:")
    for task, count in task_counts.items():
        print(f"  {task}: {count} ({count/len(merged_data)*100:.1f}%)")


def main():
    """Example usage."""

    # Set random seed for reproducibility
    random.seed(42)

    # Merge training datasets
    train_datasets = {
        'obb': './data/dota_train.jsonl',
        'vqa': './data/rsvqa_lr_train.jsonl',
        'captioning': './data/rsicd_train.jsonl'
    }

    merge_datasets(
        dataset_paths=train_datasets,
        output_path='./data/unified_train.jsonl',
        task_ratios={
            'obb': 0.4,
            'vqa': 0.3,
            'captioning': 0.3
        },
        shuffle=True
    )

    # Merge validation datasets
    val_datasets = {
        'obb': './data/dota_val.jsonl',
        'vqa': './data/rsvqa_lr_val.jsonl',
        'captioning': './data/rsicd_val.jsonl'
    }

    merge_datasets(
        dataset_paths=val_datasets,
        output_path='./data/unified_val.jsonl',
        task_ratios={
            'obb': 0.4,
            'vqa': 0.3,
            'captioning': 0.3
        },
        shuffle=True
    )


if __name__ == "__main__":
    main()
