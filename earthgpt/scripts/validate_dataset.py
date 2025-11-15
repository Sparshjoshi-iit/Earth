"""
Dataset Validation and Statistics Script

Validates JSONL format and provides statistics about the dataset.
"""

import json
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def validate_obb_format(text):
    """Validate OBB output format."""
    pattern = r'\[([^,]+),\s*\((\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+)\)\]'
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_sample(sample, line_num):
    """
    Validate a single sample.

    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    if 'image' not in sample:
        return False, f"Line {line_num}: Missing 'image' field"

    if 'conversations' not in sample:
        return False, f"Line {line_num}: Missing 'conversations' field"

    if 'task' not in sample:
        return False, f"Line {line_num}: Missing 'task' field"

    # Check task type
    if sample['task'] not in ['obb', 'vqa', 'captioning', 'unknown']:
        return False, f"Line {line_num}: Invalid task type '{sample['task']}'"

    # Check conversations
    if not isinstance(sample['conversations'], list):
        return False, f"Line {line_num}: 'conversations' must be a list"

    if len(sample['conversations']) == 0:
        return False, f"Line {line_num}: 'conversations' is empty"

    # Check first turn is from human with <image> token
    first_turn = sample['conversations'][0]
    if first_turn.get('from') != 'human':
        return False, f"Line {line_num}: First turn must be from 'human'"

    if '<image>' not in first_turn.get('value', ''):
        return False, f"Line {line_num}: First turn must contain '<image>' token"

    # Check alternating turns
    for i, turn in enumerate(sample['conversations']):
        expected_role = 'human' if i % 2 == 0 else 'gpt'
        if turn.get('from') != expected_role:
            return False, f"Line {line_num}: Expected '{expected_role}' at turn {i}, got '{turn.get('from')}'"

        if 'value' not in turn:
            return False, f"Line {line_num}: Turn {i} missing 'value' field"

    # Validate OBB format if applicable
    if sample['task'] == 'obb':
        response = sample['conversations'][-1]['value']
        if not validate_obb_format(response):
            return False, f"Line {line_num}: Invalid OBB format in response"

    return True, None


def compute_statistics(samples):
    """Compute dataset statistics."""

    stats = {
        'total_samples': len(samples),
        'task_distribution': defaultdict(int),
        'unique_images': set(),
        'avg_response_length': [],
        'samples_per_image': defaultdict(int),
        'conversation_lengths': []
    }

    for sample in samples:
        # Task distribution
        stats['task_distribution'][sample['task']] += 1

        # Unique images
        stats['unique_images'].add(sample['image'])

        # Samples per image
        stats['samples_per_image'][sample['image']] += 1

        # Response length
        response = sample['conversations'][-1]['value']
        stats['avg_response_length'].append(len(response.split()))

        # Conversation length
        stats['conversation_lengths'].append(len(sample['conversations']))

    # Convert sets to counts
    stats['num_unique_images'] = len(stats['unique_images'])
    stats['unique_images'] = None  # Remove to save memory

    # Compute averages
    if stats['avg_response_length']:
        stats['avg_response_length_words'] = sum(stats['avg_response_length']) / len(stats['avg_response_length'])
    else:
        stats['avg_response_length_words'] = 0

    if stats['samples_per_image']:
        stats['avg_samples_per_image'] = stats['total_samples'] / stats['num_unique_images']
    else:
        stats['avg_samples_per_image'] = 0

    stats['avg_conversation_length'] = sum(stats['conversation_lengths']) / len(stats['conversation_lengths']) if stats['conversation_lengths'] else 0

    return stats


def print_statistics(stats):
    """Print formatted statistics."""

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print(f"\nTotal samples: {stats['total_samples']:,}")
    print(f"Unique images: {stats['num_unique_images']:,}")
    print(f"Average samples per image: {stats['avg_samples_per_image']:.2f}")

    print("\nTask Distribution:")
    for task, count in stats['task_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  {task}: {count:,} ({percentage:.1f}%)")

    print(f"\nAverage response length: {stats['avg_response_length_words']:.1f} words")
    print(f"Average conversation length: {stats['avg_conversation_length']:.1f} turns")

    print("=" * 80 + "\n")


def validate_dataset(file_path, check_images=True, verbose=False):
    """
    Validate JSONL dataset.

    Args:
        file_path: Path to JSONL file
        check_images: Whether to check if image files exist
        verbose: Print all errors
    """
    print(f"Validating dataset: {file_path}")

    samples = []
    errors = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Validating")):
            try:
                sample = json.loads(line.strip())

                # Validate format
                is_valid, error_msg = validate_sample(sample, i + 1)

                if not is_valid:
                    errors.append(error_msg)
                    if verbose:
                        print(error_msg)
                    continue

                # Check if image exists
                if check_images:
                    if not os.path.exists(sample['image']):
                        errors.append(f"Line {i+1}: Image not found: {sample['image']}")
                        if verbose:
                            print(f"Warning: Image not found: {sample['image']}")

                samples.append(sample)

            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: JSON decode error: {e}")
                if verbose:
                    print(f"Line {i+1}: JSON decode error: {e}")
            except Exception as e:
                errors.append(f"Line {i+1}: Unexpected error: {e}")
                if verbose:
                    print(f"Line {i+1}: Unexpected error: {e}")

    # Print validation results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Print first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✅ All samples passed validation!")

    print(f"\nValid samples: {len(samples)} / {len(samples) + len(errors)}")
    print("=" * 80)

    # Compute and print statistics
    if samples:
        stats = compute_statistics(samples)
        print_statistics(stats)

    return len(errors) == 0, samples


def main():
    parser = argparse.ArgumentParser(description="Validate EarthGPT dataset")

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to JSONL dataset file"
    )

    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="Skip checking if image files exist"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all errors during validation"
    )

    args = parser.parse_args()

    # Validate
    is_valid, samples = validate_dataset(
        file_path=args.file_path,
        check_images=not args.no_check_images,
        verbose=args.verbose
    )

    # Exit code
    if is_valid:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
