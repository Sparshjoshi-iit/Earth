"""
Complete Setup Script for Your Data

This script orchestrates all preprocessing steps:
1. Process GSD metadata → GSD estimation task
2. Process VQA annotations → VQA task
3. Process OBB annotations → OBB + measurement tasks
4. Merge everything into unified training data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_preprocessing.process_gsd_data import GSDPreprocessor
from data_preprocessing.process_your_vqa_data import YourVQAPreprocessor
from data_preprocessing.process_your_obb_data import YourOBBPreprocessor
from data_preprocessing.merge_datasets import merge_datasets


def setup_your_data(
    # GSD data paths
    gsd_metadata_path: str,
    gsd_image_dir: str,

    # VQA data paths
    vqa_annotations_path: str,
    vqa_image_root: str,

    # OBB data paths (optional)
    obb_annotations_path: str = None,
    obb_image_dir: str = None,

    # Output paths
    output_dir: str = "./data/your_processed_data",

    # Options
    include_gsd_estimation: bool = True,
    include_obb: bool = False,
    include_measurements: bool = True
):
    """
    Complete data preprocessing pipeline.

    Args:
        gsd_metadata_path: Path to GSD metadata JSON
        gsd_image_dir: Directory with images for GSD data
        vqa_annotations_path: Path to VQA annotations JSONL
        vqa_image_root: Root directory for VQA images
        obb_annotations_path: Path to OBB annotations (optional)
        obb_image_dir: Directory with OBB images (optional)
        output_dir: Output directory for processed data
        include_gsd_estimation: Include GSD estimation task
        include_obb: Include OBB detection task
        include_measurements: Include GSD-aware measurements
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = {}

    # ========================================================================
    # Step 1: Process GSD Data
    # ========================================================================
    if include_gsd_estimation:
        print("\n" + "=" * 80)
        print("STEP 1: Processing GSD Data")
        print("=" * 80)

        gsd_processor = GSDPreprocessor(
            gsd_metadata_path=gsd_metadata_path,
            image_dir=gsd_image_dir,
            output_path=str(output_dir / "gsd_estimation.jsonl"),
            include_estimation=True
        )
        gsd_processor.process()
        dataset_paths['gsd_estimation'] = str(output_dir / "gsd_estimation.jsonl")

    # ========================================================================
    # Step 2: Process VQA Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Processing VQA Data")
    print("=" * 80)

    vqa_processor = YourVQAPreprocessor(
        annotations_file=vqa_annotations_path,
        image_root=vqa_image_root,
        output_path=str(output_dir / "vqa.jsonl"),
        max_qa_pairs_per_image=None,  # Use all QA pairs
        question_types=None  # Use all question types
    )
    vqa_processor.process()
    dataset_paths['vqa'] = str(output_dir / "vqa.jsonl")

    # ========================================================================
    # Step 3: Process OBB Data (if available)
    # ========================================================================
    if include_obb and obb_annotations_path:
        print("\n" + "=" * 80)
        print("STEP 3: Processing OBB Data")
        print("=" * 80)

        obb_processor = YourOBBPreprocessor(
            annotations_file=obb_annotations_path,
            image_dir=obb_image_dir,
            gsd_metadata_path=gsd_metadata_path if include_measurements else None,
            output_path=str(output_dir / "obb.jsonl"),
            include_gsd_measurements=include_measurements
        )
        obb_processor.process()
        dataset_paths['obb'] = str(output_dir / "obb.jsonl")

    # ========================================================================
    # Step 4: Merge All Datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Merging All Datasets")
    print("=" * 80)

    # Set task ratios based on what's included
    task_ratios = {}
    if 'gsd_estimation' in dataset_paths:
        task_ratios['gsd_estimation'] = 0.15  # 15% GSD estimation
    if 'vqa' in dataset_paths:
        task_ratios['vqa'] = 0.55  # 55% VQA (your main task)
    if 'obb' in dataset_paths:
        task_ratios['obb'] = 0.20  # 20% OBB
        task_ratios['obb_measurement'] = 0.10  # 10% Measurements

    merge_datasets(
        dataset_paths=dataset_paths,
        output_path=str(output_dir / "unified_train.jsonl"),
        task_ratios=task_ratios,
        shuffle=True
    )

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nUnified training data: {output_dir / 'unified_train.jsonl'}")
    print("\nNext steps:")
    print("1. Validate data: python scripts/validate_dataset.py " + str(output_dir / "unified_train.jsonl"))
    print("2. Train model: python training/train.py --training_config configs/training_config.yaml")


def main():
    """
    Example usage - MODIFY THESE PATHS FOR YOUR DATA!
    """

    setup_your_data(
        # GSD data
        gsd_metadata_path="./data/your_gsd_metadata.json",
        gsd_image_dir="./data/your_gsd_images",

        # VQA data
        vqa_annotations_path="./data/your_vqa_annotations.jsonl",
        vqa_image_root="./data",  # Root for VQA image paths

        # OBB data (if you have it)
        obb_annotations_path="./data/your_obb_annotations.json",
        obb_image_dir="./data/your_obb_images",

        # Output
        output_dir="./data/your_processed_data",

        # Options
        include_gsd_estimation=True,
        include_obb=True,  # Set to False if you don't have OBB data yet
        include_measurements=True
    )


if __name__ == "__main__":
    main()
