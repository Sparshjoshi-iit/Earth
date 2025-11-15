"""
Display EarthGPT Project Structure

Shows the complete file structure and what each component does.
"""

from pathlib import Path


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure."""

    if current_depth > max_depth:
        return

    items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))

    # Filter out unwanted directories
    items = [item for item in items if item.name not in ['.git', '__pycache__', 'venv', '.venv', 'cache', 'outputs', 'logs']]

    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")

        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1)


def print_project_info():
    """Print project information."""

    print("=" * 80)
    print("🛰️  EarthGPT Project Structure")
    print("=" * 80)
    print()

    print("📁 File Structure:")
    print()

    project_root = Path(__file__).parent.parent
    print_tree(project_root, max_depth=2)

    print()
    print("=" * 80)
    print("📚 Component Overview")
    print("=" * 80)
    print()

    components = {
        "configs/": "YAML configuration files for model and training",
        "data_preprocessing/": "Scripts to convert datasets to unified format",
        "model/": "Core EarthGPT architecture and dataset classes",
        "training/": "Training script with LoRA fine-tuning",
        "scripts/": "Utilities for inference, data generation, validation",
        "evaluation/": "Task-specific evaluation metrics",
    }

    for component, description in components.items():
        print(f"  {component:25s} {description}")

    print()
    print("=" * 80)
    print("🚀 Quick Start Options")
    print("=" * 80)
    print()

    print("Option 1: Synthetic Data (Fastest - No Downloads)")
    print("  1. pip install -r requirements.txt")
    print("  2. python scripts/generate_sample_data.py")
    print("  3. python training/train.py --training_config configs/training_config_synthetic.yaml")
    print()

    print("Option 2: Real Datasets (Production Quality)")
    print("  1. python scripts/download_datasets.py --all")
    print("  2. Download DOTA, RSVQA manually (see QUICKSTART.md)")
    print("  3. python data_preprocessing/process_*.py")
    print("  4. python data_preprocessing/merge_datasets.py")
    print("  5. python training/train.py --training_config configs/training_config.yaml")
    print()

    print("=" * 80)
    print("📖 Documentation")
    print("=" * 80)
    print()
    print("  README.md        - Complete documentation and architecture")
    print("  QUICKSTART.md    - 5-minute setup guide")
    print("  DATA_FORMAT.md   - Dataset format specification")
    print()

    print("=" * 80)
    print("🔧 Key Scripts")
    print("=" * 80)
    print()

    scripts = {
        "generate_sample_data.py": "Create synthetic satellite imagery",
        "download_datasets.py": "Auto-download available datasets",
        "validate_dataset.py": "Validate JSONL format and statistics",
        "inference.py": "Run inference on images (OBB/VQA/Captioning)",
        "test_installation.py": "Check if all packages installed",
    }

    for script, description in scripts.items():
        print(f"  {script:30s} {description}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    print_project_info()
