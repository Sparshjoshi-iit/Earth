"""
Automated Dataset Downloader for EarthGPT

Downloads publicly available remote sensing datasets from Hugging Face and other sources.

Supported datasets:
1. RSICD (via Hugging Face) - Image captioning
2. RSVQA (via Hugging Face) - Visual question answering
3. EuroSAT - Scene classification (can be adapted for captioning)
4. PatternNet - Scene classification
5. UC Merced Land Use - Scene classification

For DOTA: Requires manual download from https://captain-whu.github.io/DOTA/dataset.html
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import tarfile

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install datasets huggingface_hub")


class DatasetDownloader:
    """Download and prepare datasets for EarthGPT."""

    def __init__(self, output_dir: str = "./data/downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_from_url(self, url: str, output_path: Path, extract: bool = True):
        """Download file from URL with progress bar."""

        if output_path.exists():
            print(f"File already exists: {output_path}")
            return

        print(f"Downloading from {url}...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Downloaded to {output_path}")

        # Extract if archive
        if extract and (output_path.suffix in ['.zip', '.tar', '.gz', '.tgz']):
            print(f"Extracting {output_path}...")
            extract_dir = output_path.parent / output_path.stem
            extract_dir.mkdir(exist_ok=True)

            if output_path.suffix == '.zip':
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif output_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(output_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)

            print(f"Extracted to {extract_dir}")

    def download_rsicd_hf(self):
        """
        Download RSICD from Hugging Face (if available).

        Note: RSICD availability on HF varies. This attempts to download
        from common repositories.
        """
        if not HF_AVAILABLE:
            print("Hugging Face datasets not available. Install with: pip install datasets")
            return False

        print("=" * 80)
        print("Downloading RSICD from Hugging Face...")
        print("=" * 80)

        try:
            # Try to load from HF datasets
            # Note: Replace with actual dataset name if/when available
            dataset = load_dataset("RSICD", cache_dir=str(self.output_dir / "rsicd"))

            print("RSICD downloaded successfully!")
            print(f"Location: {self.output_dir / 'rsicd'}")
            return True

        except Exception as e:
            print(f"RSICD not available on Hugging Face: {e}")
            print("\nAlternative: Use synthetic data or download manually from:")
            print("  https://github.com/201528014227051/RSICD_optimal")
            return False

    def download_rsvqa_hf(self):
        """Download RSVQA from Hugging Face."""
        if not HF_AVAILABLE:
            print("Hugging Face datasets not available.")
            return False

        print("=" * 80)
        print("Downloading RSVQA from Hugging Face...")
        print("=" * 80)

        try:
            # Try different RSVQA versions
            for variant in ['LR', 'HR']:
                print(f"\nDownloading RSVQA-{variant}...")

                # This is a placeholder - adjust based on actual HF dataset name
                dataset = load_dataset(
                    "sylvain-RSVQA",  # Adjust to actual dataset name
                    variant,
                    cache_dir=str(self.output_dir / f"rsvqa_{variant.lower()}")
                )

                print(f"RSVQA-{variant} downloaded successfully!")

            return True

        except Exception as e:
            print(f"RSVQA not available on Hugging Face: {e}")
            print("\nAlternative: Download manually from:")
            print("  https://rsvqa.sylvainlobry.com/")
            return False

    def download_eurosat(self):
        """Download EuroSAT dataset (good for scene captioning adaptation)."""
        if not HF_AVAILABLE:
            print("Hugging Face datasets not available.")
            return False

        print("=" * 80)
        print("Downloading EuroSAT...")
        print("=" * 80)

        try:
            dataset = load_dataset(
                "timm/eurosat",
                cache_dir=str(self.output_dir / "eurosat")
            )

            print("EuroSAT downloaded successfully!")
            print("This dataset contains 27,000 labeled satellite images")
            print("Can be adapted for scene captioning")

            return True

        except Exception as e:
            print(f"Error downloading EuroSAT: {e}")
            return False

    def download_ucmerced(self):
        """Download UC Merced Land Use dataset."""

        print("=" * 80)
        print("Downloading UC Merced Land Use Dataset...")
        print("=" * 80)

        url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
        output_path = self.output_dir / "UCMerced_LandUse.zip"

        try:
            self.download_from_url(url, output_path, extract=True)
            print("UC Merced dataset downloaded successfully!")
            print("21 land use classes, 100 images each")
            return True
        except Exception as e:
            print(f"Error downloading UC Merced: {e}")
            return False

    def download_patternnet(self):
        """
        PatternNet dataset info.

        Note: This requires manual download due to terms of use.
        """
        print("=" * 80)
        print("PatternNet Dataset")
        print("=" * 80)
        print("\nPatternNet must be downloaded manually from:")
        print("  https://sites.google.com/view/zhouwx/dataset")
        print("\nDataset info:")
        print("  - 30,400 images")
        print("  - 38 classes")
        print("  - High resolution satellite imagery")
        print("\nAfter download, place in:", self.output_dir / "patternnet")
        return False

    def print_dota_instructions(self):
        """Print instructions for downloading DOTA."""
        print("=" * 80)
        print("DOTA Dataset")
        print("=" * 80)
        print("\nDOTA must be downloaded manually from:")
        print("  https://captain-whu.github.io/DOTA/dataset.html")
        print("\nDataset info:")
        print("  - Large-scale dataset for object detection in aerial images")
        print("  - 15 object categories")
        print("  - Oriented bounding boxes")
        print("\nRecommended version: DOTA-v1.5")
        print("\nAfter download, place in:", self.output_dir / "DOTA-v1.5")
        print("\nStructure should be:")
        print("  DOTA-v1.5/")
        print("    train/")
        print("      images/")
        print("      labelTxt-v1.5/")
        print("    val/")
        print("      images/")
        print("      labelTxt-v1.5/")

    def create_dataset_summary(self):
        """Create summary of downloaded datasets."""

        summary_file = self.output_dir / "DATASETS_README.md"

        content = """# Downloaded Datasets for EarthGPT

## Overview

This directory contains datasets for training EarthGPT on geospatial tasks.

## Available Datasets

### 1. Synthetic Data (Generated)
- **Location**: `../synthetic/`
- **Purpose**: Testing pipeline, proof-of-concept
- **Tasks**: OBB, VQA, Captioning
- **Size**: 300 train + 60 val samples
- **Generate with**: `python scripts/generate_sample_data.py`

### 2. EuroSAT (if downloaded)
- **Location**: `eurosat/`
- **Purpose**: Scene classification → adapted for captioning
- **Size**: 27,000 images, 10 classes
- **Source**: Hugging Face

### 3. UC Merced Land Use (if downloaded)
- **Location**: `UCMerced_LandUse/`
- **Purpose**: Scene classification → adapted for captioning
- **Size**: 2,100 images, 21 classes
- **Source**: UC Merced

### 4. RSICD (Manual Download Required)
- **Location**: `rsicd/`
- **Purpose**: Image captioning
- **Size**: 10,921 images with 5 captions each
- **Download**: https://github.com/201528014227051/RSICD_optimal
- **Alternative**: Use synthetic data or EuroSAT

### 5. RSVQA (Manual Download Recommended)
- **Location**: `rsvqa/`
- **Purpose**: Visual question answering
- **Variants**: LR (Low-Res), HR (High-Res)
- **Download**: https://rsvqa.sylvainlobry.com/

### 6. DOTA (Manual Download Required)
- **Location**: `DOTA-v1.5/`
- **Purpose**: Oriented bounding box detection
- **Size**: Large-scale, 15 object categories
- **Download**: https://captain-whu.github.io/DOTA/dataset.html

## Quick Start

### Option 1: Synthetic Data (Fastest)
```bash
# Generate synthetic data
python scripts/generate_sample_data.py

# Train on synthetic data
python training/train.py --training_config configs/training_config_synthetic.yaml
```

### Option 2: Real Datasets
```bash
# Download available datasets
python scripts/download_datasets.py --all

# Preprocess datasets
python data_preprocessing/process_*.py

# Merge datasets
python data_preprocessing/merge_datasets.py

# Train
python training/train.py --training_config configs/training_config.yaml
```

## Dataset Preprocessing

Each dataset has a corresponding preprocessing script in `data_preprocessing/`:

- `process_dota.py` - DOTA → OBB format
- `process_rsvqa.py` - RSVQA → VQA format
- `process_rsicd.py` - RSICD → Captioning format
- `merge_datasets.py` - Combine all into unified JSONL

## Data Format

All datasets are converted to unified conversational JSONL:

```json
{
    "image": "path/to/image.png",
    "conversations": [
        {"from": "human", "value": "<image>\\nQuestion or instruction"},
        {"from": "gpt", "value": "Expected response"}
    ],
    "task": "obb|vqa|captioning"
}
```

See `DATA_FORMAT.md` for complete specification.

## Storage Requirements

- Synthetic data: ~50 MB
- EuroSAT: ~2 GB
- UC Merced: ~300 MB
- RSICD: ~1 GB
- RSVQA: ~2 GB (LR) / ~10 GB (HR)
- DOTA-v1.5: ~20 GB

## Troubleshooting

### Dataset not downloading
- Check internet connection
- Try manual download from provided links
- Use synthetic data as alternative

### Out of disk space
- Start with synthetic data only
- Download individual datasets as needed
- Use smaller dataset variants (e.g., RSVQA-LR instead of HR)

### Permission errors
- Ensure write permissions in data directory
- Some datasets require registration/agreement

## Contact

For issues with dataset access, see individual dataset websites or use synthetic data for testing.
"""

        with open(summary_file, 'w') as f:
            f.write(content)

        print(f"\nDataset summary created: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for EarthGPT")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/downloads",
        help="Directory to save downloaded datasets"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Attempt to download all available datasets"
    )

    parser.add_argument(
        "--eurosat",
        action="store_true",
        help="Download EuroSAT dataset"
    )

    parser.add_argument(
        "--ucmerced",
        action="store_true",
        help="Download UC Merced dataset"
    )

    parser.add_argument(
        "--rsicd",
        action="store_true",
        help="Attempt to download RSICD from Hugging Face"
    )

    parser.add_argument(
        "--rsvqa",
        action="store_true",
        help="Attempt to download RSVQA from Hugging Face"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DatasetDownloader(output_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("EarthGPT Dataset Downloader")
    print("=" * 80 + "\n")

    # Track what was downloaded
    downloaded = []

    # Download requested datasets
    if args.all or args.eurosat:
        if downloader.download_eurosat():
            downloaded.append("EuroSAT")

    if args.all or args.ucmerced:
        if downloader.download_ucmerced():
            downloaded.append("UC Merced")

    if args.all or args.rsicd:
        if downloader.download_rsicd_hf():
            downloaded.append("RSICD")

    if args.all or args.rsvqa:
        if downloader.download_rsvqa_hf():
            downloaded.append("RSVQA")

    # Always show DOTA instructions
    print("\n")
    downloader.print_dota_instructions()

    # Create summary
    downloader.create_dataset_summary()

    # Print summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)

    if downloaded:
        print(f"\nSuccessfully downloaded: {', '.join(downloaded)}")
    else:
        print("\nNo datasets were automatically downloaded.")

    print("\nFor datasets requiring manual download, see instructions above.")
    print("\nAlternatively, use synthetic data for testing:")
    print("  python scripts/generate_sample_data.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
