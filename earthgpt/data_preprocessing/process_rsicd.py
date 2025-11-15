"""
RSICD Dataset Preprocessing for Dense Image Captioning

RSICD Format:
- JSON file with structure: {"images": [{filename: ..., sentences: [{raw: ...}]}]}
- Multiple captions per image (typically 5)

Output Format (Conversational JSONL):
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe this satellite image in detail."},
        {"from": "gpt", "value": "A residential area with dense housing and a river running through it."}
    ],
    "task": "captioning"
}
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import shutil


class RSICDPreprocessor:
    """Preprocessor for RSICD dataset to conversational format."""

    # Instruction templates for captioning diversity
    INSTRUCTION_TEMPLATES = [
        "<image>\nDescribe this satellite image in detail.",
        "<image>\nWhat do you see in this image?",
        "<image>\nProvide a detailed description of this aerial view.",
        "<image>\nGenerate a caption for this remote sensing image.",
        "<image>\nWhat is shown in this satellite imagery?",
        "<image>\nDescribe the contents of this image.",
        "<image>\nWhat features are visible in this overhead image?",
    ]

    def __init__(
        self,
        rsicd_root: str,
        output_path: str,
        image_output_dir: str,
        split: str = "train",
        use_all_captions: bool = False
    ):
        """
        Args:
            rsicd_root: Path to RSICD dataset root
            output_path: Path to output JSONL file
            image_output_dir: Directory to copy/symlink images
            split: 'train', 'val', or 'test'
            use_all_captions: If True, create separate sample for each caption
        """
        self.rsicd_root = Path(rsicd_root)
        self.output_path = Path(output_path)
        self.image_output_dir = Path(image_output_dir)
        self.split = split
        self.use_all_captions = use_all_captions

        # Paths
        self.annotation_file = self.rsicd_root / "dataset_rsicd.json"
        self.image_dir = self.rsicd_root / "RSICD_images"

        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # Load split information
        self.split_file = self.rsicd_root / f"{split}_list.txt"

    def load_annotations(self) -> Dict:
        """Load RSICD annotations."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            data = json.load(f)

        return data

    def load_split_images(self) -> List[str]:
        """Load list of images for this split."""
        if not self.split_file.exists():
            print(f"Warning: Split file {self.split_file} not found. Using all images.")
            return None

        with open(self.split_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]

        return images

    def process(self, max_samples: int = None):
        """Process RSICD dataset and create conversational JSONL."""

        annotations = self.load_annotations()
        split_images = self.load_split_images()

        output_data = []

        print(f"Processing RSICD {self.split} split...")

        for img_data in tqdm(annotations['images']):
            filename = img_data.get('filename', '')

            # Check if image is in this split
            if split_images and filename not in split_images:
                continue

            # Get captions
            sentences = img_data.get('sentences', [])
            if not sentences:
                continue

            # Get image path
            source_image_path = self.image_dir / filename

            if not source_image_path.exists():
                continue

            # Copy/symlink image to output directory
            output_image_path = self.image_output_dir / filename

            if not output_image_path.exists():
                shutil.copy2(source_image_path, output_image_path)

            # Process captions
            captions = [sent['raw'] for sent in sentences if 'raw' in sent]

            if not captions:
                continue

            if self.use_all_captions:
                # Create separate sample for each caption
                for caption in captions:
                    instruction = random.choice(self.INSTRUCTION_TEMPLATES)

                    sample = {
                        "image": str(output_image_path),
                        "conversations": [
                            {"from": "human", "value": instruction},
                            {"from": "gpt", "value": caption}
                        ],
                        "task": "captioning"
                    }
                    output_data.append(sample)
            else:
                # Use random caption for each image
                caption = random.choice(captions)
                instruction = random.choice(self.INSTRUCTION_TEMPLATES)

                sample = {
                    "image": str(output_image_path),
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": caption}
                    ],
                    "task": "captioning"
                }
                output_data.append(sample)

            if max_samples and len(output_data) >= max_samples:
                break

        # Save to JSONL
        with open(self.output_path, 'w') as f:
            for sample in output_data:
                f.write(json.dumps(sample) + '\n')

        print(f"Saved {len(output_data)} samples to {self.output_path}")
        return output_data


def main():
    """Example usage."""

    # Process training set
    train_processor = RSICDPreprocessor(
        rsicd_root="/path/to/RSICD",
        output_path="./data/rsicd_train.jsonl",
        image_output_dir="./data/images/rsicd",
        split="train",
        use_all_captions=False  # Set to True for data augmentation
    )
    train_processor.process()

    # Process validation set
    val_processor = RSICDPreprocessor(
        rsicd_root="/path/to/RSICD",
        output_path="./data/rsicd_val.jsonl",
        image_output_dir="./data/images/rsicd",
        split="val",
        use_all_captions=False
    )
    val_processor.process()


if __name__ == "__main__":
    main()
