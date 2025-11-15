"""
RSVQA Dataset Preprocessing for Visual Question Answering

RSVQA Format:
- JSON files with structure: {"questions": [...], "answers": [...], "images": [...]}
- Two variants: LR (Low Resolution) and HR (High Resolution)
- Question types: presence, count, area, comparison

Output Format (Conversational JSONL):
{
    "image": "path/to/image.tif",
    "conversations": [
        {"from": "human", "value": "<image>\nHow many ships are in the harbor?"},
        {"from": "gpt", "value": "3"}
    ],
    "task": "vqa"
}
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import shutil


class RSVQAPreprocessor:
    """Preprocessor for RSVQA dataset to conversational format."""

    def __init__(
        self,
        rsvqa_root: str,
        output_path: str,
        image_output_dir: str,
        split: str = "train",
        variant: str = "LR"  # "LR" or "HR"
    ):
        """
        Args:
            rsvqa_root: Path to RSVQA dataset root
            output_path: Path to output JSONL file
            image_output_dir: Directory to copy/symlink images
            split: 'train', 'val', or 'test'
            variant: 'LR' (Low Resolution) or 'HR' (High Resolution)
        """
        self.rsvqa_root = Path(rsvqa_root)
        self.output_path = Path(output_path)
        self.image_output_dir = Path(image_output_dir)
        self.split = split
        self.variant = variant

        # Paths
        if variant == "LR":
            self.qa_file = self.rsvqa_root / f"LR/LR_{split}.json"
            self.image_dir = self.rsvqa_root / "LR/Images"
        else:  # HR
            self.qa_file = self.rsvqa_root / f"HR/HR_{split}.json"
            self.image_dir = self.rsvqa_root / "HR/Images"

        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def load_qa_pairs(self) -> List[Dict]:
        """Load question-answer pairs from JSON."""
        if not self.qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {self.qa_file}")

        with open(self.qa_file, 'r') as f:
            data = json.load(f)

        qa_pairs = []

        # RSVQA format varies, handle both possible structures
        if isinstance(data, list):
            # Format: [{image: ..., question: ..., answer: ...}, ...]
            qa_pairs = data
        elif isinstance(data, dict):
            # Format: {questions: [...], answers: [...], images: [...]}
            if 'questions' in data and 'answers' in data:
                images = data.get('images', [None] * len(data['questions']))

                for i in range(len(data['questions'])):
                    qa_pairs.append({
                        'image': images[i] if i < len(images) else None,
                        'question': data['questions'][i],
                        'answer': data['answers'][i]
                    })

        return qa_pairs

    def process(self, max_samples: int = None):
        """Process RSVQA dataset and create conversational JSONL."""

        qa_pairs = self.load_qa_pairs()

        if max_samples:
            qa_pairs = qa_pairs[:max_samples]

        output_data = []

        print(f"Processing {len(qa_pairs)} QA pairs from RSVQA {self.variant} {self.split}...")

        for qa in tqdm(qa_pairs):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            image_name = qa.get('image', '')

            if not all([question, answer, image_name]):
                continue

            # Handle image path
            source_image_path = self.image_dir / image_name

            if not source_image_path.exists():
                # Try with different extensions
                for ext in ['.tif', '.png', '.jpg']:
                    alt_path = self.image_dir / f"{Path(image_name).stem}{ext}"
                    if alt_path.exists():
                        source_image_path = alt_path
                        break

            if not source_image_path.exists():
                continue

            # Copy/symlink image to output directory
            output_image_path = self.image_output_dir / source_image_path.name

            if not output_image_path.exists():
                shutil.copy2(source_image_path, output_image_path)

            # Format conversation
            # Add <image> token to question
            if "<image>" not in question:
                question = f"<image>\n{question}"

            sample = {
                "image": str(output_image_path),
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": str(answer)}
                ],
                "task": "vqa"
            }

            output_data.append(sample)

        # Save to JSONL
        with open(self.output_path, 'w') as f:
            for sample in output_data:
                f.write(json.dumps(sample) + '\n')

        print(f"Saved {len(output_data)} samples to {self.output_path}")
        return output_data


def main():
    """Example usage."""

    # Process LR training set
    train_processor = RSVQAPreprocessor(
        rsvqa_root="/path/to/RSVQA",
        output_path="./data/rsvqa_lr_train.jsonl",
        image_output_dir="./data/images/rsvqa_lr",
        split="train",
        variant="LR"
    )
    train_processor.process()

    # Process LR validation set
    val_processor = RSVQAPreprocessor(
        rsvqa_root="/path/to/RSVQA",
        output_path="./data/rsvqa_lr_val.jsonl",
        image_output_dir="./data/images/rsvqa_lr",
        split="val",
        variant="LR"
    )
    val_processor.process()

    # Optional: Process HR variant
    # hr_processor = RSVQAPreprocessor(
    #     rsvqa_root="/path/to/RSVQA",
    #     output_path="./data/rsvqa_hr_train.jsonl",
    #     image_output_dir="./data/images/rsvqa_hr",
    #     split="train",
    #     variant="HR"
    # )
    # hr_processor.process()


if __name__ == "__main__":
    main()
