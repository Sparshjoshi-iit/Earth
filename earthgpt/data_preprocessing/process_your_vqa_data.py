"""
Preprocessor for Your VQA Data (RSVLM-QA format)

Handles rich VQA annotations with:
- Multiple question types (spatial, object, overall, quantity, count, presence, caption)
- Tags and relations
- Detailed answers
"""

import json
from pathlib import Path
from typing import List, Dict


class YourVQAPreprocessor:
    """Preprocessor for your VQA dataset."""

    def __init__(
        self,
        annotations_file: str,
        image_root: str,
        output_path: str,
        max_qa_pairs_per_image: int = None,
        question_types: List[str] = None
    ):
        """
        Args:
            annotations_file: Path to JSONL with VQA annotations
            image_root: Root directory for images
            output_path: Output JSONL path
            max_qa_pairs_per_image: Limit QA pairs per image (None = use all)
            question_types: Filter by question types (None = use all)
        """
        self.annotations_file = Path(annotations_file)
        self.image_root = Path(image_root)
        self.output_path = Path(output_path)
        self.max_qa_pairs_per_image = max_qa_pairs_per_image
        self.question_types = question_types

    def load_annotations(self) -> List[Dict]:
        """Load VQA annotations from JSONL."""
        annotations = []
        with open(self.annotations_file, 'r') as f:
            for line in f:
                annotations.append(json.loads(line.strip()))
        return annotations

    def should_include_qa(self, qa_pair: Dict) -> bool:
        """Check if QA pair should be included based on filters."""
        if self.question_types is None:
            return True
        return qa_pair.get('question_type') in self.question_types

    def process(self):
        """Process VQA annotations and create training samples."""

        annotations = self.load_annotations()
        samples = []

        print(f"Processing {len(annotations)} images with VQA annotations...")

        for annotation in annotations:
            image_path = annotation['image']
            vqa_pairs = annotation.get('vqa_pairs', [])

            # Full path to image
            full_image_path = self.image_root / image_path

            # Check if image exists
            if not full_image_path.exists():
                print(f"Warning: Image not found: {full_image_path}")
                continue

            # Filter and limit QA pairs
            filtered_pairs = [qa for qa in vqa_pairs if self.should_include_qa(qa)]

            if self.max_qa_pairs_per_image:
                filtered_pairs = filtered_pairs[:self.max_qa_pairs_per_image]

            # Create samples for each QA pair
            for qa_pair in filtered_pairs:
                question = qa_pair['question']
                answer = qa_pair['answer']
                question_type = qa_pair.get('question_type', 'unknown')

                # Add <image> token if not present
                if '<image>' not in question:
                    question = f"<image>\n{question}"

                sample = {
                    "image": str(full_image_path),
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer}
                    ],
                    "task": "vqa",
                    "metadata": {
                        "question_id": qa_pair.get('question_id'),
                        "question_type": question_type,
                        "image_id": annotation.get('id')
                    }
                }

                samples.append(sample)

        # Save to JSONL
        with open(self.output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Created {len(samples)} VQA samples")
        print(f"Saved to {self.output_path}")

        # Print question type distribution
        question_type_counts = {}
        for sample in samples:
            qtype = sample['metadata']['question_type']
            question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1

        print("\nQuestion Type Distribution:")
        for qtype, count in sorted(question_type_counts.items()):
            print(f"  {qtype}: {count}")


def main():
    """Example usage."""

    # Process all question types
    processor = YourVQAPreprocessor(
        annotations_file="./data/your_vqa_annotations.jsonl",
        image_root="./data",
        output_path="./data/your_vqa_train.jsonl",
        max_qa_pairs_per_image=None,  # Use all QA pairs
        question_types=None  # Use all question types
    )

    processor.process()

    # Or filter specific question types
    # processor = YourVQAPreprocessor(
    #     annotations_file="./data/your_vqa_annotations.jsonl",
    #     image_root="./data",
    #     output_path="./data/your_vqa_spatial_train.jsonl",
    #     question_types=['spatial', 'object', 'count']
    # )


if __name__ == "__main__":
    main()
