"""
Preprocessor for GSD Estimation Data

Converts your GSD metadata to conversational format where the model learns to:
1. Estimate GSD from visual appearance
2. Use GSD for measurement tasks
"""

import json
import random
from pathlib import Path
from typing import Dict, List


class GSDPreprocessor:
    """Preprocessor for GSD estimation task."""

    # Instruction templates for GSD estimation
    GSD_ESTIMATION_TEMPLATES = [
        "<image>\nWhat is the ground sample distance (GSD) of this image in meters per pixel?",
        "<image>\nEstimate the spatial resolution (GSD) of this satellite image.",
        "<image>\nWhat is the resolution of this image in meters/pixel?",
        "<image>\nDetermine the GSD value for this aerial imagery.",
    ]

    # Templates for GSD-aware measurements
    MEASUREMENT_TEMPLATES = [
        "<image>\nGSD: {gsd:.4f}m/pixel. Calculate the area of the object at coordinates {coords}.",
        "<image>\nGiven GSD of {gsd:.4f}m/pixel, what is the length of the building from {p1} to {p2}?",
        "<image>\nWith resolution {gsd:.4f}m/pixel, estimate the size of this structure.",
    ]

    def __init__(
        self,
        gsd_metadata_path: str,
        image_dir: str,
        output_path: str,
        include_estimation: bool = True,
        include_measurements: bool = False
    ):
        """
        Args:
            gsd_metadata_path: Path to JSON with GSD metadata
            image_dir: Directory containing images
            output_path: Output JSONL path
            include_estimation: Include GSD estimation samples
            include_measurements: Include measurement tasks (requires OBB data)
        """
        self.gsd_metadata_path = Path(gsd_metadata_path)
        self.image_dir = Path(image_dir)
        self.output_path = Path(output_path)
        self.include_estimation = include_estimation
        self.include_measurements = include_measurements

    def load_gsd_metadata(self) -> Dict:
        """Load GSD metadata from JSON."""
        with open(self.gsd_metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def format_gsd_value(self, gsd: float) -> str:
        """Format GSD value for model output."""
        # Round to 4 decimal places
        return f"{gsd:.4f}"

    def create_gsd_estimation_sample(
        self,
        image_name: str,
        metadata: Dict
    ) -> Dict:
        """Create a sample for GSD estimation."""

        image_path = self.image_dir / image_name
        gsd = metadata['gsd']

        instruction = random.choice(self.GSD_ESTIMATION_TEMPLATES)

        # Format response with units
        response = f"The ground sample distance (GSD) is {self.format_gsd_value(gsd)} meters per pixel."

        return {
            "image": str(image_path),
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": response}
            ],
            "task": "gsd_estimation",
            "metadata": {
                "gsd": gsd,
                "width": metadata['width'],
                "height": metadata['height']
            }
        }

    def process(self):
        """Process GSD metadata and create training samples."""

        metadata = self.load_gsd_metadata()
        samples = []

        print(f"Processing {len(metadata)} images with GSD metadata...")

        for image_name, img_metadata in metadata.items():
            # Check if image exists
            image_path = self.image_dir / image_name
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # Create GSD estimation sample
            if self.include_estimation:
                sample = self.create_gsd_estimation_sample(image_name, img_metadata)
                samples.append(sample)

        # Save to JSONL
        with open(self.output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Created {len(samples)} samples")
        print(f"Saved to {self.output_path}")

        # Print GSD statistics
        gsds = [metadata[img]['gsd'] for img in metadata]
        print(f"\nGSD Statistics:")
        print(f"  Min: {min(gsds):.4f} m/pixel")
        print(f"  Max: {max(gsds):.4f} m/pixel")
        print(f"  Mean: {sum(gsds)/len(gsds):.4f} m/pixel")


def main():
    """Example usage."""

    processor = GSDPreprocessor(
        gsd_metadata_path="./data/your_gsd_metadata.json",
        image_dir="./data/your_images",
        output_path="./data/gsd_estimation_train.jsonl",
        include_estimation=True
    )

    processor.process()


if __name__ == "__main__":
    main()
