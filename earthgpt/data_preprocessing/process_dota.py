"""
DOTA Dataset Preprocessing for Oriented Bounding Box (OBB) Detection

DOTA Format:
- Annotations are in text files with format: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
- Images are high-resolution (often 4000x4000+)

Output Format (Conversational JSONL):
{
    "image": "path/to/image.png",
    "conversations": [
        {"from": "human", "value": "<image>\nDetect all objects and provide oriented bounding boxes."},
        {"from": "gpt", "value": "[plane, (120,340,190,342,188,401,118,399)] [ship, (50,50,100,50,100,100,50,100)]"}
    ],
    "task": "obb"
}
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import cv2
import numpy as np


class DOTAPreprocessor:
    """Preprocessor for DOTA dataset to conversational format."""

    # DOTA class names
    CLASSES = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

    # Instruction templates for diversity
    INSTRUCTION_TEMPLATES = [
        "<image>\nDetect all objects and provide oriented bounding boxes.",
        "<image>\nIdentify all objects with their precise locations.",
        "<image>\nWhat objects are visible? Provide bounding box coordinates.",
        "<image>\nLocalize all objects in this satellite image.",
        "<image>\nFind all objects and their orientations.",
    ]

    def __init__(
        self,
        dota_root: str,
        output_path: str,
        image_output_dir: str,
        split: str = "train",
        tile_size: int = 1024,
        overlap: int = 200
    ):
        """
        Args:
            dota_root: Path to DOTA dataset (contains 'images' and 'labelTxt-v1.5' folders)
            output_path: Path to output JSONL file
            image_output_dir: Directory to save cropped/tiled images
            split: 'train' or 'val'
            tile_size: Size to crop large images (DOTA images are huge)
            overlap: Overlap between tiles
        """
        self.dota_root = Path(dota_root)
        self.output_path = Path(output_path)
        self.image_output_dir = Path(image_output_dir)
        self.split = split
        self.tile_size = tile_size
        self.overlap = overlap

        self.image_dir = self.dota_root / split / "images"
        self.label_dir = self.dota_root / split / "labelTxt-v1.5"

        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def parse_annotation(self, ann_path: Path) -> List[Dict]:
        """Parse DOTA annotation file."""
        objects = []

        with open(ann_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                category = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 else 0

                objects.append({
                    'bbox': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                    'category': category,
                    'difficulty': difficulty
                })
            except (ValueError, IndexError):
                continue

        return objects

    def tile_image_and_annotations(
        self,
        image_path: Path,
        objects: List[Dict]
    ) -> List[Tuple[np.ndarray, List[Dict], str]]:
        """
        Tile large image into smaller patches with corresponding annotations.

        Returns:
            List of (image_tile, objects_in_tile, tile_name)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        h, w = image.shape[:2]
        tiles = []

        stride = self.tile_size - self.overlap

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)

                tile = image[y:y_end, x:x_end]

                # Adjust tile size if at boundary
                if tile.shape[0] < self.tile_size // 2 or tile.shape[1] < self.tile_size // 2:
                    continue

                # Find objects in this tile
                tile_objects = []
                for obj in objects:
                    bbox = np.array(obj['bbox'])

                    # Check if bbox center is in tile
                    center_x = bbox[:, 0].mean()
                    center_y = bbox[:, 1].mean()

                    if x <= center_x < x_end and y <= center_y < y_end:
                        # Adjust coordinates to tile reference
                        adjusted_bbox = bbox - np.array([x, y])

                        # Check if bbox is mostly within tile
                        if np.all(adjusted_bbox >= 0) and \
                           np.all(adjusted_bbox[:, 0] < tile.shape[1]) and \
                           np.all(adjusted_bbox[:, 1] < tile.shape[0]):

                            tile_objects.append({
                                'bbox': adjusted_bbox.tolist(),
                                'category': obj['category'],
                                'difficulty': obj['difficulty']
                            })

                # Only keep tiles with objects
                if tile_objects:
                    tile_name = f"{image_path.stem}_tile_{y}_{x}.png"
                    tiles.append((tile, tile_objects, tile_name))

        return tiles

    def format_obb_output(self, objects: List[Dict]) -> str:
        """Format OBB annotations as text string."""
        formatted = []

        for obj in objects:
            category = obj['category']
            bbox = obj['bbox']

            # Flatten coordinates: (x1,y1,x2,y2,x3,y3,x4,y4)
            coords = []
            for point in bbox:
                coords.extend([f"{int(point[0])}", f"{int(point[1])}"])

            coords_str = ",".join(coords)
            formatted.append(f"[{category}, ({coords_str})]")

        return " ".join(formatted)

    def process(self, max_samples: int = None):
        """Process DOTA dataset and create conversational JSONL."""

        image_files = sorted(self.image_dir.glob("*.png"))
        if max_samples:
            image_files = image_files[:max_samples]

        output_data = []

        print(f"Processing {len(image_files)} images from DOTA {self.split} split...")

        for image_path in tqdm(image_files):
            # Get annotation file
            ann_path = self.label_dir / f"{image_path.stem}.txt"

            if not ann_path.exists():
                continue

            # Parse annotations
            objects = self.parse_annotation(ann_path)

            if not objects:
                continue

            # Tile image and annotations
            tiles = self.tile_image_and_annotations(image_path, objects)

            for tile_img, tile_objects, tile_name in tiles:
                # Save tile image
                output_image_path = self.image_output_dir / tile_name
                cv2.imwrite(str(output_image_path), tile_img)

                # Format conversation
                instruction = random.choice(self.INSTRUCTION_TEMPLATES)
                response = self.format_obb_output(tile_objects)

                sample = {
                    "image": str(output_image_path),
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": response}
                    ],
                    "task": "obb"
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

    # Process training set
    train_processor = DOTAPreprocessor(
        dota_root="/path/to/DOTA-v1.5",
        output_path="./data/dota_train.jsonl",
        image_output_dir="./data/images/dota_train",
        split="train",
        tile_size=1024,
        overlap=200
    )
    train_processor.process()

    # Process validation set
    val_processor = DOTAPreprocessor(
        dota_root="/path/to/DOTA-v1.5",
        output_path="./data/dota_val.jsonl",
        image_output_dir="./data/images/dota_val",
        split="val",
        tile_size=1024,
        overlap=200
    )
    val_processor.process()


if __name__ == "__main__":
    main()
