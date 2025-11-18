"""
Preprocessor for Your OBB Data with GSD-Aware Measurements

Handles OBB annotations and creates:
1. Standard OBB detection tasks
2. GSD-aware measurement tasks (area, length, perimeter)
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple


class YourOBBPreprocessor:
    """Preprocessor for OBB data with optional GSD integration."""

    OBB_TEMPLATES = [
        "<image>\nDetect all objects and provide oriented bounding boxes.",
        "<image>\nIdentify all objects with their precise locations.",
        "<image>\nLocalize all objects in this image.",
    ]

    GSD_AWARE_TEMPLATES = [
        "<image>\nGSD: {gsd:.4f}m/pixel. Detect all {class_name} and calculate their areas.",
        "<image>\nResolution: {gsd:.4f}m/pixel. Find all {class_name} with size information.",
    ]

    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        gsd_metadata_path: str,
        output_path: str,
        include_gsd_measurements: bool = True
    ):
        """
        Args:
            annotations_file: Path to OBB annotations (your format)
            image_dir: Directory with images
            gsd_metadata_path: Path to GSD metadata JSON
            output_path: Output JSONL path
            include_gsd_measurements: Include area/length calculations
        """
        self.annotations_file = Path(annotations_file)
        self.image_dir = Path(image_dir)
        self.gsd_metadata_path = Path(gsd_metadata_path)
        self.output_path = Path(output_path)
        self.include_gsd_measurements = include_gsd_measurements

        # Load GSD metadata
        if gsd_metadata_path:
            with open(gsd_metadata_path, 'r') as f:
                self.gsd_metadata = json.load(f)
        else:
            self.gsd_metadata = {}

    def calculate_obb_area(
        self,
        points: List[Tuple[int, int]],
        gsd: float
    ) -> float:
        """
        Calculate area of oriented bounding box.

        Args:
            points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            gsd: Ground sample distance in m/pixel

        Returns:
            Area in square meters
        """
        # Shoelace formula for polygon area
        n = len(points)
        area_pixels = 0.0
        for i in range(n):
            j = (i + 1) % n
            area_pixels += points[i][0] * points[j][1]
            area_pixels -= points[j][0] * points[i][1]
        area_pixels = abs(area_pixels) / 2.0

        # Convert to square meters
        area_m2 = area_pixels * (gsd ** 2)
        return area_m2

    def calculate_obb_dimensions(
        self,
        points: List[Tuple[int, int]],
        gsd: float
    ) -> Tuple[float, float]:
        """
        Calculate width and height of OBB.

        Returns:
            (width_m, height_m) in meters
        """
        # Calculate distances between adjacent points
        def distance(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        d1 = distance(points[0], points[1]) * gsd
        d2 = distance(points[1], points[2]) * gsd

        return d1, d2

    def format_obb_string(self, objects: List[Dict]) -> str:
        """Format OBB annotations as output string."""
        formatted = []
        for obj in objects:
            class_name = obj['class']
            points = obj['points']  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

            # Flatten coordinates
            coords = []
            for point in points:
                coords.extend([str(int(point[0])), str(int(point[1]))])

            coords_str = ",".join(coords)
            formatted.append(f"[{class_name}, ({coords_str})]")

        return " ".join(formatted)

    def format_obb_with_measurements(
        self,
        objects: List[Dict],
        gsd: float
    ) -> str:
        """Format OBB with area measurements."""
        formatted = []
        for obj in objects:
            class_name = obj['class']
            points = obj['points']

            # Calculate measurements
            area = self.calculate_obb_area(points, gsd)
            width, height = self.calculate_obb_dimensions(points, gsd)

            # Flatten coordinates
            coords = []
            for point in points:
                coords.extend([str(int(point[0])), str(int(point[1]))])
            coords_str = ",".join(coords)

            # Format with measurements
            formatted.append(
                f"[{class_name}, ({coords_str}), area: {area:.2f}m², "
                f"dimensions: {width:.2f}m × {height:.2f}m]"
            )

        return " ".join(formatted)

    def load_annotations(self) -> Dict:
        """
        Load your OBB annotations.

        Adapt this to your specific annotation format!
        Expected format after loading:
        {
            "image_name.png": [
                {
                    "class": "building",
                    "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                },
                ...
            ]
        }
        """
        # TODO: Adapt this to your annotation format
        # This is a placeholder - modify based on your data structure

        with open(self.annotations_file, 'r') as f:
            # If your format is JSON
            annotations = json.load(f)
            # Or if it's JSONL
            # annotations = {}
            # for line in f:
            #     data = json.loads(line)
            #     annotations[data['image']] = data['objects']

        return annotations

    def process(self):
        """Process OBB annotations and create training samples."""

        annotations = self.load_annotations()
        samples = []

        print(f"Processing OBB annotations for {len(annotations)} images...")

        for image_name, objects in annotations.items():
            image_path = self.image_dir / image_name

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # Get GSD for this image
            gsd = self.gsd_metadata.get(image_name, {}).get('gsd')

            # Create standard OBB sample
            import random
            instruction = random.choice(self.OBB_TEMPLATES)
            response = self.format_obb_string(objects)

            sample = {
                "image": str(image_path),
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": response}
                ],
                "task": "obb"
            }
            samples.append(sample)

            # Create GSD-aware measurement sample if GSD available
            if self.include_gsd_measurements and gsd:
                # Get unique classes
                classes = list(set(obj['class'] for obj in objects))

                for class_name in classes:
                    class_objects = [obj for obj in objects if obj['class'] == class_name]

                    instruction = f"<image>\nGSD: {gsd:.4f}m/pixel. Detect all {class_name} and calculate their areas."
                    response = self.format_obb_with_measurements(class_objects, gsd)

                    sample = {
                        "image": str(image_path),
                        "conversations": [
                            {"from": "human", "value": instruction},
                            {"from": "gpt", "value": response}
                        ],
                        "task": "obb_measurement",
                        "metadata": {"gsd": gsd}
                    }
                    samples.append(sample)

        # Save to JSONL
        with open(self.output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Created {len(samples)} samples")
        print(f"Saved to {self.output_path}")


def main():
    """Example usage."""

    processor = YourOBBPreprocessor(
        annotations_file="./data/your_obb_annotations.json",
        image_dir="./data/your_images",
        gsd_metadata_path="./data/your_gsd_metadata.json",
        output_path="./data/your_obb_train.jsonl",
        include_gsd_measurements=True
    )

    processor.process()


if __name__ == "__main__":
    main()
