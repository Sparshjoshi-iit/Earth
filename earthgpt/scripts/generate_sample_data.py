"""
Sample Data Generator for EarthGPT

Creates synthetic satellite-like images with annotations for:
- Oriented Bounding Box Detection (OBB)
- Visual Question Answering (VQA)
- Dense Image Captioning

This allows testing the complete pipeline without downloading large datasets.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2


class SampleDataGenerator:
    """Generate synthetic geospatial data for EarthGPT."""

    # Object classes for OBB
    OBB_CLASSES = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'vehicle', 'building'
    ]

    # Colors for different object types
    COLORS = {
        'plane': (200, 200, 200),
        'ship': (100, 100, 150),
        'storage-tank': (180, 180, 180),
        'baseball-diamond': (150, 200, 150),
        'tennis-court': (100, 150, 100),
        'basketball-court': (120, 120, 100),
        'ground-track-field': (100, 180, 100),
        'harbor': (80, 100, 120),
        'bridge': (160, 160, 160),
        'vehicle': (150, 150, 150),
        'building': (180, 150, 140)
    }

    # Captioning templates
    CAPTION_TEMPLATES = [
        "An aerial view of a {location} area with {count} {objects}.",
        "This satellite image shows {objects} in a {location} setting with {feature}.",
        "A {location} scene featuring {objects} and {feature}.",
        "{count} {objects} are visible in this {location} area near {feature}.",
        "Overhead imagery of {objects} in {location} with {feature} nearby."
    ]

    LOCATIONS = ['urban', 'suburban', 'rural', 'coastal', 'industrial', 'residential']
    FEATURES = ['a river', 'dense vegetation', 'road networks', 'parking areas', 'open fields']

    def __init__(self, output_dir: str, image_size: int = 512):
        """
        Args:
            output_dir: Directory to save generated data
            image_size: Size of generated images
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size

        # Create directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)

    def generate_background(self) -> np.ndarray:
        """Generate realistic satellite-like background."""

        # Random terrain type
        terrain_type = random.choice(['urban', 'vegetation', 'water', 'mixed'])

        if terrain_type == 'urban':
            # Gray/brown tones
            base_color = np.random.randint(100, 150, size=3)
        elif terrain_type == 'vegetation':
            # Green tones
            base_color = np.array([80, 120, 60])
        elif terrain_type == 'water':
            # Blue tones
            base_color = np.array([60, 80, 120])
        else:
            # Mixed
            base_color = np.random.randint(80, 130, size=3)

        # Create base image
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * base_color

        # Add texture/noise
        noise = np.random.randint(-20, 20, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some random roads/features
        if random.random() > 0.5:
            # Add roads
            num_roads = random.randint(1, 3)
            for _ in range(num_roads):
                if random.random() > 0.5:
                    # Horizontal road
                    y = random.randint(0, self.image_size)
                    thickness = random.randint(3, 8)
                    cv2.line(image, (0, y), (self.image_size, y),
                            (140, 140, 140), thickness)
                else:
                    # Vertical road
                    x = random.randint(0, self.image_size)
                    thickness = random.randint(3, 8)
                    cv2.line(image, (x, 0), (x, self.image_size),
                            (140, 140, 140), thickness)

        return image

    def draw_rotated_rectangle(
        self,
        image: np.ndarray,
        center: tuple,
        size: tuple,
        angle: float,
        color: tuple
    ) -> list:
        """
        Draw a rotated rectangle and return its corner coordinates.

        Returns:
            List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        cx, cy = center
        w, h = size

        # Calculate corners before rotation
        corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])

        # Rotation matrix
        angle_rad = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Rotate and translate
        rotated_corners = corners @ rot_matrix.T
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy

        # Draw
        points = rotated_corners.astype(np.int32)
        cv2.fillPoly(image, [points], color)

        # Add some detail
        cv2.polylines(image, [points], True,
                     tuple(max(0, c-30) for c in color), 2)

        return [(int(x), int(y)) for x, y in rotated_corners]

    def generate_obb_sample(self, sample_id: int) -> dict:
        """Generate OBB detection sample."""

        # Create background
        image = self.generate_background()

        # Random number of objects
        num_objects = random.randint(2, 8)

        objects = []
        bboxes = []

        for _ in range(num_objects):
            # Random object class
            obj_class = random.choice(self.OBB_CLASSES)
            color = self.COLORS[obj_class]

            # Random position
            cx = random.randint(50, self.image_size - 50)
            cy = random.randint(50, self.image_size - 50)

            # Random size based on class
            if obj_class in ['plane', 'ship']:
                w = random.randint(30, 60)
                h = random.randint(15, 30)
            elif obj_class in ['vehicle']:
                w = random.randint(8, 15)
                h = random.randint(5, 10)
            elif obj_class in ['building']:
                w = random.randint(40, 80)
                h = random.randint(40, 80)
            else:
                w = random.randint(20, 50)
                h = random.randint(20, 50)

            # Random rotation
            angle = random.randint(0, 180)

            # Draw object
            corners = self.draw_rotated_rectangle(
                image, (cx, cy), (w, h), angle, color
            )

            bboxes.append((obj_class, corners))

        # Save image
        image_path = self.output_dir / 'images' / f'obb_sample_{sample_id:04d}.png'
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Format OBB string
        obb_string = ' '.join([
            f"[{cls}, ({','.join([str(coord) for point in corners for coord in point])})]"
            for cls, corners in bboxes
        ])

        instruction = random.choice([
            "<image>\nDetect all objects and provide oriented bounding boxes.",
            "<image>\nIdentify all objects with their precise locations.",
            "<image>\nWhat objects are visible? Provide bounding box coordinates."
        ])

        return {
            'image': str(image_path),
            'conversations': [
                {'from': 'human', 'value': instruction},
                {'from': 'gpt', 'value': obb_string}
            ],
            'task': 'obb'
        }

    def generate_vqa_sample(self, sample_id: int) -> dict:
        """Generate VQA sample."""

        # Create background
        image = self.generate_background()

        # Add some objects for questions
        num_objects = random.randint(1, 6)
        obj_class = random.choice(self.OBB_CLASSES)
        color = self.COLORS[obj_class]

        for _ in range(num_objects):
            cx = random.randint(50, self.image_size - 50)
            cy = random.randint(50, self.image_size - 50)
            w = random.randint(20, 40)
            h = random.randint(20, 40)
            angle = random.randint(0, 180)

            self.draw_rotated_rectangle(image, (cx, cy), (w, h), angle, color)

        # Save image
        image_path = self.output_dir / 'images' / f'vqa_sample_{sample_id:04d}.png'
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Generate Q&A
        question_type = random.choice(['count', 'presence', 'color'])

        if question_type == 'count':
            question = f"How many {obj_class}s are in this image?"
            answer = str(num_objects)
        elif question_type == 'presence':
            question = f"Are there any {obj_class}s visible?"
            answer = "yes" if num_objects > 0 else "no"
        else:
            question = "What is the dominant color in this image?"
            answer = random.choice(['gray', 'green', 'blue', 'brown'])

        return {
            'image': str(image_path),
            'conversations': [
                {'from': 'human', 'value': f'<image>\n{question}'},
                {'from': 'gpt', 'value': answer}
            ],
            'task': 'vqa'
        }

    def generate_captioning_sample(self, sample_id: int) -> dict:
        """Generate captioning sample."""

        # Create background
        image = self.generate_background()

        # Add multiple object types
        object_types = random.sample(self.OBB_CLASSES, k=random.randint(2, 4))
        total_objects = 0

        for obj_class in object_types:
            num = random.randint(1, 4)
            total_objects += num
            color = self.COLORS[obj_class]

            for _ in range(num):
                cx = random.randint(50, self.image_size - 50)
                cy = random.randint(50, self.image_size - 50)
                w = random.randint(15, 40)
                h = random.randint(15, 40)
                angle = random.randint(0, 180)

                self.draw_rotated_rectangle(image, (cx, cy), (w, h), angle, color)

        # Save image
        image_path = self.output_dir / 'images' / f'caption_sample_{sample_id:04d}.png'
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Generate caption
        location = random.choice(self.LOCATIONS)
        feature = random.choice(self.FEATURES)
        objects = ' and '.join(object_types[:2]) + 's'

        template = random.choice(self.CAPTION_TEMPLATES)
        caption = template.format(
            location=location,
            count=total_objects,
            objects=objects,
            feature=feature
        )

        instruction = random.choice([
            "<image>\nDescribe this satellite image in detail.",
            "<image>\nWhat do you see in this image?",
            "<image>\nGenerate a caption for this remote sensing image."
        ])

        return {
            'image': str(image_path),
            'conversations': [
                {'from': 'human', 'value': instruction},
                {'from': 'gpt', 'value': caption}
            ],
            'task': 'captioning'
        }

    def generate_dataset(
        self,
        num_samples_per_task: int = 100,
        split: str = 'train'
    ):
        """
        Generate complete dataset.

        Args:
            num_samples_per_task: Number of samples to generate for each task
            split: 'train' or 'val'
        """

        print(f"Generating {split} dataset with {num_samples_per_task} samples per task...")

        all_samples = []

        # Generate OBB samples
        print("Generating OBB samples...")
        for i in range(num_samples_per_task):
            sample = self.generate_obb_sample(i)
            all_samples.append(sample)

        # Generate VQA samples
        print("Generating VQA samples...")
        for i in range(num_samples_per_task):
            sample = self.generate_vqa_sample(i)
            all_samples.append(sample)

        # Generate captioning samples
        print("Generating captioning samples...")
        for i in range(num_samples_per_task):
            sample = self.generate_captioning_sample(i)
            all_samples.append(sample)

        # Shuffle
        random.shuffle(all_samples)

        # Save to JSONL
        output_file = self.output_dir / f'synthetic_{split}.jsonl'
        with open(output_file, 'w') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Generated {len(all_samples)} samples")
        print(f"Saved to {output_file}")

        # Print statistics
        task_counts = {}
        for sample in all_samples:
            task = sample['task']
            task_counts[task] = task_counts.get(task, 0) + 1

        print("\nTask distribution:")
        for task, count in task_counts.items():
            print(f"  {task}: {count}")


def main():
    """Generate sample datasets for training and validation."""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create generator
    generator = SampleDataGenerator(output_dir='./data/synthetic')

    # Generate training set (300 samples total: 100 per task)
    generator.generate_dataset(num_samples_per_task=100, split='train')

    # Generate validation set (60 samples total: 20 per task)
    generator.generate_dataset(num_samples_per_task=20, split='val')

    print("\n" + "=" * 80)
    print("Sample dataset generation complete!")
    print("=" * 80)
    print("\nYou can now test the training pipeline with:")
    print("  python training/train.py --training_config configs/training_config_synthetic.yaml")


if __name__ == "__main__":
    main()
