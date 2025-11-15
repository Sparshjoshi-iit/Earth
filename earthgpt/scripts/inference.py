"""
Inference Script for EarthGPT

Supports:
- Oriented Bounding Box Detection (OBB)
- Visual Question Answering (VQA)
- Dense Image Captioning

Usage:
    python inference.py --image path/to/image.png --task obb
    python inference.py --image path/to/image.png --task vqa --question "How many ships?"
    python inference.py --image path/to/image.png --task captioning
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import re

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.earthgpt_model import EarthGPT


class EarthGPTInference:
    """Inference wrapper for EarthGPT model."""

    # Task prompts
    TASK_PROMPTS = {
        'obb': "<image>\nDetect all objects and provide oriented bounding boxes.",
        'vqa': "<image>\n{question}",
        'captioning': "<image>\nDescribe this satellite image in detail."
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device

        print(f"Loading EarthGPT model from {model_path}...")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.image_processor = self.model.vision_processor.image_processor

        print("Model loaded successfully!")

    def _load_model(self, model_path: str) -> EarthGPT:
        """Load trained model with LoRA adapters."""

        model_path = Path(model_path)

        # Check if this is a full model or just LoRA adapters
        if (model_path / "lora_adapters").exists():
            # Load base model with LoRA adapters
            model = EarthGPT(
                vision_model_name="google/siglip-so400m-patch14-384",
                llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
                use_lora=False,  # Will load adapters separately
                load_in_4bit=True
            )

            # Load LoRA adapters
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(
                model.llm,
                model_path / "lora_adapters"
            )

            # Load projector weights
            projector_path = model_path / "projector.pt"
            if projector_path.exists():
                model.projector.load_state_dict(torch.load(projector_path))

        else:
            # Load full fine-tuned model
            model = EarthGPT.from_pretrained(model_path)

        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values']

        return pixel_values.to(self.device), image

    def prepare_prompt(self, task: str, question: Optional[str] = None) -> str:
        """Prepare prompt for specific task."""

        if task == 'vqa':
            if question is None:
                raise ValueError("Question required for VQA task")
            prompt = self.TASK_PROMPTS['vqa'].format(question=question)
        else:
            prompt = self.TASK_PROMPTS.get(task, self.TASK_PROMPTS['captioning'])

        return prompt

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        task: str = 'captioning',
        question: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Run inference on image.

        Args:
            image_path: Path to input image
            task: One of ['obb', 'vqa', 'captioning']
            question: Question for VQA task
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response
        """
        # Load and preprocess image
        pixel_values, _ = self.preprocess_image(image_path)

        # Prepare prompt
        prompt = self.prepare_prompt(task, question)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            pixel_values=pixel_values,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response (remove prompt)
        response = generated_text.replace(prompt, "").strip()

        return response

    def parse_obb_output(self, output: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Parse OBB output string into structured format.

        Input format: "[class, (x1,y1,x2,y2,x3,y3,x4,y4)] [class2, (...)]"

        Returns:
            List of (class_name, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
        """
        boxes = []

        # Regex to match [class, (coordinates)]
        pattern = r'\[([^,]+),\s*\(([^)]+)\)\]'

        matches = re.findall(pattern, output)

        for class_name, coords_str in matches:
            # Parse coordinates
            coords = [int(x.strip()) for x in coords_str.split(',')]

            if len(coords) == 8:
                # Convert to list of points
                points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7])
                ]

                boxes.append((class_name.strip(), points))

        return boxes

    def visualize_obb(
        self,
        image_path: str,
        output: str,
        save_path: Optional[str] = None
    ):
        """Visualize OBB predictions on image."""

        # Load image
        image = Image.open(image_path)

        # Parse OBB output
        boxes = self.parse_obb_output(output)

        # Create plot
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)

        # Draw each box
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, (class_name, points) in enumerate(boxes):
            color = colors[i % len(colors)]

            # Create polygon
            polygon = patches.Polygon(
                points,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(polygon)

            # Add label
            x, y = points[0]
            ax.text(
                x, y - 10,
                class_name,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7)
            )

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description="EarthGPT Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=['obb', 'vqa', 'captioning'],
        default='captioning',
        help="Task to perform"
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for VQA task"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization (for OBB task)"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Initialize inference
    inference = EarthGPTInference(model_path=args.model_path)

    # Run prediction
    print(f"\nRunning {args.task} inference on {args.image}...")

    response = inference.predict(
        image_path=args.image,
        task=args.task,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Visualize OBB if applicable
    if args.task == 'obb':
        output_path = args.output or f"{Path(args.image).stem}_obb_result.png"
        inference.visualize_obb(
            image_path=args.image,
            output=response,
            save_path=output_path
        )


if __name__ == "__main__":
    main()
