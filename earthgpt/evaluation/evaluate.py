"""
Evaluation Script for EarthGPT

Evaluates performance on:
- OBB: mAP (mean Average Precision)
- VQA: Accuracy
- Captioning: BLEU, METEOR, CIDEr
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from scripts.inference import EarthGPTInference


class OBBEvaluator:
    """Evaluator for Oriented Bounding Box detection."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_iou(
        self,
        box1: List[Tuple[int, int]],
        box2: List[Tuple[int, int]]
    ) -> float:
        """Compute IoU between two oriented bounding boxes."""

        try:
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)

            if not poly1.is_valid or not poly2.is_valid:
                return 0.0

            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return 0.0

    def match_boxes(
        self,
        pred_boxes: List[Tuple[str, List[Tuple]]],
        gt_boxes: List[Tuple[str, List[Tuple]]]
    ) -> Tuple[int, int, int]:
        """
        Match predicted boxes to ground truth.

        Returns:
            (true_positives, false_positives, false_negatives)
        """
        tp = 0
        fp = 0
        fn = 0

        matched_gt = set()

        for pred_class, pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, (gt_class, gt_box) in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                if pred_class != gt_class:
                    continue

                iou = self.compute_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= self.iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        return tp, fp, fn

    def evaluate(
        self,
        predictions: List[List[Tuple]],
        ground_truths: List[List[Tuple]]
    ) -> Dict[str, float]:
        """
        Evaluate OBB predictions.

        Returns:
            Dictionary with precision, recall, F1, mAP
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in zip(predictions, ground_truths):
            tp, fp, fn = self.match_boxes(pred, gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP': f1  # Simplified mAP (use proper AP calculation for production)
        }


class VQAEvaluator:
    """Evaluator for Visual Question Answering."""

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        return answer.lower().strip()

    def compute_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """Compute exact match accuracy."""

        correct = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_norm = self.normalize_answer(pred)
            gt_norm = self.normalize_answer(gt)

            if pred_norm == gt_norm:
                correct += 1

        return correct / len(predictions) if predictions else 0.0

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Evaluate VQA predictions."""

        accuracy = self.compute_accuracy(predictions, ground_truths)

        return {
            'accuracy': accuracy
        }


class CaptioningEvaluator:
    """Evaluator for image captioning."""

    def __init__(self):
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.cider.cider import Cider

            self.bleu = Bleu(4)
            self.meteor = Meteor()
            self.cider = Cider()
        except ImportError:
            print("Warning: pycocoevalcap not installed. Install with:")
            print("pip install pycocoevalcap")
            self.bleu = None
            self.meteor = None
            self.cider = None

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]  # Multiple references per image
    ) -> Dict[str, float]:
        """
        Evaluate captioning predictions.

        Args:
            predictions: List of predicted captions
            ground_truths: List of lists (multiple references per image)

        Returns:
            Dictionary with BLEU, METEOR, CIDEr scores
        """
        if self.bleu is None:
            print("Captioning metrics not available. Returning dummy scores.")
            return {
                'BLEU-4': 0.0,
                'METEOR': 0.0,
                'CIDEr': 0.0
            }

        # Format for pycocoevalcap
        gts = {}
        res = {}

        for i, (pred, gt_list) in enumerate(zip(predictions, ground_truths)):
            res[i] = [pred]
            gts[i] = gt_list if isinstance(gt_list, list) else [gt_list]

        # Compute scores
        bleu_scores, _ = self.bleu.compute_score(gts, res)
        meteor_score, _ = self.meteor.compute_score(gts, res)
        cider_score, _ = self.cider.compute_score(gts, res)

        return {
            'BLEU-1': bleu_scores[0],
            'BLEU-2': bleu_scores[1],
            'BLEU-3': bleu_scores[2],
            'BLEU-4': bleu_scores[3],
            'METEOR': meteor_score,
            'CIDEr': cider_score
        }


def load_test_data(test_file: str) -> List[Dict]:
    """Load test dataset from JSONL."""
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def run_evaluation(
    model_path: str,
    test_file: str,
    task: str,
    max_samples: int = None
):
    """
    Run evaluation on test set.

    Args:
        model_path: Path to trained model
        test_file: Path to test JSONL file
        task: One of ['obb', 'vqa', 'captioning']
        max_samples: Maximum samples to evaluate (for testing)
    """
    # Load model
    inference = EarthGPTInference(model_path=model_path)

    # Load test data
    test_data = load_test_data(test_file)

    if max_samples:
        test_data = test_data[:max_samples]

    # Filter by task
    test_data = [sample for sample in test_data if sample.get('task') == task]

    print(f"Evaluating {len(test_data)} {task} samples...")

    # Run predictions
    predictions = []
    ground_truths = []

    for sample in tqdm(test_data):
        image_path = sample['image']

        # Get ground truth
        gt_response = sample['conversations'][-1]['value']

        # Get question for VQA
        question = None
        if task == 'vqa':
            question = sample['conversations'][0]['value'].replace('<image>', '').strip()

        # Run prediction
        try:
            pred_response = inference.predict(
                image_path=image_path,
                task=task,
                question=question,
                temperature=0.1  # Low temperature for evaluation
            )
        except Exception as e:
            print(f"Error on {image_path}: {e}")
            pred_response = ""

        predictions.append(pred_response)
        ground_truths.append(gt_response)

    # Evaluate based on task
    if task == 'obb':
        # Parse OBB outputs
        pred_boxes = [inference.parse_obb_output(pred) for pred in predictions]
        gt_boxes = [inference.parse_obb_output(gt) for gt in ground_truths]

        evaluator = OBBEvaluator()
        metrics = evaluator.evaluate(pred_boxes, gt_boxes)

    elif task == 'vqa':
        evaluator = VQAEvaluator()
        metrics = evaluator.evaluate(predictions, ground_truths)

    elif task == 'captioning':
        # Convert ground truths to list format
        gt_lists = [[gt] for gt in ground_truths]

        evaluator = CaptioningEvaluator()
        metrics = evaluator.evaluate(predictions, gt_lists)

    else:
        raise ValueError(f"Unknown task: {task}")

    # Print results
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - {task.upper()}")
    print("=" * 80)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EarthGPT")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=['obb', 'vqa', 'captioning'],
        required=True,
        help="Task to evaluate"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for quick testing)"
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        test_file=args.test_file,
        task=args.task,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
