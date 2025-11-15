"""
Unified Dataset and DataCollator for EarthGPT Multi-Task Training.

Handles conversational format with image tokens for:
- Oriented Bounding Box Detection (OBB)
- Visual Question Answering (VQA)
- Dense Image Captioning
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor


class EarthGPTDataset(Dataset):
    """Unified dataset for multi-task geospatial VLM training."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        image_processor: AutoProcessor,
        image_token: str = "<image>",
        max_length: int = 2048,
    ):
        """
        Args:
            data_path: Path to JSONL file with conversational format
            tokenizer: Tokenizer for language model
            image_processor: Processor for vision encoder
            image_token: Special token for image placeholder
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = image_token
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'image': PIL.Image,
                'input_ids': torch.Tensor,
                'labels': torch.Tensor,
                'pixel_values': torch.Tensor
            }
        """
        sample = self.data[idx]

        # Load image
        try:
            image = Image.open(sample['image']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image']}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (384, 384), color='white')

        # Process image
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'][0]

        # Build conversation text
        conversations = sample['conversations']
        text = self._format_conversation(conversations)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        input_ids = encoding['input_ids']

        # Create labels (mask out user turns, only train on assistant responses)
        labels = self._create_labels(conversations, input_ids)

        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'task': sample.get('task', 'unknown')
        }

    def _format_conversation(self, conversations: List[Dict]) -> str:
        """
        Format conversation into text with special tokens.

        Example:
            <image>\nUser: Detect all ships.\nAssistant: [ship, (120,340,...)]
        """
        text = ""

        for turn in conversations:
            role = turn['from']
            content = turn['value']

            if role == 'human':
                text += f"{content}\n"
            elif role == 'gpt':
                text += f"{content}"

        return text

    def _create_labels(self, conversations: List[Dict], input_ids: List[int]) -> List[int]:
        """
        Create labels by masking user inputs (only train on assistant responses).

        Returns labels same length as input_ids, with -100 for tokens to ignore.
        """
        labels = [-100] * len(input_ids)

        # Convert conversations to text for each turn
        current_pos = 0

        for i, turn in enumerate(conversations):
            role = turn['from']
            content = turn['value']

            # Tokenize this turn
            turn_text = content
            if role == 'human':
                turn_text += "\n"

            turn_tokens = self.tokenizer(turn_text, add_special_tokens=False)['input_ids']
            turn_len = len(turn_tokens)

            # If this is assistant turn, unmask the labels
            if role == 'gpt':
                for j in range(current_pos, min(current_pos + turn_len, len(labels))):
                    if j < len(input_ids):
                        labels[j] = input_ids[j]

            current_pos += turn_len

        return labels


@dataclass
class EarthGPTDataCollator:
    """
    Data collator for batching EarthGPT samples.

    Handles:
    - Padding of input_ids and labels
    - Stacking of pixel_values
    """

    tokenizer: AutoTokenizer
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            features: List of dicts from dataset __getitem__

        Returns:
            Batched tensors ready for model forward pass
        """
        # Separate images and text
        pixel_values = torch.stack([f['pixel_values'] for f in features])

        # Prepare text features for padding
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]

        # Pad input_ids
        input_ids_padded = self._pad_sequence(
            input_ids,
            padding_value=self.tokenizer.pad_token_id
        )

        # Pad labels
        labels_padded = self._pad_sequence(
            labels,
            padding_value=-100
        )

        # Create attention mask
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded
        }

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        padding_value: int
    ) -> torch.Tensor:
        """Pad sequences to same length."""

        # Find max length
        if self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = max(len(seq) for seq in sequences)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad each sequence
        padded = []
        for seq in sequences:
            seq_len = len(seq)
            if seq_len < max_len:
                # Right padding
                padding = torch.full(
                    (max_len - seq_len,),
                    padding_value,
                    dtype=seq.dtype
                )
                padded_seq = torch.cat([seq, padding])
            else:
                # Truncate if too long
                padded_seq = seq[:max_len]

            padded.append(padded_seq)

        return torch.stack(padded)


def test_dataset():
    """Test dataset loading and collation."""
    from transformers import AutoTokenizer, AutoProcessor

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Add special tokens
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.add_tokens(['<image>'])

    # Create dataset
    dataset = EarthGPTDataset(
        data_path="./data/unified_train.jsonl",
        tokenizer=tokenizer,
        image_processor=image_processor.image_processor
    )

    # Test single sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample['pixel_values'].shape)
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Labels shape:", sample['labels'].shape)

    # Test collator
    collator = EarthGPTDataCollator(tokenizer=tokenizer)
    batch = collator([dataset[i] for i in range(4)])

    print("\nBatch keys:", batch.keys())
    print("Batch pixel_values shape:", batch['pixel_values'].shape)
    print("Batch input_ids shape:", batch['input_ids'].shape)
    print("Batch labels shape:", batch['labels'].shape)


if __name__ == "__main__":
    test_dataset()
