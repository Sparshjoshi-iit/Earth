"""
Training Script for EarthGPT with LoRA Fine-Tuning

Features:
- Q-LoRA (4-bit quantization + LoRA)
- Multi-task training (OBB + VQA + Captioning)
- Efficient gradient checkpointing
- Mixed precision training (BF16)
- WandB logging
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict

import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.earthgpt_model import EarthGPT
from model.dataset import EarthGPTDataset, EarthGPTDataCollator


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class EarthGPTTrainer(Trainer):
    """Custom Trainer for EarthGPT."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for multi-task training.

        The model already computes cross-entropy loss internally.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def prepare_model(model_config: Dict, lora_config: Dict):
    """Initialize EarthGPT model with LoRA."""

    model = EarthGPT(
        vision_model_name=model_config['vision_encoder']['name'],
        llm_model_name=model_config['language_model']['name'],
        projector_hidden_dim=model_config['projector']['hidden_dim'],
        use_lora=True,
        lora_config=lora_config,
        load_in_4bit=True,
        device_map="auto"
    )

    return model


def prepare_datasets(
    train_data_path: str,
    val_data_path: str,
    tokenizer: AutoTokenizer,
    image_processor: AutoProcessor,
    max_length: int = 2048
):
    """Prepare train and validation datasets."""

    train_dataset = EarthGPTDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length
    )

    val_dataset = EarthGPTDataset(
        data_path=val_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length
    )

    return train_dataset, val_dataset


def main(args):
    """Main training function."""

    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    # Extract configs
    train_params = training_config['training']
    lora_params = model_config['lora']

    # Initialize wandb
    if train_params.get('report_to') == 'wandb':
        wandb.init(
            project="earthgpt",
            name=args.run_name,
            config={**model_config, **training_config}
        )

    # Initialize model
    print("=" * 80)
    print("Initializing EarthGPT model...")
    print("=" * 80)

    model = prepare_model(model_config['model'], lora_params)

    # Get tokenizer and processor from model
    tokenizer = model.tokenizer
    image_processor = model.vision_processor.image_processor

    # Prepare datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)

    train_dataset, val_dataset = prepare_datasets(
        train_data_path=train_params['data_path'],
        val_data_path=train_params['val_data_path'],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=2048
    )

    # Data collator
    data_collator = EarthGPTDataCollator(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_params['output_dir'],
        num_train_epochs=train_params['num_train_epochs'],
        per_device_train_batch_size=train_params['per_device_train_batch_size'],
        per_device_eval_batch_size=train_params['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_params['gradient_accumulation_steps'],
        learning_rate=train_params['learning_rate'],
        warmup_steps=train_params['warmup_steps'],
        lr_scheduler_type=train_params['lr_scheduler_type'],
        optim=train_params['optim'],
        weight_decay=train_params['weight_decay'],
        max_grad_norm=train_params['max_grad_norm'],
        fp16=train_params['fp16'],
        bf16=train_params['bf16'],
        logging_steps=train_params['logging_steps'],
        save_strategy=train_params['save_strategy'],
        save_steps=train_params['save_steps'],
        save_total_limit=train_params['save_total_limit'],
        evaluation_strategy=train_params['evaluation_strategy'],
        eval_steps=train_params['eval_steps'],
        dataloader_num_workers=train_params['dataloader_num_workers'],
        remove_unused_columns=False,
        report_to=train_params['report_to'],
        logging_dir=train_params['logging_dir'],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        seed=train_params['seed']
    )

    # Initialize Trainer
    trainer = EarthGPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)

    final_save_path = Path(train_params['output_dir']) / "final_model"
    trainer.save_model(final_save_path)

    # Save LoRA adapters separately
    if hasattr(model.llm, 'save_pretrained'):
        lora_save_path = Path(train_params['output_dir']) / "lora_adapters"
        model.llm.save_pretrained(lora_save_path)
        print(f"LoRA adapters saved to {lora_save_path}")

    # Save projector weights
    projector_save_path = Path(train_params['output_dir']) / "projector.pt"
    torch.save(model.projector.state_dict(), projector_save_path)
    print(f"Projector weights saved to {projector_save_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EarthGPT model")

    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration"
    )

    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="earthgpt-lora",
        help="Name for this training run"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    main(args)
