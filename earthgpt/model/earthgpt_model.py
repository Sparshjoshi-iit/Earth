"""
EarthGPT: Multimodal Vision-Language Model for Geospatial Intelligence

Architecture:
    Vision Encoder (SigLIP-ViT) → Projector (MLP) → Language Model (Llama-3.2-3B)

Tasks:
    - Oriented Bounding Box Detection (OBB)
    - Visual Question Answering (VQA)
    - Dense Image Captioning
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class VisionProjector(nn.Module):
    """MLP projector to map vision features to LLM embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        # Final projection to LLM dimension
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)

        Returns:
            projected_features: (batch_size, num_patches, llm_dim)
        """
        return self.projector(vision_features)


class EarthGPT(nn.Module):
    """
    EarthGPT: Multimodal VLM for geospatial imagery.

    Architecture:
        1. Vision Encoder: SigLIP-ViT extracts image features
        2. Projector: MLP projects vision features to LLM space
        3. Language Model: Llama-3.2-3B generates text outputs
    """

    def __init__(
        self,
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        projector_hidden_dim: int = 3072,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        super().__init__()

        self.vision_model_name = vision_model_name
        self.llm_model_name = llm_model_name

        # Load vision encoder (SigLIP)
        print(f"Loading vision encoder: {vision_model_name}")
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_name)
        self.vision_encoder = self._load_vision_encoder(vision_model_name)

        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.vision_encoder.eval()

        # Get dimensions
        vision_hidden_size = self.vision_encoder.config.hidden_size  # 1152 for SigLIP-SO400M
        llm_hidden_size = 3072  # Llama-3.2-3B

        # Create projector
        print("Creating vision-language projector...")
        self.projector = VisionProjector(
            input_dim=vision_hidden_size,
            hidden_dim=projector_hidden_dim,
            output_dim=llm_hidden_size,
            num_layers=2
        )

        # Load LLM with quantization
        print(f"Loading language model: {llm_model_name}")

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Add special tokens
        special_tokens = {'pad_token': '<pad>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.add_tokens(['<image>'])

        # Resize token embeddings
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if specified
        if use_lora:
            print("Applying LoRA to language model...")
            self.llm = self._apply_lora(self.llm, lora_config or {})

        # Get special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')

        print("EarthGPT model initialized successfully!")

    def _load_vision_encoder(self, model_name: str):
        """Load and configure vision encoder."""
        from transformers import AutoModel

        vision_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        return vision_model

    def _apply_lora(self, model, lora_params: Dict):
        """Apply LoRA to the language model."""

        # Prepare model for k-bit training if using quantization
        model = prepare_model_for_kbit_training(model)

        # Default LoRA config
        lora_config = LoraConfig(
            r=lora_params.get('r', 64),
            lora_alpha=lora_params.get('lora_alpha', 128),
            target_modules=lora_params.get('target_modules', [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_params.get('lora_dropout', 0.05),
            bias=lora_params.get('bias', 'none'),
            task_type=lora_params.get('task_type', 'CAUSAL_LM')
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder and projector.

        Args:
            pixel_values: (batch_size, channels, height, width)

        Returns:
            image_embeds: (batch_size, num_patches, llm_hidden_size)
        """
        with torch.no_grad():
            # Get vision features
            vision_outputs = self.vision_encoder.vision_model(pixel_values)
            vision_features = vision_outputs.last_hidden_state  # (B, num_patches, vision_dim)

        # Project to LLM space
        image_embeds = self.projector(vision_features)  # (B, num_patches, llm_dim)

        return image_embeds

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare input embeddings by replacing <image> tokens with vision features.

        Args:
            input_ids: (batch_size, seq_len)
            pixel_values: (batch_size, channels, height, width)
            attention_mask: (batch_size, seq_len)

        Returns:
            inputs_embeds: (batch_size, new_seq_len, hidden_size)
            attention_mask: (batch_size, new_seq_len) or None
        """
        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # If no images, return text embeddings directly
        if pixel_values is None:
            return inputs_embeds, attention_mask

        # Encode images
        image_embeds = self.encode_images(pixel_values)  # (B, num_patches, hidden_size)
        num_image_patches = image_embeds.shape[1]

        # Replace <image> tokens with image embeddings
        batch_size = input_ids.shape[0]
        new_inputs_embeds = []
        new_attention_masks = [] if attention_mask is not None else None

        for i in range(batch_size):
            # Find <image> token positions
            image_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                # No image token, keep original embeddings
                new_inputs_embeds.append(inputs_embeds[i])
                if attention_mask is not None:
                    new_attention_masks.append(attention_mask[i])
                continue

            # For simplicity, assume one <image> token per sample
            # Replace it with all image patch embeddings
            image_pos = image_positions[0].item()

            # Split embeddings: before_image | image_embeds | after_image
            before_image = inputs_embeds[i, :image_pos]
            after_image = inputs_embeds[i, image_pos + 1:]

            # Concatenate: text before image + image patches + text after image
            new_embed = torch.cat([
                before_image,
                image_embeds[i],
                after_image,
            ], dim=0)

            new_inputs_embeds.append(new_embed)

            # Update attention mask if provided
            if attention_mask is not None:
                before_mask = attention_mask[i, :image_pos]
                after_mask = attention_mask[i, image_pos + 1:]
                # Image patches are always attended to (mask = 1)
                image_mask = torch.ones(num_image_patches, dtype=attention_mask.dtype, device=attention_mask.device)

                new_mask = torch.cat([
                    before_mask,
                    image_mask,
                    after_mask,
                ], dim=0)

                new_attention_masks.append(new_mask)

        # Stack back into tensor
        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

        if new_attention_masks is not None:
            attention_mask = torch.stack(new_attention_masks, dim=0)

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            pixel_values: (batch_size, 3, H, W)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len)

        Returns:
            CausalLMOutput with loss and logits
        """
        # Get input embeddings with vision features and updated attention mask
        inputs_embeds, attention_mask = self.prepare_inputs_embeds(
            input_ids, pixel_values, attention_mask
        )

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the language model.

        This reduces memory usage during training at the cost of computation.
        """
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        elif hasattr(self.llm, 'enable_input_require_grads'):
            # For PEFT models
            self.llm.enable_input_require_grads()
            if hasattr(self.llm.base_model, 'gradient_checkpointing_enable'):
                self.llm.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the language model.
        """
        if hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()
        elif hasattr(self.llm.base_model, 'gradient_checkpointing_disable'):
            self.llm.base_model.gradient_checkpointing_disable()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ):
        """
        Generate text given image and prompt.

        Args:
            input_ids: (batch_size, seq_len)
            pixel_values: (batch_size, 3, H, W)
            attention_mask: (batch_size, seq_len)
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        # Get input embeddings and updated attention mask
        inputs_embeds, attention_mask = self.prepare_inputs_embeds(
            input_ids, pixel_values, attention_mask
        )

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )

        return outputs


def test_model():
    """Test model initialization and forward pass."""
    import torch

    # Initialize model
    model = EarthGPT(
        vision_model_name="google/siglip-so400m-patch14-384",
        llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
        use_lora=True,
        load_in_4bit=True
    )

    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    img_size = 384

    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    pixel_values = torch.randn(batch_size, 3, img_size, img_size).cuda()
    attention_mask = torch.ones(batch_size, seq_len).cuda()
    labels = input_ids.clone()

    # Forward pass
    print("Running forward pass...")
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")

    print("Model test passed!")


if __name__ == "__main__":
    test_model()
