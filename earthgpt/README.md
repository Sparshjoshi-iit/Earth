# 🛰️ EarthGPT: Geospatial Vision-Language Model

A production-ready multimodal Vision-Language Model (VLM) fine-tuned for satellite and aerial imagery analysis. EarthGPT performs three core geospatial intelligence tasks with high precision:

1. **Oriented Bounding Box Detection (OBB)** - Detect objects with rotated bounding boxes
2. **Visual Question Answering (VQA)** - Answer questions about satellite imagery
3. **Dense Image Captioning** - Generate detailed descriptions of aerial scenes

---

## ⚡ **Quick Start (No Downloads Required!)**

Get started in 5 minutes with **synthetic data**:

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Generate synthetic satellite data (300 samples)
python scripts/generate_sample_data.py

# 3. Train
python training/train.py --training_config configs/training_config_synthetic.yaml

# 4. Test inference
python scripts/inference.py \
    --model_path outputs/earthgpt-synthetic/final_model \
    --image data/synthetic/images/obb_sample_0001.png \
    --task obb
```

**No large dataset downloads needed!** See [QUICKSTART.md](QUICKSTART.md) for detailed guide.

---

## 🎯 Key Features

- **Efficient Fine-Tuning**: Q-LoRA (4-bit quantization + LoRA) for training on single GPU
- **State-of-the-Art Models**: SigLIP-ViT vision encoder + Llama-3.2-3B LLM
- **Multi-Task Learning**: Unified training on OBB, VQA, and captioning
- **High-Resolution Support**: Optimized for 0.5m-10m/pixel satellite imagery
- **Production-Ready**: Complete pipeline from data preprocessing to deployment

## 🏗️ Architecture

```
┌──────────────────────┐
│  Satellite Image     │
│  (384×384)           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vision Encoder      │
│  SigLIP-ViT-SO400M   │
│  (Frozen)            │
└──────────┬───────────┘
           │ vision features
           ▼            (1152-dim)
┌──────────────────────┐
│  Projector (MLP)     │
│  (Trainable)         │
└──────────┬───────────┘
           │ projected features
           ▼            (3072-dim)
┌──────────────────────┐
│  Language Model      │
│  Llama-3.2-3B        │
│  (LoRA Fine-tuned)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output              │
│  • OBB coordinates   │
│  • VQA answers       │
│  • Captions          │
└──────────────────────┘
```

### Model Components

| Component | Model | Parameters | Training |
|-----------|-------|------------|----------|
| Vision Encoder | SigLIP-ViT-SO400M | 400M | Frozen |
| Projector | 2-layer MLP | 3.5M | Trainable |
| Language Model | Llama-3.2-3B-Instruct | 3B | LoRA (64 rank) |

**Total Trainable Parameters**: ~70M (2.3% of total model)

## 📦 Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU)
- 40GB+ GPU RAM (for training with Q-LoRA)
- 16GB+ GPU RAM (for inference)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/earthgpt.git
cd earthgpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional captioning metrics (optional)
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

## 📊 Dataset Preparation

### Option 1: Synthetic Data (Recommended for Testing) ⚡

**No downloads required!** Generate synthetic satellite-like images instantly:

```bash
python scripts/generate_sample_data.py
```

**Generated:**
- 300 training samples (100 OBB + 100 VQA + 100 Captioning)
- 60 validation samples
- Synthetic satellite-like images
- Location: `data/synthetic/`

**Pros:**
- ✅ Instant generation (2 minutes)
- ✅ No large downloads
- ✅ Tests complete pipeline
- ✅ Good for development and debugging

**Cons:**
- ❌ Not suitable for production deployment
- ❌ Lower realism than real satellite data

### Option 2: Real Datasets (For Production)

#### Automatically Downloadable

```bash
# Download datasets from Hugging Face (where available)
python scripts/download_datasets.py --all

# Specific datasets:
python scripts/download_datasets.py --eurosat    # Scene classification
python scripts/download_datasets.py --ucmerced   # Land use classification
```

#### Manual Downloads Required

Some datasets require manual download due to access restrictions:

1. **DOTA v1.5** - Oriented bounding box detection
   - Download: [DOTA Dataset](https://captain-whu.github.io/DOTA/dataset.html)
   - ~15 classes of objects in aerial images
   - **Size**: ~20 GB

2. **RSVQA** - Remote sensing visual question answering
   - Download: [RSVQA](https://rsvqa.sylvainlobry.com/)
   - Low Resolution (LR) and High Resolution (HR) variants
   - **Size**: ~2 GB (LR) / ~10 GB (HR)

3. **RSICD** - Remote sensing image captioning dataset
   - **Status**: Currently difficult to access
   - **Alternative**: Use synthetic data or EuroSAT for captioning
   - **Size**: ~1 GB

### Data Preprocessing (Real Datasets)

#### Step 1: Preprocess Individual Datasets

```bash
# Process DOTA for OBB
python data_preprocessing/process_dota.py

# Process RSVQA for VQA
python data_preprocessing/process_rsvqa.py

# Process RSICD for Captioning (if available)
python data_preprocessing/process_rsicd.py
```

**Important**: Edit the paths in each script to point to your downloaded datasets.

#### Step 2: Merge into Unified Dataset

```bash
python data_preprocessing/merge_datasets.py
```

This creates:
- `data/unified_train.jsonl` - Training set
- `data/unified_val.jsonl` - Validation set

**Note**: For quick testing, use synthetic data first! See [QUICKSTART.md](QUICKSTART.md)

### Unified Data Format

All tasks use the same conversational JSONL format:

```jsonl
{"image": "path/to/img.png", "conversations": [{"from": "human", "value": "<image>\nDetect all ships."}, {"from": "gpt", "value": "[ship, (120,340,190,342,188,401,118,399)] [ship, (50,50,100,50,100,100,50,100)]"}], "task": "obb"}
{"image": "path/to/img2.png", "conversations": [{"from": "human", "value": "<image>\nHow many planes are visible?"}, {"from": "gpt", "value": "3"}], "task": "vqa"}
{"image": "path/to/img3.png", "conversations": [{"from": "human", "value": "<image>\nDescribe this image."}, {"from": "gpt", "value": "A residential area with dense housing near a river."}], "task": "captioning"}
```

## 🚀 Training

### Configuration

Edit `configs/training_config.yaml` to set:
- Data paths
- Batch size and accumulation steps
- Learning rate and schedule
- Output directory

### Start Training

```bash
python training/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --run_name earthgpt-lora-v1
```

### Training with DeepSpeed (Optional)

For multi-GPU training:

```bash
deepspeed --num_gpus=4 training/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --run_name earthgpt-lora-v1
```

### Expected Training Time

| Setup | Hardware | Time (3 epochs) |
|-------|----------|-----------------|
| Single A100 (40GB) | 1x A100 | ~24 hours |
| 4x A100 | 4x A100 | ~6 hours |
| Single RTX 4090 | 1x RTX 4090 | ~36 hours |

### Monitoring

Training metrics are logged to WandB by default. View at https://wandb.ai

## 🔮 Inference

### Command-Line Interface

#### OBB Detection

```bash
python scripts/inference.py \
    --model_path outputs/earthgpt-lora/final_model \
    --image path/to/satellite_image.png \
    --task obb \
    --output result_obb.png
```

**Output**:
```
[ship, (120,340,190,342,188,401,118,399)] [plane, (450,220,520,225,518,280,448,275)]
```

#### Visual Question Answering

```bash
python scripts/inference.py \
    --model_path outputs/earthgpt-lora/final_model \
    --image path/to/satellite_image.png \
    --task vqa \
    --question "How many ships are in the harbor?"
```

**Output**:
```
3
```

#### Dense Captioning

```bash
python scripts/inference.py \
    --model_path outputs/earthgpt-lora/final_model \
    --image path/to/satellite_image.png \
    --task captioning
```

**Output**:
```
A large harbor with multiple ships docked along the piers. Dense urban development surrounds the port area with several large warehouse buildings visible.
```

### Python API

```python
from scripts.inference import EarthGPTInference

# Load model
model = EarthGPTInference(model_path="outputs/earthgpt-lora/final_model")

# Run OBB detection
response = model.predict(
    image_path="image.png",
    task="obb"
)

# Visualize results
model.visualize_obb(
    image_path="image.png",
    output=response,
    save_path="result.png"
)
```

## 📈 Evaluation

### Run Evaluation

```bash
# Evaluate OBB
python evaluation/evaluate.py \
    --model_path outputs/earthgpt-lora/final_model \
    --test_file data/unified_val.jsonl \
    --task obb

# Evaluate VQA
python evaluation/evaluate.py \
    --model_path outputs/earthgpt-lora/final_model \
    --test_file data/unified_val.jsonl \
    --task vqa

# Evaluate Captioning
python evaluation/evaluate.py \
    --model_path outputs/earthgpt-lora/final_model \
    --test_file data/unified_val.jsonl \
    --task captioning
```

### Metrics

| Task | Metrics |
|------|---------|
| OBB | Precision, Recall, F1, mAP |
| VQA | Accuracy |
| Captioning | BLEU-1/2/3/4, METEOR, CIDEr |

## 🎓 Expected Performance

Baseline performance on validation sets (with proper training):

| Task | Metric | Expected Score |
|------|--------|----------------|
| OBB (DOTA) | mAP@0.5 | 0.65-0.75 |
| VQA (RSVQA-LR) | Accuracy | 0.75-0.85 |
| Captioning (RSICD) | BLEU-4 | 0.25-0.35 |
| Captioning (RSICD) | CIDEr | 1.5-2.5 |

*Note: Actual performance depends on dataset size, training hyperparameters, and compute budget.*

## 🛠️ Project Structure

```
earthgpt/
├── configs/
│   ├── model_config.yaml           # Model architecture config
│   └── training_config.yaml        # Training hyperparameters
├── data_preprocessing/
│   ├── process_dota.py             # DOTA → JSONL
│   ├── process_rsvqa.py            # RSVQA → JSONL
│   ├── process_rsicd.py            # RSICD → JSONL
│   └── merge_datasets.py           # Merge into unified format
├── model/
│   ├── earthgpt_model.py           # Core VLM architecture
│   └── dataset.py                  # Dataset & DataCollator
├── training/
│   └── train.py                    # Training script
├── scripts/
│   └── inference.py                # Inference CLI & API
├── evaluation/
│   └── evaluate.py                 # Evaluation metrics
├── requirements.txt
└── README.md
```

## 🔧 Advanced Usage

### Custom Tasks

To add custom tasks, modify:

1. **Data Preprocessing**: Create `data_preprocessing/process_custom.py`
2. **Instruction Templates**: Add to `INSTRUCTION_TEMPLATES` in preprocessing script
3. **Task Format**: Follow conversational JSONL format
4. **Training**: Add to `merge_datasets.py` with desired ratio

### Fine-Tuning on Custom Data

```python
# 1. Create custom JSONL file
custom_data = [
    {
        "image": "path/to/image.png",
        "conversations": [
            {"from": "human", "value": "<image>\nYour question"},
            {"from": "gpt", "value": "Expected answer"}
        ],
        "task": "custom"
    }
]

# 2. Update training config
# Set data_path to your custom JSONL

# 3. Train
python training/train.py --training_config configs/training_config.yaml
```

### LoRA Hyperparameter Tuning

Key hyperparameters in `configs/model_config.yaml`:

- `r`: LoRA rank (higher = more capacity, slower training)
  - Recommended: 32-64
- `lora_alpha`: Scaling factor (typically 2x rank)
- `lora_dropout`: Regularization (0.05-0.1)
- `target_modules`: Which layers to adapt

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. Reduce batch size in `training_config.yaml`
2. Increase `gradient_accumulation_steps`
3. Enable `load_in_4bit=True` (default)
4. Use smaller LoRA rank

### Slow Training

**Solutions**:
1. Use `bf16=True` instead of `fp16`
2. Increase batch size if memory allows
3. Use DeepSpeed for multi-GPU
4. Reduce `dataloader_num_workers` if CPU-bound

### Poor OBB Performance

**Solutions**:
1. Increase OBB task ratio in `merge_datasets.py`
2. Use larger LoRA rank (64-128)
3. Increase training epochs
4. Add more DOTA data (use FAIR1M dataset)

## 📚 Citation

If you use EarthGPT in your research, please cite:

```bibtex
@software{earthgpt2024,
  title={EarthGPT: Geospatial Vision-Language Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/earthgpt}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **SigLIP**: Superior vision-language pretraining ([Paper](https://arxiv.org/abs/2303.15343))
- **Llama 3.2**: Meta's instruction-tuned language model
- **DOTA**: Benchmark for object detection in aerial images
- **RSVQA**: Remote sensing VQA dataset
- **RSICD**: Remote sensing captioning dataset
- **LLaVA**: Inspiration for vision-language projection architecture

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com]
