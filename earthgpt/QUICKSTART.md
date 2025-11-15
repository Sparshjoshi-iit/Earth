# 🚀 EarthGPT Quick Start Guide

Get EarthGPT running in minutes with synthetic data!

## ⚡ Fast Track (5 Minutes)

### Step 1: Setup Environment

```bash
cd earthgpt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Synthetic Data

No downloads needed! Generate synthetic satellite-like images:

```bash
python scripts/generate_sample_data.py
```

This creates:
- **300 training samples** (100 OBB + 100 VQA + 100 Captioning)
- **60 validation samples** (20 per task)
- Synthetic satellite-like images in `data/synthetic/images/`

**Output:**
```
data/synthetic/
├── images/
│   ├── obb_sample_0000.png
│   ├── vqa_sample_0000.png
│   ├── caption_sample_0000.png
│   └── ...
├── synthetic_train.jsonl  (300 samples)
└── synthetic_val.jsonl    (60 samples)
```

### Step 3: Validate Data

```bash
python scripts/validate_dataset.py data/synthetic/synthetic_train.jsonl
```

**Expected output:**
```
✅ All samples passed validation!
Valid samples: 300 / 300

DATASET STATISTICS
Total samples: 300
Unique images: 300
Task Distribution:
  obb: 100 (33.3%)
  vqa: 100 (33.3%)
  captioning: 100 (33.3%)
```

### Step 4: Train Model

```bash
python training/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config_synthetic.yaml \
    --run_name earthgpt-synthetic-test
```

**Training time:**
- **RTX 4090**: ~2 hours
- **A100 40GB**: ~1 hour
- **V100 16GB**: ~3 hours

**What happens:**
- Downloads SigLIP-ViT and Llama-3.2-3B (first time only)
- Trains with Q-LoRA (4-bit quantization)
- Saves checkpoints every 100 steps
- Logs to tensorboard

**Monitor training:**
```bash
tensorboard --logdir logs
```

### Step 5: Test Inference

After training (or stop early to test):

```bash
# OBB Detection
python scripts/inference.py \
    --model_path outputs/earthgpt-synthetic/final_model \
    --image data/synthetic/images/obb_sample_0001.png \
    --task obb \
    --output result.png

# VQA
python scripts/inference.py \
    --model_path outputs/earthgpt-synthetic/final_model \
    --image data/synthetic/images/vqa_sample_0001.png \
    --task vqa \
    --question "How many objects are visible?"

# Captioning
python scripts/inference.py \
    --model_path outputs/earthgpt-synthetic/final_model \
    --image data/synthetic/images/caption_sample_0001.png \
    --task captioning
```

---

## 📦 Using Real Datasets (Advanced)

### Option 1: Download Available Datasets

```bash
# Download what's available automatically
python scripts/download_datasets.py --all

# Or download specific datasets:
python scripts/download_datasets.py --eurosat
python scripts/download_datasets.py --ucmerced
```

### Option 2: Manual Downloads

Some datasets require manual download:

#### RSICD (Image Captioning)
1. Visit: https://github.com/201528014227051/RSICD_optimal
2. Download dataset
3. Extract to: `data/downloads/rsicd/`
4. Preprocess:
   ```bash
   # Edit process_rsicd.py to set correct paths
   python data_preprocessing/process_rsicd.py
   ```

#### RSVQA (Visual Question Answering)
1. Visit: https://rsvqa.sylvainlobry.com/
2. Download LR or HR variant
3. Extract to: `data/downloads/rsvqa/`
4. Preprocess:
   ```bash
   # Edit process_rsvqa.py to set correct paths
   python data_preprocessing/process_rsvqa.py
   ```

#### DOTA (Object Detection)
1. Visit: https://captain-whu.github.io/DOTA/dataset.html
2. Download DOTA-v1.5
3. Extract to: `data/downloads/DOTA-v1.5/`
4. Preprocess:
   ```bash
   # Edit process_dota.py to set correct paths
   python data_preprocessing/process_dota.py
   ```

### Merge Datasets

After preprocessing individual datasets:

```bash
# Edit merge_datasets.py to set correct input paths
python data_preprocessing/merge_datasets.py
```

This creates:
- `data/unified_train.jsonl`
- `data/unified_val.jsonl`

### Train on Real Data

```bash
python training/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --run_name earthgpt-real-data
```

---

## 🔧 Configuration

### Training Parameters

Edit `configs/training_config_synthetic.yaml`:

```yaml
training:
  num_train_epochs: 5              # Number of epochs
  per_device_train_batch_size: 2   # Batch size (reduce if OOM)
  gradient_accumulation_steps: 4    # Effective batch = 2*4 = 8
  learning_rate: 2e-4              # Learning rate
  save_steps: 100                  # Save checkpoint every N steps
```

### Model Parameters

Edit `configs/model_config.yaml`:

```yaml
lora:
  r: 64                  # LoRA rank (32-128)
  lora_alpha: 128        # LoRA alpha (typically 2x rank)
  lora_dropout: 0.05     # Dropout for regularization
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Symptom**: CUDA out of memory error

**Solutions:**
```yaml
# In training_config_synthetic.yaml:
per_device_train_batch_size: 1  # Reduce to 1
gradient_accumulation_steps: 8   # Increase to maintain effective batch size
```

Or:
```yaml
# In model_config.yaml:
lora:
  r: 32  # Reduce LoRA rank from 64 to 32
```

### Model Download Fails

**Symptom**: Cannot download SigLIP or Llama models

**Solution:**
```bash
# Pre-download models
python -c "from transformers import AutoModel, AutoTokenizer; \
  AutoModel.from_pretrained('google/siglip-so400m-patch14-384'); \
  AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"
```

### Slow Training

**Symptom**: Training too slow

**Solutions:**
1. Use fewer workers:
   ```yaml
   dataloader_num_workers: 2  # Reduce from 4
   ```

2. Use smaller dataset:
   ```bash
   # Generate less synthetic data
   # Edit generate_sample_data.py:
   generator.generate_dataset(num_samples_per_task=50, split='train')
   ```

3. Enable DeepSpeed (multi-GPU):
   ```bash
   deepspeed --num_gpus=2 training/train.py ...
   ```

### Validation Errors

**Symptom**: Dataset validation fails

**Check:**
```bash
# Validate dataset format
python scripts/validate_dataset.py data/synthetic/synthetic_train.jsonl --verbose

# Check image paths exist
python scripts/validate_dataset.py data/synthetic/synthetic_train.jsonl --check-images
```

---

## 📊 Expected Results

### Synthetic Data (Quick Test)

After 5 epochs on synthetic data:

| Task | Metric | Expected |
|------|--------|----------|
| OBB | Basic detection | ✅ Works |
| VQA | Count accuracy | ~70% |
| Captioning | Coherent text | ✅ Works |

**Note**: Synthetic data is for testing the pipeline, not for production-quality results.

### Real Data (Production)

With full DOTA + RSVQA + RSICD:

| Task | Metric | Expected |
|------|--------|----------|
| OBB | mAP@0.5 | 0.65-0.75 |
| VQA | Accuracy | 0.75-0.85 |
| Captioning | BLEU-4 | 0.25-0.35 |

---

## 🎯 Next Steps

### 1. Test on Your Own Images

```bash
python scripts/inference.py \
    --model_path outputs/earthgpt-synthetic/final_model \
    --image your_satellite_image.png \
    --task captioning
```

### 2. Fine-tune on Custom Data

Create your own dataset in JSONL format:

```python
import json

samples = [
    {
        "image": "path/to/your/image.png",
        "conversations": [
            {"from": "human", "value": "<image>\nYour question"},
            {"from": "gpt", "value": "Expected answer"}
        ],
        "task": "vqa"
    }
]

with open('custom_data.jsonl', 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')
```

Then train:
```bash
# Update training config with your data path
python training/train.py --training_config your_config.yaml
```

### 3. Scale Up

- Use real datasets (DOTA, RSVQA, RSICD)
- Increase LoRA rank for better capacity
- Train for more epochs
- Use multi-GPU training

---

## 💡 Tips

1. **Start Small**: Use synthetic data to verify everything works
2. **Monitor Training**: Use tensorboard to watch loss curves
3. **Save Checkpoints**: Training can be resumed from checkpoints
4. **Experiment**: Try different LoRA ranks and learning rates
5. **Evaluate Often**: Run inference on validation set to check quality

---

## 📚 Resources

- **Full Documentation**: See `README.md`
- **Data Format**: See `DATA_FORMAT.md`
- **Model Architecture**: See `model/earthgpt_model.py`
- **Issues**: Open a GitHub issue if you encounter problems

---

## ⏱️ Timeline Summary

| Task | Time | Requirements |
|------|------|--------------|
| Setup environment | 5 min | Internet connection |
| Generate synthetic data | 2 min | - |
| Train on synthetic | 1-3 hours | GPU with 16GB+ VRAM |
| Download real datasets | 1-2 hours | Internet + 30GB disk |
| Preprocess real data | 30 min | - |
| Train on real data | 6-24 hours | GPU with 40GB VRAM |

**Total (synthetic)**: ~2-4 hours from zero to working model
**Total (real data)**: ~10-30 hours from zero to production model

---

Happy training! 🛰️🚀
