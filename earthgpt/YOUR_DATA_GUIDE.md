# 🎯 Guide for Your Custom Data

This guide shows how to train EarthGPT on **your specific data**:
- ✅ **GSD Estimation**: Model predicts GSD from visual appearance
- ✅ **VQA**: Spatial, object, count, presence questions
- ✅ **OBB Detection**: With optional area/length measurements
- ✅ **Multi-GSD Support**: Images from 0.136m to 1.36m/pixel

---

## 📊 Your Data Format

### 1. GSD Metadata (JSON)

```json
{
    "P2735_0029.png": {
        "width": 1444,
        "height": 1444,
        "gsd": 0.37017914620710485
    },
    "P2116_0006.png": {
        "width": 973,
        "height": 973,
        "gsd": 0.13669312211667115
    }
}
```

**What the model will learn:**
- Input: Image
- Output: "The ground sample distance (GSD) is 0.3702 meters per pixel."

### 2. VQA Annotations (JSONL)

```json
{
    "id": "0",
    "image": "RSVLM-QA/INRIA-Aerial-Image-Labeling/train/images/austin11.tif",
    "vqa_pairs": [
        {
            "question_id": "1",
            "question_type": "spatial",
            "question": "Where is the highway interchange located in the image?",
            "answer": "The highway interchange is located in the central portion of the image."
        },
        {
            "question_id": "8",
            "question_type": "count",
            "question": "How many buildings are there in the image?",
            "answer": "There are 1588 buildings in the image."
        }
    ]
}
```

**Question types supported:**
- ✅ `spatial`: "Where is X?"
- ✅ `object`: "What can you see?"
- ✅ `count`: "How many X?"
- ✅ `presence`: "Are there any X?"
- ✅ `overall`: "What kind of area?"
- ✅ `quantity`: Quantitative reasoning
- ✅ `caption`: Full scene description

### 3. OBB Annotations (Your Format)

You'll need to provide OBB data in your format. The preprocessor expects:

```python
# After loading your annotations:
{
    "image_name.png": [
        {
            "class": "building",
            "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        }
    ]
}
```

---

## 🚀 Quick Start

### Step 1: Organize Your Data

```
your_data/
├── images/                     # Your images
│   ├── P2735_0029.png
│   ├── P2116_0006.png
│   └── ...
├── gsd_metadata.json           # GSD metadata
├── vqa_annotations.jsonl       # VQA Q&A pairs
└── obb_annotations.json        # OBB annotations (if you have)
```

### Step 2: Edit Setup Script

Open `scripts/setup_your_data.py` and update paths:

```python
setup_your_data(
    # GSD data
    gsd_metadata_path="./your_data/gsd_metadata.json",
    gsd_image_dir="./your_data/images",

    # VQA data
    vqa_annotations_path="./your_data/vqa_annotations.jsonl",
    vqa_image_root="./your_data",

    # OBB data (if available)
    obb_annotations_path="./your_data/obb_annotations.json",
    obb_image_dir="./your_data/images",

    # Output
    output_dir="./data/your_processed_data",

    # Options
    include_gsd_estimation=True,
    include_obb=False,  # Set to True when you have OBB data
    include_measurements=True
)
```

### Step 3: Process Data

```bash
python scripts/setup_your_data.py
```

**Output:**
```
STEP 1: Processing GSD Data
Created X samples
Saved to ./data/your_processed_data/gsd_estimation.jsonl

STEP 2: Processing VQA Data
Created Y samples
Saved to ./data/your_processed_data/vqa.jsonl

STEP 4: Merging All Datasets
Total samples: Z
Task Distribution:
  gsd_estimation: XX (15%)
  vqa: YY (85%)

Unified training data: ./data/your_processed_data/unified_train.jsonl
```

### Step 4: Validate Data

```bash
python scripts/validate_dataset.py ./data/your_processed_data/unified_train.jsonl
```

### Step 5: Train Model

```bash
python training/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --run_name earthgpt-your-data
```

---

## 📝 Output Examples

### Example 1: GSD Estimation

**Input:**
```
<image>
What is the ground sample distance (GSD) of this image in meters per pixel?
```

**Output:**
```
The ground sample distance (GSD) is 0.3702 meters per pixel.
```

### Example 2: VQA - Spatial Question

**Input:**
```
<image>
Where is the highway interchange located in the image?
```

**Output:**
```
The highway interchange is located in the central portion of the image.
```

### Example 3: VQA - Count Question

**Input:**
```
<image>
How many buildings are there in the image?
```

**Output:**
```
There are 1588 buildings in the image.
```

### Example 4: OBB with Measurements (Advanced)

**Input:**
```
<image>
GSD: 0.3702m/pixel. Detect all buildings and calculate their areas.
```

**Output:**
```
[building, (100,200,150,205,148,250,98,245), area: 458.23m², dimensions: 18.5m × 24.8m]
[building, (300,400,380,405,378,480,298,475), area: 1205.67m², dimensions: 29.6m × 40.7m]
```

---

## 🎯 Task Breakdown

### Tasks You'll Train:

| Task | Percentage | Purpose |
|------|-----------|---------|
| GSD Estimation | 15% | Model learns to predict GSD from visual cues |
| VQA | 85% | Your primary task - diverse question answering |
| OBB | (Optional) | Object detection with rotated boxes |
| Measurements | (Optional) | GSD-aware area/length calculation |

---

## 🔧 Customization Options

### 1. Adjust Task Ratios

In `scripts/setup_your_data.py`:

```python
task_ratios = {
    'gsd_estimation': 0.10,  # Reduce GSD to 10%
    'vqa': 0.70,             # More VQA
    'obb': 0.20              # Add OBB
}
```

### 2. Filter VQA Question Types

If you only want specific question types:

```python
vqa_processor = YourVQAPreprocessor(
    annotations_file=vqa_annotations_path,
    image_root=vqa_image_root,
    output_path=output_path,
    question_types=['spatial', 'count', 'presence']  # Only these types
)
```

### 3. Limit QA Pairs per Image

To avoid over-representation:

```python
vqa_processor = YourVQAPreprocessor(
    ...,
    max_qa_pairs_per_image=5  # Max 5 questions per image
)
```

---

## 📊 Expected Results

### After Training (3 epochs on your data):

**GSD Estimation:**
- Mean Absolute Error: ~0.05-0.1 m/pixel
- Model learns to estimate GSD from visual features

**VQA:**
- Count questions: ~70-80% accuracy
- Presence questions: ~85-90% accuracy
- Spatial questions: ~60-70% accuracy
- Caption generation: BLEU-4 ~0.3-0.4

**OBB (if included):**
- mAP@0.5: ~0.50-0.65 (depends on annotation quality)
- With measurements: Area estimation ±10-20% error

---

## 🐛 Troubleshooting

### "Image not found" errors

**Problem:** Image paths don't match
**Solution:** Check that `image_root` in VQA preprocessor matches your directory structure

```python
# If your VQA annotations have:
"image": "RSVLM-QA/train/images/austin11.tif"

# Set image_root to:
image_root = "./data"  # So full path is ./data/RSVLM-QA/train/images/austin11.tif
```

### Different image sizes

**Problem:** Your images vary from 363×363 to 1444×1444
**Solution:** ✅ No problem! The model resizes all to 384×384 automatically

### Missing GSD for some images

**Problem:** Not all images in VQA have GSD metadata
**Solution:** That's fine! GSD estimation task uses GSD images, VQA uses all images

### Want to add OBB data later

**Solution:**
1. Process OBB data separately: `python data_preprocessing/process_your_obb_data.py`
2. Merge with existing data: `python data_preprocessing/merge_datasets.py`
3. Continue training from checkpoint

---

## 🎓 Advanced: GSD-Aware Measurements

The model can use predicted GSD for measurements:

### Workflow:

```python
# Step 1: Predict GSD
gsd_prediction = model.predict(
    image=image,
    task="gsd_estimation"
)
# Output: "The ground sample distance (GSD) is 0.3702 meters per pixel."

# Step 2: Extract GSD value
gsd = 0.3702  # Parse from output

# Step 3: Use GSD for measurement task
measurement = model.predict(
    image=image,
    task="obb_measurement",
    prompt=f"GSD: {gsd}m/pixel. Detect all buildings and calculate their areas."
)
```

---

## 📈 Training Timeline

| Stage | Time | Hardware | Description |
|-------|------|----------|-------------|
| Data Preprocessing | 5-10 min | CPU | Process all annotations |
| Training (3 epochs) | 4-8 hours | A100 40GB | Full training |
| Evaluation | 10-20 min | GPU | Test on validation set |

---

## 🎯 Next Steps

1. **Start with VQA + GSD**: Your primary tasks
   ```bash
   include_obb=False  # In setup script
   ```

2. **Train initial model**: Get baseline performance

3. **Add OBB later**: When you have annotations
   ```bash
   include_obb=True
   ```

4. **Fine-tune**: Continue training with OBB data

---

## 📞 Need Help?

If you encounter issues:

1. **Check data format**: Use `validate_dataset.py`
2. **Verify paths**: Make sure all images exist
3. **Start small**: Test with 50-100 samples first
4. **Check outputs**: Look at generated JSONL files

---

**You're ready to train on your custom data!** 🚀

The model will learn to:
- ✅ Estimate GSD from visual appearance
- ✅ Answer diverse VQA questions (spatial, count, presence, etc.)
- ✅ Handle multi-GSD imagery (0.136m to 1.36m range)
- ✅ Optionally detect objects and calculate real-world measurements
