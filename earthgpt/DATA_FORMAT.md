# 📋 EarthGPT Data Format Specification

This document describes the unified conversational JSONL format used for training EarthGPT.

## Overview

All tasks (OBB, VQA, Captioning) use the same conversational format to enable multi-task learning in a single model.

## Unified JSONL Format

Each line in the JSONL file represents one training sample:

```json
{
    "image": "path/to/image.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nUser instruction or question"
        },
        {
            "from": "gpt",
            "value": "Model response"
        }
    ],
    "task": "task_type"
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Absolute or relative path to image file |
| `conversations` | array | Yes | List of conversation turns |
| `task` | string | Yes | Task type: "obb", "vqa", or "captioning" |

### Conversation Format

Each conversation is a list of turns alternating between "human" and "gpt":

- **`from`**: Either "human" (user) or "gpt" (assistant)
- **`value`**: The text content
  - Human turns must include `<image>` token
  - GPT turns contain the expected output

## Task-Specific Formats

### 1. Oriented Bounding Box Detection (OBB)

**Purpose**: Detect objects with rotated bounding boxes

**Format**: `[class_name, (x1,y1,x2,y2,x3,y3,x4,y4)]`

**Example**:

```json
{
    "image": "data/images/dota_train/P0001_tile_0_0.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nDetect all objects and provide oriented bounding boxes."
        },
        {
            "from": "gpt",
            "value": "[ship, (120,340,190,342,188,401,118,399)] [ship, (450,220,520,225,518,280,448,275)] [plane, (800,600,850,605,848,650,798,645)]"
        }
    ],
    "task": "obb"
}
```

**OBB Coordinate System**:
- Coordinates are in image pixel space
- Four corners in clockwise order: (x1,y1), (x2,y2), (x3,y3), (x4,y4)
- Typically starts from top-left corner
- Multiple boxes separated by spaces

**Supported Instructions**:
```
- "<image>\nDetect all objects and provide oriented bounding boxes."
- "<image>\nIdentify all objects with their precise locations."
- "<image>\nWhat objects are visible? Provide bounding box coordinates."
- "<image>\nLocalize all objects in this satellite image."
- "<image>\nFind all objects and their orientations."
```

### 2. Visual Question Answering (VQA)

**Purpose**: Answer questions about image content

**Format**: Free-form text answer

**Example**:

```json
{
    "image": "data/images/rsvqa_lr/image_001.tif",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nHow many ships are in the harbor?"
        },
        {
            "from": "gpt",
            "value": "3"
        }
    ],
    "task": "vqa"
}
```

**Answer Types**:
- **Count**: Numeric answers (e.g., "3", "15")
- **Binary**: Yes/No answers
- **Category**: Single word/phrase (e.g., "residential", "forest")
- **Comparison**: Comparative answers (e.g., "larger", "east")

**Example Questions**:
```
- "How many ships are in the harbor?"
- "What is the dominant land use in this area?"
- "Is there a runway visible in the image?"
- "What color is the roof of the largest building?"
- "Are there more cars or trucks in the parking lot?"
```

### 3. Dense Image Captioning

**Purpose**: Generate detailed descriptions of satellite/aerial imagery

**Format**: Free-form descriptive text

**Example**:

```json
{
    "image": "data/images/rsicd/airport_001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nDescribe this satellite image in detail."
        },
        {
            "from": "gpt",
            "value": "A large commercial airport with two parallel runways. Multiple aircraft are visible on the tarmac near several terminal buildings. The airport is surrounded by urban development with a major highway running along the eastern edge."
        }
    ],
    "task": "captioning"
}
```

**Caption Guidelines**:
- Describe major features (buildings, roads, water bodies)
- Mention spatial relationships (north of, adjacent to)
- Include counts for prominent objects
- Describe land use/cover types
- Note distinctive features or patterns

**Supported Instructions**:
```
- "<image>\nDescribe this satellite image in detail."
- "<image>\nWhat do you see in this image?"
- "<image>\nProvide a detailed description of this aerial view."
- "<image>\nGenerate a caption for this remote sensing image."
- "<image>\nWhat is shown in this satellite imagery?"
```

## Multi-Turn Conversations (Advanced)

For more complex interactions, you can have multiple turns:

```json
{
    "image": "data/images/complex_scene.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat is the main feature in this image?"
        },
        {
            "from": "gpt",
            "value": "A large harbor with multiple piers."
        },
        {
            "from": "human",
            "value": "How many ships can you count?"
        },
        {
            "from": "gpt",
            "value": "There are 7 ships visible in the harbor."
        }
    ],
    "task": "vqa"
}
```

## Data Validation

### Required Checks

1. **Image exists**: File path in `image` field must point to valid image
2. **Conversations not empty**: At least one human-gpt turn pair
3. **Image token present**: First human turn must contain `<image>`
4. **Task type valid**: Must be one of: "obb", "vqa", "captioning"
5. **Alternating turns**: Conversations should alternate human-gpt

### OBB-Specific Validation

```python
import re

def validate_obb_format(text):
    """Validate OBB output format."""
    pattern = r'\[([^,]+),\s*\((\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+)\)\]'
    matches = re.findall(pattern, text)
    return len(matches) > 0

# Valid
validate_obb_format("[ship, (120,340,190,342,188,401,118,399)]")  # True

# Invalid
validate_obb_format("[ship, (120,340)]")  # False (not 8 coords)
```

## Dataset Statistics

Example statistics for a well-balanced training set:

```
Total samples: 50,000
├── OBB: 20,000 (40%)
├── VQA: 15,000 (30%)
└── Captioning: 15,000 (30%)

Images: 35,000 unique
├── DOTA tiles: 15,000
├── RSVQA: 10,000
└── RSICD: 10,000

Average samples per image: 1.43
```

## Best Practices

### 1. Instruction Diversity

Use varied instructions for the same task to improve generalization:

```python
INSTRUCTION_TEMPLATES = [
    "<image>\nDetect all objects and provide oriented bounding boxes.",
    "<image>\nIdentify all objects with their precise locations.",
    "<image>\nWhat objects are visible? Provide bounding box coordinates.",
]

import random
instruction = random.choice(INSTRUCTION_TEMPLATES)
```

### 2. Data Augmentation

For captioning tasks, use multiple reference captions:

```python
# Original: 5 captions per image
# Option 1: Use all (5x data)
for caption in captions:
    create_sample(image, caption)

# Option 2: Random selection (1x data, more diverse)
caption = random.choice(captions)
create_sample(image, caption)
```

### 3. Task Balancing

Control task distribution in merged dataset:

```python
task_ratios = {
    'obb': 0.4,        # 40% OBB
    'vqa': 0.3,        # 30% VQA
    'captioning': 0.3  # 30% Captioning
}
```

Adjust based on:
- Downstream task importance
- Dataset size per task
- Performance requirements

### 4. Image Paths

Use relative paths for portability:

```json
{
    "image": "./data/images/sample.png"  // Good
    "image": "/home/user/data/images/sample.png"  // Bad (not portable)
}
```

### 5. Quality Control

Filter out problematic samples:

```python
def is_valid_sample(sample):
    # Check image exists
    if not os.path.exists(sample['image']):
        return False

    # Check conversations not empty
    if len(sample['conversations']) == 0:
        return False

    # Check OBB format (if OBB task)
    if sample['task'] == 'obb':
        response = sample['conversations'][-1]['value']
        if not validate_obb_format(response):
            return False

    return True
```

## Example: Creating Custom Dataset

```python
import json

def create_custom_sample(image_path, task, instruction, response):
    """Create a sample in EarthGPT format."""
    sample = {
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{instruction}"
            },
            {
                "from": "gpt",
                "value": response
            }
        ],
        "task": task
    }
    return sample

# Example usage
samples = []

# OBB sample
samples.append(create_custom_sample(
    image_path="./data/my_image.png",
    task="obb",
    instruction="Detect all buildings.",
    response="[building, (100,200,150,205,148,250,98,245)] [building, (300,400,350,405,348,450,298,445)]"
))

# VQA sample
samples.append(create_custom_sample(
    image_path="./data/my_image2.png",
    task="vqa",
    instruction="Is there a river in this image?",
    response="Yes"
))

# Save to JSONL
with open('custom_dataset.jsonl', 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')
```

## Tools for Validation

```python
def validate_jsonl(file_path):
    """Validate JSONL dataset."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)

                # Check required fields
                assert 'image' in sample
                assert 'conversations' in sample
                assert 'task' in sample

                # Check task type
                assert sample['task'] in ['obb', 'vqa', 'captioning']

                # Check conversations
                assert len(sample['conversations']) > 0
                assert '<image>' in sample['conversations'][0]['value']

            except Exception as e:
                print(f"Error on line {i+1}: {e}")

    print(f"Validation complete. Processed {i+1} samples.")
```

## Summary

The unified conversational format enables:

1. **Multi-task learning**: Single model for all tasks
2. **Flexible instruction following**: Varied prompts improve generalization
3. **Easy extension**: Add new tasks with same format
4. **Standard training**: Use with any conversational LLM framework

For questions or issues with data formatting, see README.md or open an issue.
