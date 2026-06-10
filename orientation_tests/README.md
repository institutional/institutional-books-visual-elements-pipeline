# Orientation Correction

Train and evaluate models that predict the rotation needed to make book page images upright.

## Label Semantics

All labels (in `manual_labels.json` and model predictions) represent the **action needed to correct** an image to upright orientation. The classes are:

| Class | Name | Correction Action |
|-------|------|-------------------|
| 0 | `upright` | No correction needed |
| 1 | `rotated_90_clockwise` | Apply 90 degrees clockwise rotation |
| 2 | `rotated_180` | Apply 180 degree rotation |
| 3 | `rotated_90_counterclockwise` | Apply 90 degrees counter-clockwise rotation |

At inference time, the model prediction can be applied directly as the correction — no inversion step is needed.

## Manual Labeling (`manual_labeler.py`)

A Gradio web GUI for labeling image orientations.

```bash
python orientation_tests/manual_labeler.py --sample-size 200 --resume
```

**How it works:**
1. Downloads random images from the HuggingFace bucket
2. Displays each image one at a time
3. You rotate the image until it appears upright using keyboard/buttons
4. The rotation you applied becomes the label (the correction needed)
5. Results save incrementally to `manual_labels.json`

**The `--resume` flag (recommended):**
- Loads the existing output file and pre-populates the results list
- Skips already-labeled images and starts from the first unlabeled one
- New labels are appended to the existing results
- The file always contains the full accumulated set of labels

**Without `--resume`:**
- Already-labeled filenames are excluded from the session (won't be shown again)
- However, the results list starts empty — when you save, the file is **overwritten** with only the new session's labels
- Always use `--resume` if you want to keep previous labels

**Controls:**
- Right arrow / D: Rotate 90 degrees clockwise
- Left arrow / A: Rotate 90 degrees counter-clockwise
- Down arrow / S: Rotate 180 degrees
- Enter / Space: Confirm as upright (save label)
- U: Undo rotation (reset to original)
- Back button: Go to the previous image — removes its label so you can re-classify it. Can be pressed multiple times to undo several labels in sequence.

## Training

### EfficientNet-V2-S

```bash
python orientation_tests/train_orientation_model.py \
    --labels orientation_tests/manual_labels.json \
    --epochs 20 \
    --eval-pdf orientation_tests/eval_efficientnet.pdf
```

### YOLO26m-cls

```bash
python orientation_tests/train_yolo_cls.py \
    --labels orientation_tests/manual_labels.json \
    --epochs 50 \
    --eval-pdf orientation_tests/eval_yolo.pdf
```

### Training Modes

**Default (synthetic augmentation):**
- Each labeled image is first corrected to upright using its label
- Then all 4 rotations are created as training samples
- Each sample is labeled with the correction needed to undo the applied rotation
- This 4x multiplies the dataset and ensures balanced classes

**`--no-synthetic`:**
- Images are used as-is from the bucket (no correction applied)
- The manual label is used directly as the class
- Dataset is not balanced (most images are upright)
- Better matches real-world inference conditions

Both modes produce models with identical prediction semantics: output = correction to apply.

### Data Split

By default, labeled data is split 80/10/10 into train, val, and test sets:
- **Train (80%):** Used for training
- **Val (10%):** Used to select the best checkpoint
- **Test (10%):** Held out entirely — only used for eval PDF generation

Configurable via `--train-split` and `--val-split` (test = remainder).

### `--eval-pdf`

When provided, the training script generates a PDF after training completes:
- Runs the best model checkpoint on the held-out **test set**
- Shows only images where a correction was predicted (non-upright)
- Each entry shows original and corrected side-by-side with the model's confidence score
- Confidence = softmax probability of predicted class (EfficientNet) or `probs.top1conf` (YOLO)

## Model Files

Models are saved to `orientation_tests/model/`:
- `orientation_model_best_X.XXXX.pth` — EfficientNet best checkpoint
- `yolo26m_orientation/weights/best.pt` — YOLO best checkpoint
- `images/` — cached training images (downloaded from HF bucket)
