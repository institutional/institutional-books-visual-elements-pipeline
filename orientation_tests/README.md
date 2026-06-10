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

### `--eval-pdf`

When provided, the training script generates a PDF after training completes:
- Samples ~500 images not in the training set
- Runs the best model checkpoint on them
- Shows only images where a correction was predicted (non-upright)
- PDF displays original alongside the model-corrected version

## Model Files

Models are saved to `orientation_tests/model/`:
- `orientation_model_best_X.XXXX.pth` — EfficientNet best checkpoint
- `yolo26m_orientation/weights/best.pt` — YOLO best checkpoint
- `images/` — cached training images (downloaded from HF bucket)
