# Manuscript Layout Detector

An Ultralytics YOLO detector for localizing manuscript layout regions. The project uses automatically generated YOLO annotations; original manuscript images and the established train/validation/test split are retained unchanged.

## Dataset

The dataset has 259 manuscript images from 11 sources, split into 203 training, 29 validation, and 27 test images.

```text
data/
├── raw/                         # original source images; never modified
└── dataset/
    ├── data.yaml                # Ultralytics configuration and class mapping
    ├── split_manifest.csv       # fixed split manifest
    ├── images/
    │   ├── train/               # 203 images
    │   ├── val/                 # 29 images
    │   └── test/                # 27 images used for inference
    ├── labels/
    │   ├── train/               # normalized YOLO labels
    │   └── val/                 # normalized YOLO labels
    └── auto_annotated_previews/ # rendered training-label previews
```

The five class IDs are:

| ID | Class | Meaning |
| --- | --- | --- |
| 0 | `header` | top-margin title, running header, or folio text |
| 1 | `footer` | bottom-margin catchword, signature, or page text |
| 2 | `main_text` | primary manuscript text block or column |
| 3 | `side_text` | distinct marginal commentary or gloss |
| 4 | `filler` | isolated decorative, stamp, or non-manuscript addition |

## Automatic annotations

`src/data/auto_annotate.py` uses the production leaf-aware detector in `src/data/layout_detector.py`. It identifies parchment leaves before measuring ink, uses projection profiles and real dark gutters to split facing or stacked folios, trims boxes to conservative ink boundaries, and searches margins separately for `side_text`. It detects eligible blue/white additions as `filler` and rejects weak texture-only leaves and text boxes wider than 80% of the image.

This is automatic annotation, not manual ground truth. Faded script, stamps, damaged leaves, unusual page geometry, and marginalia can be missed or misclassified. The generated labels should be reviewed or replaced with expert annotation before using the model for scholarly or production decisions.

Regenerate labels and previews without changing images or the split:

```bash
python src/data/auto_annotate.py --input data/dataset/images/train --output data/dataset/labels/train --preview-dir data/dataset/auto_annotated_previews
python src/data/auto_annotate.py --input data/dataset/images/val --output data/dataset/labels/val --preview-dir data/dataset/auto_annotated_previews_val
```

## Training and evaluation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the configured YOLO11n model:

```bash
python train.py
```

The best checkpoint is copied to `models/best.pt`; training artifacts are in `models/runs/train/`. To evaluate the current checkpoint explicitly:

```bash
python -c "from ultralytics import YOLO; YOLO('models/best.pt').val(data='data/dataset/data.yaml', imgsz=640, split='val')"
```

The training run reached 11 completed epochs. The current `models/best.pt` is the validation-selected best checkpoint and has the same SHA-256 hash as `models/runs/train/weights/best.pt` (rather than the later `last.pt` snapshot). It was measured on the 29-image validation split:

| Metric | Measured value |
| --- | ---: |
| Precision | 0.9170 |
| Recall | 0.1411 |
| mAP50 | 0.1386 |
| mAP50-95 | 0.1013 |

These metrics reflect automatically generated labels and a small, class-imbalanced validation set. In particular, the current model has low recall and should be treated as an early baseline, not as a reliable all-class layout detector.

## Inference

Run the trained checkpoint on the fixed test split:

```bash
python inference.py --input data/dataset/images/test --output results --weights models/best.pt
```

For one image:

```bash
python inference.py --input path/to/image.jpg --output results --weights models/best.pt
```

Inference writes, without modifying input images:

- `results/annotated/` — rendered predictions
- `results/metadata.json` — image name, dimensions, class ID/name, confidence, and clipped `x_min`, `y_min`, `x_max`, `y_max` boxes

The inference post-processing clips every box to image bounds and rejects page-wide `main_text` predictions above 80% of image width to suppress the known blank-leaf/texture failure mode.
