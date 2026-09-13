# Manuscript Layout Region Detection Dataset

## Overview
This dataset directory is configured for training and evaluating an Ultralytics YOLO object detection model to identify structural layout regions in manuscript pages.

## Class Mapping
The dataset identifies the following 5 target classes:
- **0**: `header` (Top metadata, title, page numbers, or running headers)
- **1**: `footer` (Bottom metadata, page numbers, footnotes, or running footers)
- **2**: `main_text` (Primary body of text content)
- **3**: `side_text` (Margin text, side notes, or commentary along column edges)
- **4**: `filler` (Decorative elements, blank spatial regions, illustrations, or non-text filler)

## Dataset Directory Structure
```
data/
├── raw/                      # Raw manuscript source images (kept intact)
└── dataset/
    ├── data.yaml             # Ultralytics YOLO dataset configuration file
    ├── README.md             # Dataset documentation
    ├── images/
    │   ├── train/            # 203 training images
    │   ├── val/              # 29 validation images
    │   └── test/             # 27 test images
    └── labels/
        ├── train/            # YOLO labels for train
        └── val/              # YOLO labels for val
```

## Expected YOLO Label Format
Each image in `images/train/` or `images/val/` will correspond to a `.txt` label file in `labels/train/` or `labels/val/` with matching file stem.

Each line in a YOLO label file contains normalized coordinates:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Where:
- `<class_id>`: Integer from 0 to 4 corresponding to the 5 target classes.
- `<x_center>`, `<y_center>`: Bounding box center normalized relative to image width and height (`0.0` to `1.0`).
- `<width>`, `<height>`: Bounding box dimensions normalized relative to image width and height (`0.0` to `1.0`).

## Notes
- Source manuscript images are stored separately in `data/raw/` and are kept unchanged.
- Splits are populated. Train/val labels are auto-generated YOLO boxes.
- Test images are used for inference (`data/dataset/images/test`, also exposed as `data/test_images`).
