Manuscript Specific Layout Region Detection

An end-to-end computer vision pipeline for detecting and classifying layout regions in historical manuscript images.

The system identifies five manuscript-specific regions — header, footer, main text, side text, and filler — and produces both visually annotated images and structured JSON predictions containing bounding boxes, class labels, and confidence scores.

Overview

Historical manuscripts often contain complex page layouts, multiple text regions, marginal annotations, decorative elements, faded ink, stains, bleed-through, and other forms of degradation.

This project provides a modular Python pipeline designed to:

Detect manuscript layout regions automatically
Classify detected regions into five target classes
Process a single image or an entire folder of images
Generate annotated output images
Generate structured JSON metadata
Keep original input images unchanged
Handle different page layouts and degraded manuscript conditions
Provide heuristic fallback detection for regions that are difficult for the learned detector to identify
Target Classes
Class	Description
header	Top-margin text, running headers, section titles, folio numbers
footer	Bottom-margin text, catchwords, page numbers, signatures
main_text	Main body text of the manuscript
side_text	Marginalia, annotations, and commentary along page margins
filler	Decorative elements, English text, and pencil text

Class IDs used throughout the project:

text
0 → header
1 → footer
2 → main_text
3 → side_text
4 → filler
Pipeline Architecture
text
                 Manuscript Images
                        │
                        ▼
              ┌─────────────────────┐
              │ Image Preprocessing │
              │                     │
              │ • Grayscale         │
              │ • Contrast handling │
              │ • Thresholding      │
              │ • Morphology        │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    ┌─────────────────┐     ┌──────────────────┐
    │ YOLO Detector   │     │ Heuristic Layout │
    │                 │     │ Detection        │
    │ Learned visual  │     │                  │
    │ region detector │     │ Projection /     │
    │                 │     │ ink / position   │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌─────────────────┐
                │ Post-processing │
                │                 │
                │ • Box clipping  │
                │ • Validation    │
                │ • Deduplication │
                └────────┬────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Final Predictions   │
              ├─────────────────────┤
              │ Annotated Images    │
              │ JSON Metadata       │
              └─────────────────────┘
Key Features
1. YOLO-based Region Detection

The learned detector is based on Ultralytics YOLO11n, trained for the five manuscript layout classes.

The model produces:

Bounding boxes
Class predictions
Confidence scores
2. Heuristic Layout Analysis

A complementary OpenCV-based detection layer is used to improve robustness for layout regions that may be difficult for the learned model to detect.

The heuristic pipeline uses visual/layout cues such as:

Adaptive thresholding
Dark-ink extraction
Morphological operations
Projection profiles
Row/column density
Page-relative positioning
Margin component analysis

This is particularly useful for manuscript regions where the visual appearance can vary significantly across pages.

3. Bounding Box Validation

Predictions are validated before being written to the final output.

The validation layer checks:

Valid class IDs
Finite coordinates
Positive width and height
Page boundary constraints
Oversized detections
Duplicate detections
Header/footer positional consistency

All final bounding boxes are clipped to the image boundaries.

Dataset

The project uses manuscript page images collected from multiple manuscript sources.

After dataset preparation:

Split	Images
Training	203
Validation	29
Test	27
Total	259

The dataset contains pages from 11 manuscript sources, providing variation in page structure, typography, writing styles, and manuscript layouts.

The dataset is intentionally kept outside version control to keep the repository lightweight and avoid committing raw image data.

Automatic Annotation

Because manually annotating a large manuscript collection is expensive and time-consuming, an automated annotation pipeline was developed.

The annotation process uses visual and layout analysis to generate initial YOLO-format annotations.

The pipeline includes:

Page/leaf detection
Adaptive thresholding
Dark-ink extraction
Morphological processing
Projection-profile analysis
Multi-column body-text detection
Margin component analysis
Side-text detection
Conservative filler detection
Large-box rejection
Annotation validation

Generated annotations are stored in YOLO format:

text
class_id center_x center_y width height

All coordinates are normalized to the image dimensions.

Annotation Validation

The generated annotations are automatically validated before training.

The validation process checks for:

Missing annotation files
Invalid annotation lines
Invalid class IDs
Out-of-bound coordinates
Oversized main-text boxes
Invalid header/footer positions
Extremely small filler boxes
Duplicate or invalid bounding boxes

This ensures that malformed annotations do not enter the training pipeline.

Model
Architecture
Model: YOLO11n
Framework: Ultralytics
Task: Object Detection
Input Size: 640 × 640

The submitted checkpoint is:

text
models/best.pt

The final checkpoint was selected based on validation performance and retained instead of using a later fine-tuning experiment that produced weaker validation results.

Validation Results

Validation performance of the submitted checkpoint:

Metric	Score
Precision	91.70%
Recall	14.11%
mAP@50	13.86%
mAP@50–95	10.13%

The high precision indicates that the learned detector is conservative about its predictions, while the relatively low recall indicates that some manuscript regions remain difficult to detect consistently.

The heuristic inference layer is therefore used as a complementary mechanism rather than relying exclusively on the YOLO predictions.

Inference

The main inference entry point is:

text
inference.py

It supports both individual images and folders containing multiple manuscript pages.

Example
bash
python inference.py --input ./data/test_images --output ./results

For the prepared test split:

bash
python inference.py --input ./data/dataset/images/test --output ./results
Optional Confidence Threshold

The confidence threshold can be adjusted using:

bash
python inference.py --input ./data/test_images --output ./results --conf 0.25
Output

Inference generates two primary outputs.

Annotated Images

text
results/
└── annotated/
    ├── page_001.jpg
    ├── page_002.jpg
    └── ...

The original input images are never modified.

JSON Metadata

text
results/
└── metadata.json

Example:

json
[
  {
    "image_name": "page_001.jpg",
    "width": 1800,
    "height": 964,
    "detections": [
      {
        "class_id": 2,
        "class_name": "main_text",
        "confidence": 0.9349,
        "bbox": {
          "x_min": 291,
          "y_min": 227,
          "x_max": 1573,
          "y_max": 780
        }
      }
    ]
  }
]

Bounding boxes are represented using pixel coordinates:

x_min
y_min
x_max
y_max
Project Structure
text
manuscript-layout-detector/
│
├── data/
│   ├── dataset/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── data.yaml
│   │   ├── README.md
│   │   └── ANNOTATION_GUIDELINES.md
│   │
│   └── raw/
│
├── models/
│   ├── best.pt
│   ├── val_metrics.json
│   └── runs/
│
├── src/
│   └── data/
│       ├── __init__.py
│       ├── auto_annotate.py
│       ├── create_contact_sheet.py
│       ├── inspect_images.py
│       ├── layout_detector.py
│       ├── prepare_dataset.py
│       └── validate_annotations.py
│
├── inference.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
Installation

Clone the repository and install the required dependencies:

bash
pip install -r requirements.txt

It is recommended to use a virtual environment.

bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Training

The training pipeline is provided through:

text
train.py

The model can be trained using the prepared YOLO-format dataset.

The dataset configuration is available at:

text
data/dataset/data.yaml

Training artifacts and large dataset files are excluded from version control.

Robustness Considerations

Historical manuscripts present several challenges that differ from conventional object-detection datasets.

The pipeline considers conditions including:

Faded ink
Uneven illumination
Bleed-through
Blur
Skewed pages
Stains
Page damage
Multiple text columns
Marginal annotations
Irregular page layouts

The preprocessing and heuristic stages use image-level and layout-level information to complement the learned detector under these conditions.

Limitations

The current system is a compact prototype developed for a limited manuscript dataset.

The main limitations are:

Limited number of training images
Class imbalance, particularly for header and footer
Automatically generated annotations are not equivalent to expert annotations
Some highly degraded or unusual layouts may remain difficult to detect
Recall of the learned detector is currently lower than desired

These limitations leave clear opportunities for improvement with a larger, expert-annotated dataset and additional training experiments.

Future Improvements

Potential improvements include:

Expanding the dataset with more manuscript sources and layouts
Increasing the number of examples for underrepresented classes
Creating a larger expert-verified annotation set
Experimenting with larger YOLO architectures
Applying manuscript-specific augmentation for degradation and illumination
Improving header/footer detection using stronger positional and layout reasoning
Evaluating performance separately across different manuscript styles and degradation levels
Exploring document-layout-specific transformer architectures
Reproducibility

The repository contains:

Training code
Inference code
Dataset preparation utilities
Automatic annotation pipeline
Annotation validation
Model checkpoint
Validation metrics
Dependency specification
Documentation

Large raw datasets and generated image folders are excluded from Git using .gitignore.

Assignment Deliverables
Requirement	Status
Python source code	✓
Modular pipeline	✓
Single-image inference	✓
Batch/folder inference	✓
Five target classes	✓
Bounding box localization	✓
Confidence scores	✓
Annotated output images	✓
JSON metadata output	✓
Original images preserved	✓
Bounding boxes within page boundaries	✓
Dataset preparation	✓
Annotation validation	✓
README documentation	✓
requirements.txt	✓
Trained model checkpoint	✓
Conclusion

This project combines a lightweight YOLO-based object detector with image-processing and layout-analysis heuristics to provide an end-to-end manuscript layout detection pipeline.

The focus is not only on detecting text regions, but on distinguishing where different types of content occur on a manuscript page and producing structured, machine-readable results suitable for downstream document analysis.