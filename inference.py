"""
Manuscript layout detection inference.

Usage:
    python inference.py --input ./data/test_images --output ./results
    python inference.py --input ./data/dataset/images/test --output ./results
    python inference.py --input path/to/image.jpg --output ./results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler",
}

CLASS_COLORS = {
    0: (255, 255, 0),
    1: (255, 0, 255),
    2: (0, 255, 0),
    3: (0, 165, 255),
    4: (0, 255, 255),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(round(min(max(0.0, x1), width - 1)))
    y1 = int(round(min(max(0.0, y1), height - 1)))
    x2 = int(round(min(max(0.0, x2), width - 1)))
    y2 = int(round(min(max(0.0, y2), height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_detections(image, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = (
            det["bbox"]["x_min"],
            det["bbox"]["y_min"],
            det["bbox"]["x_max"],
            det["bbox"]["y_max"],
        )
        class_id = det["class_id"]
        color = CLASS_COLORS.get(class_id, (0, 255, 0))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, th + 6)
        cv2.rectangle(image, (x1, label_y - th - 6), (x1 + tw + 6, label_y + baseline), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def run_inference(
    input_path: Path,
    output_path: Path,
    weights_path: Path,
    conf: float,
) -> Path:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found at: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_path / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    metadata: list[dict] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        detections: list[dict] = []

        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = clip_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3], width, height)
                    class_id = int(box.cls[0].item())
                    # Preserve the annotation pipeline's conservative policy:
                    # page-wide main-text regions are usually parchment texture
                    # or a blank stamped leaf, not a localized text layout.
                    if class_id == 2 and (x2 - x1) / max(width, 1) > 0.80:
                        continue
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                            "confidence": float(box.conf[0].item()),
                            "bbox": {
                                "x_min": x1,
                                "y_min": y1,
                                "x_max": x2,
                                "y_max": y2,
                            },
                        }
                    )

        annotated = image.copy()
        draw_detections(annotated, detections)
        annotated_path = annotated_dir / image_path.name
        cv2.imwrite(str(annotated_path), annotated)

        metadata.append(
            {
                "image_name": image_path.name,
                "width": width,
                "height": height,
                "detections": detections,
            }
        )

    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run manuscript layout detection inference.")
    parser.add_argument("--input", required=True, help="Path to an image file or directory of images.")
    parser.add_argument("--output", required=True, help="Directory for annotated images and JSON metadata.")
    parser.add_argument(
        "--weights",
        default=str(project_dir / "models" / "best.pt"),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fallback = project_dir / "data" / "dataset" / "images" / "test"
        print(f"[ERROR] Input path not found: {input_path.resolve()}")
        print(f"[INFO] Dataset test images are at: {fallback}")
        print("[INFO] Example: python inference.py --input ./data/dataset/images/test --output ./results")
        sys.exit(1)

    metadata_path = run_inference(
        input_path=input_path,
        output_path=Path(args.output),
        weights_path=Path(args.weights),
        conf=args.conf,
    )
    print(f"[SUCCESS] Inference complete. Metadata: {metadata_path.resolve()}")
    print(f"[INFO] Annotated images: {(Path(args.output) / 'annotated').resolve()}")
    print("[INFO] Original input images were not modified.")


if __name__ == "__main__":
    main()
