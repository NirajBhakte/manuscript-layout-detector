"""
Manuscript Layout Region Detection — YOLO training pipeline.

Trains a pretrained lightweight Ultralytics YOLO model on data/dataset/data.yaml
and copies the best weights to models/best.pt.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PRETRAINED_MODEL = "yolo11n.pt"
EPOCHS = 40
IMGSZ = 640


def extract_metrics(results) -> dict[str, float]:
    metrics = getattr(results, "results_dict", {}) or {}
    return {
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    data_yaml = project_dir / "data" / "dataset" / "data.yaml"
    if not data_yaml.exists():
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 0 if sys.platform.startswith("win") else 8
    batch = 8 if device == "cpu" else 16

    print("=" * 65)
    print(" Manuscript Layout Detection — YOLO Training")
    print("=" * 65)
    print(f" Pretrained Model       : {PRETRAINED_MODEL}")
    print(f" Dataset Config         : {data_yaml}")
    print(f" Device                 : {device.upper()}")
    print(f" Epochs                 : {EPOCHS}")
    print(f" Image Size             : {IMGSZ}")
    print(f" Batch                  : {batch}")
    print("=" * 65)

    model = YOLO(PRETRAINED_MODEL)
    runs_dir = project_dir / "models" / "runs"
    model.train(
        data=str(data_yaml),
        plots=True,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device,
        batch=batch,
        workers=workers,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best_weights_path = runs_dir / "train" / "weights" / "best.pt"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_best_pt = models_dir / "best.pt"
    if not best_weights_path.exists():
        print(f"[ERROR] Best weights not found at: {best_weights_path}")
        sys.exit(1)
    shutil.copy2(best_weights_path, target_best_pt)

    eval_model = YOLO(str(target_best_pt))
    val_results = eval_model.val(data=str(data_yaml), imgsz=IMGSZ, device=device, workers=workers, split="val")
    metrics = extract_metrics(val_results)

    metrics_path = models_dir / "val_metrics.json"
    payload = {
        "best_model": str(target_best_pt),
        "epochs": EPOCHS,
        "pretrained_model": PRETRAINED_MODEL,
        **metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 65)
    print(" Validation Metrics")
    print(f"   Best Model Path  : {target_best_pt}")
    print(f"   Precision        : {metrics['precision']:.4f}")
    print(f"   Recall           : {metrics['recall']:.4f}")
    print(f"   mAP50            : {metrics['mAP50']:.4f}")
    print(f"   mAP50-95         : {metrics['mAP50-95']:.4f}")
    print(f"   Metrics JSON     : {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
