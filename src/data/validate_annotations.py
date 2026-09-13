"""
Validate existing YOLO label files for format and coordinate issues.

Fixes only technical problems (out-of-range coordinates, extra whitespace,
non-finite values). Does not invent new boxes or change class IDs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_and_fix_line(line: str) -> tuple[str | None, str | None]:
    """
    Return (fixed_line, error). If the line should be dropped, fixed_line is None.
    """
    stripped = line.strip()
    if not stripped:
        return None, None

    parts = stripped.split()
    if len(parts) != 5:
        return None, f"expected 5 fields, got {len(parts)}"

    try:
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:])
    except ValueError:
        return None, "non-numeric values"

    if class_id not in CLASS_NAMES:
        return None, f"invalid class_id {class_id}"

    if not all(map(lambda v: v == v and abs(v) != float("inf"), (xc, yc, bw, bh))):
        return None, "non-finite coordinate"

    xmin = xc - bw / 2.0
    xmax = xc + bw / 2.0
    ymin = yc - bh / 2.0
    ymax = yc + bh / 2.0

    xmin = min(1.0, max(0.0, xmin))
    xmax = min(1.0, max(0.0, xmax))
    ymin = min(1.0, max(0.0, ymin))
    ymax = min(1.0, max(0.0, ymax))

    real_w = xmax - xmin
    real_h = ymax - ymin
    if real_w <= 1e-6 or real_h <= 1e-6:
        return None, "zero-area box after clipping"

    real_xc = xmin + real_w / 2.0
    real_yc = ymin + real_h / 2.0
    return f"{class_id} {real_xc:.6f} {real_yc:.6f} {real_w:.6f} {real_h:.6f}", None


def validate_split(images_dir: Path, labels_dir: Path, write_fixes: bool) -> dict:
    image_files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

    missing_labels = []
    empty_labels = []
    invalid_lines = 0
    fixed_files = 0
    box_count = 0
    class_counts = Counter()

    for image_path in image_files:
        label_path = label_files.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path.name)
            continue

        original = label_path.read_text(encoding="utf-8")
        kept: list[str] = []
        file_had_error = False
        for raw_line in original.splitlines():
            fixed, err = parse_and_fix_line(raw_line)
            if err:
                invalid_lines += 1
                file_had_error = True
                continue
            if fixed is None:
                continue
            kept.append(fixed)
            box_count += 1
            class_counts[int(fixed.split()[0])] += 1

        if not kept:
            empty_labels.append(label_path.name)

        rewritten = "\n".join(kept)
        if kept:
            rewritten += "\n"
        if write_fixes and rewritten != original:
            label_path.write_text(rewritten, encoding="utf-8")
            fixed_files += 1
        elif file_had_error:
            fixed_files += 1

    extra_labels = sorted(stem for stem in label_files if stem not in {p.stem for p in image_files})

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "extra_labels": extra_labels,
        "invalid_lines": invalid_lines,
        "fixed_files": fixed_files,
        "boxes": box_count,
        "class_counts": class_counts,
    }


def print_split_report(name: str, stats: dict) -> None:
    print(f" Split: {name}")
    print(f"   Images                 : {stats['images']}")
    print(f"   Label files            : {stats['labels']}")
    print(f"   Missing labels         : {len(stats['missing_labels'])}")
    print(f"   Empty label files      : {len(stats['empty_labels'])}")
    print(f"   Extra label files      : {len(stats['extra_labels'])}")
    print(f"   Invalid lines dropped  : {stats['invalid_lines']}")
    print(f"   Files rewritten        : {stats['fixed_files']}")
    print(f"   Valid boxes            : {stats['boxes']}")
    for cid in range(5):
        print(f"     {cid} {CLASS_NAMES[cid]:<10}: {stats['class_counts'][cid]}")
    if stats["missing_labels"][:5]:
        print(f"   Missing examples       : {stats['missing_labels'][:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO annotations.")
    parser.add_argument("--dataset", type=str, default="data/dataset")
    parser.add_argument("--no-write", action="store_true", help="Report only; do not rewrite files.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print("=" * 65)
    print(" YOLO Annotation Validation")
    print("=" * 65)

    ok = True
    for split in ("train", "val", "test"):
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        if not images_dir.exists():
            print(f" Split: {split} — images directory missing")
            ok = False
            continue
        stats = validate_split(images_dir, labels_dir, write_fixes=not args.no_write)
        print_split_report(split, stats)
        if split in ("train", "val") and stats["missing_labels"]:
            ok = False
        if split == "train" and stats["boxes"] == 0:
            ok = False
        if split == "val" and stats["boxes"] == 0:
            print("   [WARN] Validation split has no boxes; metrics will be undefined.")
        print("-" * 65)

    if not ok:
        print("[ERROR] Annotation validation failed.")
        sys.exit(1)
    print("[SUCCESS] Annotation format validation complete.")


if __name__ == "__main__":
    main()
