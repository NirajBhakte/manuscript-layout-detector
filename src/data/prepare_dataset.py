"""
Dataset Preparation and Stratified Train/Validation/Test Split Pipeline.

Recursively discovers raw manuscript images from input directory (`data/raw/`),
verifies image readability, categorizes images by manuscript source (including
distinguishing root-level filename patterns), performs a stratified randomized
split into train, val, and test subsets with reproducible random seed, copies
selected images into destination split folders, and generates a CSV split manifest.

Safety Features:
- Does NOT alter, delete, or rename anything inside `data/raw/`.
- Does NOT generate fake label files or empty YOLO annotation files.
- Fails safely if destination split folders already contain files unless `--force` is passed.
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """
    Validate that train, validation, and test ratios sum to 1.0.

    Args:
        train_ratio (float): Target ratio for training set.
        val_ratio (float): Target ratio for validation set.
        test_ratio (float): Target ratio for test set.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within tolerance 1e-5).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0. Given: train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio:.4f})"
        )


def check_destination_safety(output_dir: Path, force: bool) -> None:
    """
    Check if destination dataset directories already contain files and handle safety logic.

    Args:
        output_dir (Path): Destination dataset root path.
        force (bool): If True, clear existing split files.

    Raises:
        SystemExit: If existing files are found and force is False.
    """
    images_dir = output_dir / "images"
    manifest_path = output_dir / "split_manifest.csv"

    splits = ["train", "val", "test"]
    existing_files = []

    for split in splits:
        split_dir = images_dir / split
        if split_dir.exists() and split_dir.is_dir():
            files = [f for f in split_dir.iterdir() if f.is_file()]
            if files:
                existing_files.extend(files)

    if manifest_path.exists():
        existing_files.append(manifest_path)

    if existing_files:
        if not force:
            print("[ERROR] Existing dataset files detected in destination directory:")
            print(f"        Output directory: {output_dir.resolve()}")
            print(f"        Found {len(existing_files)} existing file(s).")
            print("[ERROR] Re-running will overwrite the dataset split.")
            print("        Pass --force to explicitly rebuild/overwrite the dataset split.")
            sys.exit(1)
        else:
            print("[INFO] --force flag supplied. Cleaning previous dataset split files...")
            for split in splits:
                split_dir = images_dir / split
                if split_dir.exists():
                    shutil.rmtree(split_dir)
            if manifest_path.exists():
                manifest_path.unlink()


def get_source_manuscript(file_path: Path, input_root: Path) -> str:
    """
    Determine manuscript source identifier for an image.

    - Subfolder images under input_root use their parent relative directory name.
    - Root-level images directly under input_root are categorized by filename prefix
      before the first underscore (e.g., '7003_*' -> 'raw_7003', '9688_*' -> 'raw_9688').

    Args:
        file_path (Path): Absolute or relative image path.
        input_root (Path): Root raw input directory.

    Returns:
        str: Manuscript source identifier.
    """
    rel_path = file_path.relative_to(input_root)
    if rel_path.parent != Path("."):
        return rel_path.parts[0]

    name_parts = file_path.name.split("_")
    if len(name_parts) > 1 and name_parts[0]:
        return f"raw_{name_parts[0]}"
    return "raw_other"


def discover_and_verify_images(
    input_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Recursively discover supported image files and verify readability.

    Args:
        input_dir (Path): Input raw dataset path.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Lists of readable and unreadable records.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input raw directory not found: {input_dir}")

    discovered_files = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    readable_records = []
    unreadable_records = []

    for file_path in discovered_files:
        rel_path = file_path.relative_to(input_dir)
        source_manuscript = get_source_manuscript(file_path, input_dir)

        record = {
            "file_path": file_path,
            "relative_path": rel_path.as_posix(),
            "filename": file_path.name,
            "source_manuscript": source_manuscript,
            "width": None,
            "height": None,
            "error": None,
        }

        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                w, h = img.size
                record["width"] = w
                record["height"] = h
                readable_records.append(record)
        except Exception as e:
            record["error"] = str(e)
            unreadable_records.append(record)

    return readable_records, unreadable_records


def allocate_source_splits(
    count: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[int, int, int]:
    """
    Calculate integer split counts (train, val, test) for a source group of size N
    using Hamilton (Largest Remainder) allocation with minimum representation rules.

    Args:
        count (int): Number of images in source group.
        train_ratio (float): Target train ratio.
        val_ratio (float): Target val ratio.
        test_ratio (float): Target test ratio.

    Returns:
        Tuple[int, int, int]: Allocations for (train_count, val_count, test_count).
    """
    if count <= 0:
        return (0, 0, 0)

    f_train = count * train_ratio
    f_val = count * val_ratio
    f_test = count * test_ratio

    n_train = int(f_train)
    n_val = int(f_val)
    n_test = int(f_test)

    if count >= 3:
        if val_ratio > 0 and n_val == 0:
            n_val = 1
        if test_ratio > 0 and n_test == 0:
            n_test = 1
        if n_train + n_val + n_test > count:
            n_train = count - n_val - n_test
    elif count == 2:
        n_train, n_val, n_test = 1, 1, 0
    elif count == 1:
        n_train, n_val, n_test = 1, 0, 0

    remainder = count - (n_train + n_val + n_test)
    if remainder > 0:
        fracs = [
            (f_train - int(f_train), 0, "train"),
            (f_val - int(f_val), 1, "val"),
            (f_test - int(f_test), 2, "test"),
        ]
        fracs.sort(key=lambda x: (-x[0], x[1]))
        for i in range(remainder):
            target_split = fracs[i][2]
            if target_split == "train":
                n_train += 1
            elif target_split == "val":
                n_val += 1
            elif target_split == "test":
                n_test += 1

    return (n_train, n_val, n_test)


def perform_stratified_split(
    readable_records: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    Perform randomized stratified splitting per manuscript source.

    Args:
        readable_records (List[Dict[str, Any]]): All readable image records.
        seed (int): Random seed for reproducibility.
        train_ratio (float): Train ratio.
        val_ratio (float): Val ratio.
        test_ratio (float): Test ratio.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
            Annotated records with split assigned and per-source split breakdown statistics.
    """
    source_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in readable_records:
        src = record["source_manuscript"]
        source_groups.setdefault(src, []).append(record)

    rng = random.Random(seed)
    split_records = []
    per_source_stats = {}

    for src in sorted(source_groups.keys()):
        group = source_groups[src]
        group_sorted = sorted(group, key=lambda x: x["relative_path"])
        rng.shuffle(group_sorted)

        count = len(group_sorted)
        n_train, n_val, n_test = allocate_source_splits(
            count, train_ratio, val_ratio, test_ratio
        )

        train_set = group_sorted[:n_train]
        val_set = group_sorted[n_train : n_train + n_val]
        test_set = group_sorted[n_train + n_val :]

        for item in train_set:
            item["split"] = "train"
            split_records.append(item)
        for item in val_set:
            item["split"] = "val"
            split_records.append(item)
        for item in test_set:
            item["split"] = "test"
            split_records.append(item)

        per_source_stats[src] = {
            "total": count,
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        }

    return split_records, per_source_stats


def copy_images_and_generate_manifest(
    split_records: List[Dict[str, Any]], output_dir: Path
) -> Path:
    """
    Copy split images to destination folders and create split_manifest.csv.

    Args:
        split_records (List[Dict[str, Any]]): List of assigned image records.
        output_dir (Path): Output dataset path.

    Returns:
        Path: Path to generated split_manifest.csv.
    """
    images_dir = output_dir / "images"
    for s in ["train", "val", "test"]:
        (images_dir / s).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "split_manifest.csv"
    used_filenames = set()
    manifest_rows = []

    for record in split_records:
        src_path = record["file_path"]
        src_manuscript = record["source_manuscript"]
        split = record["split"]
        orig_filename = record["filename"]

        rel_path = Path(record["relative_path"])
        if rel_path.parent != Path("."):
            dst_filename = f"{src_manuscript}_{orig_filename}"
        else:
            dst_filename = orig_filename

        if dst_filename in used_filenames:
            dst_filename = f"{src_manuscript}_{orig_filename}"
        used_filenames.add(dst_filename)

        record["image_filename"] = dst_filename
        dst_path = images_dir / split / dst_filename

        shutil.copy2(src_path, dst_path)

        orig_raw_posix = f"data/raw/{record['relative_path']}"

        manifest_rows.append(
            {
                "image_filename": dst_filename,
                "original_raw_path": orig_raw_posix,
                "source_manuscript": src_manuscript,
                "split": split,
                "width": record["width"],
                "height": record["height"],
            }
        )

    fieldnames = [
        "image_filename",
        "original_raw_path",
        "source_manuscript",
        "split",
        "width",
        "height",
    ]

    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


def print_summary(
    total_discovered: int,
    readable_records: List[Dict[str, Any]],
    unreadable_records: List[Dict[str, Any]],
    per_source_stats: Dict[str, Dict[str, int]],
    seed: int,
    output_dir: Path,
) -> None:
    """
    Print formatted execution summary and per-source split breakdown table.

    Args:
        total_discovered (int): Total discovered images.
        readable_records (List[Dict[str, Any]]): Readable image records.
        unreadable_records (List[Dict[str, Any]]): Unreadable image records.
        per_source_stats (Dict[str, Dict[str, int]]): Per-source counts dictionary.
        seed (int): Random seed used.
        output_dir (Path): Destination output path.
    """
    total_readable = len(readable_records)
    total_unreadable = len(unreadable_records)

    tot_train = sum(stats["train"] for stats in per_source_stats.values())
    tot_val = sum(stats["val"] for stats in per_source_stats.values())
    tot_test = sum(stats["test"] for stats in per_source_stats.values())

    p_train = (tot_train / total_readable * 100) if total_readable > 0 else 0.0
    p_val = (tot_val / total_readable * 100) if total_readable > 0 else 0.0
    p_test = (tot_test / total_readable * 100) if total_readable > 0 else 0.0

    print("=" * 65)
    print(" Dataset Preparation & Stratified Split Summary")
    print("=" * 65)
    print(f" Destination Directory    : {output_dir.resolve()}")
    print(f" Random Seed               : {seed} (Split is reproducible)")
    print(f" Total Discovered Images   : {total_discovered}")
    print(f" Readable Images           : {total_readable}")
    print(f" Unreadable Images         : {total_unreadable}")
    print(f" Number of Source Manuscripts: {len(per_source_stats)}")
    print("-" * 65)
    print(f" Final Train Count         : {tot_train} ({p_train:.2f}%)")
    print(f" Final Validation Count    : {tot_val} ({p_val:.2f}%)")
    print(f" Final Test Count          : {tot_test} ({p_test:.2f}%)")
    print("=" * 65)
    print("\n Per-Source Manuscript Split Breakdown Table:")
    print("=" * 65)
    print(
        f" {'Source Manuscript':<22} | {'Total':>6} | {'Train':>6} | {'Val':>5} | {'Test':>5} |"
    )
    print(f" {'-'*22}-|-{'-'*6}-|-{'-'*6}-|-{'-'*5}-|-{'-'*5}-|")

    for src in sorted(per_source_stats.keys()):
        st = per_source_stats[src]
        print(
            f" {src:<22} | {st['total']:>6} | {st['train']:>6} | {st['val']:>5} | {st['test']:>5} |"
        )

    print(f" {'-'*22}-|-{'-'*6}-|-{'-'*6}-|-{'-'*5}-|-{'-'*5}-|")
    print(
        f" {'TOTAL':<22} | {total_readable:>6} | {tot_train:>6} | {tot_val:>5} | {tot_test:>5} |"
    )
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset by performing randomized stratified train/val/test split across manuscript sources."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Path to raw dataset input directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset",
        help="Path to output dataset directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Target proportion for training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Target proportion for validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Target proportion for test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly overwrite/rebuild existing output dataset split.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    check_destination_safety(output_dir, args.force)

    readable_records, unreadable_records = discover_and_verify_images(input_dir)
    total_discovered = len(readable_records) + len(unreadable_records)

    split_records, per_source_stats = perform_stratified_split(
        readable_records,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    manifest_path = copy_images_and_generate_manifest(split_records, output_dir)

    print_summary(
        total_discovered,
        readable_records,
        unreadable_records,
        per_source_stats,
        args.seed,
        output_dir,
    )
    print(f"\n[SUCCESS] Dataset preparation complete.")
    print(f"[INFO] Manifest written to: {manifest_path.resolve()}\n")


if __name__ == "__main__":
    main()
