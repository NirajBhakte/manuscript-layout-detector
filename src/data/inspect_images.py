"""
Dataset Image Inspection Utility.

Recursively scans a specified raw dataset directory for image files, collects metadata 
(dimensions, aspect ratio, file size, readability status), identifies unreadable/corrupted images, 
and produces a formatted summary report and optional CSV inventory export.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def inspect_single_image(file_path: Path, input_root: Path) -> Dict[str, Any]:
    """
    Inspect a single image file for metadata and readability.

    Args:
        file_path (Path): Path to the target image file.
        input_root (Path): Root directory for relative path calculation.

    Returns:
        Dict[str, Any]: Dictionary containing metadata attributes or error status.
    """
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    rel_path = file_path.relative_to(input_root)
    source_folder = str(rel_path.parent) if rel_path.parent != Path(".") else input_root.name

    metadata = {
        "source_folder": source_folder,
        "filename": file_path.name,
        "relative_path": str(rel_path),
        "file_size_mb": round(file_size_mb, 4),
        "status": "readable",
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "error": None,
    }

    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            width, height = img.size
            metadata["width"] = width
            metadata["height"] = height
            metadata["aspect_ratio"] = round(width / height, 4) if height > 0 else 0.0
    except Exception as e:
        metadata["status"] = "corrupted"
        metadata["error"] = str(e)

    return metadata


def scan_dataset(input_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Recursively scan input_dir for supported images and inspect each file.

    Args:
        input_dir (Path): Root input directory to scan.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Lists of readable and unreadable records.
    """
    readable_records = []
    corrupted_records = []

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    all_files = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    for file_path in all_files:
        record = inspect_single_image(file_path, input_dir)
        if record["status"] == "readable":
            readable_records.append(record)
        else:
            corrupted_records.append(record)

    return readable_records, corrupted_records


def generate_summary(
    readable: List[Dict[str, Any]], corrupted: List[Dict[str, Any]], input_dir: Path
) -> Dict[str, Any]:
    """
    Generate aggregate summary statistics from inspection results.

    Args:
        readable (List[Dict[str, Any]]): Readable image records.
        corrupted (List[Dict[str, Any]]): Unreadable image records.
        input_dir (Path): Root dataset input directory.

    Returns:
        Dict[str, Any]: Consolidated metrics dictionary.
    """
    total_count = len(readable) + len(corrupted)
    readable_count = len(readable)
    unreadable_count = len(corrupted)

    source_folders = set()
    total_size_mb = 0.0

    widths = []
    heights = []
    aspect_ratios = []

    for r in readable + corrupted:
        source_folders.add(r["source_folder"])
        total_size_mb += r["file_size_mb"]

    for r in readable:
        widths.append(r["width"])
        heights.append(r["height"])
        aspect_ratios.append(r["aspect_ratio"])

    min_dim = (min(widths), min(heights)) if widths and heights else (0, 0)
    max_dim = (max(widths), max(heights)) if widths and heights else (0, 0)
    aspect_ratio_range = (min(aspect_ratios), max(aspect_ratios)) if aspect_ratios else (0.0, 0.0)

    return {
        "total_image_count": total_count,
        "readable_count": readable_count,
        "unreadable_count": unreadable_count,
        "source_folder_count": len(source_folders),
        "min_dimensions": min_dim,
        "max_dimensions": max_dim,
        "aspect_ratio_range": aspect_ratio_range,
        "total_size_mb": round(total_size_mb, 2),
    }


def print_summary(summary: Dict[str, Any], input_dir: Path, corrupted: List[Dict[str, Any]]) -> None:
    """
    Print formatted summary to standard output.

    Args:
        summary (Dict[str, Any]): Summary metrics dictionary.
        input_dir (Path): Input dataset path.
        corrupted (List[Dict[str, Any]]): List of corrupted file records.
    """
    print("=" * 60)
    print(f" Dataset Image Inspection Summary: {input_dir.resolve()}")
    print("=" * 60)
    print(f" Total image count        : {summary['total_image_count']}")
    print(f" Readable image count     : {summary['readable_count']}")
    print(f" Unreadable image count   : {summary['unreadable_count']}")
    print(f" Manuscript/source folders: {summary['source_folder_count']}")
    print(f" Min dimensions (W x H)   : {summary['min_dimensions'][0]} x {summary['min_dimensions'][1]} px")
    print(f" Max dimensions (W x H)   : {summary['max_dimensions'][0]} x {summary['max_dimensions'][1]} px")
    print(f" Aspect ratio range       : {summary['aspect_ratio_range'][0]} - {summary['aspect_ratio_range'][1]}")
    print(f" Total dataset size       : {summary['total_size_mb']} MB")
    print("=" * 60)

    if corrupted:
        print("\n[WARNING] Unreadable / Corrupted Files Found:")
        for c in corrupted:
            print(f" - {c['relative_path']} (Error: {c['error']})")
        print("=" * 60)


def save_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save inspection records to a CSV file.

    Args:
        records (List[Dict[str, Any]]): List of image records.
        output_path (Path): Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_folder",
        "filename",
        "relative_path",
        "status",
        "width",
        "height",
        "aspect_ratio",
        "file_size_mb",
        "error",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] Inventory successfully saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset manuscript images for metadata and readability.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset directory.")
    parser.add_argument("--output", type=str, required=False, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    readable, corrupted = scan_dataset(input_dir)
    summary = generate_summary(readable, corrupted, input_dir)

    all_records = readable + corrupted
    if args.output:
        save_csv(all_records, Path(args.output))

    print_summary(summary, input_dir, corrupted)


if __name__ == "__main__":
    main()
