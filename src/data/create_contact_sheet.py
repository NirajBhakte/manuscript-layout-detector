"""
Contact Sheet Generator Utility.

Recursively scans a dataset directory for manuscript images and generates a visual grid (contact sheet)
with aspect-ratio-preserved thumbnails and filenames for visual inspection and page selection.
"""

import argparse
import math
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def get_image_files(input_dir: Path) -> List[Path]:
    """
    Recursively scan input directory for supported image files.

    Args:
        input_dir (Path): Root directory to scan.

    Returns:
        List[Path]: Sorted list of image file paths.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or invalid: {input_dir}")

    files = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    return files


def create_contact_sheet(
    image_paths: List[Path],
    output_path: Path,
    columns: int = 10,
    thumb_width: int = 200,
    thumb_height: int = 280,
    padding: int = 10,
    label_height: int = 25,
) -> int:
    """
    Generate and save a visual contact sheet containing thumbnails of images.

    Args:
        image_paths (List[Path]): List of image paths.
        output_path (Path): Destination image output file path.
        columns (int): Number of thumbnail columns per row.
        thumb_width (int): Maximum thumbnail width in pixels.
        thumb_height (int): Maximum thumbnail height in pixels.
        padding (int): Padding around each cell in pixels.
        label_height (int): Height allocated for filename label in pixels.

    Returns:
        int: Total number of successfully rendered images in the contact sheet.
    """
    if not image_paths:
        print("[WARNING] No image files found to render.")
        return 0

    readable_images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            readable_images.append(path)
        except Exception as e:
            print(f"[WARNING] Skipping unreadable image {path.name}: {e}")

    total_images = len(readable_images)
    if total_images == 0:
        print("[WARNING] No readable images found.")
        return 0

    rows = math.ceil(total_images / columns)
    cell_width = thumb_width + (2 * padding)
    cell_height = thumb_height + label_height + (2 * padding)

    sheet_width = columns * cell_width
    sheet_height = rows * cell_height

    canvas = Image.new("RGB", (sheet_width, sheet_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    rendered_count = 0
    for idx, path in enumerate(readable_images):
        col = idx % columns
        row = idx // columns

        cell_x = col * cell_width + padding
        cell_y = row * cell_height + padding

        try:
            with Image.open(path) as img:
                img_copy = img.convert("RGB")
                img_copy.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                w, h = img_copy.size

                # Center thumbnail inside thumbnail container cell
                offset_x = cell_x + (thumb_width - w) // 2
                offset_y = cell_y + (thumb_height - h) // 2
                canvas.paste(img_copy, (offset_x, offset_y))

                # Draw subtle cell bounding box
                draw.rectangle(
                    [cell_x, cell_y, cell_x + thumb_width, cell_y + thumb_height],
                    outline=(220, 220, 220),
                    width=1,
                )

                # Format and center label string below thumbnail
                filename = path.name
                if len(filename) > 25:
                    filename = filename[:22] + "..."

                bbox = font.getbbox(filename)
                text_w = bbox[2] - bbox[0]
                text_x = cell_x + max(0, (thumb_width - text_w) // 2)
                text_y = cell_y + thumb_height + 5

                draw.text((text_x, text_y), filename, fill=(40, 40, 40), font=font)
                rendered_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to render image {path.name}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)
    print(f"[INFO] Contact sheet saved to: {output_path.resolve()}")
    return rendered_count


def main():
    parser = argparse.ArgumentParser(description="Create a visual contact sheet of manuscript images.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing raw images.")
    parser.add_argument("--output", type=str, required=True, help="Output image path (e.g., results/contact_sheet.jpg).")
    parser.add_argument("--columns", type=int, default=10, help="Number of columns in grid.")
    parser.add_argument("--thumb-width", type=int, default=200, help="Thumbnail max width in pixels.")
    parser.add_argument("--thumb-height", type=int, default=280, help="Thumbnail max height in pixels.")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    image_paths = get_image_files(input_dir)
    rendered = create_contact_sheet(
        image_paths=image_paths,
        output_path=output_path,
        columns=args.columns,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )
    print(f"[SUMMARY] Rendered contact sheet containing {rendered} images out of {len(image_paths)} found.")


if __name__ == "__main__":
    main()
