"""
Manuscript Layout Auto-Annotation Utility.

Generates initial YOLO object detection annotations for manuscript images by combining
document image analysis (adaptive binarization, projection profile gap splitting,
morphological component analysis, tight ink trimming) with conservative layout classification.

Target Classes (0-4):
- 0: header     (Top-margin titles, running headers, top folio numbers)
- 1: footer     (Bottom-margin catchwords, quire signatures, page numbers)
- 2: main_text  (Primary manuscript body text blocks / columns / panels)
- 3: side_text  (Marginalia, glosses, side commentary in outer margins)
- 4: filler     (Isolated decorative ornaments, English/pencil annotations)

Features:
- Projection profile gap detection to split multi-column pages and two-page spreads into separate text blocks.
- Strict 80% area safety rule: rejects any box covering >80% of the image area to eliminate giant full-page boxes.
- Tight ink boundary trimming to exclude blank parchment background margins.
- Exports normalized YOLO format .txt label files into destination directory.
- Renders color-coded annotated preview images in preview directory.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import cv2
import numpy as np

CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler",
}

CLASS_COLORS = {
    0: (255, 255, 0),    # Cyan / Yellow-Cyan for Header
    1: (255, 0, 255),    # Magenta for Footer
    2: (0, 255, 0),      # Green for Main Text
    3: (0, 165, 255),    # Orange for Side Text
    4: (0, 255, 255),    # Yellow for Filler
}


def _legacy_extract_layout_boxes(
    img_path: Path,
) -> Tuple[int, int, List[Tuple[int, str, float, float, float, float]]]:
    """
    Extract layout region bounding boxes for a manuscript image using
    projection profile gap splitting, morphological component analysis, and tight ink trimming.

    Args:
        img_path (Path): Path to target manuscript image.

    Returns:
        Tuple[int, int, List[Tuple[int, str, float, float, float, float]]]:
            Image height H, width W, and list of detected boxes:
            (class_id, class_name, x_center, y_center, width, height)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image file: {img_path}")

    H, W, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to separate ink from parchment background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Fine morphological kernel to group character strokes into line components
    kw_line = max(10, int(W * 0.012))
    kh_line = max(3, int(H * 0.003))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_line, kh_line))
    dilated_line = cv2.dilate(thresh, kernel_line, iterations=2)

    contours_line, _ = cv2.findContours(
        dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    line_rects = []
    for cnt in contours_line:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > W * 0.015 and h > H * 0.005 and (w * h) > (W * H * 0.0003):
            line_rects.append((x, y, w, h))

    if not line_rects:
        return H, W, []

    # Build ink binary mask
    ink_mask = np.zeros((H, W), dtype=np.uint8)
    for x, y, w, h in line_rects:
        cv2.rectangle(ink_mask, (x, y), (x + w, y + h), 255, -1)

    # Projection profile analysis to split page spreads or multi-column layouts
    row_sums = np.sum(ink_mask == 255, axis=1)
    col_sums = np.sum(ink_mask == 255, axis=0)

    # Check horizontal gap near middle (y = 0.35 to 0.65) for vertical page spreads
    mid_start, mid_end = int(H * 0.35), int(H * 0.65)
    h_split = None
    if mid_end > mid_start:
        min_row_sum = np.min(row_sums[mid_start:mid_end])
        if min_row_sum < W * 0.04:
            gap_indices = np.where(row_sums[mid_start:mid_end] == min_row_sum)[0]
            if len(gap_indices) > 0:
                h_split = mid_start + gap_indices[len(gap_indices) // 2]

    # Check vertical gap near middle (x = 0.35 to 0.65) for two-column pages
    mid_col_start, mid_col_end = int(W * 0.35), int(W * 0.65)
    v_split = None
    if mid_col_end > mid_col_start:
        min_col_sum = np.min(col_sums[mid_col_start:mid_col_end])
        if min_col_sum < H * 0.04:
            col_gap_indices = np.where(col_sums[mid_col_start:mid_col_end] == min_col_sum)[0]
            if len(col_gap_indices) > 0:
                v_split = mid_col_start + col_gap_indices[len(col_gap_indices) // 2]

    # Partition page into layout panels/regions if gaps exist
    panels = []
    if h_split is not None:
        panels.append((0, 0, W, h_split))
        panels.append((0, h_split, W, H - h_split))
    elif v_split is not None:
        panels.append((0, 0, v_split, H))
        panels.append((v_split, 0, W - v_split, H))
    else:
        panels.append((0, 0, W, H))

    raw_boxes = []

    for px, py, pw, ph in panels:
        panel_rects = [
            r
            for r in line_rects
            if r[0] >= px
            and (r[0] + r[2]) <= (px + pw)
            and r[1] >= py
            and (r[1] + r[3]) <= (py + ph)
        ]
        if not panel_rects:
            continue

        headers = []
        footers = []
        side_texts = []
        fillers = []
        main_line_rects = []

        for rx, ry, rw, rh in panel_rects:
            r_nx, r_ny, r_nw, r_nh = rx / W, ry / H, rw / W, rh / H
            r_cx, r_cy = r_nx + r_nw / 2.0, r_ny + r_nh / 2.0

            # Header check: Top margin
            if (r_ny <= 0.08 or r_cy <= 0.12) and r_nh <= 0.10 and r_nw <= 0.80:
                headers.append((rx, ry, rw, rh))
            # Footer check: Bottom margin
            elif (r_ny + r_nh >= 0.92 or r_cy >= 0.88) and r_nh <= 0.10 and r_nw <= 0.80:
                footers.append((rx, ry, rw, rh))
            # Side text check: Left or Right margin
            elif (r_cx <= 0.18 or r_cx >= 0.82) and 0.12 <= r_cy <= 0.88 and r_nw <= 0.28:
                side_texts.append((rx, ry, rw, rh))
            # Filler check: Small isolated decorative or pencil elements
            elif (r_nw * r_nh) <= 0.003 and (
                r_cx <= 0.20 or r_cx >= 0.80 or r_cy <= 0.15 or r_cy >= 0.85
            ):
                fillers.append((rx, ry, rw, rh))
            else:
                main_line_rects.append((rx, ry, rw, rh))

        def fit_tight_box(r_list: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
            if not r_list:
                return None
            x1 = min(r[0] for r in r_list)
            y1 = min(r[1] for r in r_list)
            x2 = max(r[0] + r[2] for r in r_list)
            y2 = max(r[1] + r[3] for r in r_list)
            return (x1, y1, x2 - x1, y2 - y1)

        # 0: header
        h_box = fit_tight_box(headers)
        if h_box:
            mx, my, mw, mh = h_box
            raw_boxes.append((0, "header", (mx + mw / 2.0) / W, (my + mh / 2.0) / H, mw / W, mh / H))

        # 1: footer
        f_box = fit_tight_box(footers)
        if f_box:
            mx, my, mw, mh = f_box
            raw_boxes.append((1, "footer", (mx + mw / 2.0) / W, (my + mh / 2.0) / H, mw / W, mh / H))

        # 3: side_text
        s_box = fit_tight_box(side_texts)
        if s_box:
            mx, my, mw, mh = s_box
            raw_boxes.append((3, "side_text", (mx + mw / 2.0) / W, (my + mh / 2.0) / H, mw / W, mh / H))

        # 4: filler
        fl_box = fit_tight_box(fillers)
        if fl_box:
            mx, my, mw, mh = fl_box
            raw_boxes.append((4, "filler", (mx + mw / 2.0) / W, (my + mh / 2.0) / H, mw / W, mh / H))

        # 2: main_text (coarse morphological aggregation of body line components)
        if main_line_rects:
            p_mask = np.zeros((H, W), dtype=np.uint8)
            for rx, ry, rw, rh in main_line_rects:
                cv2.rectangle(p_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)

            kw_m = max(15, int(W * 0.02))
            kh_m = max(10, int(H * 0.012))
            kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_m, kh_m))
            p_dilated = cv2.dilate(p_mask, kernel_m, iterations=2)

            p_cnts, _ = cv2.findContours(
                p_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in p_cnts:
                bx, by, bw, bh = cv2.boundingRect(cnt)

                # Trim box to actual tight ink points inside this contour
                mask_cnt = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                ink_pts = np.where((p_mask > 0) & (mask_cnt > 0))

                if len(ink_pts[0]) > 0:
                    y_min, y_max = np.min(ink_pts[0]), np.max(ink_pts[0])
                    x_min, x_max = np.min(ink_pts[1]), np.max(ink_pts[1])
                    bx, by, bw, bh = (
                        x_min,
                        y_min,
                        x_max - x_min + 1,
                        y_max - y_min + 1,
                    )

                bnx, bny, bnw, bnh = bx / W, by / H, bw / W, bh / H
                area = bnw * bnh

                if area >= 0.008:
                    bcx, bcy = bnx + bnw / 2.0, bny + bnh / 2.0
                    raw_boxes.append((2, "main_text", bcx, bcy, bnw, bnh))

    # Clip coordinates strictly within [0.0, 1.0] & apply 80% max area safety rule
    clipped_boxes = []
    for cid, cname, xc, yc, bw, bh in raw_boxes:
        xmin = max(0.0, xc - bw / 2.0)
        xmax = min(1.0, xc + bw / 2.0)
        ymin = max(0.0, yc - bh / 2.0)
        ymax = min(1.0, yc + bh / 2.0)

        real_w = max(0.001, xmax - xmin)
        real_h = max(0.001, ymax - ymin)

        # STRICT SAFETY RULE: Reject any proposed text-region box whose area is > 80% of image area
        if real_w * real_h > 0.80:
            continue

        real_xc = xmin + real_w / 2.0
        real_yc = ymin + real_h / 2.0

        clipped_boxes.append(
            (
                cid,
                cname,
                round(real_xc, 6),
                round(real_yc, 6),
                round(real_w, 6),
                round(real_h, 6),
            )
        )

    return H, W, clipped_boxes


# The original morphology-only detector is retained above for reference, but it
# merged neighbouring folios into page-wide regions.  The production detector
# first identifies individual parchment leaves, then uses ink profiles and real
# gutters to localize regions inside each leaf.
try:  # Package import when invoked with ``python -m``.
    from .layout_detector import extract_layout_boxes
except ImportError:  # Direct-script execution from ``src/data``.
    from layout_detector import extract_layout_boxes


def write_yolo_annotation_file(
    label_path: Path,
    boxes: List[Tuple[int, str, float, float, float, float]],
) -> None:
    """
    Write normalized YOLO bounding box label file (.txt).

    Args:
        label_path (Path): Path to output .txt file.
        boxes (List[Tuple[int, str, float, float, float, float]]): List of box tuples.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, mode="w", encoding="utf-8") as f:
        for cid, _, xc, yc, bw, bh in boxes:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def generate_preview_image(
    img_path: Path,
    boxes: List[Tuple[int, str, float, float, float, float]],
    preview_path: Path,
) -> None:
    """
    Generate an annotated visual preview image with color-coded bounding boxes and labels.

    Args:
        img_path (Path): Original image path.
        boxes (List[Tuple[int, str, float, float, float, float]]): Detected boxes.
        preview_path (Path): Output preview image path.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return

    H, W, _ = img.shape
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    for cid, cname, xc, yc, bw, bh in boxes:
        xmin = int((xc - bw / 2.0) * W)
        ymin = int((yc - bh / 2.0) * H)
        xmax = int((xc + bw / 2.0) * W)
        ymax = int((yc + bh / 2.0) * H)

        color = CLASS_COLORS.get(cid, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        # Format label text
        label_text = f"{cid}:{cname}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        label_ymin = max(ymin, th + 5)
        cv2.rectangle(
            img,
            (xmin, label_ymin - th - 5),
            (xmin + tw + 6, label_ymin + baseline),
            color,
            -1,
        )
        cv2.putText(
            img,
            label_text,
            (xmin + 3, label_ymin - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(preview_path), img)


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial YOLO object detection annotations for manuscript layout regions."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/dataset/images/train",
        help="Input directory containing training manuscript images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset/labels/train",
        help="Output directory for YOLO label files (.txt).",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="data/dataset/auto_annotated_previews",
        help="Directory to save annotated preview images.",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=0,
        help="Maximum number of preview images to render (0 for all).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    preview_dir = Path(args.preview_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        ]
    )

    total_images = len(image_paths)
    images_with_detections = 0
    total_boxes = 0
    rejected_over_80 = 0
    class_counts = Counter()
    errors = []

    print("=" * 65)
    print(" Manuscript Layout Auto-Annotation Execution (Improved Layout Pipeline)")
    print("=" * 65)
    print(f" Input Images Directory  : {input_dir.resolve()}")
    print(f" Output Labels Directory  : {output_dir.resolve()}")
    print(f" Preview Images Directory : {preview_dir.resolve()}")
    print(f" Total Images Found       : {total_images}")
    print("-" * 65)

    for idx, img_path in enumerate(image_paths):
        try:
            H, W, boxes = extract_layout_boxes(img_path)

            if boxes:
                images_with_detections += 1
                total_boxes += len(boxes)
                for b in boxes:
                    class_counts[b[0]] += 1
                    if (b[4] * b[5]) > 0.80:
                        rejected_over_80 += 1

                label_path = output_dir / f"{img_path.stem}.txt"
                write_yolo_annotation_file(label_path, boxes)

            else:
                label_path = output_dir / f"{img_path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(label_path, mode="w", encoding="utf-8") as f:
                    pass

            # Render preview image for all images
            if args.max_previews == 0 or idx < args.max_previews:
                preview_path = preview_dir / f"{img_path.stem}_preview.jpg"
                generate_preview_image(img_path, boxes, preview_path)

        except Exception as e:
            errors.append(f"{img_path.name}: {str(e)}")

    print(f" Processed Images            : {total_images}")
    print(f" Images with Detections      : {images_with_detections}")
    print(f" Total Boxes Generated       : {total_boxes}")
    print(f" Boxes Rejected (Area > 80%) : {rejected_over_80}")
    print(f" Errors Encountered          : {len(errors)}")
    print("-" * 65)
    print(" Boxes per Class Breakdown:")
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        cnum = class_counts[cid]
        print(f"   Class {cid} ({cname:9}): {cnum:4} boxes")
    print("=" * 65)

    if errors:
        print("\n[WARNING] Errors during processing:")
        for err in errors:
            print(f"  - {err}")

    print(f"\n[SUCCESS] Auto-annotation process complete.")
    print(f"[INFO] YOLO label files written to: {output_dir.resolve()}")
    print(f"[INFO] Annotated previews saved to: {preview_dir.resolve()}\n")


if __name__ == "__main__":
    main()
