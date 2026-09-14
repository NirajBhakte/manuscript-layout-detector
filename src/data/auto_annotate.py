"""Evidence-based pseudo-annotation for manuscript layout images.

The labels produced here are deliberately conservative.  They describe visibly
separate layout regions, rather than trying to make every dark mark on a page an
object.  In particular, a top/bottom band is *not* a header/footer on its own:
there must be a compact ink region in that margin and a visible gap to a body
text block.

Class mapping (kept stable for the YOLO dataset):
    0 header, 1 footer, 2 main_text, 3 side_text, 4 filler.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Tuple

import cv2
import numpy as np


CLASS_NAMES = {0: "header", 1: "footer", 2: "main_text", 3: "side_text", 4: "filler"}
CLASS_COLORS = {0: (255, 255, 0), 1: (255, 0, 255), 2: (0, 255, 0), 3: (0, 165, 255), 4: (0, 255, 255)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
Rect = Tuple[int, int, int, int]
YoloBox = Tuple[int, str, float, float, float, float]


def _odd(value: int) -> int:
    return max(3, int(value) | 1)


def _note_rejection(diagnostics: Optional[MutableMapping[str, int]], reason: str) -> None:
    if diagnostics is not None:
        diagnostics["rejected"] += 1
        diagnostics[f"rejected_{reason}"] += 1


def _read_bgr(image_path: Path) -> np.ndarray:
    """Read grayscale, BGR, or BGRA input as a reliable BGR image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image file: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape} for {image_path}")
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return (image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported channel count for {image_path}")


def _normalise_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten slow lighting/shadow changes but retain dark handwriting."""
    h, w = gray.shape
    kernel = _odd(min(151, max(31, int(min(h, w) * 0.10))))
    background = np.maximum(cv2.GaussianBlur(gray, (kernel, kernel), 0), 1)
    return cv2.divide(gray, background, scale=220)


def _find_leaf_rectangles(gray: np.ndarray) -> List[Rect]:
    """Find paper leaves and exclude the dark scanner surround.

    A scan can contain two leaves stacked vertically; handling each leaf
    independently prevents a page-height text box spanning both of them.
    """
    h, w = gray.shape
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    threshold = max(80.0, float(np.median(border)) + 24.0)
    bright = (gray > threshold).astype(np.uint8) * 255
    k = max(5, int(min(h, w) * 0.012))
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k))), iterations=2,
    )
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(max(3, k // 2)), _odd(max(3, k // 2)))),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    leaves: List[Rect] = []
    for index in range(1, count):
        x, y, rw, rh, area = (int(value) for value in stats[index])
        if area >= 0.045 * w * h and rw >= 0.18 * w and rh >= 0.14 * h:
            leaves.append((x, y, rw, rh))
    return sorted(leaves, key=lambda rect: (rect[1], rect[0])) or [(0, 0, w, h)]


def _clean_ink_mask(gray: np.ndarray, corrected: np.ndarray, leaf: Rect) -> np.ndarray:
    """Extract strong ink; reject paper grain, edges, rules, and speckles."""
    h, w = gray.shape
    lx, ly, lw, lh = leaf
    mask = np.zeros((h, w), dtype=np.uint8)
    # The physical page frame and torn perimeter are frequent false positives
    # on this collection; a four-percent trim removes them while retaining
    # genuine marginal writing set inside the parchment.
    inset_x, inset_y = max(2, int(lw * 0.040)), max(2, int(lh * 0.040))
    x0, x1, y0, y1 = lx + inset_x, lx + lw - inset_x, ly + inset_y, ly + lh - inset_y
    if x1 - x0 < 12 or y1 - y0 < 12:
        return mask
    roi = corrected[y0:y1, x0:x1]
    background = float(np.percentile(roi, 70))
    dark_limit = min(195.0, max(62.0, background - 48.0))
    dark = (roi < dark_limit).astype(np.uint8) * 255
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(roi, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, _odd(min(91, max(31, int(min(roi.shape) * 0.06)))), 9,
    )
    ink = cv2.morphologyEx(cv2.bitwise_and(dark, adaptive), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(7, int(0.000006 * lw * lh))
    cleaned = np.zeros_like(ink)
    for index in range(1, count):
        _, _, cw, ch, area = (int(value) for value in stats[index])
        if area < min_area:
            continue
        if (cw > 0.30 * lw and ch <= max(3, int(0.006 * lh))) or (ch > 0.35 * lh and cw <= max(3, int(0.006 * lw))):
            continue
        cleaned[labels == index] = 255
    mask[y0:y1, x0:x1] = cleaned
    return mask


def _runs(active: np.ndarray) -> List[Tuple[int, int]]:
    values = np.r_[False, active, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(values) == 1).astype(int), np.flatnonzero(np.diff(values) == -1).astype(int)))


def _dense_spans(profile: np.ndarray, min_len: int, merge_gap: int, ratio: float) -> List[Tuple[int, int]]:
    """Convert a smoothed ink-density profile into gap-separated spans."""
    if profile.size == 0 or float(profile.max()) <= 0:
        return []
    width = _odd(max(5, int(profile.size * 0.014)))
    smooth = np.convolve(profile.astype(np.float32), np.ones(width, dtype=np.float32) / width, mode="same")
    spans = _runs(smooth >= max(1.0, float(smooth.max()) * ratio))
    if not spans:
        return []
    merged: List[List[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= min_len]


def _tight_ink_box(ink: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Optional[Rect]:
    """Return a robust, ink-tight rectangle clipped to a proposed region."""
    h, w = ink.shape
    x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    ys, xs = np.where(ink[y0:y1, x0:x1] > 0)
    if xs.size < 25:
        return None
    left, right = np.percentile(xs, (1.0, 99.0))
    top, bottom = np.percentile(ys, (1.0, 99.0))
    bx0, by0 = max(0, x0 + int(np.floor(left)) - 1), max(0, y0 + int(np.floor(top)) - 1)
    bx1, by1 = min(w, x0 + int(np.ceil(right)) + 2), min(h, y0 + int(np.ceil(bottom)) + 2)
    return (bx0, by0, bx1 - bx0, by1 - by0) if bx1 > bx0 and by1 > by0 else None


def _rect_ink_mass(ink: np.ndarray, rect: Rect) -> int:
    x, y, w, h = rect
    return int(cv2.countNonZero(ink[y:y + h, x:x + w]))


def _horizontal_overlap(a: Rect, b: Rect) -> float:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) / max(1, min(aw, bw))


def _merge_body_blocks(blocks: Iterable[Rect], ink: np.ndarray, leaf: Rect) -> List[Rect]:
    """Merge fragments of one column across short vertical gaps, never gutters."""
    _, _, _, lh = leaf
    pending = sorted(blocks, key=lambda rect: (rect[0], rect[1]))
    changed = True
    while changed:
        changed, merged = False, []
        used = [False] * len(pending)
        for i, rect in enumerate(pending):
            if used[i]:
                continue
            x, y, w, h = rect
            used[i] = True
            for j in range(i + 1, len(pending)):
                ox, oy, ow, oh = pending[j]
                if used[j] or _horizontal_overlap((x, y, w, h), pending[j]) < 0.58:
                    continue
                gap = max(oy - (y + h), y - (oy + oh), 0)
                if gap <= max(10, int(0.070 * lh)):
                    x1, y1, x2, y2 = min(x, ox), min(y, oy), max(x + w, ox + ow), max(y + h, oy + oh)
                    x, y, w, h, used[j], changed = x1, y1, x2 - x1, y2 - y1, True, True
            merged.append(_tight_ink_box(ink, x, y, x + w, y + h) or (x, y, w, h))
        pending = merged
    return pending


def _main_text_blocks(ink: np.ndarray, leaf: Rect) -> List[Rect]:
    """Use row/column density profiles to extract central text blocks/columns."""
    lx, ly, lw, lh = leaf
    roi = ink[ly:ly + lh, lx:lx + lw]
    rows = _dense_spans(np.count_nonzero(roi, axis=1), max(10, int(0.055 * lh)), max(12, int(0.050 * lh)), 0.055)
    candidates: List[Rect] = []
    for ya, yb in rows:
        if yb - ya < 0.075 * lh:
            continue
        cols = _dense_spans(np.count_nonzero(roi[ya:yb], axis=0), max(16, int(0.120 * lw)), max(12, int(0.038 * lw)), 0.075)
        for xa, xb in cols:
            if xb - xa < 0.145 * lw:
                continue
            candidate = _tight_ink_box(ink, lx + xa, ly + ya, lx + xb, ly + yb)
            if candidate is None:
                continue
            x, _, w, h = candidate
            centre_x = (x + w / 2.0 - lx) / max(1, lw)
            if 0.14 <= centre_x <= 0.86 and w >= 0.145 * lw and h >= 0.070 * lh and _rect_ink_mass(ink, candidate) >= max(100, int(0.00020 * lw * lh)):
                candidates.append(candidate)
    candidates = _merge_body_blocks(candidates, ink, leaf)
    if not candidates:
        return []
    masses = [_rect_ink_mass(ink, rect) for rect in candidates]
    largest = max(masses)
    out: List[Rect] = []
    for rect, mass in zip(candidates, masses):
        x, _, w, h = rect
        centre_x = (x + w / 2.0 - lx) / max(1, lw)
        density = mass / max(1, w * h)
        if h >= 0.095 * lh and w >= 0.155 * lw and density >= 0.004 and mass >= 0.14 * largest and 0.16 <= centre_x <= 0.84:
            out.append(rect)
    return out


def _margin_boxes(ink: np.ndarray, leaf: Rect, main_boxes: List[Rect]) -> List[Rect]:
    """Find marginal text outside the left/right extent of central body text."""
    if not main_boxes:
        return []
    lx, ly, lw, lh = leaf
    gap = max(4, int(0.020 * lw))
    left = min(rect[0] for rect in main_boxes) - gap
    right = max(rect[0] + rect[2] for rect in main_boxes) + gap
    main_top = min(rect[1] for rect in main_boxes)
    main_bottom = max(rect[1] + rect[3] for rect in main_boxes)
    # Do not mine the ragged physical leaf edge for marginalia.  The remaining
    # band still includes normal notes/folio marks set inside the margin.
    regions = [(lx + max(3, int(0.045 * lw)), ly, left, ly + lh), (right, ly, lx + lw - max(3, int(0.045 * lw)), ly + lh)]
    boxes: List[Rect] = []
    for x0, y0, x1, y1 in regions:
        if x1 - x0 < max(10, int(0.018 * lw)):
            continue
        margin = ink[y0:y1, x0:x1]
        expanded = cv2.dilate(margin, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.009 * lw)), max(3, int(0.035 * lh)))))
        count, _, stats, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)
        for index in range(1, count):
            x, y, w, h, _ = (int(value) for value in stats[index])
            candidate = _tight_ink_box(ink, x0 + x, y0 + y, x0 + x + w, y0 + y + h)
            if candidate is None:
                continue
            bx, by, bw, bh = candidate
            rel_y = (by + bh / 2.0 - ly) / max(1, lh)
            mass = _rect_ink_mass(ink, candidate)
            clear_horizontal_gap = bx + bw <= min(rect[0] for rect in main_boxes) - int(0.025 * lw) or bx >= max(rect[0] + rect[2] for rect in main_boxes) + int(0.025 * lw)
            alongside_body = by < main_bottom and by + bh > main_top
            if alongside_body and clear_horizontal_gap and 0.06 <= rel_y <= 0.94 and 0.012 * lw <= bw <= 0.24 * lw and 0.012 * lh <= bh <= 0.75 * lh and mass >= max(45, int(0.00005 * lw * lh)):
                boxes.append(candidate)
    return boxes


def _header_footer_boxes(ink: np.ndarray, leaf: Rect, main_boxes: List[Rect], image_height: int) -> Tuple[List[Rect], List[Rect]]:
    """Require separated, evidenced top/bottom ink before assigning 0 or 1."""
    if not main_boxes:
        return [], []
    lx, ly, lw, lh = leaf
    roi = ink[ly:ly + lh, lx:lx + lw]
    rows = _dense_spans(np.count_nonzero(roi, axis=1), max(5, int(0.010 * lh)), max(5, int(0.022 * lh)), 0.045)
    first_main, last_main = min(rect[1] for rect in main_boxes), max(rect[1] + rect[3] for rect in main_boxes)
    separation = max(9, int(0.032 * lh))
    headers: List[Rect] = []
    footers: List[Rect] = []
    for ya, yb in rows:
        if yb - ya > 0.150 * lh:
            continue
        global_y = (ly + (ya + yb) / 2.0) / max(1, image_height)
        relative_y = (ya + yb) / (2.0 * max(1, lh))
        header = global_y < 0.33 and 0.07 <= relative_y < 0.25 and ly + yb + separation < first_main
        footer = global_y > 0.67 and 0.75 < relative_y <= 0.92 and ly + ya - separation > last_main
        if not (header or footer):
            continue
        cols = _dense_spans(np.count_nonzero(roi[ya:yb], axis=0), max(4, int(0.012 * lw)), max(5, int(0.018 * lw)), 0.050)
        for xa, xb in cols:
            candidate = _tight_ink_box(ink, lx + xa, ly + ya, lx + xb, ly + yb)
            if candidate is None:
                continue
            _, _, bw, bh = candidate
            # A page frame or torn edge can have ink evidence, but is normally
            # nearly leaf-wide.  It is not a tight header/footer text region.
            if bw >= 0.010 * lw and bw <= 0.70 * lw and bh >= 0.008 * lh and _rect_ink_mass(ink, candidate) >= max(35, int(0.000035 * lw * lh)):
                (headers if header else footers).append(candidate)
    return headers, footers


def _blue_or_green_filler(image: np.ndarray, leaf: Rect) -> List[Rect]:
    """Keep only isolated cool-colour decorations/stamps as high-confidence filler.

    Without OCR, ordinary dark writing cannot safely be called English or pencil,
    so arbitrary ink/noise is deliberately never promoted to class 4.
    """
    lx, ly, lw, lh = leaf
    hsv = cv2.cvtColor(image[ly:ly + lh, lx:lx + lw], cv2.COLOR_BGR2HSV)
    cool = cv2.inRange(hsv, (85, 70, 45), (145, 255, 255))
    ix, iy = max(2, int(0.012 * lw)), max(2, int(0.012 * lh))
    cool[:iy], cool[-iy:], cool[:, :ix], cool[:, -ix:] = 0, 0, 0, 0
    size = _odd(max(3, int(min(lw, lh) * 0.030)))
    cool = cv2.morphologyEx(cool, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cool, connectivity=8)
    out: List[Rect] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area >= 0.0015 * lw * lh and w >= 0.025 * lw and h >= 0.025 * lh and w <= 0.30 * lw and h <= 0.35 * lh and area / max(1, w * h) >= 0.12:
            out.append((lx + x, ly + y, w, h))
    return out


def _iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a[2] - a[4] / 2.0, a[3] - a[5] / 2.0, a[2] + a[4] / 2.0, a[3] + a[5] / 2.0
    bx1, by1, bx2, by2 = b[2] - b[4] / 2.0, b[3] - b[5] / 2.0, b[2] + b[4] / 2.0, b[3] + b[5] / 2.0
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[4] * a[5] + b[4] * b[5] - inter
    return inter / union if union else 0.0


def _sanitise_boxes(raw: Iterable[Tuple[int, Rect]], image_width: int, image_height: int, diagnostics: Optional[MutableMapping[str, int]]) -> List[YoloBox]:
    """Clip, validate, de-duplicate, and de-conflict region candidates."""
    boxes: List[YoloBox] = []
    for class_id, (x, y, w, h) in raw:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(image_width, x + w), min(image_height, y + h)
        if x1 <= x0 or y1 <= y0:
            _note_rejection(diagnostics, "invalid")
            continue
        bw, bh = (x1 - x0) / image_width, (y1 - y0) / image_height
        xc, yc = (x0 + x1) / (2.0 * image_width), (y0 + y1) / (2.0 * image_height)
        if bw < 0.010 or bh < 0.010:
            _note_rejection(diagnostics, "tiny")
            continue
        if bw * bh > 0.62 or bw > 0.92 or bh > 0.94:
            _note_rejection(diagnostics, "oversize")
            continue
        # These global rules explicitly prevent the historical middle-image
        # header/footer failure, even for scans with stacked leaves.
        if class_id == 0 and yc >= 0.33:
            _note_rejection(diagnostics, "header_position")
            continue
        if class_id == 1 and yc <= 0.67:
            _note_rejection(diagnostics, "footer_position")
            continue
        boxes.append((class_id, CLASS_NAMES[class_id], xc, yc, bw, bh))
    unique: List[YoloBox] = []
    for box in sorted(boxes, key=lambda value: value[4] * value[5], reverse=True):
        if any(box[0] == kept[0] and _iou(box, kept) > 0.68 for kept in unique):
            _note_rejection(diagnostics, "duplicate")
            continue
        unique.append(box)
    priority = {0: 0, 1: 0, 4: 1, 3: 2, 2: 3}
    kept: List[YoloBox] = []
    for box in sorted(unique, key=lambda value: (priority[value[0]], -(value[4] * value[5]))):
        if any(box[0] != previous[0] and _iou(box, previous) > 0.30 for previous in kept):
            _note_rejection(diagnostics, "cross_class_overlap")
            continue
        kept.append(box)
    return sorted(kept, key=lambda value: (value[0], value[3], value[2]))


def extract_layout_boxes(img_path: Path, diagnostics: Optional[MutableMapping[str, int]] = None) -> Tuple[int, int, List[YoloBox]]:
    """Extract conservative manuscript-layout pseudo-labels from one image."""
    image = _read_bgr(img_path)
    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corrected = _normalise_illumination(gray)
    raw: List[Tuple[int, Rect]] = []
    for leaf in _find_leaf_rectangles(gray):
        ink = _clean_ink_mask(gray, corrected, leaf)
        if cv2.countNonZero(ink) < max(100, int(0.0004 * leaf[2] * leaf[3])):
            continue
        main = _main_text_blocks(ink, leaf)
        headers, footers = _header_footer_boxes(ink, leaf, main, image_height)
        raw.extend((0, rect) for rect in headers)
        raw.extend((1, rect) for rect in footers)
        raw.extend((2, rect) for rect in main)
        raw.extend((3, rect) for rect in _margin_boxes(ink, leaf, main))
        raw.extend((4, rect) for rect in _blue_or_green_filler(image, leaf))
    return image_height, image_width, _sanitise_boxes(raw, image_width, image_height, diagnostics)


def write_yolo_annotation_file(label_path: Path, boxes: List[YoloBox]) -> None:
    """Write one normalized YOLO label file, including intentionally empty files."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as output:
        for class_id, _, xc, yc, bw, bh in boxes:
            output.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def generate_preview_image(img_path: Path, boxes: List[YoloBox], preview_path: Path) -> None:
    """Render optional colored previews for inspection."""
    image = _read_bgr(img_path)
    h, w = image.shape[:2]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    for class_id, name, xc, yc, bw, bh in boxes:
        x0, y0, x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = CLASS_COLORS[class_id]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
        label = f"{class_id}:{name}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y0, th + 5)
        cv2.rectangle(image, (x0, text_y - th - 5), (x0 + tw + 6, text_y + baseline), color, -1)
        cv2.putText(image, label, (x0 + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(preview_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conservative YOLO pseudo-labels for manuscript layout regions.")
    parser.add_argument("--input", type=str, default="data/dataset/images/train", help="Input image directory.")
    parser.add_argument("--output", type=str, default="data/dataset/labels/train", help="Output YOLO-label directory.")
    parser.add_argument("--preview-dir", type=str, default="data/dataset/auto_annotated_previews", help="Directory for optional previews.")
    parser.add_argument("--max-previews", type=int, default=0, help="Preview count (0 = all, -1 = none).")
    args = parser.parse_args()
    input_dir, output_dir, preview_dir = Path(args.input), Path(args.output), Path(args.preview_dir)
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    diagnostics: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    images_per_class: Counter[int] = Counter()
    errors: List[str] = []
    empty = 0
    print("=" * 65)
    print(" Manuscript Layout Auto-Annotation")
    print("=" * 65)
    print(f"Input images : {input_dir.resolve()}")
    print(f"Output labels: {output_dir.resolve()}")
    print(f"Images found : {len(paths)}")
    print("-" * 65)
    for index, path in enumerate(paths):
        try:
            _, _, boxes = extract_layout_boxes(path, diagnostics)
            write_yolo_annotation_file(output_dir / f"{path.stem}.txt", boxes)
            if not boxes:
                empty += 1
            for class_id in {box[0] for box in boxes}:
                images_per_class[class_id] += 1
            for box in boxes:
                counts[box[0]] += 1
            if args.max_previews == 0 or (args.max_previews > 0 and index < args.max_previews):
                generate_preview_image(path, boxes, preview_dir / f"{path.stem}_preview.jpg")
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    print(f"Images processed: {len(paths)}")
    print("\nGenerated instances:")
    for class_id in range(5):
        print(f"{CLASS_NAMES[class_id]}: {counts[class_id]}")
    print("\nImages with each class:")
    for class_id in range(5):
        print(f"{CLASS_NAMES[class_id]}: {images_per_class[class_id]}")
    print(f"\nRejected candidate boxes: {diagnostics['rejected']}")
    for reason in ("tiny", "oversize", "invalid", "duplicate", "cross_class_overlap", "header_position", "footer_position"):
        if diagnostics[f"rejected_{reason}"]:
            print(f"  {reason}: {diagnostics[f'rejected_{reason}']}")
    print(f"Invalid/empty annotations: {empty}")
    print(f"Read/processing errors: {len(errors)}")
    print(f"Label files regenerated: {len(paths) - len(errors)}")
    if errors:
        print("\n[WARNING] Files not regenerated due to errors:")
        for error in errors:
            print(f"  - {error}")
    print("=" * 65)


if __name__ == "__main__":
    main()
