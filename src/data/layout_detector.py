"""Prototype: leaf segmentation + projection-profile text blocks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "header", 1: "footer", 2: "main_text", 3: "side_text", 4: "filler"}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _merge_runs(runs: list[tuple[int, int]], max_gap: int, min_len: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    kept = [(a, b) for a, b in merged if (b - a) >= min_len]
    return kept if kept else [(merged[0][0], merged[-1][1])]


def _smooth(profile: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    return np.convolve(profile.astype(np.float32), np.ones(k, np.float32) / k, mode="same")


def _dense_spans(profile: np.ndarray, valley_ratio: float, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    if profile.size == 0 or float(profile.max()) <= 0:
        return []
    sm = _smooth(profile, max(5, len(profile) // 50))
    peak = float(sm.max())
    active = sm >= peak * valley_ratio
    return _merge_runs(_runs(active), max_gap=max_gap, min_len=min_len)


def _leaf_rects(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    thr = max(bg + 30.0, 75.0)
    leaf = (gray > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k, iterations=2)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, rw, rh = cv2.boundingRect(c)
        if rw * rh < 0.06 * w * h:
            continue
        if rw < 0.18 * w and rh < 0.18 * h:
            continue
        rects.append((x, y, rw, rh))
    if not rects:
        return [(0, 0, w, h)]
    rects.sort(key=lambda r: (r[1], r[0]))
    return rects


def _split_stacked_or_facing(gray: np.ndarray, leaves: list[tuple[int, int, int, int]], parchment_mask: np.ndarray):
    """Split a leaf bbox when a dark gutter separates stacked or facing pages."""
    out = []
    for lx, ly, lw, lh in leaves:
        roi = parchment_mask[ly : ly + lh, lx : lx + lw]
        row = np.sum(roi > 0, axis=1).astype(np.float32)
        col = np.sum(roi > 0, axis=0).astype(np.float32)
        split_done = False

        if lh >= 0.45 * gray.shape[0]:
            sm = _smooth(row, max(7, lh // 40))
            mid0, mid1 = int(0.30 * lh), int(0.70 * lh)
            if mid1 > mid0 and sm.max() > 0:
                mid = sm[mid0:mid1]
                if mid.min() < 0.16 * sm.max() and mid.min() < 0.22 * lw:
                    split = mid0 + int(np.argmin(mid))
                    if split > 0.22 * lh and (lh - split) > 0.22 * lh:
                        out.append((lx, ly, lw, split))
                        out.append((lx, ly + split, lw, lh - split))
                        split_done = True

        if not split_done and lw >= 0.55 * gray.shape[1]:
            sm = _smooth(col, max(7, lw // 40))
            mid0, mid1 = int(0.32 * lw), int(0.68 * lw)
            if mid1 > mid0 and sm.max() > 0:
                mid = sm[mid0:mid1]
                if mid.min() < 0.16 * sm.max() and mid.min() < 0.22 * lh:
                    split = mid0 + int(np.argmin(mid))
                    if split > 0.22 * lw and (lw - split) > 0.22 * lw:
                        out.append((lx, ly, split, lh))
                        out.append((lx + split, ly, lw - split, lh))
                        split_done = True

        if not split_done:
            out.append((lx, ly, lw, lh))
    out.sort(key=lambda r: (r[1], r[0]))
    return out


def _ink_mask(gray: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    leaf_pix = gray[leaf_mask > 0]
    if leaf_pix.size == 0:
        return np.zeros_like(gray)
    parchment = float(np.percentile(leaf_pix, 62))
    dark = (gray < (parchment - 28)).astype(np.uint8) * 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 16
    )
    ink = cv2.bitwise_and(cv2.bitwise_and(adaptive, dark), leaf_mask)
    # Remove only long ruling lines, not text strokes.
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(28, int(h * 0.28))))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(w * 0.42)), 2))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk))
    ink = cv2.medianBlur(ink, 3)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _tight(ink: np.ndarray, x0, y0, x1, y1, px=3.0, py=2.0):
    h, w = ink.shape
    x0 = int(max(0, x0)); y0 = int(max(0, y0))
    x1 = int(min(w, x1)); y1 = int(min(h, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ys, xs = np.where(ink[y0:y1, x0:x1] > 0)
    if xs.size < 25:
        return None
    xa, xb = np.percentile(xs, [px, 100 - px])
    ya, yb = np.percentile(ys, [py, 100 - py])
    bx0 = x0 + int(np.floor(xa))
    by0 = y0 + int(np.floor(ya))
    bx1 = x0 + int(np.ceil(xb)) + 1
    by1 = y0 + int(np.ceil(yb)) + 1
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 6 or bh < 5:
        return None
    return bx0, by0, bw, bh


def _blue_stamps(img, leaf):
    lx, ly, lw, lh = leaf
    hsv = cv2.cvtColor(img[ly:ly + lh, lx:lx + lw], cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 60, 50), (140, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < 22:
            continue
        if w * h < 0.003 * lw * lh:
            continue
        if w > 0.28 * lw or h > 0.35 * lh:
            continue
        out.append((lx + x, ly + y, w, h))
    return out


def _white_stickers(img, gray, leaf):
    lx, ly, lw, lh = leaf
    roi_g = gray[ly:ly + lh, lx:lx + lw]
    hsv = cv2.cvtColor(img[ly:ly + lh, lx:lx + lw], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 210), (180, 40, 255))
    med = float(np.median(roi_g)) if roi_g.size else 150
    mask = ((roi_g > med + 40) & (white > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 28:
            continue
        area = cv2.contourArea(c)
        if area < 0.006 * lw * lh:
            continue
        if w > 0.32 * lw or h > 0.22 * lh:
            continue
        rect_area = w * h
        if rect_area > 0 and area / rect_area < 0.55:
            continue
        out.append((lx + x, ly + y, w, h))
    return out


def extract_layout_boxes(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read {img_path}")
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leaves = _leaf_rects(gray)
    border_med = float(np.median(np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])))
    parchment_mask = (gray > max(border_med + 30.0, 75.0)).astype(np.uint8) * 255
    leaves = _split_stacked_or_facing(gray, leaves, parchment_mask)
    raw = []

    def emit(cid, name, box):
        if box is None:
            return
        bx, by, bw, bh = box
        nw, nh = bw / W, bh / H
        # A page-wide text box is the failure mode this detector is explicitly
        # designed to avoid.  Width is checked independently of area because a
        # shallow, full-spread region can otherwise evade an area-only limit.
        if nw < 0.012 or nh < 0.008 or nw > 0.80 or nw * nh > 0.80:
            return
        raw.append((cid, name, (bx + bw / 2) / W, (by + bh / 2) / H, nw, nh))

    for leaf in leaves:
        lx, ly, lw, lh = leaf
        leaf_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(leaf_mask, (lx, ly), (lx + lw, ly + lh), 255, -1)
        leaf_mask = cv2.bitwise_and(leaf_mask, parchment_mask)
        if cv2.countNonZero(leaf_mask) < 0.02 * W * H:
            cv2.rectangle(leaf_mask, (lx, ly), (lx + lw, ly + lh), 255, -1)

        ink = _ink_mask(gray, leaf_mask)
        roi = ink[ly:ly + lh, lx:lx + lw]
        ink_frac = cv2.countNonZero(roi) / max(1, lw * lh)
        # The adaptive mask can still treat parchment grain, ruled borders, and
        # scanner texture as ink.  Require a modest amount of genuinely dark
        # material before calling a leaf body text; decorative detections above
        # remain eligible on otherwise blank leaves.
        leaf_pixels = gray[leaf_mask > 0]
        parchment_level = float(np.percentile(leaf_pixels, 62)) if leaf_pixels.size else 255.0
        strict_ink_frac = float(np.count_nonzero((gray[ly:ly + lh, lx:lx + lw] < parchment_level - 60) & (leaf_mask[ly:ly + lh, lx:lx + lw] > 0))) / max(1, cv2.countNonZero(leaf_mask[ly:ly + lh, lx:lx + lw]))

        for box in _blue_stamps(img, leaf) + _white_stickers(img, gray, leaf):
            emit(4, "filler", box)

        if ink_frac < 0.0020 or strict_ink_frac < 0.010:
            continue

        col = np.sum(roi > 0, axis=0)
        row = np.sum(roi > 0, axis=1)
        y_spans = _dense_spans(row, 0.14, max_gap=max(8, int(0.10 * lh)), min_len=max(6, int(0.018 * lh)))
        x_spans_all = _dense_spans(col, 0.12, max_gap=max(14, int(0.10 * lw)), min_len=max(16, int(0.18 * lw)))
        if not y_spans:
            continue

        y_ink = [float(row[a:b].sum()) for a, b in y_spans]
        y_tot = sum(y_ink) + 1e-6
        headers, footers, mains = [], [], []
        for (ya, yb), ys in zip(y_spans, y_ink):
            rel_h = (yb - ya) / max(lh, 1)
            rel_top = ya / max(lh, 1)
            rel_bot = yb / max(lh, 1)
            share = ys / y_tot
            small = rel_h <= 0.13 and share <= 0.18
            if small and rel_top <= 0.16 and len(y_spans) > 1:
                headers.append((ya, yb))
            elif small and rel_bot >= 0.86 and len(y_spans) > 1:
                footers.append((ya, yb))
            else:
                mains.append((ya, yb))
        if not mains:
            mains = y_spans
            headers, footers = [], []

        # dominant x-spans: keep those with enough mass or width
        if x_spans_all:
            x_ink = [float(col[a:b].sum()) for a, b in x_spans_all]
            x_tot = sum(x_ink) + 1e-6
            main_x = [
                (a, b)
                for (a, b), s in zip(x_spans_all, x_ink)
                if (s / x_tot >= 0.18) or ((b - a) >= 0.30 * lw)
            ]
            if not main_x:
                main_x = [max(x_spans_all, key=lambda ab: ab[1] - ab[0])]
        else:
            main_x = [(int(0.08 * lw), int(0.92 * lw))]

        for ya, yb in headers:
            for xa, xb in main_x:
                emit(0, "header", _tight(ink, lx + xa, ly + ya, lx + xb, ly + yb, px=5.0, py=2.0))
        for ya, yb in footers:
            for xa, xb in main_x:
                emit(1, "footer", _tight(ink, lx + xa, ly + ya, lx + xb, ly + yb, px=5.0, py=2.0))

        for ya, yb in mains:
            band_col = np.sum(roi[ya:yb, :] > 0, axis=0)
            spans = _dense_spans(
                band_col,
                0.14,
                max_gap=max(16, int(0.10 * lw)),
                min_len=max(16, int(0.18 * lw)),
            )
            if not spans:
                spans = main_x
            # keep only substantial column spans
            masses = [float(band_col[a:b].sum()) for a, b in spans]
            tot = sum(masses) + 1e-6
            spans = [
                (a, b)
                for (a, b), m in zip(spans, masses)
                if m / tot >= 0.16 or (b - a) >= 0.28 * lw
            ] or spans

            for xa, xb in spans:
                box = _tight(ink, lx + xa, ly + ya, lx + xb, ly + yb, px=3.5, py=2.0)
                if box is not None and box[2] / W > 0.82:
                    box = _tight(ink, lx + xa, ly + ya, lx + xb, ly + yb, px=7.0, py=2.5)
                if box is not None and box[2] / W > 0.80:
                    bx, by, bw, bh = box
                    sub = ink[by:by + bh, bx:bx + bw]
                    cprof = np.cumsum(np.sum(sub > 0, axis=0))
                    if cprof[-1] > 0:
                        lo = int(np.searchsorted(cprof, 0.06 * cprof[-1]))
                        hi = int(np.searchsorted(cprof, 0.94 * cprof[-1]))
                        box = (bx + lo, by, max(6, hi - lo), bh)
                emit(2, "main_text", box)

        # side text from leftover margin components
        mx0 = min(a for a, b in main_x)
        mx1 = max(b for a, b in main_x)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
        lefts, rights = [], []
        for i in range(1, nlab):
            x, y, ww, hh, area = stats[i]
            if area < 60 or ww > 0.20 * lw:
                continue
            cx, cy = x + ww / 2, y + hh / 2
            if cy < 0.08 * lh or cy > 0.92 * lh:
                continue
            if cx < min(mx0 + 0.02 * lw, 0.22 * lw) and ww <= 0.20 * lw:
                lefts.append((x, y, ww, hh))
            elif cx > max(mx1 - 0.02 * lw, 0.78 * lw) and ww <= 0.20 * lw:
                rights.append((x, y, ww, hh))

        def cluster_margin(ccs):
            if not ccs:
                return []
            ccs = sorted(ccs, key=lambda r: r[1])
            groups = [[ccs[0]]]
            for r in ccs[1:]:
                g = groups[-1]
                gy2 = max(t[1] + t[3] for t in g)
                if r[1] - gy2 <= max(14, int(0.07 * lh)):
                    g.append(r)
                else:
                    groups.append([r])
            boxes = []
            for g in groups:
                x1 = min(t[0] for t in g)
                y1 = min(t[1] for t in g)
                x2 = max(t[0] + t[2] for t in g)
                y2 = max(t[1] + t[3] for t in g)
                if (y2 - y1) < 0.018 * lh and (x2 - x1) < 0.04 * lw:
                    continue
                boxes.append((lx + x1, ly + y1, x2 - x1, y2 - y1))
            return boxes

        for box in cluster_margin(lefts) + cluster_margin(rights):
            emit(3, "side_text", _tight(ink, box[0], box[1], box[0] + box[2], box[1] + box[3], px=1.0, py=1.0))

    out = []
    for cid, name, xc, yc, bw, bh in raw:
        xmin, xmax = max(0.0, xc - bw / 2), min(1.0, xc + bw / 2)
        ymin, ymax = max(0.0, yc - bh / 2), min(1.0, yc + bh / 2)
        rw, rh = xmax - xmin, ymax - ymin
        if rw <= 0.004 or rh <= 0.004 or rw > 0.80 or rw * rh > 0.80:
            continue
        out.append((cid, name, xmin + rw / 2, ymin + rh / 2, rw, rh))

    def iou(a, b):
        ax1, ay1 = a[2] - a[4] / 2, a[3] - a[5] / 2
        ax2, ay2 = a[2] + a[4] / 2, a[3] + a[5] / 2
        bx1, by1 = b[2] - b[4] / 2, b[3] - b[5] / 2
        bx2, by2 = b[2] + b[4] / 2, b[3] + b[5] / 2
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = a[4] * a[5] + b[4] * b[5] - inter
        return inter / ua if ua > 0 else 0.0

    kept = []
    for box in sorted(out, key=lambda b: b[4] * b[5], reverse=True):
        if any(box[0] == k[0] and iou(box, k) > 0.5 for k in kept):
            continue
        kept.append(box)
    return H, W, kept


if __name__ == "__main__":
    img_dir = Path("data/dataset/images/train")
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    samples = [
        "mscoll390_item1271_2057_0001_web.jpg",
        "mscoll390_item529_5770_0000_web.jpg",
        "mscoll390_item496_3594_0000_web.jpg",
        "mscoll390_item15_6357_0000_web.jpg",
        "mscoll390_item1411_2086_0001_web.jpg",
        "mscoll390_item1557_2119_0001_web.jpg",
        "mscoll390_item461_3537_0000_web.jpg",
        "mscoll390_item1354_5296_0001_web.jpg",
        "mscoll390_item1258_2009_0001_web.jpg",
        "mscoll390_item1564_4117_0000_web.jpg",
        "mscoll390_item1603_2204_0000_web.jpg",
    ]
    print("=== samples ===")
    for name in samples:
        _, _, boxes = extract_layout_boxes(img_dir / name)
        print(name, "n", len(boxes))
        for b in boxes:
            print(f"  {b[0]}:{b[1]:10s} xc={b[2]:.3f} yc={b[3]:.3f} w={b[4]:.3f} h={b[5]:.3f}")

    all_boxes = []
    empty = 0
    n_main = []
    n_all = []
    for p in paths:
        _, _, boxes = extract_layout_boxes(p)
        empty += int(not boxes)
        all_boxes.extend(boxes)
        n_main.append(sum(1 for b in boxes if b[0] == 2))
        n_all.append(len(boxes))
    ws = np.array([b[4] for b in all_boxes])
    hs = np.array([b[5] for b in all_boxes])
    print("\n=== full train ===")
    print("images", len(paths), "empty", empty, "boxes", len(all_boxes))
    print("boxes/image p50/p90/max", np.percentile(n_all, [50, 90, 100]))
    print("main/image p50/p90/max", np.percentile(n_main, [50, 90, 100]))
    print("class", dict(Counter(b[0] for b in all_boxes)))
    print("width p10/p50/p90/p95/max", np.round(np.percentile(ws, [10, 50, 90, 95, 100]), 3))
    print("height p10/p50/p90/p95/max", np.round(np.percentile(hs, [10, 50, 90, 95, 100]), 3))
    print("width>0.8", int((ws > 0.8).sum()), "width>0.9", int((ws > 0.9).sum()))
    for cid in range(5):
        sub = [b for b in all_boxes if b[0] == cid]
        if not sub:
            print(cid, CLASS_NAMES[cid], 0)
            continue
        w = np.array([b[4] for b in sub])
        h = np.array([b[5] for b in sub])
        print(cid, CLASS_NAMES[cid], len(sub), "w>0.8", int((w > 0.8).sum()),
              "w_p50", round(float(np.median(w)), 3), "w_max", round(float(w.max()), 3),
              "h_p50", round(float(np.median(h)), 3), "h_max", round(float(h.max()), 3))
