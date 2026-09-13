# Manuscript Layout Region Detection — Annotation Guidelines

## Overview & Purpose

This document defines the official manual annotation protocol for the Manuscript Layout Region Detection project. Annotators must follow these guidelines to produce consistent, high-quality bounding-box annotations compatible with the **Ultralytics YOLO object detection format**.

The goal of this dataset is to identify structural layout regions in historical manuscript pages (e.g., main text blocks, headers, footers, marginalia, and decorative elements).

---

## Target Class Summary

The dataset defines exactly **five object detection classes** (Class IDs `0` through `4`). Class IDs and class names must be assigned strictly according to the table below.

| Class ID | Class Name | Description | Included Elements | Excluded Elements |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `header` | Top-margin structural and navigational text | Running headers, section titles, top-margin chapter headers, top-region folio numbers | Body text, marginal commentary, top decorative borders |
| **1** | `footer` | Bottom-margin structural and navigational text | Catchwords, page/folio numbers in bottom margin, quire signatures, bottom-margin notes | Main body text, bottom decorative borders, stains/damage |
| **2** | `main_text` | Primary body text of the manuscript page | Main text blocks, individual text columns in multi-column layouts | Running headers, catchwords, side marginalia, separate filler |
| **3** | `side_text` | Marginal text written along left/right margins | Marginalia, glosses, commentary, marginal notes, side annotations | Main body text, top running headers, bottom catchwords |
| **4** | `filler` | Decorative elements, non-manuscript text, and pencil notes | Illustrations, decorative ornaments, illuminations, English text, pencil annotations | Normal manuscript body text, manuscript marginalia |

---

## Detailed Class Definitions & Positive Examples

### 1. Class 0: `header`
- **Location**: Situated in the upper margin of the manuscript page.
- **Function**: Provides structural navigation, chapter/section titles, or page numbering.
- **Positive Examples**:
  - Running title at the very top center of the page.
  - Section or chapter heading separated from the main body text block.
  - Folio or page number written in the top-right or top-left corner.
- **Bounding Box Policy**: Draw a single tight rectangular box enclosing the header text.

### 2. Class 1: `footer`
- **Location**: Situated in the lower margin of the manuscript page.
- **Function**: Page navigation, quire assembly marks, or bottom-margin notes.
- **Positive Examples**:
  - Catchwords written at the bottom-right corner of a page.
  - Page numbers or folio numbers located in the bottom margin.
  - Quire signatures or binder marks centered at the bottom edge.
- **Bounding Box Policy**: Enclose the bottom-margin text element tightly.

### 3. Class 2: `main_text`
- **Location**: Central region(s) of the manuscript page.
- **Function**: Contains the primary narrative, religious, or core text content.
- **Positive Examples**:
  - The central rectangular block of manuscript text.
  - In a **multi-column layout** (e.g., 2 or 3 columns), **each column is annotated as a separate `main_text` box**.
- **Bounding Box Policy**: Enclose the full extent of the text block/column. Include line-initial capitals and integrated drop-caps that cannot be separated.

### 4. Class 3: `side_text`
- **Location**: Situated in the left or right outer margins alongside the main text.
- **Function**: Marginal commentary, scholar annotations, glosses, or side notes.
- **Positive Examples**:
  - Commentary text written vertically or horizontally in the side margin.
  - Scholar corrections or interlinear/marginal glosses extending into the margin.
- **Bounding Box Policy**: Draw a separate tight box around each distinct block of marginal text.

### 5. Class 4: `filler`
- **Location**: Any region on the page containing decorative or non-manuscript additions.
- **Function**: Non-text visual elements or non-original script additions specified by project guidelines.
- **Positive Examples**:
  - Visual illustrations, miniatures, floral borders, and decorative tailpieces.
  - Later English text additions or catalog notes written on the page.
  - Pencil marks, pencil annotations, or library catalog numbers.
- **Bounding Box Policy**: Enclose the decorative element or non-manuscript text tightly.

---

## Negative Cases (What NOT to Annotate)

Do **NOT** draw bounding boxes for any of the following:
1. **Paper Damage & Aging Artifacts**: Stains, watermarks, ink bleed-through from the reverse side, paper tears, holes, mold, or foxing.
2. **Page Boundaries & Blank Space**: Large blank areas, margins, or background binding.
3. **Integrated Decorative Drop-Caps**: Decorative initials that are embedded directly inside a `main_text` block and cannot be separated cleanly should remain inside the `main_text` box.
4. **Normal Manuscript Text**: Never label normal manuscript body text as `filler` simply because it is written in a different ink color (e.g., rubricated red text) or script style.
5. **Individual Lines or Characters**: Do not annotate individual words or lines unless that single line constitutes an entire separate layout element (e.g., a 1-line running header or 1-word catchword).

---

## Specific Layout Scenario Rules

### Multi-Column Pages
- If a page contains multiple distinct columns of main text, **draw a separate `main_text` bounding box for each column**.
- Do **NOT** draw one single overarching box surrounding all columns if clear column gutters exist.

### Marginalia (`side_text`)
- If a marginal note is visually distinct from the main body text, label it as `side_text`.
- If multiple separate marginal notes exist along the side margin, annotate each as an independent `side_text` box.

### Header vs. Footer vs. Side Text Ambiguity
- **Top Margin**: Text located clearly above the main text block functioning as title/header/folio $\rightarrow$ `header`.
- **Bottom Margin**: Text located clearly below the main text block functioning as catchword/signature/page number $\rightarrow$ `footer`.
- **Side Margins**: Text written in left/right outer margins functioning as commentary/gloss $\rightarrow$ `side_text`.

### Decorative Filler, English Text, & Pencil Writing
- Visually separate decorative illustrations, borders, and miniatures $\rightarrow$ `filler`.
- Any English text or pencil annotations present on the manuscript page $\rightarrow$ `filler` (per project specification).
- If a decorative motif is directly integrated into a text region and cannot be logically separated without overlapping text lines, include it inside the text region rather than creating an artificial `filler` box.

### Damaged, Faded, or Blur-Affected Pages
- If text is partially faded, stained, or hard to read, but its visual position clearly indicates its layout role, **annotate the visible region**.
- Classification is based on **visual/spatial layout position**, NOT OCR or full text legibility.

---

## Fundamental Annotation Rules

Annotators must adhere strictly to the following 20 rules:

1. Draw **tight rectangular bounding boxes** around the actual region boundary.
2. **Never** draw one giant box around the entire manuscript page.
3. Do **not** annotate individual characters or individual lines unless the entire region itself is an independent layout element.
4. Do **not** label normal manuscript body text as `filler` merely because it is visually distinct or rubricated.
5. A class should **only** be annotated when the corresponding region actually exists on the page.
6. An image may contain **any subset** of the five classes (e.g., only `main_text`, or `main_text` + `header` + `footer`).
7. Do **not** force all five classes onto every image.
8. If a marginal note is clearly separate from the body text, label it `side_text`.
9. If text is clearly in the top margin and functions as a header, section title, or top folio, label it `header`.
10. If text is clearly in the bottom margin and functions as a footer, catchword, page number, or signature, label it `footer`.
11. If a decorative element is visually separate from the text region, label it `filler`.
12. English text or pencil writing must be labeled as `filler`, even if it occurs inside an otherwise manuscript-style page.
13. If a decorative element is integrated into a larger text region and cannot reasonably be separated, do not create an artificial `filler` box.
14. Keep bounding boxes **completely inside image boundaries** ($0.0 \le x, y \le 1.0$).
15. Do **not** intentionally include large amounts of empty page margin inside bounding boxes.
16. If a region is partially damaged, faded, blurred, or difficult to read but its layout position clearly identifies its class, annotate the visible region.
17. Do **not** use OCR or text reading to decide whether a region exists. Classification must be based primarily on visual/layout characteristics.
18. Do **not** annotate stains, tears, paper damage, shadows, or bleed-through as objects unless they are part of a defined class.
19. If text overlaps or touches decorative elements, use the most logically separable regions possible.
20. Maintain the **same annotation policy consistently** across all manuscript sources.

---

## YOLO Format Specification & Coordinates

Annotations must be exported in **Ultralytics YOLO object detection format**:
- One `.txt` label file per image, sharing the exact same file stem (e.g., `image_001.jpg` $\rightarrow$ `image_001.txt`).
- Each line in the `.txt` file represents one bounding box:
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
- All coordinates must be normalized floating-point values in the range `[0.0, 1.0]` relative to image width ($W$) and height ($H$):
  $$\text{x\_center} = \frac{x_{\text{min}} + x_{\text{max}}}{2 \cdot W}, \quad \text{y\_center} = \frac{y_{\text{min}} + y_{\text{max}}}{2 \cdot H}$$
  $$\text{width} = \frac{x_{\text{max}} - x_{\text{min}}}{W}, \quad \text{height} = \frac{y_{\text{max}} - y_{\text{min}}}{H}$$

---

## Final Annotator Checklist

Before saving annotations for a manuscript page, perform this quick quality check:

- [ ] Are Class IDs strictly integers from `0` to `4`?
- [ ] Is `header` (Class 0) assigned only to top-margin navigational/title text?
- [ ] Is `footer` (Class 1) assigned only to bottom-margin catchwords/signatures/page numbers?
- [ ] Is `main_text` (Class 2) assigned to main body text blocks (separate boxes per column in multi-column pages)?
- [ ] Is `side_text` (Class 3) assigned to marginal notes along page edges?
- [ ] Is `filler` (Class 4) assigned to decorative ornaments, pencil writing, and English text?
- [ ] Are all bounding boxes tight around regions without unnecessary white space?
- [ ] Have all stains, tears, shadows, and bleed-through artifacts been excluded?
- [ ] Are coordinates properly normalized between `0.0` and `1.0`?
