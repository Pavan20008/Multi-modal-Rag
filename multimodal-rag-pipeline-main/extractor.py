"""
Extracts text (via OCR, to bypass broken font/ToUnicode encodings found in
some PDFs), structured tables (via PyMuPDF's native table detection, which
preserves row/column association far more reliably than OCR does on
complex/merged-cell tables), and cropped diagram/table images from a PDF.

Images are cropped tightly to their actual visual content (not the full
page) before being handed to the VLM. This matters because CLIP-ViT-based
vision models (e.g. LLaVA) internally downscale every input image to a
small fixed resolution (~336x336) regardless of source resolution - so a
full page with a small diagram surrounded by margins/whitespace makes the
diagram itself unreadable after that internal downscale. Cropping to the
diagram's actual bounding box lets it fill that fixed resolution budget
instead of competing with empty space.

Tables get a second, separate extraction pass beyond plain-text OCR.
OCR reads dense tables with merged/spanning cells (e.g. a lubricants table
where "5.6 L" applies to two sub-rows) by linearizing pixels in reading
order, which can scramble which number belongs to which row - verified
directly on this document's "Recommended Lubricants and Capacities" table,
where OCR corrupted a real "5.92 US qt." into a nonexistent "6.92 US qt."
and it fed straight into a wrong answer. PyMuPDF's find_tables() reads the
PDF's actual structure instead of pixels, so row/column association stays
correct even for merged cells.
"""
import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from tqdm import tqdm

import config

# --- Windows: point pytesseract at the Tesseract binary if it's not on PATH ---
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

RENDER_ZOOM = 2.0             # ~144 DPI; used for the full page (OCR pass)
CROP_ZOOM = 4.0                # ~288 DPI; used when re-rendering just the cropped
                                # diagram region, since it covers a smaller area
CROP_PADDING = 15              # points of padding around the detected visual bbox
MIN_DRAWINGS_FOR_IMAGE = 5     # pages with >= this many vector paths are treated
                                # as having a diagram worth saving
MIN_CROP_AREA_RATIO = 0.02     # ignore bboxes that are basically the whole page
                                # or basically nothing (noise)
MIN_TABLE_ROWS = 2              # ignore tiny/spurious table detections
MIN_TABLE_COLS = 2


def _clean_cell(text):
    """Strip PDF control-code artifacts left by broken font encodings
    (some glyphs in this document map to non-printable control chars)."""
    if text is None:
        return ""
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return cleaned.strip().replace("\n", " ")


def _table_to_text(table_data):
    """Render a PyMuPDF-extracted table (list of rows, cells possibly None
    for merged/spanning cells) as clean, row-associated text. Blank cells
    are forward-filled from the value above in the same column, since
    PyMuPDF represents a vertically-merged cell as a value in its first
    row and None in the continuation rows."""
    filled = [[_clean_cell(c) for c in row] for row in table_data]
    n_cols = max(len(r) for r in filled)
    last_seen = [""] * n_cols

    for row in filled:
        for c in range(len(row)):
            if row[c]:
                last_seen[c] = row[c]
            else:
                row[c] = last_seen[c]

    lines = []
    for row in filled:
        line = " | ".join(c for c in row if c)
        if line.strip(" |"):
            lines.append(line)
    return "\n".join(lines)


def _extract_tables(page):
    """Detect and extract structured tables from a page's native PDF
    content (not OCR). Returns a list of clean text blocks, one per table."""
    tables_text = []
    try:
        tabs = page.find_tables()
    except Exception:
        return tables_text

    for t in tabs.tables:
        data = t.extract()
        if len(data) < MIN_TABLE_ROWS or len(data[0]) < MIN_TABLE_COLS:
            continue
        text = _table_to_text(data)
        if text.strip():
            tables_text.append(text)

    return tables_text


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _get_visual_bbox(page, raster_images, drawings):
    """Bounding box of the single largest visual element on the page (by
    area) - typically the actual diagram/table, as opposed to smaller
    decorative elements (rule lines, small icons) or unrelated photos
    (e.g. a small "here's where this is located" inset) that would
    otherwise get unioned in and dilute the crop. Padded a bit. Returns a
    fitz.Rect, or None if nothing usable was found."""
    rects = []

    for img in raster_images:
        try:
            rects.append(page.get_image_bbox(img))
        except Exception:
            continue

    for d in drawings:
        rect = d.get("rect")
        if rect is not None:
            rects.append(rect)

    if not rects:
        return None

    largest = max(rects, key=lambda r: r.width * r.height)

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    largest_area = largest.width * largest.height
    if page_area <= 0 or largest_area / page_area < MIN_CROP_AREA_RATIO:
        return None

    x0 = max(largest.x0 - CROP_PADDING, page_rect.x0)
    y0 = max(largest.y0 - CROP_PADDING, page_rect.y0)
    x1 = min(largest.x1 + CROP_PADDING, page_rect.x1)
    y1 = min(largest.y1 + CROP_PADDING, page_rect.y1)

    return fitz.Rect(x0, y0, x1, y1)


def extract_pdf(pdf_path):
    """
    Extract OCR'd text chunks and cropped diagram/table images from a PDF.

    Returns:
        text_items: list of dicts {id, content, page}
        image_items: list of dicts {id, path, page}
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    text_items = []
    image_items = []
    text_id, image_id = 0, 0

    for page_num in tqdm(range(len(doc)), desc="Extracting pages"):
        page = doc[page_num]

        # ---- Full-page render for OCR ----
        full_pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        full_page_path = os.path.join(config.IMAGE_DIR, f"page{page_num + 1}_render.png")
        full_pix.save(full_page_path)

        pil_img = Image.open(full_page_path)
        ocr_text = pytesseract.image_to_string(pil_img)

        for chunk in chunk_text(ocr_text):
            text_items.append({
                "id": f"text_{text_id}",
                "content": chunk,
                "page": page_num + 1,
            })
            text_id += 1

        # ---- Structured table extraction (native PDF structure, not OCR) ----
        for table_text in _extract_tables(page):
            text_items.append({
                "id": f"text_{text_id}",
                "content": f"[TABLE on page {page_num + 1}]\n{table_text}",
                "page": page_num + 1,
            })
            text_id += 1

        # ---- Decide whether this page has meaningful visual content ----
        raster_images = page.get_images(full=True)
        drawings = page.get_drawings()
        has_visual_content = len(raster_images) > 0 or len(drawings) >= MIN_DRAWINGS_FOR_IMAGE

        if has_visual_content:
            bbox = _get_visual_bbox(page, raster_images, drawings)

            if bbox is not None:
                # Re-render just the cropped region at higher zoom, so the
                # diagram fills the model's fixed input resolution instead
                # of competing with page margins/whitespace.
                crop_pix = page.get_pixmap(matrix=fitz.Matrix(CROP_ZOOM, CROP_ZOOM), clip=bbox)
                crop_path = os.path.join(config.IMAGE_DIR, f"page{page_num + 1}_crop.png")
                crop_pix.save(crop_path)
                image_path = crop_path
                os.remove(full_page_path)  # keep only the crop for this page
            else:
                # Couldn't isolate a tight region (e.g. content spans the
                # whole page) - fall back to the full-page render.
                image_path = full_page_path

            image_items.append({
                "id": f"image_{image_id}",
                "path": image_path,
                "page": page_num + 1,
            })
            image_id += 1
        else:
            os.remove(full_page_path)  # plain text page, no image needed

    doc.close()
    print(f"Extracted {len(text_items)} text chunks and {len(image_items)} images.")
    return text_items, image_items


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    texts, images = extract_pdf(pdf)
    print(texts[:2])
    print(images[:2])