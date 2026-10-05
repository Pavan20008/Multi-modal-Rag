# Multi-Modal RAG Pipeline (Free & Open-Source, Local-Only)

A local, no-API-key Retrieval-Augmented Generation pipeline that reads
**text, tables, and diagrams** from a PDF and answers questions about it
using only free, open-source models running on your own machine.

This was built and hardened against a real 554-page vehicle manual with
messy real-world problems: a broken font encoding that garbled text
extraction, vector-drawn diagrams with no text layer, dense multi-cell
fuse-panel grids, and multi-row tables with merged cells. The pipeline
below is the result of diagnosing and fixing each of those in turn - see
"Design notes" for what each piece is actually protecting against.

```
PDF
 │
 ├─▶ OCR (Tesseract, on rendered pages)  ───────▶ Text chunks
 ├─▶ Native table detection (PyMuPDF)    ───────▶ Table chunks
 └─▶ Cropped diagram/table images        ───────▶ Image chunks (CLIP-embedded)
                                                        │
                     ChromaDB (vector store) ◀──────────┘
                     BM25 keyword index
                                │
              Hybrid retrieval: dense + BM25 → cross-encoder rerank
                                │
              Cross-modal filter: images only kept if corroborated
              by the text retrieval, or a strong CLIP match alone
                                │
                                ▼
                Local VLM (Qwen2.5-VL via Ollama) ──▶ Grounded answer
```

## 1. Install system dependencies

- **Ollama** (runs the VLM locally): https://ollama.com/download
  ```bash
  ollama pull qwen2.5vl:7b
  ollama serve   # if it doesn't already run as a background service
  ```
- **Tesseract OCR**: https://github.com/UB-Mannheim/tesseract/wiki (Windows build).
  Default install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`) is
  auto-detected by `extractor.py`; edit `DEFAULT_TESSERACT_PATH` there if
  yours differs.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

First run downloads the embedding model weights (sentence-transformers,
CLIP, cross-encoder reranker - a few hundred MB total, one-time).

## 3. Ingest your PDF

```bash
python main.py ingest --pdf /path/to/your_document.pdf
```

This will, per page:
1. Render the page and run **OCR** for text (bypasses broken font
   encodings some PDFs have - OCR reads what's visually on the page
   regardless of how the underlying text is encoded).
2. Run **native table detection** (`page.find_tables()`) separately from
   OCR, since OCR reliably scrambles row/column association on tables
   with merged/spanning cells. Each detected table becomes its own
   clearly-tagged `[TABLE on page N]` chunk.
3. If the page has meaningful visual content (raster images or enough
   vector drawings to suggest a diagram), **crop tightly** to the single
   largest visual element and save that as an image chunk, embedded with
   CLIP for retrieval.

Extracted images land in `data/images/`; the vector store and BM25 index
live in `vector_db/`.

## 4. Ask questions

```bash
python main.py query --q "What is the engine oil capacity?"
```

Retrieval, per query:
1. Embed the query with `all-MiniLM-L6-v2` (dense) and score it against
   BM25 (keyword) over all text chunks; merge both candidate pools.
2. Rerank the merged pool with a cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keep the true top-k.
3. Pull in other chunks from the same page(s) as the top hits, so a list
   or table split across chunk boundaries at ingestion doesn't lose
   pieces at query time.
4. Retrieve images by CLIP similarity, but only trust one if its page is
   corroborated by the text retrieval above, or its CLIP match is strong
   enough to stand on its own - this stops visually-similar-but-
   irrelevant pages (e.g. two different icon/grid-style diagrams) from
   being pulled in on layout alone.
5. Send everything to **Qwen2.5-VL** (via Ollama) with an instruction set
   that asks it to actually describe attached diagrams (not just lean on
   text), enumerate every list/table row it's given rather than
   summarizing a subset, avoid pairing a number with the wrong label, and
   flag when a value should be visually double-checked.

## Project structure

| File               | Purpose                                                  |
|--------------------|-----------------------------------------------------------|
| `config.py`        | Model names, paths, chunk size, top-k settings            |
| `extractor.py`     | OCR + native table extraction + cropped diagram images    |
| `embeddings.py`    | Text embedder (sentence-transformers) + CLIP embedder     |
| `vector_store.py`  | ChromaDB wrapper (separate text/image collections)        |
| `bm25_index.py`    | BM25 keyword index over ingested text chunks              |
| `retriever.py`     | Hybrid retrieval: BM25 + dense → rerank → page expansion; cross-modal image filtering |
| `generator.py`     | Builds the prompt and calls the local VLM via Ollama       |
| `pipeline.py`      | Orchestrates ingest / query end-to-end                     |
| `main.py`          | CLI entry point                                            |

## Known limitations

- **Dense numeric grids** (e.g. a fuse panel with 40+ small amperage
  labels) are hard for any free local VLM to read with full numeric
  precision, even when cropped tightly and rendered at high resolution -
  this is a real capability ceiling, not a retrieval bug. The model is
  prompted to flag this and suggest visually confirming exact numbers.
- **Exhaustive list enumeration** (e.g. "list every condition that
  indicates X") can under-report on a 7B model even when the full list is
  correctly present in context - verified directly by inspecting what
  actually reached the model. Larger models (e.g. `qwen2.5vl:32b`, if your
  hardware supports it) would likely help; more aggressive prompting was
  tested and made this worse, not better, so it's left as a known,
  accepted limitation rather than over-engineered around.

## Tuning

- `CHUNK_SIZE` / `CHUNK_OVERLAP` (config.py) - text chunk granularity.
- `TOP_K_TEXT` / `TOP_K_IMAGES` / `BM25_TOP_K` - how much context reaches
  the model per query.
- `MIN_DRAWINGS_FOR_IMAGE` / `MIN_CROP_AREA_RATIO` (extractor.py) - how
  aggressively pages are flagged as "has a diagram worth saving."
- `IMAGE_DISTANCE_THRESHOLD` (retriever.py) - how strong an uncorroborated
  CLIP match needs to be to be trusted on its own.
- `VLM_MODEL` (config.py) - swap in any Ollama vision model
  (`llava:13b`, `bakllava`, etc.) if you want to trade speed for accuracy
  differently than the current `qwen2.5vl:7b` default.
