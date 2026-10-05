"""
Simple local web UI for the Multi-Modal RAG pipeline.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.

This is a thin wrapper around the existing pipeline (ingest / answer_query)
- no changes to the core retrieval/generation logic. Everything still runs
locally: Ollama for the VLM, Tesseract for OCR, ChromaDB for vectors.
"""
import os
import json
import fitz
from flask import Flask, request, render_template_string, send_from_directory

import config
from pipeline import ingest, answer_query

app = Flask(__name__)

UPLOAD_DIR = os.path.join(config.BASE_DIR, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

INGESTED_INFO_PATH = os.path.join(config.VECTOR_DB_DIR, "ingested_info.json")


def _read_ingested_info():
    if not os.path.exists(INGESTED_INFO_PATH):
        return None
    try:
        with open(INGESTED_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_ingested_info(filename, page_count):
    with open(INGESTED_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({"filename": filename, "pages": page_count}, f)

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-Modal RAG</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px;
           margin: 40px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }
    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
    form { background: #f7f7f8; padding: 16px; border-radius: 10px; margin-bottom: 20px;
           border: 1px solid #e5e5e5; }
    input[type=text], input[type=file] { width: 100%; padding: 9px; margin: 6px 0 12px;
           box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; }
    button { background: #2563eb; color: white; border: none; padding: 10px 20px;
             border-radius: 6px; cursor: pointer; font-size: 0.95rem; }
    button:hover { background: #1d4ed8; }
    .status { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.92rem; }
    .status.ok { background: #dcfce7; color: #166534; }
    .status.err { background: #fee2e2; color: #991b1b; }
    .answer { background: #fff; border: 1px solid #e5e5e5; border-radius: 10px;
              padding: 18px; white-space: pre-wrap; line-height: 1.5; }
    .sources { font-size: 0.85rem; color: #555; margin-top: 12px; }
    .sources div { padding: 2px 0; }
    .images { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
    .images img { max-width: 280px; border: 1px solid #ddd; border-radius: 8px; }
    .hint { font-size: 0.82rem; color: #777; margin: 4px 0 10px; }
    .loaded-doc { font-size: 0.88rem; color: #444; background: #eef2ff; border: 1px solid #dbe3fb;
                  padding: 8px 14px; border-radius: 8px; margin-bottom: 18px; }
    .loading { display: none; align-items: center; gap: 10px; font-size: 0.9rem; color: #444;
               margin-top: 10px; }
    .loading.active { display: flex; }
    .spinner { width: 16px; height: 16px; border: 2px solid #cbd5e1; border-top-color: #2563eb;
               border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
  <script>
    function showLoading(id, msg) {
      var el = document.getElementById(id);
      el.querySelector('.loading-text').textContent = msg;
      el.classList.add('active');
      var btn = el.parentElement.querySelector('button');
      if (btn) { btn.disabled = true; btn.textContent = 'Working...'; }
    }
  </script>
</head>
<body>
  <h1>Multi-Modal RAG</h1>

  {% if ingested_info %}
    <div class="loaded-doc">Currently loaded: <strong>{{ ingested_info.filename }}</strong> ({{ ingested_info.pages }} pages)</div>
  {% else %}
    <div class="loaded-doc">No document ingested yet - upload a PDF below to get started.</div>
  {% endif %}

  {% if status_message %}
    <div class="status {{ 'ok' if status_ok else 'err' }}">{{ status_message }}</div>
  {% endif %}

  <h2>1. Ingest a PDF</h2>
  <form action="/ingest" method="post" enctype="multipart/form-data"
        onsubmit="showLoading('ingest-loading', 'Ingesting document - this can take several minutes for large PDFs, please wait...')">
    <label>Upload a PDF file</label>
    <input type="file" name="pdf_file" accept=".pdf" required>
    <div class="hint">
      Re-processes the whole document (OCR + native table extraction + cropped
      diagram images). Large PDFs can take a while.
    </div>
    <button type="submit">Ingest PDF</button>
    <div class="loading" id="ingest-loading"><div class="spinner"></div><span class="loading-text"></span></div>
  </form>

  <h2>2. Ask a question</h2>
  <form action="/query" method="post"
        onsubmit="showLoading('query-loading', 'Thinking - retrieving context and generating an answer...')">
    <label>Your question</label>
    <input type="text" name="question" placeholder="e.g. What is the engine oil capacity?"
           value="{{ last_question or '' }}" required>
    <button type="submit">Ask</button>
    <div class="loading" id="query-loading"><div class="spinner"></div><span class="loading-text"></span></div>
  </form>

  {% if answer %}
    <h2>Answer</h2>
    <div class="answer">{{ answer }}</div>

    {% if images %}
    <div class="images">
      {% for img in images %}
        <img src="/images/{{ img }}" alt="retrieved diagram">
      {% endfor %}
    </div>
    {% endif %}

    <div class="sources">
      <strong>Sources</strong>
      {% for s in sources %}<div>Page {{ s.page }} ({{ s.type }})</div>{% endfor %}
    </div>
  {% endif %}

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE_TEMPLATE, ingested_info=_read_ingested_info())


def _friendly_error(e):
    """Turn common exceptions into a message a non-developer can act on,
    instead of a raw stack-trace-looking string."""
    msg = str(e)
    if "Failed to connect to Ollama" in msg or "ConnectionError" in type(e).__name__:
        return "Can't reach Ollama - make sure it's running (try 'ollama list' in a terminal), then try again."
    if "Tesseract" in msg or "tesseract" in msg:
        return "Can't reach Tesseract OCR - check it's installed and the path in extractor.py is correct."
    return f"Something went wrong: {msg}"


@app.route("/ingest", methods=["POST"])
def do_ingest():
    file = request.files.get("pdf_file")
    if not file or file.filename == "":
        return render_template_string(
            PAGE_TEMPLATE, status_message="No file selected.", status_ok=False,
            ingested_info=_read_ingested_info(),
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        ingest(save_path)
        doc = fitz.open(save_path)
        page_count = len(doc)
        doc.close()
        _write_ingested_info(file.filename, page_count)
        return render_template_string(
            PAGE_TEMPLATE,
            status_message=f"Ingested '{file.filename}' successfully. You can now ask questions below.",
            status_ok=True,
            ingested_info=_read_ingested_info(),
        )
    except Exception as e:
        return render_template_string(
            PAGE_TEMPLATE,
            status_message=_friendly_error(e),
            status_ok=False,
            ingested_info=_read_ingested_info(),
        )


@app.route("/query", methods=["POST"])
def do_query():
    question = request.form.get("question", "").strip()
    ingested_info = _read_ingested_info()

    if not question:
        return render_template_string(
            PAGE_TEMPLATE, status_message="Please enter a question.", status_ok=False,
            ingested_info=ingested_info,
        )

    try:
        result = answer_query(question)
    except Exception as e:
        return render_template_string(
            PAGE_TEMPLATE,
            status_message=_friendly_error(e),
            status_ok=False,
            last_question=question,
            ingested_info=ingested_info,
        )

    sources = []
    image_filenames = []
    for c in result["text_chunks"]:
        sources.append({"page": c["page"], "type": "text"})
    for img in result["images"]:
        sources.append({"page": img["page"], "type": "image"})
        image_filenames.append(os.path.basename(img["path"]))

    return render_template_string(
        PAGE_TEMPLATE,
        answer=result["answer"],
        sources=sources,
        images=image_filenames,
        last_question=question,
        ingested_info=ingested_info,
    )


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(config.IMAGE_DIR, filename)


if __name__ == "__main__":
    print("Starting Multi-Modal RAG web UI at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)