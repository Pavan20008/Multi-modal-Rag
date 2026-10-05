"""
End-to-end orchestration: ingestion and querying.

Retrieval: hybrid text (BM25 + dense embeddings + cross-encoder rerank,
with same-page chunk expansion) plus CLIP-based image retrieval for
diagrams/tables/illustrations. Both feed into the same VLM for the final
answer.

(Note: a ColPali visual-page-retrieval route was evaluated but dropped -
on CPU-only hardware, indexing hundreds of pages with a ~3B-parameter
vision model was impractical. The CLIP image path covers the same need
at a fraction of the compute cost.)
"""
from extractor import extract_pdf
from embeddings import TextEmbedder, ClipEmbedder
from vector_store import VectorStore
from retriever import Retriever
from generator import generate_answer
import bm25_index


def ingest(pdf_path, reset=True):
    """Extract, embed, and index text + images from a PDF."""
    text_items, image_items = extract_pdf(pdf_path)

    store = VectorStore()
    if reset:
        store.reset()

    text_embedder = TextEmbedder()
    clip_embedder = ClipEmbedder()

    if text_items:
        print("Embedding text chunks...")
        text_embeddings = text_embedder.embed([t["content"] for t in text_items])
        store.add_text_chunks(text_items, text_embeddings)

        print("Building BM25 keyword index...")
        bm25_index.build_index(text_items)

    if image_items:
        print("Embedding page images (CLIP)...")
        image_embeddings = clip_embedder.embed_images([i["path"] for i in image_items])
        store.add_images(image_items, image_embeddings)

    print("Ingestion complete.")


def answer_query(query):
    """Retrieve hybrid text + image context and generate an answer."""
    retriever = Retriever()

    text_chunks = retriever.retrieve_text(query)
    relevant_pages = {c["page"] for c in text_chunks}
    images = retriever.retrieve_images(query, relevant_pages=relevant_pages)

    print(f"Retrieved {len(text_chunks)} text chunks and {len(images)} images.")

    answer = generate_answer(query, text_chunks, images)
    return {
        "answer": answer,
        "text_chunks": text_chunks,
        "images": images,
    }