"""
Retrieves the most relevant text chunks and images for a query.

Text retrieval is now hybrid, three stages:
  1. Recall: dense (embedding) search from ChromaDB + BM25 keyword search,
     merged into one candidate pool (dense catches semantic matches, BM25
     catches exact terms like part numbers/fuse ratings that embeddings
     under-weight)
  2. Rerank: score every candidate against the query with a cross-encoder
     and keep the true top-k
  3. Page expansion: pull in other chunks from the same page(s) as the
     top hits, so a list/section split across chunk boundaries at
     ingestion time doesn't lose pieces at query time

Image retrieval (CLIP-based) is unchanged and is used for the "text" route
in pipeline.py's hybrid architecture; the "visual" route uses ColPali
instead (see colpali_retriever.py).
"""
import config
from embeddings import TextEmbedder, ClipEmbedder
from vector_store import VectorStore
from sentence_transformers import CrossEncoder
from bm25_index import BM25Retriever

RERANK_POOL_MULTIPLIER = 4          # fetch this many x top_k_text as dense candidates
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHUNKS_PER_PAGE = 4             # cap on how many chunks to pull per matched page
IMAGE_POOL_MULTIPLIER = 4           # fetch this many x top_k_images as CLIP candidates
IMAGE_DISTANCE_THRESHOLD = 0.35     # cosine distance cutoff for an uncorroborated image match (lower = stricter; tune against your own results)


class Retriever:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.clip_embedder = ClipEmbedder()
        self.store = VectorStore()
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        try:
            self.bm25 = BM25Retriever()
        except FileNotFoundError:
            print("Warning: BM25 index not found - falling back to dense-only retrieval.")
            self.bm25 = None

    def retrieve(self, query, top_k_text=config.TOP_K_TEXT, top_k_images=config.TOP_K_IMAGES):
        text_chunks = self.retrieve_text(query, top_k_text)
        relevant_pages = {c["page"] for c in text_chunks}
        images = self.retrieve_images(query, top_k_images, relevant_pages=relevant_pages)
        return text_chunks, images

    def retrieve_text(self, query, top_k_text=config.TOP_K_TEXT):
        # --- Stage 1: wide recall, dense + BM25 merged ---
        pool_size = max(top_k_text * RERANK_POOL_MULTIPLIER, top_k_text)
        text_query_emb = self.text_embedder.embed(query)[0]
        results = self.store.query_text(text_query_emb, pool_size)

        candidates = []
        seen = set()
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                candidates.append({"content": doc, "page": meta["page"]})
                seen.add(doc)

        if self.bm25:
            for hit in self.bm25.search(query, top_k=config.BM25_TOP_K):
                if hit["content"] not in seen:
                    candidates.append({"content": hit["content"], "page": hit["page"]})
                    seen.add(hit["content"])

        if not candidates:
            return []

        # --- Stage 2: rerank the merged pool with a cross-encoder ---
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        candidates.sort(key=lambda c: c["score"], reverse=True)

        top_hits = candidates[:top_k_text]
        top_pages = {c["page"] for c in top_hits}

        # --- Stage 3: expand to other chunks on the same page(s) ---
        final_chunks = list(top_hits)
        seen_content = {c["content"] for c in final_chunks}

        for page in top_pages:
            same_page_chunks = [c for c in candidates if c["page"] == page and c["content"] not in seen_content]
            for c in same_page_chunks[:MAX_CHUNKS_PER_PAGE]:
                final_chunks.append(c)
                seen_content.add(c["content"])

        final_chunks.sort(key=lambda c: c["page"])
        return final_chunks

    def retrieve_images(self, query, top_k_images=config.TOP_K_IMAGES, relevant_pages=None):
        """
        Retrieves images by CLIP similarity, but cross-checks against
        `relevant_pages` (typically the pages the text retriever already
        trusts for this query). An image only gets in "for free" if its
        page is corroborated by text retrieval; otherwise it needs a
        strong enough CLIP match on its own to be trusted. This prevents
        visually-similar-but-content-irrelevant pages (e.g. two different
        icon/grid-style diagrams) from being pulled in on layout alone.
        """
        pool_size = max(top_k_images * IMAGE_POOL_MULTIPLIER, top_k_images)
        clip_query_emb = self.clip_embedder.embed_text(query)[0]
        image_results = self.store.query_images(clip_query_emb, pool_size)

        candidates = []
        if image_results["documents"] and image_results["documents"][0]:
            distances = image_results.get("distances", [[]])[0]
            for doc, meta, dist in zip(
                image_results["documents"][0], image_results["metadatas"][0], distances
            ):
                candidates.append({"path": meta["path"], "page": meta["page"], "distance": dist})

        if not candidates:
            return []

        if relevant_pages:
            corroborated = [c for c in candidates if c["page"] in relevant_pages]
            others = [c for c in candidates if c["page"] not in relevant_pages]
        else:
            corroborated, others = candidates, []

        corroborated.sort(key=lambda c: c["distance"])
        others.sort(key=lambda c: c["distance"])

        selected = corroborated[:top_k_images]
        if len(selected) < top_k_images:
            remaining = top_k_images - len(selected)
            strong_others = [c for c in others if c["distance"] <= IMAGE_DISTANCE_THRESHOLD]
            selected += strong_others[:remaining]

        return [{"path": c["path"], "page": c["page"]} for c in selected]