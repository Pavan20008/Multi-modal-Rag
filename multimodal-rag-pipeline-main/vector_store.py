"""
Thin wrapper around ChromaDB for storing/retrieving text and image embeddings.
Two separate collections are kept since text and image embeddings live in
different vector spaces (sentence-transformers vs. CLIP).
"""
import chromadb

import config


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_DIR)

        self.text_collection = self.client.get_or_create_collection(
            name=config.TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name=config.IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add_text_chunks(self, items, embeddings):
        self.text_collection.add(
            ids=[i["id"] for i in items],
            embeddings=embeddings,
            documents=[i["content"] for i in items],
            metadatas=[{"page": i["page"]} for i in items],
        )

    def add_images(self, items, embeddings):
        self.image_collection.add(
            ids=[i["id"] for i in items],
            embeddings=embeddings,
            documents=[i["path"] for i in items],
            metadatas=[{"page": i["page"], "path": i["path"]} for i in items],
        )

    def query_text(self, embedding, top_k):
        return self.text_collection.query(query_embeddings=[embedding], n_results=top_k)

    def query_images(self, embedding, top_k):
        return self.image_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def reset(self):
        self.client.delete_collection(config.TEXT_COLLECTION)
        self.client.delete_collection(config.IMAGE_COLLECTION)
        self.__init__()