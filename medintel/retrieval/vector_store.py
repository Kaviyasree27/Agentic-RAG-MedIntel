import os
import shutil
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from medintel.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

class DenseVectorStore:
    """Manages dense embedding generation and ChromaDB vector indexing."""

    def __init__(self, collection_name: str = "medintel_collection", persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.embed_model_name = EMBEDDING_MODEL
        self._embedder = None

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def clear(self):
        """Resets the vector collection."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds and indexes document chunks into ChromaDB."""
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [
            {
                "source": str(chunk.get("source", "unknown")),
                "page_number": int(chunk.get("page_number", 1)),
                "section": str(chunk.get("section", "General")),
            }
            for chunk in chunks
        ]

        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return len(ids)

    def search(self, query: str, top_k: int = 6, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs dense semantic vector search strictly honoring doc_filter."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()
        
        where_clause = None
        clean_name = ""
        if doc_filter:
            clean_name = doc_filter.split("(")[0].strip().lower()

        results = None
        # Attempt filtered query first
        if clean_name:
            try:
                all_data = self.collection.get(include=["metadatas"])
                matching_sources = set()
                for m in all_data.get("metadatas", []):
                    src = m.get("source", "")
                    if clean_name in src.lower() or src.lower() in clean_name:
                        matching_sources.add(src)
                
                if len(matching_sources) == 1:
                    where_clause = {"source": list(matching_sources)[0]}
                elif len(matching_sources) > 1:
                    where_clause = {"source": {"$in": list(matching_sources)}}
                
                if where_clause:
                    results = self.collection.query(
                        query_embeddings=query_embedding,
                        n_results=min(top_k, self.collection.count()),
                        where=where_clause,
                        include=["documents", "metadatas", "distances"]
                    )
            except Exception:
                results = None

        if not results:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k * 2, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )

        dense_hits = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                similarity = 1.0 - distance

                dense_hits.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "section": meta.get("section", "General"),
                    "score": float(similarity),
                    "retrieval_method": "vector_dense"
                })
        
        # STRICT post-filtering: if doc_filter is provided, NEVER leak chunks from other files
        if clean_name:
            filtered = [h for h in dense_hits if clean_name in h.get("source", "").lower() or h.get("source", "").lower() in clean_name]
            return filtered

        return dense_hits[:top_k]
