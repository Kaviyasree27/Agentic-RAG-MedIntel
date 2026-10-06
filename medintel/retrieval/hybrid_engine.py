import logging
from typing import List, Dict, Any, Union, Optional
from medintel.ingestion.parser import DocumentParser
from medintel.ingestion.chunker import ClinicalChunker
from medintel.retrieval.vector_store import DenseVectorStore
from medintel.retrieval.bm25_search import BM25SearchEngine
from medintel.retrieval.rrf import ReciprocalRankFusion
from medintel.retrieval.reranker import MedicalReranker
from medintel.config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RERANKED

logger = logging.getLogger(__name__)

class HybridRetrievalEngine:
    """
    End-to-End Hybrid RAG Engine implementing:
    Dense Vector Search + BM25 Sparse Search -> Reciprocal Rank Fusion -> Cross-Encoder Reranker
    """

    def __init__(self, collection_name: str = "medintel_hybrid"):
        self.parser = DocumentParser()
        self.chunker = ClinicalChunker()
        self.vector_store = DenseVectorStore(collection_name=collection_name)
        self.bm25_engine = BM25SearchEngine()
        self.reranker = MedicalReranker()
        self.indexed_chunks: List[Dict[str, Any]] = []

    def reset(self):
        """Clears both dense and sparse indices."""
        self.vector_store.clear()
        self.bm25_engine.clear()
        self.indexed_chunks = []

    def ingest_document(self, file_input: Union[str, bytes], filename: str = "doc") -> Dict[str, Any]:
        """Parses, chunks, and simultaneously indexes a document into both Vector and BM25 stores."""
        parsed_pages = self.parser.parse_file(file_input, filename=filename)
        if not parsed_pages:
            return {"status": "error", "message": "No extractable text found", "chunks_count": 0}

        chunks = self.chunker.chunk_documents(parsed_pages)
        if not chunks:
            return {"status": "error", "message": "Chunking produced 0 chunks", "chunks_count": 0}

        dense_count = self.vector_store.add_chunks(chunks)
        self.indexed_chunks.extend(chunks)
        sparse_count = self.bm25_engine.index_chunks(self.indexed_chunks)

        logger.info(f"Ingested {filename}: {dense_count} dense vectors, {sparse_count} sparse documents indexed.")
        return {
            "status": "success",
            "filename": filename,
            "pages_count": len(parsed_pages),
            "chunks_count": len(chunks),
            "total_corpus_chunks": len(self.indexed_chunks)
        }

    def get_document_content(self, filename: str) -> Dict[str, Any]:
        """Returns all chunks and concatenated text of an indexed document with fuzzy filename matching."""
        clean_name = filename.split("(")[0].strip().lower()
        chunks = [
            c for c in self.indexed_chunks 
            if clean_name in c.get("source", "").lower() or c.get("source", "").lower() in clean_name
        ]
        if not chunks and self.indexed_chunks:
            chunks = [c for c in self.indexed_chunks if c.get("source") == filename]

        if not chunks:
            return {"status": "error", "message": "Document not found", "chunks": [], "full_text": ""}
        
        full_text = "\n\n".join(c.get("text", "") for c in chunks)
        actual_name = chunks[0].get("source", filename)
        return {
            "status": "success",
            "filename": actual_name,
            "chunks_count": len(chunks),
            "pages_count": len(set(c.get("page_number", 1) for c in chunks)),
            "full_text": full_text
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns unique filenames and chunk counts in active memory, newest uploaded first."""
        doc_stats = {}
        for chunk in reversed(self.indexed_chunks):
            src = chunk.get("source", "Unknown")
            if src not in doc_stats:
                doc_stats[src] = {"filename": src, "chunks_count": 0, "pages": set()}
            doc_stats[src]["chunks_count"] += 1
            doc_stats[src]["pages"].add(chunk.get("page_number", 1))
        
        result = []
        for src, data in doc_stats.items():
            result.append({
                "filename": src,
                "chunks_count": data["chunks_count"],
                "pages_count": len(data["pages"])
            })
        return result

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Deletes all chunks of a specific document from vector store and BM25 index."""
        try:
            self.vector_store.collection.delete(where={"source": filename})
        except Exception as e:
            logger.warning(f"Chroma delete error for {filename}: {e}")

        original_count = len(self.indexed_chunks)
        self.indexed_chunks = [c for c in self.indexed_chunks if c.get("source") != filename]
        deleted_chunks = original_count - len(self.indexed_chunks)

        self.bm25_engine.clear()
        if self.indexed_chunks:
            self.bm25_engine.index_chunks(self.indexed_chunks)

        return {
            "status": "success",
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "remaining_total_chunks": len(self.indexed_chunks)
        }

    def search(
        self,
        query: str,
        top_k_dense: int = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
        top_k_final: int = TOP_K_RERANKED,
        doc_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes hybrid retrieval strictly isolating results when doc_filter is provided."""
        dense_hits = self.vector_store.search(query, top_k=top_k_dense, doc_filter=doc_filter)
        sparse_hits = self.bm25_engine.search(query, top_k=top_k_sparse * 2)

        clean_f = ""
        if doc_filter:
            clean_f = doc_filter.split("(")[0].strip().lower()
            # STRICT filter for BM25: only keep chunks belonging to active doc
            sparse_hits = [h for h in sparse_hits if clean_f in h.get("source", "").lower() or h.get("source", "").lower() in clean_f]

        # If sparse and dense both returned 0 for this query but we have chunks in indexed_chunks for doc_filter, return doc chunks!
        if clean_f and not dense_hits and not sparse_hits:
            doc_chunks = [c for c in self.indexed_chunks if clean_f in c.get("source", "").lower() or c.get("source", "").lower() in clean_f]
            if doc_chunks:
                dense_hits = [{
                    "chunk_id": c.get("chunk_id", f"c_{i}"),
                    "text": c.get("text", ""),
                    "source": c.get("source", doc_filter),
                    "page_number": c.get("page_number", 1),
                    "section": c.get("section", "General"),
                    "score": 0.95,
                    "retrieval_method": "direct_document_memory"
                } for i, c in enumerate(doc_chunks[:top_k_final])]

        fused_candidates = ReciprocalRankFusion.fuse(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            top_k=top_k_dense + top_k_sparse
        )

        # Strict filter on fused candidates as well
        if clean_f:
            fused_candidates = [c for c in fused_candidates if clean_f in c.get("source", "").lower() or c.get("source", "").lower() in clean_f]

        final_top_chunks = self.reranker.rerank(
            query=query,
            candidate_chunks=fused_candidates,
            top_k=top_k_final
        )

        # Final guarantee of strict scoping
        if clean_f:
            final_top_chunks = [c for c in final_top_chunks if clean_f in c.get("source", "").lower() or c.get("source", "").lower() in clean_f]
            if not final_top_chunks and fused_candidates:
                final_top_chunks = fused_candidates[:top_k_final]

        return {
            "query": query,
            "dense_hits_count": len(dense_hits),
            "sparse_hits_count": len(sparse_hits),
            "fused_candidates_count": len(fused_candidates),
            "dense_results": dense_hits,
            "sparse_results": sparse_hits,
            "fused_results": fused_candidates,
            "reranked_results": final_top_chunks
        }
