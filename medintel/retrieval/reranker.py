import logging
from typing import List, Dict, Any, Optional
from medintel.config import TOP_K_RERANKED

logger = logging.getLogger(__name__)

class MedicalReranker:
    """
    Reranks candidate chunks against the clinical query using ultra-fast cross-encoder models.
    Supports FlashRank (lightweight, ~10ms CPU) and SentenceTransformers CrossEncoder.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._flashrank_client = None
        self._cross_encoder = None
        self._init_engine()

    def _init_engine(self):
        try:
            from flashrank import Ranker
            self._flashrank_client = Ranker(model_name=self.model_name, cache_dir="./models_cache")
            logger.info("FlashRank reranker initialized successfully.")
        except Exception as e:
            logger.warning(f"FlashRank init failed, falling back to CrossEncoder: {e}")
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as ce:
                logger.error(f"CrossEncoder fallback also failed: {ce}")

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = TOP_K_RERANKED) -> List[Dict[str, Any]]:
        """Scores candidate chunks against the query and returns top-K with highest precision."""
        if not candidate_chunks:
            return []

        if len(candidate_chunks) == 1:
            candidate_chunks[0]["rerank_score"] = 1.0
            return candidate_chunks

        # Fast path with FlashRank
        if self._flashrank_client is not None:
            try:
                from flashrank import RerankRequest
                passages = [{"id": i, "text": chunk["text"], "meta": chunk} for i, chunk in enumerate(candidate_chunks)]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = self._flashrank_client.rerank(rerank_req)

                reranked_chunks = []
                for res in results[:top_k]:
                    chunk_meta = res["meta"].copy()
                    chunk_meta["rerank_score"] = round(float(res.get("score", 0.0)), 4)
                    chunk_meta["retrieval_method"] = "hybrid_rrf_reranked"
                    reranked_chunks.append(chunk_meta)
                return reranked_chunks
            except Exception as e:
                logger.warning(f"FlashRank scoring failed: {e}")

        # Fallback with CrossEncoder
        if self._cross_encoder is not None:
            try:
                pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
                scores = self._cross_encoder.predict(pairs)
                scored_chunks = []
                for chunk, score in zip(candidate_chunks, scores):
                    c = chunk.copy()
                    c["rerank_score"] = round(float(score), 4)
                    c["retrieval_method"] = "hybrid_rrf_reranked"
                    scored_chunks.append(c)
                scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_chunks[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder scoring failed: {e}")

        # Graceful fallback: return top candidates sorted by existing RRF score
        sorted_fallback = sorted(candidate_chunks, key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return sorted_fallback[:top_k]
