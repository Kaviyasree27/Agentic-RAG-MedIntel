from typing import List, Dict, Any
from medintel.config import RRF_K_CONSTANT

class ReciprocalRankFusion:
    """Combines rankings from multiple retrieval sources (Dense Vector + BM25) using RRF."""

    @staticmethod
    def fuse(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k_constant: int = RRF_K_CONSTANT,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Computes reciprocal rank fusion scores for documents retrieved across methods.
        Formula: RRF_Score(d) = sum(1 / (k + rank(d)))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        provenance: Dict[str, List[str]] = {}

        # Process Dense results
        for rank_idx, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            chunk_lookup[cid] = doc
            provenance.setdefault(cid, []).append(f"dense(rank={rank_idx+1})")
            score_contribution = 1.0 / (k_constant + (rank_idx + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution

        # Process Sparse BM25 results
        for rank_idx, doc in enumerate(sparse_results):
            cid = doc["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = doc
            provenance.setdefault(cid, []).append(f"bm25(rank={rank_idx+1})")
            score_contribution = 1.0 / (k_constant + (rank_idx + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution

        # Sort combined results by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results = []
        for cid in sorted_cids[:top_k]:
            base_doc = chunk_lookup[cid].copy()
            base_doc["rrf_score"] = round(rrf_scores[cid], 5)
            base_doc["retrieval_method"] = "rrf_fusion"
            base_doc["provenance"] = ", ".join(provenance.get(cid, []))
            fused_results.append(base_doc)

        return fused_results
