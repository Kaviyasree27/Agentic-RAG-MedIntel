import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

class BM25SearchEngine:
    """Sparse keyword search engine powered by BM25 with medical-aware tokenization."""

    def __init__(self):
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        # Tokenizer preserves alphanumerics like eGFR, HbA1c, 10mg, ICD-10, etc.
        self.token_pattern = re.compile(r"[A-Za-z0-9\-\.\/]+", re.IGNORECASE)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text preserving clinical terms and converting to lowercase."""
        return [tok.lower() for tok in self.token_pattern.findall(text) if len(tok) > 1]

    def clear(self):
        """Resets the BM25 index."""
        self.corpus_chunks = []
        self.bm25 = None

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes chunks into the BM25 inverted index."""
        if not chunks:
            return 0
        self.corpus_chunks = chunks
        tokenized_corpus = [self.tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return len(chunks)

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Searches BM25 index and returns top-K scored chunks."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Pair with chunks and sort descending
        scored_pairs = list(zip(self.corpus_chunks, doc_scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        sparse_hits = []
        for chunk, score in scored_pairs[:top_k]:
            if score <= 0:
                continue
            sparse_hits.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "page_number": chunk.get("page_number", 1),
                "section": chunk.get("section", "General"),
                "score": float(score),
                "retrieval_method": "bm25_sparse"
            })
        return sparse_hits
