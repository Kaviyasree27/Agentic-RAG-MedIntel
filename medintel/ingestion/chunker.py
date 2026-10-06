import re
import uuid
from typing import List, Dict, Any
from medintel.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class ClinicalChunker:
    """Performs section-aware and token/word-aware chunking on parsed documents with globally unique chunk IDs."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Common clinical header patterns
        self.header_pattern = re.compile(
            r"(?:^|\n)(#{1,4}\s+.*|[A-Z][A-Za-z0-9\s\/\-]{2,30}:|(?:HISTORY OF PRESENT ILLNESS|PAST MEDICAL HISTORY|MEDICATIONS|PHYSICAL EXAMINATION|LABORATORY|IMPRESSION|ASSESSMENT|PLAN|DIAGNOSIS|CONTRAINDICATIONS|DOSAGE AND ADMINISTRATION|WARNINGS|INDICATIONS)[\s\:])",
            re.IGNORECASE
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits a list of parsed page documents into granular chunks with unique chunk IDs."""
        all_chunks = []
        chunk_id_counter = 0

        for doc in documents:
            text = doc["text"]
            source = doc.get("source", "unknown")
            page_num = doc.get("page_number", 1)
            clean_src = re.sub(r'[^a-zA-Z0-9_]', '_', source)

            # Split into words/tokens
            words = text.split()
            if not words:
                continue

            i = 0
            while i < len(words):
                end_idx = min(i + self.chunk_size, len(words))
                chunk_text = " ".join(words[i:end_idx])

                # Extract likely section context
                section_matches = self.header_pattern.findall(chunk_text)
                section = section_matches[0].strip(": \n#") if section_matches else "General"

                chunk_id_counter += 1
                unique_cid = f"{clean_src}_p{page_num}_c{chunk_id_counter}_{uuid.uuid4().hex[:6]}"
                
                chunk_obj = {
                    "chunk_id": unique_cid,
                    "text": chunk_text,
                    "source": source,
                    "page_number": page_num,
                    "section": section,
                    "token_count": len(words[i:end_idx])
                }
                all_chunks.append(chunk_obj)

                if end_idx >= len(words):
                    break
                i += max(1, self.chunk_size - self.chunk_overlap)

        return all_chunks
