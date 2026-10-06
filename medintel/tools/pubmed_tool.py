import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

class PubMedTool:
    """
    Direct client for NCBI Entrez E-Utilities API.
    Searches PubMed for clinical trials, guidelines, and biomedical literature in real-time.
    100% Free public API.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    @classmethod
    def search_literature(cls, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Searches PubMed for relevant papers and returns structured metadata + titles + abstracts."""
        clean_query = query.strip()
        if not clean_query:
            return []

        try:
            # 1. Search for PMIDs
            search_params = {
                "db": "pubmed",
                "term": clean_query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance"
            }
            res = requests.get(cls.ESEARCH_URL, params=search_params, timeout=6)
            if res.status_code != 200:
                logger.warning(f"PubMed search returned status code {res.status_code}")
                return []

            data = res.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # 2. Fetch summaries for PMIDs
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            sum_res = requests.get(cls.ESUMMARY_URL, params=summary_params, timeout=6)
            if sum_res.status_code != 200:
                return []

            sum_data = sum_res.json().get("result", {})
            articles = []

            for pmid in id_list:
                item = sum_data.get(pmid, {})
                if not item:
                    continue

                title = item.get("title", "Untitled")
                pubdate = item.get("pubdate", "Unknown Date")
                source = item.get("source", "PubMed")
                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "pub_date": pubdate,
                    "journal": source,
                    "authors": author_str,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": f"PubMed (PMID: {pmid})"
                })

            return articles

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
