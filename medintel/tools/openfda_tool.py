import logging
import urllib.parse
from typing import Dict, Any, List
import requests

logger = logging.getLogger(__name__)

class OpenFDATool:
    """
    Interacts with the US Food and Drug Administration (OpenFDA) public API.
    Provides verified drug labeling, contraindications, drug-drug interactions, and boxed warnings.
    100% Free & Fast (No API key required for standard usage).
    """

    BASE_URL = "https://api.fda.gov/drug/label.json"

    @classmethod
    def get_drug_info(cls, drug_name: str) -> Dict[str, Any]:
        """Fetches indications, contraindications, drug interactions, and warnings for a given medication."""
        clean_drug = drug_name.strip()
        if not clean_drug:
            return {"error": "Drug name is required"}

        try:
            # Query openFDA for active ingredient or brand name
            query_str = f'openfda.generic_name:"{clean_drug}" OR openfda.brand_name:"{clean_drug}" OR active_ingredient:"{clean_drug}"'
            params = {
                "search": query_str,
                "limit": 1
            }
            response = requests.get(cls.BASE_URL, params=params, timeout=5)

            if response.status_code == 404:
                # Fallback broader search
                broad_query = f'description:"{clean_drug}"'
                response = requests.get(cls.BASE_URL, params={"search": broad_query, "limit": 1}, timeout=5)

            if response.status_code != 200:
                return {
                    "drug_name": clean_drug,
                    "found": False,
                    "status_code": response.status_code,
                    "message": f"No openFDA label records found for '{clean_drug}'"
                }

            data = response.json()
            if not data.get("results"):
                return {"drug_name": clean_drug, "found": False}

            result = data["results"][0]

            def _get_field(field_name: str) -> str:
                val = result.get(field_name, [])
                if isinstance(val, list) and val:
                    return "\n".join(val)[:1200]  # truncate for concise context
                return "Not specified in official label"

            return {
                "drug_name": clean_drug,
                "found": True,
                "generic_name": result.get("openfda", {}).get("generic_name", [clean_drug])[0],
                "brand_name": result.get("openfda", {}).get("brand_name", ["N/A"])[0],
                "boxed_warning": _get_field("boxed_warning"),
                "contraindications": _get_field("contraindications"),
                "drug_interactions": _get_field("drug_interactions"),
                "warnings_and_cautions": _get_field("warnings_and_cautions"),
                "dosage_and_administration": _get_field("dosage_and_administration"),
                "source": "US FDA Official Drug Label (OpenFDA)"
            }

        except Exception as e:
            logger.error(f"OpenFDA API query failed for {drug_name}: {e}")
            return {
                "drug_name": clean_drug,
                "found": False,
                "error": str(e)
            }

    @classmethod
    def check_drug_interaction(cls, drug_a: str, drug_b: str) -> Dict[str, Any]:
        """Checks if drug_a's FDA label mentions interactions with drug_b."""
        info_a = cls.get_drug_info(drug_a)
        info_b = cls.get_drug_info(drug_b)

        interaction_alerts: List[str] = []

        if info_a.get("found"):
            interactions_text = info_a.get("drug_interactions", "").lower()
            warnings_text = info_a.get("warnings_and_cautions", "").lower()
            contra_text = info_a.get("contraindications", "").lower()
            boxed = info_a.get("boxed_warning", "").lower()

            target = drug_b.lower()
            if target in interactions_text or target in warnings_text or target in contra_text or target in boxed:
                interaction_alerts.append(f"FDA label for {drug_a} contains interaction/warning warnings mentioning {drug_b}.")

        if info_b.get("found"):
            interactions_text = info_b.get("drug_interactions", "").lower()
            warnings_text = info_b.get("warnings_and_cautions", "").lower()
            target = drug_a.lower()
            if target in interactions_text or target in warnings_text:
                interaction_alerts.append(f"FDA label for {drug_b} contains interaction/warning warnings mentioning {drug_a}.")

        has_alert = len(interaction_alerts) > 0

        return {
            "drug_pair": f"{drug_a} + {drug_b}",
            "interaction_detected": has_alert,
            "alerts": interaction_alerts if has_alert else [f"No direct contraindication keywords found between {drug_a} and {drug_b} in current label excerpts."],
            "drug_a_data": info_a,
            "drug_b_data": info_b
        }
