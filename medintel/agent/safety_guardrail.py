import re
from typing import Dict, Any, List

class SafetyGuardrail:
    """
    Clinical safety evaluator and guardrail guard.
    Identifies high-risk clinical red flags, dangerous drug combinations, and enforces disclaimers.
    """

    CRITICAL_CONTRAINDICATIONS = [
        ("lisinopril", "spironolactone", "Risk of severe/fatal Hyperkalemia and acute kidney injury with dual RAAS blockade without close electrolyte monitoring."),
        ("warfarin", "aspirin", "Elevated risk of major gastrointestinal and systemic hemorrhage."),
        ("warfarin", "ibuprofen", "NSAIDs increase bleeding risk and may cause severe GI ulceration with Warfarin."),
        ("metformin", "ckd stage 4", "Metformin is contraindicated with eGFR < 30 mL/min due to lactic acidosis risk."),
        ("metformin", "egfr < 30", "Metformin is contraindicated with severe renal impairment (eGFR < 30 mL/min)."),
        ("viagra", "nitroglycerin", "Fatal hypotension risk with concurrent PDE-5 inhibitor and nitrate therapy."),
        ("sildenafil", "nitroglycerin", "Fatal hypotension risk with concurrent PDE-5 inhibitor and nitrate therapy.")
    ]

    DISCLAIMER = (
        "\n\n---\n"
        "⚖️ **Clinical Decision Support Notice & Disclaimer:**\n"
        "*MedIntel is an AI-augmented evidence retrieval and clinical intelligence system intended for informational "
        "and decision-support use by qualified healthcare professionals. It does not replace individualized clinical judgment, "
        "formal medical diagnosis, or personalized patient assessment.*"
    )

    @classmethod
    def check_safety(cls, draft_answer: str, query: str = "", tool_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs rule-based and content-based safety checks."""
        combined_text = (query + " " + draft_answer).lower()
        flagged_alerts = []

        # Check known severe contraindications
        for drug_a, drug_b, warning in cls.CRITICAL_CONTRAINDICATIONS:
            if drug_a in combined_text and drug_b in combined_text:
                flagged_alerts.append({
                    "type": "CONTRAINDICATION_ALERT",
                    "combination": f"{drug_a.capitalize()} + {drug_b.capitalize()}",
                    "warning": warning
                })

        # Check for critical lab warning terms
        if "critical_high" in str(tool_data).lower() or "critical_low" in str(tool_data).lower():
            flagged_alerts.append({
                "type": "CRITICAL_LAB_ALERT",
                "warning": "One or more laboratory values are in the critical panic range requiring immediate clinical triage."
            })

        is_safe = len(flagged_alerts) == 0 or all(alert["type"] != "CRITICAL_FATAL" for alert in flagged_alerts)

        safety_banner = ""
        if flagged_alerts:
            safety_banner = "\n\n### ⚠️ Critical Clinical Safety Alerts\n"
            for alert in flagged_alerts:
                if "combination" in alert:
                    safety_banner += f"- **{alert['combination']}**: {alert['warning']}\n"
                else:
                    safety_banner += f"- **Alert**: {alert['warning']}\n"

        return {
            "is_safe": is_safe,
            "alerts_count": len(flagged_alerts),
            "alerts": flagged_alerts,
            "safety_banner": safety_banner,
            "disclaimer": cls.DISCLAIMER
        }
