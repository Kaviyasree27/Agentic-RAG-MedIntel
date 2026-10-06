import json
import re
from typing import Dict, Any, List
from medintel.agent.llm_client import LLMClient

class ClinicalPlanner:
    """Decomposes complex medical queries and identifies required tools & RAG sub-queries."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, query: str, patient_context: Dict[str, Any] = None, doc_filter: str = None) -> Dict[str, Any]:
        """Produces a structured execution plan strictly constrained to active document context."""
        context_str = json.dumps(patient_context or {})
        
        doc_topic = ""
        doc_constraint_prompt = ""
        if doc_filter:
            doc_topic = doc_filter.split(".")[0].replace("_", " ").strip()
            doc_constraint_prompt = (
                f"\nCRITICAL MANDATORY SCOPE:\n"
                f"Active Focus Document: '{doc_filter}' (Primary Medical Subject: '{doc_topic}').\n"
                f"The user query must be interpreted strictly in the context of '{doc_topic}'.\n"
                f"All sub_queries and the clinical_intent MUST focus exclusively on '{doc_topic}' and NEVER on any other condition (such as asthma, diabetes, heart failure, etc. unless explicitly mentioned in the user query).\n"
            )

        system_prompt = (
            "You are an expert Chief Medical Officer and Clinical AI Planner. "
            "Your job is to analyze clinical queries, resolve conversational context against the active document, "
            "extract drug names, identify needed calculations, and decide which tools to execute. Output purely valid JSON."
        )

        prompt = f"""
Analyze the following clinical scenario and output a JSON plan:
User Query: "{query}"
{doc_constraint_prompt}
Patient Context: {context_str}

Available Tools:
- "openfda_tool": for drug contraindications, interactions, boxed warnings, dosing.
- "pubmed_tool": for latest biomedical research and clinical trial literature.
- "clinical_calculators": for eGFR, BMI, CHA2DS2-VASc, Wells score.
- "lab_interpreter": for interpreting blood panels and metabolic tests.

Return JSON in this exact structure:
{{
    "clinical_intent": "<brief description of the goal focused on active document topic>",
    "sub_queries": ["<sub query 1 strictly relevant to {doc_topic or 'the query'}>", "<sub query 2 strictly relevant to {doc_topic or 'the query'}>"],
    "drugs_identified": ["<drug1>", "<drug2>"],
    "tools_needed": ["openfda_tool", "clinical_calculators"],
    "calculations_needed": ["egfr", "bmi"],
    "labs_to_interpret": {{"creatinine": 1.8, "potassium": 5.2}},
    "reasoning": "<why these steps were chosen>"
}}
"""
        response_text = self.llm.generate(prompt, system_prompt=system_prompt)
        
        # Parse JSON
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
            else:
                plan_data = json.loads(response_text)
        except Exception:
            resolved_query = query
            if doc_topic:
                if doc_topic.lower() not in query.lower():
                    resolved_query = f"{query} for {doc_topic}"

            drugs = []
            common_drugs = ["lisinopril", "spironolactone", "metformin", "atorvastatin", "warfarin", "aspirin", "ibuprofen", "losartan", "amlodipine", "empagliflozin"]
            for d in common_drugs:
                if d in resolved_query.lower():
                    drugs.append(d.capitalize())

            tools = []
            if drugs:
                tools.append("openfda_tool")
            if "egfr" in resolved_query.lower() or "creatinine" in resolved_query.lower():
                tools.append("clinical_calculators")

            plan_data = {
                "clinical_intent": f"Evaluate {resolved_query}",
                "sub_queries": [resolved_query],
                "drugs_identified": drugs,
                "tools_needed": tools,
                "calculations_needed": [],
                "labs_to_interpret": {},
                "reasoning": f"Scoped retrieval strictly for {doc_filter or 'general knowledge'}"
            }

        # Post-enforce document topic in sub_queries if doc_filter is active
        if doc_topic:
            filtered_sub_queries = []
            for sq in plan_data.get("sub_queries", []):
                if doc_topic.lower() not in sq.lower():
                    filtered_sub_queries.append(f"{sq} {doc_topic}")
                else:
                    filtered_sub_queries.append(sq)
            if not filtered_sub_queries:
                filtered_sub_queries = [f"{query} {doc_topic}"]
            plan_data["sub_queries"] = filtered_sub_queries

        return plan_data
