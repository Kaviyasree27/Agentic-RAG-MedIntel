from typing import Dict, Any, List
from medintel.agent.llm_client import LLMClient

class AnswerGenerator:
    """Generates a clinically rigorous, structured response with explicit inline citations [1], [2]."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_answer(
        self,
        query: str,
        fused_evidence_text: str,
        patient_context: Dict[str, Any] = None,
        doc_filter: str = None
    ) -> str:
        """Generates the grounded draft answer strictly based on retrieved documents and clinical tools."""
        doc_topic = ""
        doc_focus_instruction = ""
        if doc_filter:
            doc_topic = doc_filter.split(".")[0].replace("_", " ").strip()
            doc_focus_instruction = (
                f"\nCRITICAL DOCUMENT FOCUS CONSTRAINTS:\n"
                f"- Active Document Scope: '{doc_filter}' (Topic: '{doc_topic}').\n"
                f"- The user's question must be answered strictly and exclusively regarding {doc_topic}.\n"
                f"- Do NOT discuss unrelated conditions (such as asthma, diabetes, heart failure) unless requested by the user.\n"
            )

        system_prompt = (
            "You are MedIntel, a state-of-the-art Clinical Decision Support & Document Intelligence AI. "
            "Your output must be evidence-based, medically precise, and strictly grounded in the provided evidence. "
            "Every medical claim, fact, or recommendation MUST cite the corresponding evidence index (e.g. [1], [2]). "
            "Do NOT speculate, generalize, or switch the topic to other diseases. "
            "Directly answer the user's specific question strictly for the active document topic."
        )

        patient_snippet = f"\nPatient Profile: {patient_context}\n" if patient_context else ""

        prompt = f"""
Clinical Question: {query}{patient_snippet}{doc_focus_instruction}
Available Evidence & Sources:
{fused_evidence_text if fused_evidence_text else 'No specific document chunks or tool outputs retrieved.'}

Please provide a structured clinical response for {doc_topic or 'the inquiry'}:
### 1. Clinical Assessment & Recommendation
- Direct, comprehensive answer to the question regarding {doc_topic or 'the condition'} grounded in the retrieved evidence above with inline citations [1], [2].

### 2. Key Findings, Safety & Contraindications
- Summarize any prevention strategies, drug interactions, safety alerts, or critical findings mentioned in the evidence.

### 3. Evidentiary Grounding & Actionable Summary
- Highlight the source evidence rationale and actionable takeaways.
"""
        return self.llm.generate(prompt, system_prompt=system_prompt)
