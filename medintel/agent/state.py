from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """LangGraph state schema representing the complete lifecycle of a clinical query."""
    # Input
    query: str
    patient_context: Optional[Dict[str, Any]]
    doc_filter: Optional[str]
    
    # Planner & Tool decisions
    plan: Dict[str, Any]
    selected_tools: List[str]
    
    # RAG & Evidence
    retrieved_docs: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    fused_evidence: Dict[str, Any]
    
    # Generation & Evaluation
    draft_answer: str
    grounding_eval: Dict[str, Any]
    safety_eval: Dict[str, Any]
    
    # Final Output & Observability
    final_response: str
    citations: List[Dict[str, Any]]
    trace: List[str]
    retry_count: int
