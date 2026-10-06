import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from medintel.agent.state import AgentState
from medintel.agent.llm_client import LLMClient
from medintel.agent.planner import ClinicalPlanner
from medintel.agent.evidence_fusion import EvidenceFusionEngine
from medintel.agent.answer_generator import AnswerGenerator
from medintel.agent.grounding_checker import GroundingChecker
from medintel.agent.safety_guardrail import SafetyGuardrail
from medintel.retrieval.hybrid_engine import HybridRetrievalEngine
from medintel.tools.openfda_tool import OpenFDATool
from medintel.tools.pubmed_tool import PubMedTool
from medintel.tools.clinical_calculators import ClinicalCalculators
from medintel.tools.lab_interpreter import LabInterpreter

logger = logging.getLogger(__name__)

class MedIntelAgent:
    """
    Main Orchestrator implementing the full Agentic RAG architecture:
    Planning ➜ Hybrid RAG & Tools ➜ Evidence Fusion ➜ Generation ➜ Grounding Check ➜ Safety Check ➜ Response
    """

    def __init__(self, hybrid_engine: HybridRetrievalEngine = None):
        self.hybrid_engine = hybrid_engine or HybridRetrievalEngine()
        self.llm_client = LLMClient()
        self.planner = ClinicalPlanner(self.llm_client)
        self.answer_generator = AnswerGenerator(self.llm_client)
        self.grounding_checker = GroundingChecker(self.llm_client)
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("planning_node", self.planning_node)
        workflow.add_node("rag_retrieval_node", self.rag_retrieval_node)
        workflow.add_node("tools_execution_node", self.tools_execution_node)
        workflow.add_node("evidence_fusion_node", self.evidence_fusion_node)
        workflow.add_node("answer_generation_node", self.answer_generation_node)
        workflow.add_node("verification_guardrails_node", self.verification_guardrails_node)

        # Build Flow
        workflow.set_entry_point("planning_node")

        workflow.add_edge("planning_node", "rag_retrieval_node")
        workflow.add_edge("rag_retrieval_node", "tools_execution_node")
        workflow.add_edge("tools_execution_node", "evidence_fusion_node")
        workflow.add_edge("evidence_fusion_node", "answer_generation_node")
        workflow.add_edge("answer_generation_node", "verification_guardrails_node")

        # Conditional Edge after Verification (Self-correction or End)
        workflow.add_conditional_edges(
            "verification_guardrails_node",
            self.route_after_verification,
            {
                "retry_refine": "answer_generation_node",
                "finalize": END
            }
        )

        return workflow.compile()

    # --- Node Implementations ---

    def planning_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        query = state["query"]
        p_context = state.get("patient_context", {})
        doc_filter = state.get("doc_filter")

        plan = self.planner.plan(query, p_context, doc_filter=doc_filter)
        trace.append(f"🧠 [Planning Node] Identified intent: '{plan.get('clinical_intent')}'. Sub-queries: {len(plan.get('sub_queries', []))}, Tools: {plan.get('tools_needed', [])}")

        return {
            "plan": plan,
            "selected_tools": plan.get("tools_needed", []),
            "trace": trace,
            "retry_count": state.get("retry_count", 0)
        }

    def rag_retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        plan = state.get("plan", {})
        sub_queries = plan.get("sub_queries", [state["query"]])

        all_retrieved = []
        for sq in sub_queries[:2]:
            doc_filter = state.get("doc_filter")
            search_out = self.hybrid_engine.search(sq, doc_filter=doc_filter)
            top_chunks = search_out.get("reranked_results", [])
            all_retrieved.extend(top_chunks)
            trace.append(f"🔍 [Hybrid RAG Node] Query: '{sq[:40]}...' ➜ Dense hits: {search_out.get('dense_hits_count')}, BM25 hits: {search_out.get('sparse_hits_count')}, RRF Fused: {search_out.get('fused_candidates_count')}, Reranked Top-K: {len(top_chunks)}")

        # Deduplicate retrieved chunks by chunk_id
        unique_chunks = {}
        for c in all_retrieved:
            unique_chunks[c["chunk_id"]] = c

        return {
            "retrieved_docs": list(unique_chunks.values()),
            "trace": trace
        }

    def tools_execution_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        plan = state.get("plan", {})
        tools_needed = plan.get("tools_needed", [])
        drugs = plan.get("drugs_identified", [])
        p_context = state.get("patient_context", {})

        tool_outputs = []

        # 1. OpenFDA Tool
        if "openfda_tool" in tools_needed and drugs:
            if len(drugs) >= 2:
                trace.append(f"💊 [Tool Execution] Querying OpenFDA for drug-drug interaction: {drugs[0]} + {drugs[1]}")
                ddi_res = OpenFDATool.check_drug_interaction(drugs[0], drugs[1])
                tool_outputs.append({"tool_name": "openfda_tool", "data": ddi_res})
            else:
                for drug in drugs[:2]:
                    trace.append(f"💊 [Tool Execution] Querying OpenFDA label for: {drug}")
                    d_info = OpenFDATool.get_drug_info(drug)
                    tool_outputs.append({"tool_name": "openfda_tool", "data": d_info})

        # 2. Clinical Calculators Tool
        if "clinical_calculators" in tools_needed:
            # Check eGFR inputs
            creatinine = p_context.get("creatinine") or plan.get("labs_to_interpret", {}).get("creatinine")
            age = p_context.get("age", 60)
            is_female = p_context.get("is_female", False)

            if creatinine:
                egfr_res = ClinicalCalculators.calculate_egfr(float(creatinine), int(age), bool(is_female))
                trace.append(f"🧮 [Tool Execution] eGFR Calculator calculated: {egfr_res.get('egfr_value_ml_min')} mL/min ({egfr_res.get('stage')})")
                tool_outputs.append({"tool_name": "clinical_calculators", "data": egfr_res})

            # Check BMI inputs
            weight = p_context.get("weight_kg")
            height = p_context.get("height_cm")
            if weight and height:
                bmi_res = ClinicalCalculators.calculate_bmi(float(weight), float(height))
                trace.append(f"🧮 [Tool Execution] BMI Calculator: {bmi_res.get('bmi')} ({bmi_res.get('category')})")
                tool_outputs.append({"tool_name": "clinical_calculators", "data": bmi_res})

        # 3. Lab Interpreter Tool
        labs = p_context.get("labs") or plan.get("labs_to_interpret")
        if labs and isinstance(labs, dict):
            lab_results = LabInterpreter.interpret_panel(labs)
            trace.append(f"🧪 [Tool Execution] Interpreted {len(lab_results)} laboratory test values.")
            tool_outputs.append({"tool_name": "lab_interpreter", "data": lab_results})

        # 4. PubMed Tool
        if "pubmed_tool" in tools_needed:
            pubmed_q = drugs[0] if drugs else state["query"]
            trace.append(f"📚 [Tool Execution] PubMed API Searching: '{pubmed_q}'")
            articles = PubMedTool.search_literature(pubmed_q, max_results=2)
            if articles:
                tool_outputs.append({"tool_name": "pubmed_tool", "data": articles})

        return {
            "tool_outputs": tool_outputs,
            "trace": trace
        }

    def evidence_fusion_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        retrieved_docs = state.get("retrieved_docs", [])
        tool_outputs = state.get("tool_outputs", [])

        fusion_res = EvidenceFusionEngine.fuse_evidence(retrieved_docs, tool_outputs)
        trace.append(f"🔗 [Evidence Fusion] Merged {len(retrieved_docs)} document chunks and {len(tool_outputs)} tool streams into {fusion_res.get('total_sources_count')} indexed citation points.")

        return {
            "fused_evidence": fusion_res,
            "citations": fusion_res.get("citations", []),
            "trace": trace
        }

    def answer_generation_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        query = state["query"]
        fused_evidence = state.get("fused_evidence", {})
        p_context = state.get("patient_context", {})

        doc_filter = state.get("doc_filter")
        draft = self.answer_generator.generate_answer(
            query=query,
            fused_evidence_text=fused_evidence.get("fused_evidence_text", ""),
            patient_context=p_context,
            doc_filter=doc_filter
        )
        trace.append(f"✍️ [Answer Generation] Formulated clinical assessment with citation links.")

        return {
            "draft_answer": draft,
            "trace": trace
        }

    def verification_guardrails_node(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        draft = state.get("draft_answer", "")
        fused_evidence = state.get("fused_evidence", {})
        query = state.get("query", "")
        tool_outputs = state.get("tool_outputs", [])

        # 1. Grounding Check
        grounding_eval = self.grounding_checker.verify_grounding(
            draft_answer=draft,
            fused_evidence_text=fused_evidence.get("fused_evidence_text", "")
        )
        trace.append(f"🛡️ [Grounding Check] Faithfulness Score: {round(grounding_eval.get('faithfulness_score', 0)*100)}% | Verdict: {grounding_eval.get('verdict')}")

        # 2. Safety Check
        safety_eval = SafetyGuardrail.check_safety(
            draft_answer=draft,
            query=query,
            tool_data=tool_outputs
        )
        trace.append(f"🚨 [Safety Check] Flagged Alerts: {safety_eval.get('alerts_count')} | Guardrails Status: {'Passed' if safety_eval.get('is_safe') else 'Attention Required'}")

        # Assemble Final Response
        final_text = draft + safety_eval.get("safety_banner", "") + safety_eval.get("disclaimer", "")

        return {
            "grounding_eval": grounding_eval,
            "safety_eval": safety_eval,
            "final_response": final_text,
            "trace": trace,
            "retry_count": state.get("retry_count", 0) + 1
        }

    def route_after_verification(self, state: AgentState) -> str:
        """Corrective loop router: checks if draft needs self-refinement."""
        grounding_eval = state.get("grounding_eval", {})
        retry_count = state.get("retry_count", 0)

        # Self-correction condition: if grounding failed and retry count < 2
        if not grounding_eval.get("is_grounded", True) and retry_count < 2:
            return "retry_refine"
        return "finalize"

    def run(self, query: str, patient_context: Dict[str, Any] = None, doc_filter: str = None) -> Dict[str, Any]:
        """Executes the full agent graph and returns final state."""
        initial_state: AgentState = {
            "query": query,
            "patient_context": patient_context or {},
            "doc_filter": doc_filter,
            "plan": {},
            "selected_tools": [],
            "retrieved_docs": [],
            "tool_outputs": [],
            "fused_evidence": {},
            "draft_answer": "",
            "grounding_eval": {},
            "safety_eval": {},
            "final_response": "",
            "citations": [],
            "trace": ["🚀 [MedIntel Agent] Clinical query initiated."],
            "retry_count": 0
        }

        return self.app.invoke(initial_state)
