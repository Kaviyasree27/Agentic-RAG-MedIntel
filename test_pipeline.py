import os
import sys

# Ensure UTF-8 output encoding for Windows command prompt / PowerShell
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from medintel.retrieval.hybrid_engine import HybridRetrievalEngine
from medintel.agent.graph import MedIntelAgent
from medintel.tools.openfda_tool import OpenFDATool
from medintel.tools.pubmed_tool import PubMedTool
from medintel.tools.clinical_calculators import ClinicalCalculators
from medintel.tools.lab_interpreter import LabInterpreter

def test_full_pipeline():
    print("=" * 70)
    print("MEDINTEL: AGENTIC HEALTHCARE RAG PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Initialize Hybrid Engine
    print("\n[Step 1] Initializing Hybrid RAG Engine (Vector + BM25 + RRF + Reranker)...")
    engine = HybridRetrievalEngine(collection_name="test_medintel")
    engine.reset()

    # 2. Ingest Sample Documents
    guideline_path = "sample_data/sample_clinical_guideline.txt"
    case_path = "sample_data/sample_patient_case.txt"
    
    res1 = engine.ingest_document(guideline_path, filename="KDIGO_ADA_2024_Guideline.txt")
    res2 = engine.ingest_document(case_path, filename="Patient_Discharge_Case.txt")
    print(f"-> Ingested Guideline: {res1['chunks_count']} chunks")
    print(f"-> Ingested Patient Case: {res2['chunks_count']} chunks")

    # 3. Test Hybrid Search
    test_query = "What is the safety of Lisinopril and Spironolactone combination in CKD with eGFR thresholds?"
    print(f"\n[Step 2] Testing Hybrid Search for: '{test_query}'")
    search_res = engine.search(test_query)
    
    print("\n--- TOP RERANKED CHUNKS ---")
    for idx, c in enumerate(search_res["reranked_results"]):
        score = c.get("rerank_score", c.get("rrf_score", 0.0))
        print(f"  [{idx + 1}] Source: {c['source']} (Page {c['page_number']}) | Score: {score}")
        print(f"      Excerpt: {c['text'][:140]}...")

    # 4. Test Clinical Tools Directly
    print("\n[Step 3] Testing Clinical Tools (Free Public APIs & Calculators)...")
    
    # 4a. eGFR Calculator
    egfr = ClinicalCalculators.calculate_egfr(serum_creatinine=1.8, age=58, is_female=False)
    print(f"  [Tool: eGFR] Value={egfr['egfr_value_ml_min']} mL/min | Stage={egfr['stage']}")

    # 4b. OpenFDA Drug Interaction
    print("  [Tool: OpenFDA] Checking Lisinopril + Spironolactone...")
    fda_check = OpenFDATool.check_drug_interaction("Lisinopril", "Spironolactone")
    print(f"      Alert: {fda_check['alerts'][0][:110]}...")

    # 4c. Lab Interpreter
    lab_eval = LabInterpreter.interpret("potassium", 5.4)
    print(f"  [Tool: Lab] Potassium 5.4 -> Status={lab_eval['status']} | {lab_eval['interpretation']}")

    # 5. Run Full LangGraph Agent
    print("\n[Step 4] Executing Full LangGraph Agent...")
    agent = MedIntelAgent(hybrid_engine=engine)
    
    patient_context = {
        "age": 58,
        "is_female": False,
        "creatinine": 1.8,
        "labs": {"potassium": 4.8, "creatinine": 1.8, "hba1c": 7.8}
    }
    
    output = agent.run(test_query, patient_context=patient_context)

    # Print Trace Logs
    print("\n--- AGENT EXECUTION TRACE LOG ---")
    for step in output.get("trace", []):
        print(f"  {step}")

    # Print Verification
    print(f"\nGrounding Faithfulness Score: {round(output.get('grounding_eval', {}).get('faithfulness_score', 0)*100)}%")
    print(f"Safety Alerts Count: {output.get('safety_eval', {}).get('alerts_count')}")

    # Print Final Response
    print("\n" + "=" * 70)
    print("FINAL CLINICAL RESPONSE & CITATIONS:")
    print("=" * 70)
    print(output.get("final_response", ""))
    print("=" * 70)
    print("\n[SUCCESS] MedIntel Pipeline Verification Complete!")

if __name__ == "__main__":
    test_full_pipeline()
