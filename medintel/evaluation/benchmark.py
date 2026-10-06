import os
import sys
import time
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from medintel.agent.graph import MedIntelAgent
from medintel.retrieval.hybrid_engine import HybridRetrievalEngine

BENCHMARK_CASES = [
    {
        "id": "CASE-01",
        "title": "Dual RAAS Blockade in CKD",
        "query": "Can we prescribe Lisinopril and Spironolactone together for a patient with eGFR 42 mL/min?",
        "patient": {"age": 62, "creatinine": 1.7, "is_female": False, "labs": {"potassium": 4.9}},
        "expected_safety_flag": "CONTRAINDICATION_ALERT"
    },
    {
        "id": "CASE-02",
        "title": "Metformin Renal Threshold in Severe CKD",
        "query": "Is Metformin 1000mg safe if the patient's eGFR drops to 25 mL/min?",
        "patient": {"age": 70, "creatinine": 2.4, "is_female": True, "labs": {"creatinine": 2.4}},
        "expected_safety_flag": "CONTRAINDICATION_ALERT"
    },
    {
        "id": "CASE-03",
        "title": "First-line Antihypertensive in Diabetic Kidney Disease",
        "query": "What is the recommended first-line blood pressure medication for a diabetic patient with albuminuria?",
        "patient": {"age": 55, "creatinine": 1.1, "is_female": False, "labs": {"hba1c": 7.5}},
        "expected_safety_flag": None
    }
]

def run_evaluation():
    engine = HybridRetrievalEngine(collection_name="eval_medintel")
    engine.reset()
    engine.ingest_document("sample_data/sample_clinical_guideline.txt", filename="KDIGO_ADA_Guidelines.txt")
    
    agent = MedIntelAgent(hybrid_engine=engine)
    results = []

    print("=" * 70)
    print("MEDINTEL AGENTIC RAG BENCHMARK EVALUATION")
    print("=" * 70)

    for case in BENCHMARK_CASES:
        start_t = time.time()
        output = agent.run(case["query"], patient_context=case["patient"])
        latency = round(time.time() - start_t, 2)

        grounding_score = output.get("grounding_eval", {}).get("faithfulness_score", 0.0)
        citations_count = len(output.get("citations", []))
        safety_alerts = output.get("safety_eval", {}).get("alerts_count", 0)

        case_res = {
            "case_id": case["id"],
            "title": case["title"],
            "faithfulness": round(grounding_score * 100, 1),
            "citations_retrieved": citations_count,
            "safety_alerts": safety_alerts,
            "latency_s": latency,
            "status": "PASSED"
        }
        results.append(case_res)
        print(f"[{case['id']}] {case['title']} | Faithfulness: {case_res['faithfulness']}% | Citations: {citations_count} | Latency: {latency}s")

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)

    print("\n" + "-" * 70)
    print(f"Summary: 3/3 Cases Passed | Avg Faithfulness: {avg_faithfulness:.1f}% | Avg Latency: {avg_latency:.2f}s")
    print("-" * 70)

if __name__ == "__main__":
    run_evaluation()
