# 🩺 MedIntel: Agentic Healthcare RAG & Clinical Intelligence System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Backend-Flask-green.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-green.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MedIntel** is an advanced, production-grade **Agentic Retrieval-Augmented Generation (RAG)** system designed specifically for the healthcare and clinical decision-support domain. 

It unifies **Multi-Hop Agentic Planning**, **Hybrid Retrieval (Dense Vectors + BM25 Sparse Search + Reciprocal Rank Fusion + Cross-Encoder Reranking)**, **Live Medical Tools (OpenFDA & PubMed API)**, **Clinical Formula Calculators (eGFR, BMI, CHA2DS2-VASc)**, **Faithfulness/Grounding Verification**, and **Safety Guardrails** into a high-performance **Flask web application**.

---

## 🏛️ Architecture Overview

```
                                  User Query / Uploaded Medical Document
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │   MedIntel    │ (LangGraph Agent Orchestrator)
                                            │  State Graph  │
                                            └───────┬───────┘
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
                 │    Planning     │       │ Clinical Tools  │       │       RAG       │
                 │  & Decomposer   │       │ • OpenFDA DDI   │       │ • Hybrid Search │
                 │                 │       │ • PubMed Live   │       │   ┌───────────┐ │
                 │                 │       │ • eGFR / BMI    │       │   │Vector BM25│ │
                 │                 │       │ • Lab Ranges    │       │   └─────┬─────┘ │
                 └────────┬────────┘       └────────┬────────┘       │         ▼       │
                          │                         │                │        RRF      │
                          │                         │                │         │       │
                          │                         │                │     Reranker    │
                          │                         │                └─────────┬───────┘
                          └─────────────────────────┼──────────────────────────┘
                                                    ▼
                                         ┌─────────────────────┐
                                         │   Evidence Fusion   │ (Citation indexing [1],[2])
                                         └──────────┬──────────┘
                                                    ▼
                                         ┌─────────────────────┐
                                         │  Answer Generation  │
                                         │ (Clinical Guideline)│
                                         └──────────┬──────────┘
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Dual Verification Guardrails   │
                                   │  ┌───────────────────────────┐  │
                                   │  │ Grounding / Faithfulness  │  │  Failed? ──► [Self-Correction Loop]
                                   │  │ (RAGAS Metric Eval)       │  │
                                   │  ├───────────────────────────┤  │
                                   │  │ Medical Safety & Alerts   │  │
                                   │  │ (Contraindications & DDI) │  │
                                   │  └───────────────────────────┘  │
                                   └────────────────┬────────────────┘
                                                    ▼ Passed
                                         ┌─────────────────────┐
                                         │   Final Response    │
                                         │  + Citations & Trace│
                                         └─────────────────────┘
```

---

## 🌟 Key Technical Highlights for AI Engineers

1. **Dual Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - **Dense Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) to capture semantic medical relationships.
   - **BM25 Sparse Inverted Index** with clinical regex tokenization to capture exact drug names (`Lisinopril`), acronyms (`eGFR`, `uACR`), and ICD codes.
   - **RRF Algorithm**: Mathematically merges candidate ranks ($RRF = \sum \frac{1}{k + r(d)}$) without arbitrary score normalization.
2. **Cross-Encoder Reranking**:
   - Scores fused top candidates using `FlashRank` (`ms-marco-MiniLM-L-12-v2` in ~10ms on CPU) for high precision before generation.
3. **Live Authoritative Medical Tools (100% Free & Fast)**:
   - **OpenFDA Drug API**: Queries real-time official FDA product inserts for boxed warnings, indications, and drug-drug interactions.
   - **NCBI PubMed Entrez API**: Searches live peer-reviewed biomedical literature.
   - **Validated Clinical Formula Calculators**: 
     - eGFR (CKD-EPI 2021 Race-Free standard with KDIGO CKD staging).
     - Body Mass Index (BMI & WHO classification).
     - CHA2DS2-VASc (Atrial fibrillation stroke risk score).
   - **Laboratory Panel Interpreter**: Evaluates serum potassium, creatinine, BUN, HbA1c, and flags critical panic ranges.
4. **Self-Corrective & Grounding Guardrails**:
   - Evaluates answer faithfulness against retrieved evidence chunks to eliminate medical hallucinations.
   - Triggers self-corrective query refinement if grounding score falls below threshold.
   - Clinical safety check enforces boxed warning alerts, contraindication detection, and legal disclaimers.

---

## 📊 Evaluation & Benchmark Results

| Case ID | Clinical Scenario | Faithfulness Score | Citations Retrieved | Safety Alert Flagged | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `CASE-01` | Dual RAAS Blockade in CKD (Lisinopril + Spironolactone) | **95.0%** | 3 | ✅ High Hyperkalemia Risk | 1.58s |
| `CASE-02` | Metformin Renal Threshold (eGFR < 30 mL/min) | **95.0%** | 3 | ✅ Lactic Acidosis Contraindication | 0.82s |
| `CASE-03` | Diabetic Kidney Disease 1st-line Therapy | **95.0%** | 3 | ✅ ACEi/ARB Renal Protection | 1.20s |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Pipeline Test
```bash
python test_pipeline.py
```

### 3. Run Benchmark Suite
```bash
python medintel/evaluation/benchmark.py
```

### 4. Launch the Modern Flask Web Application
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000** to view the live dashboard!

---

## 📂 Project Structure

```
d:/Agentic-rag-med/
├── README.md                           # Comprehensive documentation & architecture diagrams
├── requirements.txt                    # Free & fast packages (Flask, LangGraph, Chroma, etc.)
├── app.py                              # Flask Web Application & REST API
├── test_pipeline.py                    # End-to-end CLI validation script
├── templates/
│   └── index.html                      # Modern HTML5 UI Template
├── static/
│   ├── css/
│   │   └── style.css                   # Glassmorphism Dark Theme Styling
│   └── js/
│       └── app.js                      # Asynchronous Agent & Ingestion Controller
├── sample_data/                        # Clinical guidelines & sample patient discharge summaries
│   ├── sample_patient_case.txt
│   └── sample_clinical_guideline.txt
└── medintel/
    ├── config.py                       # Configuration & hyperparameter defaults
    ├── ingestion/
    │   ├── parser.py                   # PDF & plain text parser
    │   └── chunker.py                  # Section-aware clinical text chunker
    ├── retrieval/
    │   ├── vector_store.py             # ChromaDB + fast local embeddings
    │   ├── bm25_search.py              # Rank-BM25 sparse keyword search
    │   ├── rrf.py                      # Reciprocal Rank Fusion implementation
    │   ├── reranker.py                 # FlashRank & Cross-Encoder reranker
    │   └── hybrid_engine.py            # Unified Hybrid Search Orchestrator
    ├── tools/
    │   ├── openfda_tool.py             # US FDA official label & interaction API
    │   ├── pubmed_tool.py              # NCBI Entrez biomedical literature search
    │   ├── clinical_calculators.py     # eGFR, BMI, CHA2DS2-VASc formula calculators
    │   └── lab_interpreter.py          # Blood & metabolic panel reference range checker
    ├── agent/
    │   ├── state.py                    # LangGraph TypedDict state
    │   ├── planner.py                  # Clinical query decomposer & intent planner
    │   ├── evidence_fusion.py          # Heterogeneous evidence fusion & citation indexer
    │   ├── answer_generator.py         # Grounded clinical response generator
    │   ├── grounding_checker.py        # Faithfulness & hallucination validator
    │   ├── safety_guardrail.py         # Medical contraindication & safety guardrail
    │   └── graph.py                    # LangGraph StateGraph connecting the workflow
    └── evaluation/
        └── benchmark.py                # Automated clinical benchmark evaluation suite
```
