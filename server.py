import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from medintel.retrieval.hybrid_engine import HybridRetrievalEngine
from medintel.agent.graph import MedIntelAgent

app = FastAPI(
    title="MedIntel API",
    description="Agentic Healthcare RAG System (Hybrid Search, Tools, LangGraph, Verification)",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared Engine & Agent Instances
hybrid_engine = HybridRetrievalEngine(collection_name="medintel_prod")
agent = MedIntelAgent(hybrid_engine=hybrid_engine)

# Auto-ingest sample guidelines if available
sample_guideline = BASE_DIR / "sample_data" / "sample_clinical_guideline.txt"
sample_case = BASE_DIR / "sample_data" / "sample_patient_case.txt"
if sample_guideline.exists():
    hybrid_engine.ingest_document(str(sample_guideline), filename="KDIGO_ADA_2024_Guideline.txt")
if sample_case.exists():
    hybrid_engine.ingest_document(str(sample_case), filename="Patient_Discharge_Case.txt")


class QueryRequest(BaseModel):
    query: str
    patient_context: Optional[Dict[str, Any]] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>MedIntel Backend Running</h1><p>Static index.html not found</p>")


@app.get("/api/stats")
async def get_stats():
    """Returns knowledge base chunk count and index status."""
    return {
        "status": "online",
        "total_chunks": len(hybrid_engine.indexed_chunks),
        "dense_vectors": hybrid_engine.vector_store.collection.count(),
        "sparse_documents": len(hybrid_engine.bm25_engine.corpus_chunks),
        "search_mode": "Hybrid (Dense Vector + BM25 Sparse + RRF + Cross-Encoder Reranker)"
    }


@app.post("/api/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Uploads and indexes PDF or text clinical documents into Hybrid RAG store."""
    ingested_details = []
    total_new_chunks = 0

    for file in files:
        contents = await file.read()
        res = hybrid_engine.ingest_document(contents, filename=file.filename)
        total_new_chunks += res.get("chunks_count", 0)
        ingested_details.append({
            "filename": file.filename,
            "chunks_count": res.get("chunks_count", 0),
            "status": res.get("status", "success")
        })

    return {
        "message": f"Successfully indexed {len(files)} document(s)",
        "total_new_chunks": total_new_chunks,
        "active_total_chunks": len(hybrid_engine.indexed_chunks),
        "files": ingested_details
    }


@app.post("/api/query")
async def process_query(req: QueryRequest):
    """Runs the complete LangGraph Agentic RAG pipeline for a clinical question."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()
    try:
        result = agent.run(req.query, patient_context=req.patient_context)
        latency = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "latency_seconds": latency,
            "query": req.query,
            "final_response": result.get("final_response", ""),
            "draft_answer": result.get("draft_answer", ""),
            "citations": result.get("citations", []),
            "trace": result.get("trace", []),
            "grounding_eval": result.get("grounding_eval", {}),
            "safety_eval": result.get("safety_eval", {}),
            "tool_outputs": result.get("tool_outputs", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
