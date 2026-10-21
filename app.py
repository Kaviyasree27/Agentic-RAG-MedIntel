import os
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from medintel.retrieval.hybrid_engine import HybridRetrievalEngine
from medintel.agent.graph import MedIntelAgent

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)
CORS(app)

# Initialize Hybrid RAG Engine & LangGraph Agent
hybrid_engine = HybridRetrievalEngine(collection_name="medintel_flask")
hybrid_engine.reset()
agent = MedIntelAgent(hybrid_engine=hybrid_engine)

# Pre-load sample clinical reference documents
sample_guide = BASE_DIR / "sample_data" / "sample_clinical_guideline.txt"
sample_case = BASE_DIR / "sample_data" / "sample_patient_case.txt"
if sample_guide.exists():
    hybrid_engine.ingest_document(str(sample_guide), filename="KDIGO_ADA_2024_Guideline.txt")
if sample_case.exists():
    hybrid_engine.ingest_document(str(sample_case), filename="Patient_Discharge_Case.txt")


@app.route("/")
def index():
    """Serves the MedIntel modern web interface."""
    return render_template("index.html")


@app.route("/api/documents/content", methods=["GET"])
def get_document_content():
    """Returns the text content of a specific indexed document."""
    filename = request.args.get("filename", "")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    res = hybrid_engine.get_document_content(filename)
    return jsonify(res)


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """Returns all active indexed documents in memory."""
    docs = hybrid_engine.list_documents()
    return jsonify({
        "status": "success",
        "total_documents": len(docs),
        "total_chunks": len(hybrid_engine.indexed_chunks),
        "documents": docs
    })


@app.route("/api/documents/delete", methods=["POST"])
def delete_document():
    """Deletes a specific document from active memory."""
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    res = hybrid_engine.delete_document(filename)
    return jsonify(res)


@app.route("/api/reset", methods=["POST"])
def reset_index():
    """Wipes all documents and resets the knowledge index."""
    hybrid_engine.reset()
    return jsonify({
        "status": "success",
        "message": "Memory index cleared successfully",
        "active_total_chunks": 0
    })


@app.route("/api/status", methods=["GET"])
def system_status():
    """Returns knowledge base chunk count and index status."""
    return jsonify({
        "status": "online",
        "total_chunks": len(hybrid_engine.indexed_chunks),
        "indexed_documents": hybrid_engine.list_documents()
    })


@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    """Uploads and indexes PDF or text clinical documents into Hybrid RAG store."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided in request"}), 400

    uploaded_files = request.files.getlist("files")
    ingested_details = []

    for file in uploaded_files:
        if file.filename == "":
            continue
        file_bytes = file.read()
        res = hybrid_engine.ingest_document(file_bytes, filename=file.filename)
        ingested_details.append(res)

    return jsonify({
        "status": "success",
        "message": f"Successfully indexed {len(uploaded_files)} document(s)",
        "total_new_chunks": sum(d.get("chunks_count", 0) for d in ingested_details),
        "active_total_chunks": len(hybrid_engine.indexed_chunks),
        "files": ingested_details
    })


@app.route("/api/query", methods=["POST"])
def query_agent():
    """Executes the complete LangGraph Agentic RAG pipeline for a clinical question."""
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query string is required"}), 400

    patient_context = data.get("patient_context", {})
    doc_filter = data.get("doc_filter", None)

    print(f"[QUERY HANDLER] Incoming query: '{query}', doc_filter: '{doc_filter}'")

    start_time = time.time()
    try:
        result = agent.run(query, patient_context=patient_context, doc_filter=doc_filter)
        latency = round(time.time() - start_time, 2)

        citations = result.get("citations", [])
        print(f"[QUERY HANDLER] Citations returned: {[c.get('source') for c in citations]}")

        return jsonify({
            "status": "success",
            "latency_seconds": latency,
            "query": query,
            "final_response": result.get("final_response", ""),
            "draft_answer": result.get("draft_answer", ""),
            "citations": citations,
            "trace": result.get("trace", []),
            "grounding_eval": result.get("grounding_eval", {}),
            "safety_eval": result.get("safety_eval", {}),
            "tool_outputs": result.get("tool_outputs", [])
        })
    except Exception as e:
        print(f"[QUERY HANDLER] Error: {e}")
        return jsonify({"error": f"Agent execution error: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("[MedIntel] Flask Application Running on http://127.0.0.1:5000")
    print("=" * 60)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port, debug=False)
