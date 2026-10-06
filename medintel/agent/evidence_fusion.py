from typing import List, Dict, Any

class EvidenceFusionEngine:
    """
    Fuses multiple heterogeneous evidence streams:
    1. RAG-retrieved chunks (Dense + BM25 + Cross-Encoder Reranked)
    2. OpenFDA Drug Interaction & Boxed Warning results
    3. PubMed Research Abstracts
    4. Clinical Calculations (eGFR, BMI, etc.)
    5. Lab Interpretations
    Assigns sequential citation keys [1], [2], ... for direct trace attribution.
    """

    @staticmethod
    def fuse_evidence(
        retrieved_docs: List[Dict[str, Any]],
        tool_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Consolidates evidence into a clean citation index and unified text bundle."""
        citations = []
        fusion_summary_items = []
        citation_idx = 1

        # 1. Process RAG retrieved document chunks
        for doc in retrieved_docs:
            src = doc.get("source", "Uploaded Document")
            pg = doc.get("page_number", 1)
            sec = doc.get("section", "General")
            text_snippet = doc.get("text", "").strip()
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))

            citation_entry = {
                "id": f"[{citation_idx}]",
                "type": "Uploaded Document / Guideline",
                "source": f"{src} (Page {pg}, {sec})",
                "score": score,
                "text": text_snippet[:350] + ("..." if len(text_snippet) > 350 else ""),
                "full_text": text_snippet
            }
            citations.append(citation_entry)
            fusion_summary_items.append(f"Source [{citation_idx}] ({src}, Pg {pg}):\n{text_snippet}\n")
            citation_idx += 1

        # 2. Process Live Tool Outputs
        for tool_item in tool_outputs:
            tool_name = tool_item.get("tool_name", "Tool")
            payload = tool_item.get("data", {})

            if tool_name == "openfda_tool":
                drug_pair = payload.get("drug_pair")
                if drug_pair:
                    alerts = "; ".join(payload.get("alerts", []))
                    content = f"FDA Interaction Check ({drug_pair}): {alerts}"
                else:
                    drug = payload.get("drug_name", "Drug")
                    contra = payload.get("contraindications", "")
                    boxed = payload.get("boxed_warning", "")
                    content = f"FDA Label for {drug}: Boxed Warning: {boxed[:200]} | Contraindications: {contra[:200]}"
                
                citations.append({
                    "id": f"[{citation_idx}]",
                    "type": "OpenFDA Official Label",
                    "source": "US Food & Drug Administration (OpenFDA API)",
                    "score": 1.0,
                    "text": content,
                    "full_text": content
                })
                fusion_summary_items.append(f"Source [{citation_idx}] (OpenFDA):\n{content}\n")
                citation_idx += 1

            elif tool_name == "pubmed_tool":
                articles = payload if isinstance(payload, list) else [payload]
                for art in articles[:2]:
                    art_title = art.get("title", "")
                    journal = art.get("journal", "PubMed")
                    url = art.get("url", "")
                    content = f"PubMed Article: '{art_title}' published in {journal}. Link: {url}"

                    citations.append({
                        "id": f"[{citation_idx}]",
                        "type": "PubMed Literature",
                        "source": f"PubMed ({journal})",
                        "score": 0.9,
                        "text": content,
                        "full_text": content
                    })
                    fusion_summary_items.append(f"Source [{citation_idx}] (PubMed):\n{content}\n")
                    citation_idx += 1

            elif tool_name == "clinical_calculators":
                calc_name = payload.get("calculator", "Clinical Metric")
                val = payload.get("egfr_value_ml_min", payload.get("bmi", payload.get("score", "")))
                stage = payload.get("stage", payload.get("category", payload.get("recommendation", "")))
                content = f"{calc_name} Output: Value={val}, Classification/Implication: {stage}"

                citations.append({
                    "id": f"[{citation_idx}]",
                    "type": "Clinical Formula Calculator",
                    "source": f"Validated Formula ({calc_name})",
                    "score": 1.0,
                    "text": content,
                    "full_text": content
                })
                fusion_summary_items.append(f"Source [{citation_idx}] (Calculator):\n{content}\n")
                citation_idx += 1

            elif tool_name == "lab_interpreter":
                lab_items = payload if isinstance(payload, list) else [payload]
                for l_item in lab_items:
                    t_name = l_item.get("test_name", "Lab")
                    val = l_item.get("value", "")
                    stat = l_item.get("status", "")
                    interp = l_item.get("interpretation", "")
                    content = f"Lab Panel ({t_name}): {val} -> {stat}. {interp}"

                    citations.append({
                        "id": f"[{citation_idx}]",
                        "type": "Laboratory Reference Range",
                        "source": "Standard Adult Lab Reference",
                        "score": 1.0,
                        "text": content,
                        "full_text": content
                    })
                    fusion_summary_items.append(f"Source [{citation_idx}] (Lab Interpreter):\n{content}\n")
                    citation_idx += 1

        fused_text = "\n".join(fusion_summary_items)

        return {
            "citations": citations,
            "fused_evidence_text": fused_text,
            "total_sources_count": len(citations)
        }
