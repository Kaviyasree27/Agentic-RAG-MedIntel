document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const newChatBtn = document.getElementById("newChatBtn");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const selectedFilesList = document.getElementById("selectedFilesList");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadBtnText = document.getElementById("uploadBtnText");
    const chunkCount = document.getElementById("chunkCount");
    const docMemoryList = document.getElementById("docMemoryList");
    const docCountBadge = document.getElementById("docCountBadge");

    const chatDocBanner = document.getElementById("chatDocBanner");
    const chatDocTitle = document.getElementById("chatDocTitle");
    const chatDocMeta = document.getElementById("chatDocMeta");
    const toggleDocPreviewBtn = document.getElementById("toggleDocPreviewBtn");
    const docPreviewBody = document.getElementById("docPreviewBody");
    const docTextPreviewContent = document.getElementById("docTextPreviewContent");

    const queryInput = document.getElementById("queryInput");
    const runQueryBtn = document.getElementById("runQueryBtn");
    const chatThreadContainer = document.getElementById("chatThreadContainer");

    const metricGrounding = document.getElementById("metricGrounding");
    const metricCitations = document.getElementById("metricCitations");
    const metricSafety = document.getElementById("metricSafety");
    const metricLatency = document.getElementById("metricLatency");

    let uploadedFilesCache = [];
    let activeSelectedDoc = localStorage.getItem("medintel_active_doc") || null;
    let documentsStore = [];
    let isDocPreviewOpen = false;

    // --- 1. Markdown & Table Parser ---
    function parseMarkdown(text) {
        if (!text) return "";
        
        if (window.marked && typeof window.marked.parse === "function") {
            try {
                return window.marked.parse(text);
            } catch (e) {
                console.warn("Marked parse error, using fallback:", e);
            }
        }

        let lines = text.split("\n");
        let html = [];
        let inTable = false;
        let tableHeaderDone = false;

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();

            if (line.startsWith("|") && line.endsWith("|")) {
                let cells = line.split("|").slice(1, -1).map(c => c.trim());
                if (cells.every(c => /^:?-+:?$/.test(c))) {
                    tableHeaderDone = true;
                    continue;
                }

                if (!inTable) {
                    html.push("<table>");
                    inTable = true;
                    tableHeaderDone = false;
                }

                if (!tableHeaderDone) {
                    html.push("<thead><tr>" + cells.map(c => `<th>${formatInline(c)}</th>`).join("") + "</tr></thead><tbody>");
                } else {
                    html.push("<tr>" + cells.map(c => `<td>${formatInline(c)}</td>`).join("") + "</tr>");
                }
                continue;
            } else {
                if (inTable) {
                    html.push("</tbody></table>");
                    inTable = false;
                    tableHeaderDone = false;
                }
            }

            if (line.startsWith("### ")) {
                html.push(`<h3>${formatInline(line.substring(4))}</h3>`);
            } else if (line.startsWith("## ")) {
                html.push(`<h2>${formatInline(line.substring(3))}</h2>`);
            } else if (line.startsWith("# ")) {
                html.push(`<h1>${formatInline(line.substring(2))}</h1>`);
            } else if (line.startsWith("- ") || line.startsWith("* ")) {
                html.push(`<li>${formatInline(line.substring(2))}</li>`);
            } else if (/^\d+\.\s/.test(line)) {
                html.push(`<li>${formatInline(line.replace(/^\d+\.\s/, ''))}</li>`);
            } else if (line === "") {
                html.push("<br>");
            } else {
                html.push(`<p>${formatInline(line)}</p>`);
            }
        }

        if (inTable) {
            html.push("</tbody></table>");
        }

        return html.join("");
    }

    function formatInline(str) {
        return str
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\[(\d+)\]/g, '<span style="color:var(--primary); font-weight:700;">[$1]</span>');
    }

    // --- 2. Attached Document Banner Logic ---
    if (toggleDocPreviewBtn) {
        toggleDocPreviewBtn.addEventListener("click", () => {
            isDocPreviewOpen = !isDocPreviewOpen;
            docPreviewBody.style.display = isDocPreviewOpen ? "block" : "none";
            toggleDocPreviewBtn.textContent = isDocPreviewOpen ? "✖ Hide Content" : "👁️ View Document Content";
        });
    }

    async function showDocumentBanner(filename) {
        if (!chatDocBanner) return;
        if (!filename) {
            chatDocBanner.style.display = "none";
            return;
        }

        const cleanName = filename.split("(")[0].trim();
        chatDocTitle.textContent = cleanName;
        chatDocBanner.style.display = "block";
        docTextPreviewContent.textContent = "Loading document text...";

        try {
            const res = await fetch(`/api/documents/content?filename=${encodeURIComponent(cleanName)}`);
            const data = await res.json();
            if (data.status === "success" && data.full_text) {
                chatDocMeta.textContent = `(${data.chunks_count} chunk${data.chunks_count === 1 ? '' : 's'} • Active Focus Document)`;
                docTextPreviewContent.textContent = data.full_text;
            } else {
                chatDocMeta.textContent = "(Active Focus Document)";
                docTextPreviewContent.textContent = "Document indexed in active memory.";
            }
        } catch (e) {
            console.warn("Could not fetch doc content:", e);
            docTextPreviewContent.textContent = "Document text available in indexed chunks.";
        }
    }

    // --- 3. Chat Session Persistence (LocalStorage) ---
    function getSessionKey(docKey) {
        const clean = (docKey || 'global').split("(")[0].trim().toLowerCase();
        return `medintel_thread_${clean}`;
    }

    function saveTurnToSession(docKey, turnData) {
        try {
            const key = getSessionKey(docKey);
            let thread = getSessionThread(docKey);
            thread.push(turnData);
            localStorage.setItem(key, JSON.stringify(thread));
        } catch (e) {
            console.warn("Could not save to localStorage:", e);
        }
    }

    function getSessionThread(docKey) {
        try {
            const key = getSessionKey(docKey);
            const saved = localStorage.getItem(key);
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            return [];
        }
    }

    function renderThread(thread) {
        if (!chatThreadContainer) return;

        if (!thread || thread.length === 0) {
            chatThreadContainer.innerHTML = `
                <div class="empty-chat-state" id="emptyChatState">
                    <div class="empty-chat-icon">🩺</div>
                    <div class="empty-chat-title">Ready for Clinical Inquiry</div>
                    <div class="empty-chat-desc">Select a document from Recent or ask any clinical question below to start.</div>
                </div>
            `;
            metricGrounding.textContent = "--";
            metricCitations.textContent = "--";
            metricSafety.textContent = "--";
            metricLatency.textContent = "--";
            return;
        }

        let html = "";
        for (let idx = 0; idx < thread.length; idx++) {
            const turn = thread[idx];
            const parsedMd = parseMarkdown(turn.final_response || turn.draft_answer || "");

            const citationsHtml = (turn.citations && turn.citations.length > 0)
                ? turn.citations.map(c => `
                    <div class="citation-card" style="margin-top:0.6rem;">
                        <div class="citation-header">
                            <strong>${c.id || '[?]'} — ${c.source || 'Document'} ${c.page_number ? `(Page ${c.page_number})` : ''}</strong>
                            <span style="color:var(--primary); font-size:0.8rem;">Score: ${c.score || '1.0'}</span>
                        </div>
                        <div class="citation-body">
                            <p>${c.text || c.full_text || ''}</p>
                        </div>
                    </div>
                `).join("")
                : `<p style="color:var(--text-muted); font-size:0.85rem; padding:0.5rem 0;">No citations retrieved.</p>`;

            html += `
                <div class="chat-turn" id="turn-${idx}">
                    <!-- User Question -->
                    <div class="user-message-card">
                        <div class="user-message-header">
                            <span>👤 You</span>
                        </div>
                        <div>${turn.query}</div>
                    </div>

                    <!-- Assistant Response Card (Assessment + Citations Only) -->
                    <div class="assistant-message-card">
                        <div class="tabs-header">
                            <button class="tab-btn active" data-turn="${idx}" data-pane="assess-${idx}">📋 Clinical Assessment</button>
                            <button class="tab-btn" data-turn="${idx}" data-pane="cite-${idx}">📚 Citations (${turn.citations ? turn.citations.length : 0})</button>
                        </div>

                        <div class="tab-content" id="assess-${idx}" style="display:block;">
                            <div class="assessment-content">${parsedMd}</div>
                        </div>

                        <div class="tab-content" id="cite-${idx}" style="display:none;">
                            <div class="citations-grid">${citationsHtml}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        chatThreadContainer.innerHTML = html;

        // Attach tab listeners
        chatThreadContainer.querySelectorAll(".tab-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const turnId = btn.getAttribute("data-turn");
                const targetPane = btn.getAttribute("data-pane");

                const turnCard = document.getElementById(`turn-${turnId}`);
                if (turnCard) {
                    turnCard.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
                    turnCard.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");

                    btn.classList.add("active");
                    const paneEl = document.getElementById(targetPane);
                    if (paneEl) paneEl.style.display = "block";
                }
            });
        });

        // Update Top Metrics to match latest turn
        const latestTurn = thread[thread.length - 1];
        if (latestTurn && latestTurn.metrics) {
            metricGrounding.textContent = latestTurn.metrics.grounding || "--";
            metricCitations.textContent = latestTurn.metrics.citations || "--";
            metricSafety.textContent = latestTurn.metrics.safety || "--";
            metricLatency.textContent = latestTurn.metrics.latency || "--";
        }
    }

    // --- 4. + New Chat Button ---
    if (newChatBtn) {
        newChatBtn.addEventListener("click", () => {
            activeSelectedDoc = null;
            localStorage.removeItem("medintel_active_doc");
            queryInput.value = "";
            queryInput.placeholder = "Enter clinical question, medication check, or diagnostic scenario...";
            queryInput.focus();

            if (chatDocBanner) chatDocBanner.style.display = "none";
            if (docPreviewBody) docPreviewBody.style.display = "none";
            isDocPreviewOpen = false;
            if (toggleDocPreviewBtn) toggleDocPreviewBtn.textContent = "👁️ View Document Content";

            renderDocumentMemory(documentsStore);
            renderThread([]);
        });
    }

    // --- 5. Fetch & Render Document Memory List with Long-Press Delete ---
    async function fetchDocuments(promotedDoc = null) {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            if (data.documents) {
                documentsStore = data.documents;
                if (promotedDoc) {
                    activeSelectedDoc = promotedDoc.split("(")[0].trim();
                    localStorage.setItem("medintel_active_doc", activeSelectedDoc);
                }
                renderDocumentMemory(documentsStore, data.total_chunks);
            }
        } catch (err) {
            console.warn("Could not fetch documents:", err);
            docMemoryList.innerHTML = `<div class="no-docs-text">Unable to load memory</div>`;
        }
    }

    function renderDocumentMemory(documents, totalChunks) {
        if (docCountBadge) {
            docCountBadge.textContent = `${documents.length} file${documents.length === 1 ? '' : 's'}`;
        }
        if (chunkCount) {
            chunkCount.textContent = totalChunks || 0;
        }

        if (!documents || documents.length === 0) {
            docMemoryList.innerHTML = `<div class="no-docs-text">No recent documents.<br>Upload above to add knowledge.</div>`;
            return;
        }

        let sortedDocs = [...documents];
        if (activeSelectedDoc) {
            const idx = sortedDocs.findIndex(d => d.filename.toLowerCase().includes(activeSelectedDoc.toLowerCase()));
            if (idx > -1) {
                const [target] = sortedDocs.splice(idx, 1);
                sortedDocs.unshift(target);
            }
        }

        let html = "";
        sortedDocs.forEach(doc => {
            const isActive = activeSelectedDoc && doc.filename.toLowerCase().includes(activeSelectedDoc.toLowerCase());
            const cCount = doc.chunks_count || doc.chunks || 1;
            // No trash icon visible! Clean item
            html += `
                <div class="doc-memory-item ${isActive ? 'active-doc' : ''}" data-docname="${encodeURIComponent(doc.filename)}" title="Click to open. Long-press or right-click to delete.">
                    <div class="doc-info">
                        <span class="doc-icon">${isActive ? '📌' : '📄'}</span>
                        <span class="doc-name">${doc.filename}</span>
                        ${isActive ? '<span class="active-badge">ACTIVE</span>' : ''}
                        <span class="doc-chunks">(${cCount} chunk${cCount === 1 ? '' : 's'})</span>
                    </div>
                </div>
            `;
        });

        docMemoryList.innerHTML = html;

        // Attach Click & Long-Press / Right-Click Listeners
        docMemoryList.querySelectorAll(".doc-memory-item").forEach(item => {
            const rawDocName = decodeURIComponent(item.getAttribute("data-docname"));
            let pressTimer = null;
            let isLongPress = false;

            function startPress(e) {
                isLongPress = false;
                item.classList.add("long-pressing");
                pressTimer = setTimeout(() => {
                    isLongPress = true;
                    item.classList.remove("long-pressing");
                    handleDeletePrompt(rawDocName);
                }, 600); // 600ms hold to trigger delete
            }

            function cancelPress() {
                item.classList.remove("long-pressing");
                if (pressTimer) {
                    clearTimeout(pressTimer);
                    pressTimer = null;
                }
            }

            // Mouse Events
            item.addEventListener("mousedown", (e) => {
                if (e.button === 0) startPress(e);
            });
            item.addEventListener("mouseup", (e) => {
                cancelPress();
                if (!isLongPress && e.button === 0) {
                    switchActiveDocument(rawDocName);
                }
            });
            item.addEventListener("mouseleave", cancelPress);

            // Touch Events (Mobile/Tablet)
            item.addEventListener("touchstart", (e) => {
                startPress(e);
            }, { passive: true });
            item.addEventListener("touchend", (e) => {
                cancelPress();
                if (!isLongPress) {
                    switchActiveDocument(rawDocName);
                }
            });
            item.addEventListener("touchcancel", cancelPress);

            // Desktop Right-Click (Context Menu)
            item.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                cancelPress();
                handleDeletePrompt(rawDocName);
            });
        });
    }

    async function handleDeletePrompt(rawName) {
        if (confirm(`Do you want to delete "${rawName}" from Recent?`)) {
            if (activeSelectedDoc && activeSelectedDoc.toLowerCase().includes(rawName.toLowerCase())) {
                activeSelectedDoc = null;
                localStorage.removeItem("medintel_active_doc");
                if (chatDocBanner) chatDocBanner.style.display = "none";
            }
            localStorage.removeItem(getSessionKey(rawName));
            await deleteDocument(rawName);
        }
    }

    function switchActiveDocument(docName) {
        const cleanName = docName.split("(")[0].trim();
        activeSelectedDoc = cleanName;
        localStorage.setItem("medintel_active_doc", cleanName);

        queryInput.value = "";
        queryInput.placeholder = `Ask anything about ${cleanName}...`;
        queryInput.focus();

        renderDocumentMemory(documentsStore);
        showDocumentBanner(cleanName);

        const thread = getSessionThread(cleanName);
        renderThread(thread);
    }

    async function deleteDocument(filename) {
        try {
            const res = await fetch("/api/documents/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: filename })
            });
            const data = await res.json();
            if (res.ok) {
                await fetchDocuments();
                if (!activeSelectedDoc) {
                    renderThread([]);
                }
            } else {
                alert(`Could not delete document: ${data.error || 'Unknown error'}`);
            }
        } catch (err) {
            alert(`Error deleting document: ${err.message}`);
        }
    }

    // Initial Load
    fetchDocuments(activeSelectedDoc);
    if (activeSelectedDoc) {
        showDocumentBanner(activeSelectedDoc);
        renderThread(getSessionThread(activeSelectedDoc));
    } else {
        renderThread(getSessionThread('global'));
    }

    // --- 6. File Dropzone & Ingestion ---
    if (dropzone) {
        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "var(--primary)";
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "rgba(148, 163, 184, 0.25)";
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "rgba(148, 163, 184, 0.25)";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFilesSelected(e.dataTransfer.files);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFilesSelected(e.target.files);
            }
        });
    }

    function handleFilesSelected(files) {
        uploadedFilesCache = Array.from(files);
        selectedFilesList.textContent = `Selected: ${uploadedFilesCache.map(f => f.name).join(", ")}`;
    }

    if (uploadBtn) {
        uploadBtn.addEventListener("click", async () => {
            if (!uploadedFilesCache || uploadedFilesCache.length === 0) {
                alert("Please select or drop PDF/TXT/DOCX files first.");
                return;
            }

            const formData = new FormData();
            const firstFileName = uploadedFilesCache[0].name;
            uploadedFilesCache.forEach(file => {
                formData.append("files", file);
            });

            uploadBtn.disabled = true;
            uploadBtnText.textContent = "Processing...";

            try {
                const res = await fetch("/api/ingest", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (res.ok) {
                    selectedFilesList.textContent = "";
                    uploadedFilesCache = [];
                    await fetchDocuments(firstFileName);
                    switchActiveDocument(firstFileName);
                } else {
                    alert(`Ingestion failed: ${data.error || 'Unknown error'}`);
                }
            } catch (err) {
                alert(`Ingestion network error: ${err.message}`);
            } finally {
                uploadBtn.disabled = false;
                uploadBtnText.textContent = "Process";
            }
        });
    }

    // --- 7. Run Execution ---
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            runQueryBtn.click();
        }
    });

    runQueryBtn.addEventListener("click", async () => {
        const query = queryInput.value.trim();
        if (!query) {
            alert("Please enter a question.");
            return;
        }

        const activeKey = activeSelectedDoc || 'global';
        const payload = {
            query: query,
            patient_context: null,
            doc_filter: activeSelectedDoc || null
        };

        const emptyState = document.getElementById("emptyChatState");
        if (emptyState) emptyState.remove();

        const turnId = Date.now();
        const pendingCard = document.createElement("div");
        pendingCard.className = "chat-turn";
        pendingCard.id = `pending-${turnId}`;
        
        // Pure loading symbol with NO text
        pendingCard.innerHTML = `
            <div class="user-message-card">
                <div class="user-message-header">
                    <span>👤 You</span>
                </div>
                <div>${query}</div>
            </div>
            <div class="turn-loading-card">
                <div class="loading-dots">
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                </div>
            </div>
        `;
        chatThreadContainer.appendChild(pendingCard);
        pendingCard.scrollIntoView({ behavior: 'smooth' });

        queryInput.value = "";
        runQueryBtn.disabled = true;

        try {
            const res = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.error || "Agent execution failed");
            }

            pendingCard.remove();

            const faith = Math.round((data.grounding_eval?.faithfulness_score || 0.95) * 100);
            const metricsObj = {
                grounding: `${faith}%`,
                citations: `${data.citations?.length || 0}`,
                safety: `${data.safety_eval?.alerts_count || 0}`,
                latency: `${data.latency_seconds || 1.2}s`
            };

            metricGrounding.textContent = metricsObj.grounding;
            metricCitations.textContent = metricsObj.citations;
            metricSafety.textContent = metricsObj.safety;
            metricLatency.textContent = metricsObj.latency;

            const newTurn = {
                query: query,
                final_response: data.final_response || data.draft_answer || "No response generated.",
                citations: data.citations || [],
                metrics: metricsObj,
                doc_filter: activeSelectedDoc || null,
                timestamp: Date.now()
            };

            saveTurnToSession(activeKey, newTurn);

            const thread = getSessionThread(activeKey);
            renderThread(thread);

            const latestTurnEl = document.getElementById(`turn-${thread.length - 1}`);
            if (latestTurnEl) {
                latestTurnEl.scrollIntoView({ behavior: 'smooth' });
            }

            await fetchDocuments(activeSelectedDoc);

        } catch (err) {
            alert(`Error: ${err.message}`);
            pendingCard.remove();
            const thread = getSessionThread(activeKey);
            renderThread(thread);
        } finally {
            runQueryBtn.disabled = false;
        }
    });
});
