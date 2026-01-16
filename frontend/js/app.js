/**
 * RAG Brain - Frontend Application
 * Handles API communication, UI interactions, and state management
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = window.location.origin;
const API_ENDPOINTS = {
    health: () => `${API_BASE}/health`,
    stats: () => `${API_BASE}/stats`,
    ingestText: () => `${API_BASE}/api/v1/ingest/text`,
    ingestFile: () => `${API_BASE}/api/v1/ingest/file`,
    search: () => `${API_BASE}/api/v1/search`,
    documents: () => `${API_BASE}/api/v1/documents`,
    document: (id) => `${API_BASE}/api/v1/documents/${id}`,
    file: (id) => `${API_BASE}/api/v1/files/${id}`,
    tags: () => `${API_BASE}/api/v1/tags`,
    chat: () => `${API_BASE}/api/v1/chat`,
    chatStream: () => `${API_BASE}/api/v1/chat/stream`,
    conversations: () => `${API_BASE}/api/v1/conversations`,
    conversation: (id) => `${API_BASE}/api/v1/conversations/${id}`,
    cleanup: () => `${API_BASE}/api/v1/cleanup`,
};

// ============================================================================
// State
// ============================================================================

const state = {
    currentPage: 'search',
    searchFilter: 'all',
    conversationId: null,
    isProcessing: false,
    currentDocId: null,
    stats: {
        document_count: 0,
        chunk_count: 0,
        tag_count: 0,
    },
};

// ============================================================================
// API Client
// ============================================================================

async function apiRequest(url, options = {}) {
    const defaults = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    try {
        const response = await fetch(url, { ...defaults, ...options });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || error.message || 'Request failed');
        }
        return response.json().catch(() => ({}));
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function getHealth() {
    return apiRequest(API_ENDPOINTS.health());
}

async function getStats() {
    return apiRequest(API_ENDPOINTS.stats());
}

async function searchDocuments(query, k = 5, threshold = 0.3, contentType = null) {
    return apiRequest(API_ENDPOINTS.search(), {
        method: 'POST',
        body: JSON.stringify({
            query,
            k,
            threshold,
            content_type: contentType,
        }),
    });
}

async function listDocuments(contentType = null, limit = 100) {
    const params = new URLSearchParams();
    if (contentType) params.append('content_type', contentType);
    params.append('limit', limit);
    return apiRequest(`${API_ENDPOINTS.documents()}?${params}`);
}

async function getDocument(docId) {
    return apiRequest(API_ENDPOINTS.document(docId));
}

async function deleteDocument(docId) {
    return apiRequest(API_ENDPOINTS.document(docId), { method: 'DELETE' });
}

async function listTags(limit = 50) {
    return apiRequest(`${API_ENDPOINTS.tags()}?limit=${limit}`);
}

async function ingestText(text, filename = 'text.txt', source = 'web') {
    return apiRequest(API_ENDPOINTS.ingestText(), {
        method: 'POST',
        body: JSON.stringify({ text, filename, source }),
    });
}

async function ingestFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(API_ENDPOINTS.ingestFile(), {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

async function sendChatMessage(query, conversationId = null, useRag = true, k = 5) {
    return apiRequest(API_ENDPOINTS.chat(), {
        method: 'POST',
        body: JSON.stringify({
            query,
            conversation_id: conversationId,
            use_rag: useRag,
            k,
        }),
    });
}

async function createConversation(title = 'New Chat') {
    return apiRequest(API_ENDPOINTS.conversations(), {
        method: 'POST',
        body: JSON.stringify({ title }),
    });
}

// ============================================================================
// UI Functions
// ============================================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            ${type === 'success'
                ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>'
                : type === 'error'
                ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>'
                : '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>'}
        </svg>
        <span>${escapeHtml(message)}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatScore(score) {
    return (score * 100).toFixed(0) + '%';
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateStats() {
    document.getElementById('stat-docs').textContent = state.stats.document_count;
    document.getElementById('stat-chunks').textContent = state.stats.chunk_count;
    document.getElementById('stat-tags').textContent = state.stats.tag_count;
}

function renderTags(tags, maxTags = 5) {
    if (!tags || tags.length === 0) return '';

    const displayTags = tags.slice(0, maxTags);
    const tagsHtml = displayTags.map(tag => {
        const name = typeof tag === 'string' ? tag : tag.name;
        const score = typeof tag === 'object' && tag.score ? ` (${Math.round(tag.score * 100)}%)` : '';
        return `<span class="tag">${escapeHtml(name)}${escapeHtml(score)}</span>`;
    }).join('');

    if (tags.length > maxTags) {
        return tagsHtml + `<span class="tag">+${tags.length - maxTags} more</span>`;
    }
    return tagsHtml;
}

// ============================================================================
// Page Navigation
// ============================================================================

function navigateTo(page) {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `${page}Page`);
    });

    const titles = {
        search: 'Semantic Search',
        browse: 'Browse Documents',
        chat: 'AI Chatbot',
    };
    document.getElementById('pageTitle').textContent = titles[page] || 'RAG Brain';

    state.currentPage = page;

    if (page === 'browse') {
        loadBrowsePage();
    }

    document.getElementById('sidebar').classList.remove('open');
}

// ============================================================================
// Search Page
// ============================================================================

async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const contentType = state.searchFilter === 'all' ? null : state.searchFilter;
        const response = await searchDocuments(query, 10, 0.3, contentType);

        displaySearchResults(response.results, response.total_count);
    } catch (error) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <p>Search failed: ${escapeHtml(error.message)}</p>
            </div>
        `;
    }
}

function displaySearchResults(results, count) {
    const container = document.getElementById('searchResults');
    const header = document.getElementById('resultsHeader');

    if (count === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                </svg>
                <p>No results found. Try a different query or upload more documents.</p>
            </div>
        `;
        header.innerHTML = '<span>No results found</span>';
        return;
    }

    header.innerHTML = `<span>Found ${count} result${count !== 1 ? 's' : ''}</span>`;

    container.innerHTML = results.map(result => `
        <div class="result-card" onclick="openDocDetailModal('${escapeHtml(result.doc_id)}')">
            <div class="result-header">
                <div class="result-title">
                    <span class="result-type ${result.content_type}">${result.content_type}</span>
                    ${escapeHtml(result.filename)}
                </div>
                <span class="result-score">${formatScore(result.score)}</span>
            </div>
            <div class="result-content">${escapeHtml(result.content)}</div>
        </div>
    `).join('');
}

// ============================================================================
// Document Detail Modal
// ============================================================================

async function openDocDetailModal(docId) {
    state.currentDocId = docId;
    const modal = document.getElementById('docDetailModal');
    const body = document.getElementById('docDetailBody');

    modal.classList.add('active');
    body.innerHTML = '<div class="loading-spinner" style="margin: 2rem auto;"></div>';

    try {
        const doc = await getDocument(docId);
        renderDocDetail(doc);
    } catch (error) {
        body.innerHTML = `
            <div class="empty-state">
                <p>Failed to load document: ${escapeHtml(error.message)}</p>
            </div>
        `;
    }
}

function closeDocDetailModal() {
    document.getElementById('docDetailModal').classList.remove('active');
    state.currentDocId = null;
}

function renderDocDetail(doc) {
    const body = document.getElementById('docDetailBody');
    const isImage = doc.content_type === 'image';
    const tags = doc.tags || [];
    const chunks = doc.chunks || [];

    let html = `
        <div class="doc-detail-header">
            <div class="doc-detail-title">
                <h2>${escapeHtml(doc.filename)}</h2>
                <div class="doc-detail-meta">
                    <span>${doc.content_type}</span>
                    <span>${formatDate(doc.created_at)}</span>
                    ${doc.file_size ? `<span>${(doc.file_size / 1024).toFixed(1)} KB</span>` : ''}
                </div>
            </div>
            <div class="doc-detail-actions">
                <button class="doc-action-btn" onclick="downloadDocument('${doc.doc_id}')">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Download
                </button>
                <button class="doc-action-btn" onclick="deleteCurrentDocument()">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                    </svg>
                    Delete
                </button>
            </div>
        </div>
    `;

    // Image preview for images
    if (isImage) {
        html += `
            <img src="${API_ENDPOINTS.file(doc.doc_id)}" alt="${escapeHtml(doc.filename)}" class="doc-image-preview" onclick="window.open('${API_ENDPOINTS.file(doc.doc_id)}', '_blank')">
        `;
        if (doc.caption) {
            html += `<div class="doc-caption">${escapeHtml(doc.caption)}</div>`;
        }
        if (doc.detailed_description) {
            html += `<div class="doc-detail-content">${escapeHtml(doc.detailed_description)}</div>`;
        }
    } else {
        // Text content
        if (doc.raw_text) {
            html += `
                <div class="doc-detail-section">
                    <div class="doc-detail-section-title">
                        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        Content
                    </div>
                    <div class="doc-detail-content">${escapeHtml(doc.raw_text)}</div>
                </div>
            `;
        }
        // Summary
        if (doc.summary && doc.summary !== doc.raw_text) {
            html += `
                <div class="doc-detail-section">
                    <div class="doc-detail-section-title">
                        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        Summary
                    </div>
                    <div class="doc-detail-content">${escapeHtml(doc.summary)}</div>
                </div>
            `;
        }
    }

    // Tags
    if (tags.length > 0) {
        html += `
            <div class="doc-detail-section">
                <div class="doc-detail-section-title">
                    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/>
                    </svg>
                    Tags (${tags.length})
                </div>
                <div class="doc-tags-list">
                    ${tags.map(tag => {
                        const name = typeof tag === 'string' ? tag : tag.name;
                        const score = typeof tag === 'object' && tag.score ? tag.score : 1;
                        return `<span class="doc-tag">${escapeHtml(name)}<span class="doc-tag-score">${Math.round(score * 100)}%</span></span>`;
                    }).join('')}
                </div>
            </div>
        `;
    }

    // Chunks
    if (chunks.length > 0) {
        html += `
            <div class="doc-detail-section">
                <div class="doc-detail-section-title">
                    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                    </svg>
                    Chunks (${chunks.length})
                </div>
                <div>
                    ${chunks.map(chunk => `
                        <div class="doc-chunk">
                            <div class="doc-chunk-header">
                                <span class="doc-chunk-number">Chunk #${chunk.chunk_number + 1}</span>
                                <span style="color: var(--text-muted); font-size: 0.75rem;">${chunk.token_count} tokens</span>
                            </div>
                            <div class="doc-chunk-content">${escapeHtml(chunk.content)}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    body.innerHTML = html;
}

async function downloadDocument(docId) {
    try {
        const response = await fetch(API_ENDPOINTS.file(docId));
        if (!response.ok) throw new Error('Failed to download');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Get filename from document
        const doc = await getDocument(docId);
        a.download = doc.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showToast('Document downloaded', 'success');
    } catch (error) {
        showToast('Failed to download: ' + error.message, 'error');
    }
}

async function deleteCurrentDocument() {
    if (!state.currentDocId) return;
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        await deleteDocument(state.currentDocId);
        showToast('Document deleted', 'success');
        closeDocDetailModal();
        refreshStats();
        if (state.currentPage === 'browse') loadBrowsePage();
    } catch (error) {
        showToast('Failed to delete: ' + error.message, 'error');
    }
}

function viewDocument(docId) {
    openDocDetailModal(docId);
}

// ============================================================================
// Browse Page
// ============================================================================

async function loadBrowsePage() {
    const container = document.getElementById('browseGrid');
    container.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await listDocuments(null, 100);
        displayBrowseGrid(response.documents);
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state">
                <p>Failed to load documents: ${escapeHtml(error.message)}</p>
            </div>
        `;
    }
}

function displayBrowseGrid(documents) {
    const container = document.getElementById('browseGrid');

    if (documents.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                </svg>
                <p>No documents yet. Upload your first document to get started!</p>
            </div>
        `;
        return;
    }

    container.innerHTML = documents.map(doc => {
        const isImage = doc.content_type === 'image';

        return `
            <div class="document-card" onclick="openDocDetailModal('${escapeHtml(doc.doc_id)}')">
                <div class="document-preview">
                    ${isImage
                        ? `<img src="${API_ENDPOINTS.file(doc.doc_id)}" alt="${escapeHtml(doc.filename)}" loading="lazy">`
                        : `<div class="document-icon">📄</div>`
                    }
                </div>
                <div class="document-info">
                    <div class="document-title" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="document-meta">
                        <span>${doc.content_type}</span>
                        <span>${formatDate(doc.created_at)}</span>
                    </div>
                    ${(doc.tags && doc.tags.length > 0) ? `<div style="margin-top: 0.5rem;">${renderTags(doc.tags, 3)}</div>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// Chat Page
// ============================================================================

async function sendChat() {
    const input = document.getElementById('chatInput');
    const query = input.value.trim();

    if (!query || state.isProcessing) return;

    input.value = '';
    state.isProcessing = true;

    addChatMessage('user', query);
    const loadingId = addChatLoading();

    try {
        const response = await sendChatMessage(query, state.conversationId, true, 5);

        removeChatMessage(loadingId);
        addChatMessage('assistant', response.response, response.context_docs, response.mock_mode);

        if (!state.conversationId) {
            state.conversationId = await createNewConversation();
        }

        updateContextPanel(response.context_docs);

        // Show mock mode notification
        if (response.mock_mode) {
            showToast('Running in mock mode - GLM API unavailable', 'info');
        }

    } catch (error) {
        removeChatMessage(loadingId);
        addChatMessage('assistant', `Error: ${error.message}`);
    } finally {
        state.isProcessing = false;
    }
}

async function createNewConversation() {
    const conv = await createConversation('Chat from Web UI');
    return conv.conversation_id;
}

function addChatMessage(role, content, contextDocs = null, mockMode = false) {
    const container = document.getElementById('chatMessages');

    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    const avatar = role === 'user' ? 'U' : 'AI';

    let contextHtml = '';
    if (contextDocs && contextDocs.length > 0) {
        contextHtml = `
            <div class="message-context">
                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                ${contextDocs.length} source${contextDocs.length !== 1 ? 's' : ''} retrieved
            </div>
        `;
    }

    let mockHtml = '';
    if (mockMode) {
        mockHtml = `
            <div class="message-mock">
                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                Mock Mode - GLM API unavailable
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${escapeHtml(content)}
            ${contextHtml}
            ${mockHtml}
        </div>
    `;

    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;

    return messageDiv;
}

function addChatLoading() {
    const container = document.getElementById('chatMessages');
    const loadingId = 'loading-' + Date.now();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = loadingId;

    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;

    return loadingId;
}

function removeChatMessage(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

function updateContextPanel(docIds) {
    const container = document.getElementById('contextDocs');

    if (!docIds || docIds.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No documents retrieved</p>';
        return;
    }

    container.innerHTML = docIds.map((docId, index) => `
        <div class="context-doc" onclick="openDocDetailModal('${escapeHtml(docId)}')" style="cursor: pointer;">
            <div class="context-doc-title">Source ${index + 1}</div>
            <div class="context-doc-score">${escapeHtml(docId.substring(0, 12))}...</div>
        </div>
    `).join('');
}

// ============================================================================
// Upload
// ============================================================================

async function handleFileUpload(files) {
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.style.display = 'block';

    let successCount = 0;
    let errorCount = 0;

    for (const file of files) {
        try {
            await ingestFile(file);
            successCount++;
        } catch (error) {
            console.error('Upload error:', error);
            errorCount++;
        }
    }

    progressDiv.style.display = 'none';
    closeModal();

    if (successCount > 0) {
        showToast(`Successfully uploaded ${successCount} file${successCount !== 1 ? 's' : ''}`, 'success');
        refreshStats();
        if (state.currentPage === 'browse') loadBrowsePage();
    }

    if (errorCount > 0) {
        showToast(`Failed to upload ${errorCount} file${errorCount !== 1 ? 's' : ''}`, 'error');
    }
}

// ============================================================================
// Modal
// ============================================================================

function openModal() {
    document.getElementById('uploadModal').classList.add('active');
}

function closeModal() {
    document.getElementById('uploadModal').classList.remove('active');
}

// ============================================================================
// Initialization & Events
// ============================================================================

async function refreshStats() {
    try {
        const stats = await getStats();
        state.stats = stats;
        updateStats();
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

function initEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => navigateTo(item.dataset.page));
    });

    // Mobile menu
    document.getElementById('menuBtn').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });

    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', () => {
        refreshStats();
        if (state.currentPage === 'browse') loadBrowsePage();
    });

    // Cleanup button
    document.getElementById('cleanupBtn').addEventListener('click', async () => {
        if (!confirm('Clean up orphaned data? This will remove:\n- Tags with no documents\n- Chunks with no documents\n- Rebuild FAISS index\n\nThis cannot be undone.')) {
            return;
        }

        const cleanupBtn = document.getElementById('cleanupBtn');
        cleanupBtn.disabled = true;
        cleanupBtn.innerHTML = '<span class="spinner"></span> Cleaning...';

        try {
            const response = await fetch(API_ENDPOINTS.cleanup(), {
                method: 'POST'
            });
            const data = await response.json();

            showToast(`Cleanup complete: ${data.stats.tags_removed} tags removed, FAISS ${data.stats.faiss_rebuilt ? 'rebuilt' : 'unchanged'}`, 'success');
            refreshStats();
            if (state.currentPage === 'browse') loadBrowsePage();
        } catch (error) {
            showToast(`Cleanup failed: ${error.message}`, 'error');
        } finally {
            cleanupBtn.disabled = false;
            cleanupBtn.innerHTML = `<svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 011-1h2a1 1 0 011 1v3M4 7h16"/>
                        </svg>
                        Cleanup`;
        }
    });

    // Upload button and modal
    document.getElementById('uploadBtn').addEventListener('click', openModal);
    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('uploadModal').addEventListener('click', (e) => {
        if (e.target.id === 'uploadModal') closeModal();
    });

    // Upload zone
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files);
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });

    // Search
    const searchInput = document.getElementById('searchInput');
    let searchTimeout;
    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(performSearch, 500);
    });
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(searchTimeout);
            performSearch();
        }
    });

    // Search filters
    document.querySelectorAll('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            state.searchFilter = chip.dataset.filter;
            if (searchInput.value.trim()) performSearch();
        });
    });

    // Chat
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChat();
        }
    });

    document.getElementById('chatSendBtn').addEventListener('click', sendChat);

    // Close doc detail modal when clicking overlay
    document.getElementById('docDetailModal').addEventListener('click', (e) => {
        if (e.target.id === 'docDetailModal') closeDocDetailModal();
    });
}

// ============================================================================
// Bootstrap
// ============================================================================

async function init() {
    initEventListeners();
    await refreshStats();

    try {
        await getHealth();
        console.log('RAG Brain is ready');
    } catch (error) {
        showToast('Failed to connect to server', 'error');
    }
}

document.addEventListener('DOMContentLoaded', init);
