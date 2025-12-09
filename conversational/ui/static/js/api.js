/**
 * API Client Module
 * Centralized API calls for the conversational AI pipeline UI
 */

const API_BASE = '';  // Same origin

class APIClient {
    /**
     * Make a GET request
     */
    static async get(endpoint) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API GET Error:', error);
            throw error;
        }
    }

    /**
     * Make a POST request
     */
    static async post(endpoint, data) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API POST Error:', error);
            throw error;
        }
    }

    // ========================================================================
    // CONVERSATION ENDPOINTS
    // ========================================================================

    static async getConversations() {
        return await this.get('/api/conversations');
    }

    static async getConversation(sessionId) {
        return await this.get(`/api/conversations/${sessionId}`);
    }

    static async getLatestConversation() {
        return await this.get('/api/conversations/latest');
    }

    // ========================================================================
    // LOGS ENDPOINTS
    // ========================================================================

    static async getRecentLogs(lines = 100) {
        return await this.get(`/api/logs/recent?lines=${lines}`);
    }

    static connectLogsWebSocket(onMessage, onError) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) onError(error);
        };

        return ws;
    }

    // ========================================================================
    // STAGE OUTPUTS ENDPOINTS
    // ========================================================================

    static async getStages() {
        return await this.get('/api/stages');
    }

    static async getStageOutputs(stageName) {
        return await this.get(`/api/stages/${stageName}/outputs`);
    }

    static async getFile(filePath) {
        return await this.get(`/api/files/${filePath}`);
    }

    // ========================================================================
    // TASK STATUS ENDPOINTS
    // ========================================================================

    static async getAllTasksStatus() {
        return await this.get('/api/tasks/status');
    }

    static async getTaskStatus(taskId) {
        return await this.get(`/api/tasks/${taskId}/status`);
    }

    static connectTaskProgressWebSocket(onMessage, onError) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/task-progress`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) onError(error);
        };

        return ws;
    }
}

// ========================================================================
// UTILITY FUNCTIONS
// ========================================================================

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Pretty print JSON
 */
function prettyPrintJSON(obj) {
    return JSON.stringify(obj, null, 2);
}
