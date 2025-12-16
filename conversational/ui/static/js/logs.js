/**
 * Logs Viewer Logic
 * Uses API polling to fetch logs (WebSocket not implemented)
 */

let logEntries = [];
let currentFilter = 'ALL';
let pollingInterval = null;

/**
 * Initialize the logs page
 */
function init() {
    loadLogs();
    // Poll every 3 seconds
    pollingInterval = setInterval(loadLogs, 3000);
}

/**
 * Load logs from API
 */
async function loadLogs() {
    try {
        const response = await APIClient.getRecentLogs(200);

        if (response.logs && response.logs.length > 0) {
            updateStatusBadge('Live', 'badge-running');

            // Parse logs and update display
            const newEntries = response.logs.map(log => parseLogLine(log)).filter(e => e);

            // Only update if we have new logs
            if (JSON.stringify(newEntries) !== JSON.stringify(logEntries)) {
                logEntries = newEntries;
                renderLogs();
            }
        } else if (response.error) {
            updateStatusBadge('Error', 'badge-error');
            showMessage(response.error);
        } else {
            updateStatusBadge('Live', 'badge-running');
            showMessage('No logs available yet. Start a pipeline to see logs.');
        }
    } catch (error) {
        console.error('Error loading logs:', error);
        updateStatusBadge('Disconnected', 'badge-error');
    }
}

/**
 * Parse a log line into structured format
 */
function parseLogLine(line) {
    if (!line || !line.trim()) return null;

    // Try to parse structured log format: "2025-12-16 15:53:37,777 - name - LEVEL - message"
    const match = line.match(/^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),?\d*\s*-\s*(\S+)\s*-\s*(\w+)\s*-\s*(.*)$/);

    if (match) {
        return {
            timestamp: match[1],
            source: match[2],
            level: match[3].toUpperCase(),
            message: match[4]
        };
    }

    // Fallback: simple format
    return {
        timestamp: new Date().toISOString(),
        level: 'INFO',
        message: line
    };
}

/**
 * Render all logs with current filter
 */
function renderLogs() {
    const container = document.getElementById('logContainer');
    container.innerHTML = '';

    const filteredLogs = logEntries.filter(entry =>
        currentFilter === 'ALL' || entry.level === currentFilter
    );

    if (filteredLogs.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted" style="padding: 2rem;">
                <p>No ${currentFilter === 'ALL' ? '' : currentFilter} logs found.</p>
            </div>
        `;
        return;
    }

    filteredLogs.forEach(entry => displayLogEntry(entry));

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * Display a single log entry
 */
function displayLogEntry(entry) {
    const container = document.getElementById('logContainer');
    const logDiv = document.createElement('div');
    logDiv.className = `log-entry ${entry.level}`;

    const time = entry.timestamp ? entry.timestamp.split(' ').pop() || entry.timestamp : '';

    logDiv.innerHTML = `
        <span class="log-timestamp">${time}</span>
        <span class="log-level ${entry.level}">${entry.level}</span>
        <span class="log-message">${escapeHtml(entry.message)}</span>
    `;

    container.appendChild(logDiv);
}

/**
 * Show a message in the log container
 */
function showMessage(msg) {
    const container = document.getElementById('logContainer');
    container.innerHTML = `
        <div class="text-center text-muted" style="padding: 2rem;">
            <p>${msg}</p>
        </div>
    `;
}

/**
 * Filter logs by level
 */
function filterLogs(level) {
    currentFilter = level;

    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.level === level) {
            btn.classList.add('active');
        }
    });

    renderLogs();
}

/**
 * Clear all logs
 */
function clearLogs() {
    logEntries = [];
    document.getElementById('logContainer').innerHTML = `
        <div class="text-center text-muted" style="padding: 2rem;">
            <p>Logs cleared. Waiting for new logs...</p>
        </div>
    `;
}

/**
 * Update status badge
 */
function updateStatusBadge(text, badgeClass) {
    const badge = document.getElementById('statusBadge');
    badge.className = `badge ${badgeClass}`;
    badge.innerHTML = `<span>● ${text}</span>`;
}

/**
 * Helper: escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', init);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
});
