/**
 * Logs Viewer Logic
 * Handles real-time log streaming and filtering
 */

let logsWebSocket = null;
let logEntries = [];
let currentFilter = 'ALL';

/**
 * Initialize the logs page
 */
function init() {
    connectToLogStream();
}

/**
 * Connect to log WebSocket stream
 */
function connectToLogStream() {
    try {
        logsWebSocket = APIClient.connectLogsWebSocket(
            handleLogMessage,
            handleLogError
        );

        logsWebSocket.onopen = () => {
            updateStatusBadge('Live', 'badge-running');
        };

        logsWebSocket.onclose = () => {
            updateStatusBadge('Disconnected', 'badge-error');
            // Try to reconnect after 5 seconds
            setTimeout(connectToLogStream, 5000);
        };

    } catch (error) {
        console.error('Error connecting to log stream:', error);
        updateStatusBadge('Error', 'badge-error');
    }
}

/**
 * Handle incoming log message
 */
function handleLogMessage(data) {
    if (data.type === 'log') {
        const logEntry = {
            level: data.level || 'INFO',
            timestamp: data.timestamp,
            message: data.message
        };

        addLogEntry(logEntry);
    }
}

/**
 * Handle log error
 */
function handleLogError(error) {
    console.error('Log stream error:', error);
    updateStatusBadge('Error', 'badge-error');
}

/**
 * Add a log entry to the display
 */
function addLogEntry(entry) {
    logEntries.push(entry);

    // Keep only last 1000 entries
    if (logEntries.length > 1000) {
        logEntries.shift();
    }

    // Only display if it matches current filter
    if (currentFilter === 'ALL' || entry.level === currentFilter) {
        displayLogEntry(entry);
    }
}

/**
 * Display a single log entry
 */
function displayLogEntry(entry) {
    const container = document.getElementById('logContainer');

    // Remove loading message if present
    if (container.querySelector('.spinner')) {
        container.innerHTML = '';
    }

    const logDiv = document.createElement('div');
    logDiv.className = `log-entry ${entry.level}`;

    const timestamp = new Date(entry.timestamp).toLocaleTimeString();

    logDiv.innerHTML = `
        <span class="log-timestamp">${timestamp}</span>
        <span class="log-level ${entry.level}">${entry.level}</span>
        <span class="log-message">${escapeHtml(entry.message)}</span>
    `;

    container.appendChild(logDiv);

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
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

    // Re-render logs
    const container = document.getElementById('logContainer');
    container.innerHTML = '';

    logEntries.forEach(entry => {
        if (level === 'ALL' || entry.level === level) {
            displayLogEntry(entry);
        }
    });
}

/**
 * Clear all logs
 */
function clearLogs() {
    logEntries = [];
    document.getElementById('logContainer').innerHTML = `
        <div class="text-center text-muted">
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

// Initialize when page loads
window.addEventListener('DOMContentLoaded', init);
