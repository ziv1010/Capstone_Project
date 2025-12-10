/**
 * Task Status Logic
 * Handles task progress tracking and pipeline visualization
 */

let tasks = [];
let progressWebSocket = null;

// Stage configuration
const STAGE_CONFIG = [
    { name: 'stage3', label: 'Planning', icon: '📋' },
    { name: 'stage3b', label: 'Data Prep', icon: '🔧' },
    { name: 'stage3_5a', label: 'Methods', icon: '🧪' },
    { name: 'stage3_5b', label: 'Benchmark', icon: '📊' },
    { name: 'stage4', label: 'Execute', icon: '⚡' },
    { name: 'stage5', label: 'Visualize', icon: '📈' },
];

/**
 * Initialize the status page
 */
async function init() {
    await refreshStatus();
    connectTaskProgressStream();
}

/**
 * Refresh task status
 */
async function refreshStatus() {
    try {
        const data = await APIClient.getAllTasksStatus();
        tasks = data.tasks || [];

        if (tasks.length === 0) {
            displayEmptyState();
        } else {
            displayTasksList();
            // Use first task for pipeline visualization
            displayPipelineForTask(tasks[0].task_id);
        }

        updateTaskCount();

    } catch (error) {
        console.error('Error loading task status:', error);
        showError('Failed to load task status');
    }
}

/**
 * Display empty state when no tasks
 */
function displayEmptyState() {
    document.getElementById('pipelineVisual').innerHTML = `
        <div class="text-center text-muted" style="width: 100%; padding: 2rem;">
            <p>No tasks running. Start a task from the conversation interface.</p>
        </div>
    `;

    document.getElementById('tasksList').innerHTML = `
        <p class="text-muted text-center">No active tasks</p>
    `;
}

/**
 * Display pipeline visualization for a task
 */
async function displayPipelineForTask(taskId) {
    try {
        const statusData = await APIClient.getTaskStatus(taskId);
        const stages = statusData.stages || {};

        const pipelineHTML = STAGE_CONFIG.map((stage, index) => {
            const status = stages[stage.name] || 'pending';
            const connectorClass = status === 'completed' ? 'active' : '';

            return `
                <div class="stage-node">
                    <div class="stage-circle ${status}">
                        <span>${stage.icon}</span>
                    </div>
                    <div class="stage-label">${stage.label}</div>
                </div>
                ${index < STAGE_CONFIG.length - 1 ? `<div class="stage-connector ${connectorClass}"></div>` : ''}
            `;
        }).join('');

        document.getElementById('pipelineVisual').innerHTML = pipelineHTML;

    } catch (error) {
        console.error('Error loading pipeline status:', error);
    }
}

/**
 * Display tasks list
 */
function displayTasksList() {
    const tasksList = document.getElementById('tasksList');

    tasksList.innerHTML = tasks.map(task => {
        const metrics = task.metrics || {};
        const status = task.status || 'unknown';
        const badgeClass = status === 'completed' ? 'badge-success' : 'badge-running';

        return `
            <div class="task-card">
                <div class="flex justify-between items-center mb-2">
                    <h4>${task.task_id}</h4>
                    <span class="badge ${badgeClass}">${status}</span>
                </div>
                
                <div class="text-muted mb-2">
                    Current Stage: <strong>${task.last_stage || 'N/A'}</strong>
                </div>
                
                ${Object.keys(metrics).length > 0 ? `
                    <div class="metrics-grid">
                        ${Object.entries(metrics).map(([key, value]) => `
                            <div class="metric-box">
                                <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                                <div class="metric-label">${key}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="flex gap-2 mt-3">
                    <button class="btn btn-secondary btn-sm" onclick="viewTaskDetails('${task.task_id}')">
                        View Details
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="displayPipelineForTask('${task.task_id}')">
                        Show Pipeline
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * View task details
 */
function viewTaskDetails(taskId) {
    window.location.href = `/static/task_details.html?task=${taskId}`;
}

/**
 * Update task count badge
 */
function updateTaskCount() {
    const badge = document.getElementById('taskCountBadge');
    badge.textContent = `${tasks.length} ${tasks.length === 1 ? 'task' : 'tasks'}`;
}

/**
 * Connect to task progress WebSocket
 */
function connectTaskProgressStream() {
    try {
        progressWebSocket = APIClient.connectTaskProgressWebSocket(
            handleProgressUpdate,
            handleProgressError
        );

        progressWebSocket.onclose = () => {
            // Reconnect after 5 seconds
            setTimeout(connectTaskProgressStream, 5000);
        };

    } catch (error) {
        console.error('Error connecting to progress stream:', error);
    }
}

/**
 * Handle progress update message
 */
function handleProgressUpdate(data) {
    if (data.type === 'task_update' && data.data) {
        tasks = data.data.tasks || [];

        if (tasks.length > 0) {
            displayTasksList();
            updateTaskCount();
        }
    }
}

/**
 * Handle progress error
 */
function handleProgressError(error) {
    console.error('Progress stream error:', error);
}

/**
 * Show error message
 */
function showError(message) {
    const pipelineDiv = document.getElementById('pipelineVisual');
    pipelineDiv.innerHTML = `
        <div class="text-center" style="width: 100%; color: var(--error);">
            <p>⚠️ ${escapeHtml(message)}</p>
        </div>
    `;
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', init);
