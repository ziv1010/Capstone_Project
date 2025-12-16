/**
 * Guardrails Page JavaScript
 * Handles loading and displaying model validation results with percentage scores
 */

let selectedTaskId = null;

// Utility: Format timestamp
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (e) {
        return timestamp;
    }
}

/**
 * Initialize the guardrails page
 */
document.addEventListener('DOMContentLoaded', () => {
    loadGuardrails();
    loadAvailableTasks();
});

/**
 * Load existing guardrails reports
 */
async function loadGuardrails() {
    const container = document.getElementById('guardrailsList');

    try {
        const response = await APIClient.get('/api/guardrails');
        const guardrails = response.guardrails || [];

        if (guardrails.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <div class="empty-state-icon">🛡️</div>
                    <h3>No Guardrails Reports Yet</h3>
                    <p>Run guardrails validation on a task below to see results</p>
                </div>
            `;
            return;
        }

        container.innerHTML = guardrails.map(g => {
            const score = g.validity_score || 0;
            const label = g.validity_label || (score >= 75 ? 'HIGH' : score >= 50 ? 'MEDIUM' : 'LOW');
            const colorClass = score >= 75 ? 'validity-high' : score >= 50 ? 'validity-medium' : 'validity-low';

            return `
                <div class="task-card" onclick="loadReport('${g.task_id}')">
                    <div class="task-card-header">
                        <div>
                            <span class="task-id">${g.task_id}</span>
                            <div class="task-meta">Validated: ${formatTimestamp(g.generated_at)}</div>
                        </div>
                        <div class="validity-score ${colorClass}" style="--score: ${score * 3.6}deg;">
                            <span class="value">${Math.round(score)}</span>
                            <span class="label">${label}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error loading guardrails:', error);
        container.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;">Error loading guardrails: ${error.message}</div>`;
    }
}

/**
 * Load available tasks that can have guardrails run
 */
async function loadAvailableTasks() {
    const container = document.getElementById('availableTasksList');

    try {
        const response = await APIClient.get('/api/tasks');
        const tasks = response.tasks || [];

        if (tasks.length === 0) {
            container.innerHTML = `<p style="grid-column: 1 / -1;">No tasks found. Complete a pipeline run first.</p>`;
            return;
        }

        container.innerHTML = tasks.map(task => `
            <div class="available-task-item">
                <div>
                    <strong>${task.id}</strong>
                    <div class="task-meta">${task.title || 'Untitled task'}</div>
                </div>
                <button class="run-btn" onclick="runGuardrails('${task.id}')" id="run-${task.id}">
                    🛡️ Run
                </button>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading tasks:', error);
        container.innerHTML = `<p style="grid-column: 1 / -1;">Error loading tasks: ${error.message}</p>`;
    }
}

/**
 * Run guardrails validation for a task
 */
async function runGuardrails(taskId) {
    const btn = document.getElementById(`run-${taskId}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Running...';
    }

    try {
        const response = await fetch(`/api/guardrails/${taskId}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();

        if (result.status === 'started') {
            pollForCompletion(taskId, btn);
        }

    } catch (error) {
        console.error('Error running guardrails:', error);
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '❌ Failed';
        }
    }
}

/**
 * Poll for guardrails completion
 */
async function pollForCompletion(taskId, btn, attempts = 0) {
    const maxAttempts = 60;

    if (attempts >= maxAttempts) {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '⚠️ Timeout';
        }
        return;
    }

    try {
        const response = await APIClient.get(`/api/guardrails/${taskId}`);

        if (response.status === 'found') {
            loadGuardrails();
            loadReport(taskId);
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '✅ Done';
            }
            return;
        }

        setTimeout(() => pollForCompletion(taskId, btn, attempts + 1), 5000);

    } catch (error) {
        setTimeout(() => pollForCompletion(taskId, btn, attempts + 1), 5000);
    }
}

/**
 * Load and display a guardrails report
 */
async function loadReport(taskId) {
    selectedTaskId = taskId;
    const panel = document.getElementById('reportPanel');

    try {
        const response = await APIClient.get(`/api/guardrails/${taskId}`);

        if (response.status !== 'found') {
            panel.style.display = 'none';
            return;
        }

        const report = response.report;
        panel.style.display = 'block';

        // Update header
        document.getElementById('reportTaskId').textContent = `Guardrails Report: ${report.task_id}`;
        document.getElementById('reportDate').textContent = `Generated: ${formatTimestamp(report.generated_at)}`;

        // Update big score
        const score = report.validity_score || 0;
        const label = report.validity_label || 'N/A';
        const bigScore = document.getElementById('bigScore');
        const validityLabel = document.getElementById('validityLabel');

        bigScore.className = `big-score ${score >= 75 ? 'validity-high' : score >= 50 ? 'validity-medium' : 'validity-low'}`;
        bigScore.style.background = `conic-gradient(${report.validity_color || '#10b981'} ${score * 3.6}deg, var(--bg-tertiary) 0)`;
        bigScore.querySelector('.value').textContent = Math.round(score);
        validityLabel.textContent = label;
        validityLabel.style.color = report.validity_color || '#10b981';

        // Render test cards
        renderTestCards(report.tests);

        // Render visualizations
        renderVisualizations(report.visualizations, taskId);

        // Render assessment
        document.getElementById('assessmentText').textContent = report.overall_assessment || 'No assessment provided.';

        // Scroll to report
        panel.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error loading report:', error);
    }
}

/**
 * Render test cards
 */
function renderTestCards(tests) {
    const container = document.getElementById('testsGrid');

    const testNames = {
        'correlation_analysis': '📊 Correlation Analysis',
        'propensity_score_analysis': '⚖️ Propensity Score',
        'inverse_propensity_weighting': '🔄 IPW Validation',
        'residual_analysis': '📉 Residual Analysis'
    };

    container.innerHTML = Object.entries(tests).map(([name, test]) => {
        const details = test.details || {};
        const status = (test.status || 'UNKNOWN').toUpperCase();
        const statusClass = status === 'PASS' ? 'pass' : status === 'FAIL' ? 'fail' : 'warning';
        const score = test.score !== undefined ? test.score : (status === 'PASS' ? 25 : status === 'WARNING' ? 12.5 : 0);
        const reason = details.reason || details.interpretation || 'No explanation available';

        return `
            <div class="test-card ${statusClass}">
                <div class="test-card-header">
                    <span class="test-name">${testNames[name] || name}</span>
                    <span class="test-score-badge ${statusClass}">${score}/25</span>
                </div>
                <div class="test-reason">${reason}</div>
                <div class="test-details-toggle" onclick="toggleDetails('${name}')">
                    ▼ Show Details
                </div>
                <div id="details-${name}" class="test-details" style="display: none;">
                    <pre>${JSON.stringify(details, null, 2)}</pre>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Toggle test details visibility
 */
function toggleDetails(name) {
    const details = document.getElementById(`details-${name}`);
    if (details) {
        const isHidden = details.style.display === 'none';
        details.style.display = isHidden ? 'block' : 'none';
    }
}

/**
 * Render visualizations
 */
function renderVisualizations(visualizations, taskId) {
    const container = document.getElementById('vizGallery');

    if (!visualizations || visualizations.length === 0) {
        container.innerHTML = '<p class="task-meta">No visualizations available</p>';
        return;
    }

    container.innerHTML = visualizations.map(viz => `
        <div class="viz-card">
            <img src="${viz.api_url || `/api/guardrails/${taskId}/image/${viz.filename}`}" 
                 alt="${viz.filename}"
                 onerror="this.parentElement.style.display='none'">
            <div class="viz-caption">${formatVizName(viz.filename)}</div>
        </div>
    `).join('');
}

/**
 * Format visualization filename for display
 */
function formatVizName(filename) {
    return filename
        .replace(/_/g, ' ')
        .replace('.png', '')
        .replace('guardrails ', '')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
