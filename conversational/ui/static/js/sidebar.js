/**
 * Persistent Status Sidebar
 * Auto-updates pipeline stage progress across all pages
 */

class PipelineSidebar {
    constructor() {
        this.stages = [
            { id: 'stage1', name: 'Dataset Summary', icon: '📊' },
            { id: 'stage2', name: 'Data Exploration', icon: '🔍' },
            { id: 'stage3', name: 'Task Planning', icon: '📋' },
            { id: 'stage3b', name: 'Data Preparation', icon: '🔧' },
            { id: 'stage3_5a', name: 'Method Proposal', icon: '🧪' },
            { id: 'stage3_5b', name: 'Benchmarking', icon: '⚖️' },
            { id: 'stage4', name: 'Execution', icon: '⚡' },
            { id: 'stage5', name: 'Visualization', icon: '📈' },
            { id: 'stage6', name: 'Report Generation', icon: '📝' },
            { id: 'stage7', name: 'Guardrails', icon: '🛡️' }
        ];

        this.currentTaskId = null;
        this.stageStates = {};
        this.progressWebSocket = null;
        this.isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

        this.init();
    }

    /**
     * Initialize the sidebar
     */
    init() {
        this.injectSidebarHTML();
        this.attachEventListeners();
        this.connectWebSocket();
        this.loadCurrentTask();

        // Apply collapsed state
        if (this.isCollapsed) {
            this.toggleSidebar(false);
        }
    }

    /**
     * Inject sidebar HTML into the page
     */
    injectSidebarHTML() {
        const sidebar = document.createElement('div');
        sidebar.id = 'pipelineSidebar';
        sidebar.className = 'status-sidebar';

        sidebar.innerHTML = `
            <div class="sidebar-header">
                <span class="sidebar-title">Pipeline Status</span>
                <button class="sidebar-toggle" id="sidebarToggle" title="Collapse sidebar">
                    ◀
                </button>
            </div>

            <div class="pipeline-stages" id="pipelineStages">
                ${this.renderStages()}
            </div>

            <div class="sidebar-footer">
                <div class="task-id-display" id="currentTaskId">
                    No active task
                </div>
                <button class="refresh-status-btn" onclick="pipelineSidebar.refreshStatus()">
                    🔄 Refresh Status
                </button>
            </div>
        `;

        document.body.insertBefore(sidebar, document.body.firstChild);
    }

    /**
     * Render stage items
     */
    renderStages() {
        return this.stages.map(stage => {
            const state = this.stageStates[stage.id] || 'pending';
            return `
                <div class="stage-item ${state}" data-stage="${stage.id}">
                    <div class="stage-icon">${stage.icon}</div>
                    <div class="stage-info">
                        <div class="stage-name">${stage.name}</div>
                        <div class="stage-status">${state}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Update stage display
     */
    updateStageDisplay() {
        const container = document.getElementById('pipelineStages');
        if (container) {
            container.innerHTML = this.renderStages();
        }
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        const toggleBtn = document.getElementById('sidebarToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleSidebar());
        }
    }

    /**
     * Toggle sidebar collapsed state
     */
    toggleSidebar(save = true) {
        const sidebar = document.getElementById('pipelineSidebar');
        const navbar = document.querySelector('.navbar');
        const pageWrapper = document.querySelector('.page-wrapper');
        const toggleBtn = document.getElementById('sidebarToggle');

        this.isCollapsed = !this.isCollapsed;

        if (this.isCollapsed) {
            sidebar.classList.add('collapsed');
            navbar?.classList.add('sidebar-collapsed');
            pageWrapper?.classList.add('sidebar-collapsed');
            toggleBtn.innerHTML = '▶';
            toggleBtn.title = 'Expand sidebar';
        } else {
            sidebar.classList.remove('collapsed');
            navbar?.classList.remove('sidebar-collapsed');
            pageWrapper?.classList.remove('sidebar-collapsed');
            toggleBtn.innerHTML = '◀';
            toggleBtn.title = 'Collapse sidebar';
        }

        if (save) {
            localStorage.setItem('sidebarCollapsed', this.isCollapsed);
        }
    }

    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/task-progress`;

            this.progressWebSocket = new WebSocket(wsUrl);

            this.progressWebSocket.onopen = () => {
                console.log('Pipeline sidebar WebSocket connected');
            };

            this.progressWebSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleProgressUpdate(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };

            this.progressWebSocket.onerror = (error) => {
                console.error('Sidebar WebSocket error:', error);
            };

            this.progressWebSocket.onclose = () => {
                console.log('Sidebar WebSocket closed, reconnecting in 5s...');
                setTimeout(() => this.connectWebSocket(), 5000);
            };

        } catch (error) {
            console.error('Failed to connect sidebar WebSocket:', error);
        }
    }

    /**
     * Handle progress update from WebSocket
     */
    handleProgressUpdate(data) {
        if (data.type === 'stage_update' && data.stage) {
            this.updateStage(data.stage, data.status || 'running');
        } else if (data.type === 'task_update' && data.task_id) {
            this.setCurrentTask(data.task_id);
            if (data.stages) {
                Object.keys(data.stages).forEach(stage => {
                    this.updateStage(stage, data.stages[stage]);
                });
            }
        } else if (data.type === 'task_complete') {
            this.markAllCompleted();
        }
    }

    /**
     * Load current task from API
     */
    async loadCurrentTask() {
        try {
            const response = await fetch('/api/tasks/status');
            const data = await response.json();

            if (data.tasks && data.tasks.length > 0) {
                const latestTask = data.tasks[0];
                this.setCurrentTask(latestTask.task_id);

                // Load detailed status for this task
                await this.loadTaskStatus(latestTask.task_id);
            }
        } catch (error) {
            console.error('Failed to load current task:', error);
        }
    }

    /**
     * Load task status from API
     */
    async loadTaskStatus(taskId) {
        try {
            const response = await fetch(`/api/tasks/${taskId}/status`);
            const data = await response.json();

            if (data.stages) {
                this.stageStates = data.stages;
                this.updateStageDisplay();
            }
        } catch (error) {
            console.error('Failed to load task status:', error);
        }
    }

    /**
     * Set current task ID
     */
    setCurrentTask(taskId) {
        this.currentTaskId = taskId;
        const display = document.getElementById('currentTaskId');
        if (display) {
            display.innerHTML = `<strong>Task:</strong> ${taskId}`;
        }
    }

    /**
     * Update a specific stage
     */
    updateStage(stageId, status) {
        this.stageStates[stageId] = status;

        const stageEl = document.querySelector(`[data-stage="${stageId}"]`);
        if (stageEl) {
            stageEl.className = `stage-item ${status}`;
            const statusEl = stageEl.querySelector('.stage-status');
            if (statusEl) {
                statusEl.textContent = status;
            }
        }
    }

    /**
     * Mark all stages as completed
     */
    markAllCompleted() {
        this.stages.forEach(stage => {
            this.updateStage(stage.id, 'completed');
        });
    }

    /**
     * Refresh status manually
     */
    async refreshStatus() {
        if (this.currentTaskId) {
            await this.loadTaskStatus(this.currentTaskId);
        } else {
            await this.loadCurrentTask();
        }
    }
}

// Global instance
let pipelineSidebar = null;

// Initialize sidebar when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        pipelineSidebar = new PipelineSidebar();
    });
} else {
    pipelineSidebar = new PipelineSidebar();
}
