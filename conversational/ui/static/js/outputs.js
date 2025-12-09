/**
 * Outputs Browser Logic
 * Handles stage output file browsing and viewing
 */

let stages = [];
let currentStage = null;
let currentFiles = [];

/**
 * Initialize the outputs page
 */
async function init() {
    await loadStages();
}

/**
 * Load all available stages
 */
async function loadStages() {
    try {
        const data = await APIClient.getStages();
        stages = data.stages || [];

        displayStages();
    } catch (error) {
        console.error('Error loading stages:', error);
        showError('Failed to load stages');
    }
}

/**
 * Display stages as clickable cards
 */
function displayStages() {
    const grid = document.getElementById('stageGrid');

    if (stages.length === 0) {
        grid.innerHTML = '<div class="text-center text-muted">No stages available</div>';
        return;
    }

    grid.innerHTML = stages.map(stage => `
        <div class="stage-card" onclick="selectStage('${stage.name}')">
            <h4>${stage.title}</h4>
            <p class="text-muted" style="font-size: 0.875rem; margin-top: 0.5rem;">
                ${stage.name}
            </p>
        </div>
    `).join('');
}

/**
 * Select a stage and load its outputs
 */
async function selectStage(stageName) {
    currentStage = stageName;

    // Update selected state
    document.querySelectorAll('.stage-card').forEach(card => {
        card.classList.remove('selected');
    });

    event.target.closest('.stage-card').classList.add('selected');

    // Load files for this stage
    await loadStageFiles(stageName);
}

/**
 * Load files for a specific stage
 */
async function loadStageFiles(stageName) {
    try {
        const data = await APIClient.getStageOutputs(stageName);
        currentFiles = data.outputs || [];

        // Update UI
        const stageInfo = stages.find(s => s.name === stageName);
        document.getElementById('stageTitle').textContent = stageInfo.title;
        document.getElementById('fileCount').textContent = `${currentFiles.length} files`;

        displayFiles();

        // Show files section
        document.getElementById('filesSection').style.display = 'block';

    } catch (error) {
        console.error('Error loading stage files:', error);
        showError('Failed to load files');
    }
}

/**
 * Display files for current stage
 */
function displayFiles() {
    const fileList = document.getElementById('fileList');

    if (currentFiles.length === 0) {
        fileList.innerHTML = '<p class="text-muted">No files available for this stage</p>';
        return;
    }

    fileList.innerHTML = currentFiles.map(file => {
        const icon = getFileIcon(file.type);
        return `
            <div class="file-item" onclick="viewFile('${file.path}', '${file.type}', '${file.filename}')">
                <div class="flex items-center">
                    <span class="file-icon">${icon}</span>
                    <div class="file-info">
                        <div class="file-name">${file.filename}</div>
                        <div class="file-meta">
                            ${formatFileSize(file.size)} • Modified: ${formatTimestamp(file.modified)}
                        </div>
                    </div>
                </div>
                <span>→</span>
            </div>
        `;
    }).join('');
}

/**
 * Get appropriate icon for file type
 */
function getFileIcon(type) {
    const icons = {
        'json': '📄',
        'parquet': '📊',
        'image': '🖼️'
    };
    return icons[type] || '📎';
}

/**
 * View a specific file
 */
async function viewFile(filePath, fileType, filename) {
    try {
        document.getElementById('fileTitle').textContent = filename;
        document.getElementById('fileViewer').style.display = 'block';

        const content = document.getElementById('fileContent');
        content.innerHTML = '<div class="text-center"><div class="spinner" style="margin: 2rem auto;"></div></div>';

        if (fileType === 'json') {
            const data = await APIClient.getFile(filePath);
            content.innerHTML = `
                <div class="json-viewer">
                    <pre><code>${escapeHtml(prettyPrintJSON(data))}</code></pre>
                </div>
            `;
        } else if (fileType === 'image') {
            content.innerHTML = `
                <div class="image-viewer">
                    <img src="/api/files/${filePath}" alt="${filename}">
                </div>
            `;
        } else if (fileType === 'parquet') {
            content.innerHTML = `
                <div class="text-center text-muted">
                    <p>Parquet files cannot be displayed directly</p>
                    <a href="/api/files/${filePath}" class="btn btn-primary" download>
                        📥 Download File
                    </a>
                </div>
            `;
        }

        // Scroll to viewer
        document.getElementById('fileViewer').scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error viewing file:', error);
        document.getElementById('fileContent').innerHTML = `
            <div class="text-center" style="color: var(--error);">
                <p>⚠️ Failed to load file</p>
            </div>
        `;
    }
}

/**
 * Close file viewer
 */
function closeFileViewer() {
    document.getElementById('fileViewer').style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    const grid = document.getElementById('stageGrid');
    grid.innerHTML = `
        <div class="text-center" style="grid-column: 1 / -1; color: var(--error);">
            <p>⚠️ ${escapeHtml(message)}</p>
        </div>
    `;
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', init);
