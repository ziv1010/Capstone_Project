/**
 * Chat Interface Logic
 * Handles conversation history display and live chat
 */

let currentSessionId = null;
let conversations = [];
let chatMode = 'live'; // Default to live
let liveMessages = [];

// Storage key for persisting chat
const CHAT_STORAGE_KEY = 'ai_pipeline_chat_messages';

/**
 * Save messages to sessionStorage
 */
function saveChatToStorage() {
    try {
        sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(liveMessages));
    } catch (e) {
        console.warn('Failed to save chat to storage:', e);
    }
}

/**
 * Load messages from sessionStorage
 */
function loadChatFromStorage() {
    try {
        const stored = sessionStorage.getItem(CHAT_STORAGE_KEY);
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.warn('Failed to load chat from storage:', e);
    }
    return null;
}

/**
 * Start a new chat session
 */
function startNewChat() {
    liveMessages = [{
        role: 'assistant',
        content: '👋 Welcome! I\'m your AI pipeline assistant with **EDA capabilities**.\n\n' +
            '🔍 **EDA Queries:** Ask about columns, statistics, correlations, or create visualizations\n' +
            '📊 **Pipeline Tasks:** Request analysis or run forecasting tasks\n\n' +
            'Try the quick actions above or ask me anything about your data!',
        timestamp: new Date().toISOString()
    }];
    saveChatToStorage();
    displayLiveMessages();

    // Scroll to top
    const container = document.getElementById('chatMessages');
    container.scrollTop = 0;
}

/**
 * Initialize the chat page
 */
async function init() {
    // Start in live mode
    const toggleBtn = document.getElementById('modeToggle');
    const sessionCard = document.getElementById('sessionSelectorCard');
    const inputContainer = document.getElementById('chatInputContainer');

    toggleBtn.innerHTML = '📜 History';
    toggleBtn.classList.add('btn-primary');
    toggleBtn.classList.remove('btn-secondary');
    sessionCard.style.display = 'none';
    inputContainer.style.display = 'block';

    document.getElementById('sessionBadge').textContent = 'Live Chat';
    document.getElementById('sessionBadge').className = 'badge badge-running';

    // Try to restore chat from storage, or show welcome message
    const storedMessages = loadChatFromStorage();
    if (storedMessages && storedMessages.length > 0) {
        liveMessages = storedMessages;
    } else {
        liveMessages = [{
            role: 'assistant',
            content: '👋 Welcome! I\'m your AI pipeline assistant with **EDA capabilities**.\n\n' +
                '🔍 **EDA Queries:** Ask about columns, statistics, correlations, or create visualizations\n' +
                '📊 **Pipeline Tasks:** Request analysis or run forecasting tasks\n\n' +
                'Try the quick actions above or ask me anything about your data!',
            timestamp: new Date().toISOString()
        }];
    }
    displayLiveMessages();

    // Load conversations in background for history mode
    await loadConversations();
}

/**
 * Send a quick query from the EDA action buttons
 */
function sendQuickQuery(query) {
    const input = document.getElementById('messageInput');
    input.value = query;
    sendMessage();
}

/**
 * Toggle between history and live chat modes
 */
function toggleChatMode() {
    const toggleBtn = document.getElementById('modeToggle');
    const sessionCard = document.getElementById('sessionSelectorCard');
    const inputContainer = document.getElementById('chatInputContainer');
    const edaCard = document.getElementById('edaActionsCard');

    if (chatMode === 'history') {
        // Switch to live mode
        chatMode = 'live';
        toggleBtn.innerHTML = '📜 History';
        toggleBtn.classList.add('btn-primary');
        toggleBtn.classList.remove('btn-secondary');
        sessionCard.style.display = 'none';
        inputContainer.style.display = 'block';
        if (edaCard) edaCard.style.display = 'block';

        // Restore from storage instead of resetting
        const storedMessages = loadChatFromStorage();
        if (storedMessages && storedMessages.length > 0) {
            liveMessages = storedMessages;
        } else {
            liveMessages = [{
                role: 'assistant',
                content: '👋 Welcome! I\'m your AI pipeline assistant with **EDA capabilities**.\n\n' +
                    '🔍 **EDA Queries:** Ask about columns, statistics, correlations, or create visualizations\n' +
                    '📊 **Pipeline Tasks:** Request analysis or run forecasting tasks\n\n' +
                    'Try the quick actions above or ask me anything about your data!',
                timestamp: new Date().toISOString()
            }];
        }
        displayLiveMessages();

        document.getElementById('sessionBadge').textContent = 'Live Chat';
        document.getElementById('sessionBadge').className = 'badge badge-running';

    } else {
        // Switch to history mode
        chatMode = 'history';
        toggleBtn.innerHTML = '📝 Live Chat';
        toggleBtn.classList.remove('btn-primary');
        toggleBtn.classList.add('btn-secondary');
        sessionCard.style.display = 'block';
        inputContainer.style.display = 'none';
        if (edaCard) edaCard.style.display = 'none';

        // Reload conversation history
        loadConversations();

        document.getElementById('sessionBadge').textContent = `${conversations.length} sessions`;
        document.getElementById('sessionBadge').className = 'badge badge-info';
    }
}

/**
 * Handle Enter key press in input
 */
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Refresh datasets - scan for new data and auto-summarize
 */
async function refreshDatasets() {
    const btn = document.getElementById('refreshDataBtn');
    const originalText = btn.innerHTML;

    btn.innerHTML = '🔄 Scanning...';
    btn.disabled = true;

    try {
        const response = await APIClient.post('/api/data/refresh', {});

        let resultMsg;
        if (response.success) {
            if (response.new_datasets && response.new_datasets.length > 0) {
                const datasetList = response.new_datasets.map(d =>
                    `  • ${d.filename} (${d.rows} rows, ${d.columns} cols)`
                ).join('\n');
                resultMsg = `✅ **${response.message}**\n\n${datasetList}\n\nYou can now query these datasets!`;
            } else {
                resultMsg = `✅ ${response.message}\n\nTotal datasets available: ${response.total_datasets}`;
            }
        } else {
            resultMsg = `❌ Refresh failed: ${response.error}`;
        }

        // Add result to chat
        liveMessages.push({
            role: 'system',
            content: resultMsg,
            timestamp: new Date().toISOString()
        });
        displayLiveMessages();

    } catch (error) {
        console.error('Refresh error:', error);
        liveMessages.push({
            role: 'system',
            content: `❌ Error: ${error.message}`,
            timestamp: new Date().toISOString()
        });
        displayLiveMessages();
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

/**
 * Send a message to the pipeline
 */
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to display
    const userMsg = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
    };
    liveMessages.push(userMsg);
    saveChatToStorage();  // Save immediately so it persists if user navigates away
    displayLiveMessages();

    // Clear input and disable
    input.value = '';
    input.disabled = true;
    document.getElementById('sendButton').disabled = true;
    document.getElementById('sendButton').innerHTML = '<span>Sending...</span> <div class="spinner" style="width: 16px; height: 16px;"></div>';

    try {
        // Send to backend
        const response = await APIClient.post('/api/chat/send', {
            message: message,
            session_id: currentSessionId
        });

        // Update current session ID
        if (response.session_id) {
            currentSessionId = response.session_id;
        }

        // Add assistant response with visualizations
        const assistantMsg = {
            role: 'assistant',
            content: response.response,
            timestamp: new Date().toISOString(),
            metadata: response.metadata,
            visualizations: response.visualizations || []
        };
        liveMessages.push(assistantMsg);
        saveChatToStorage();  // Persist chat
        displayLiveMessages();

        // Show info if a task was created
        if (response.task_id) {
            const infoMsg = {
                role: 'system',
                content: `🚀 Task ${response.task_id} created. Monitor progress in the Status page.`,
                timestamp: new Date().toISOString()
            };
            liveMessages.push(infoMsg);
            saveChatToStorage();  // Persist chat
            displayLiveMessages();
        }

    } catch (error) {
        console.error('Error sending message:', error);

        const errorMsg = {
            role: 'system',
            content: `⚠️ Error: ${error.message}. Please try again.`,
            timestamp: new Date().toISOString()
        };
        liveMessages.push(errorMsg);
        displayLiveMessages();
    } finally {
        // Re-enable input
        input.disabled = false;
        input.focus();
        document.getElementById('sendButton').disabled = false;
        document.getElementById('sendButton').innerHTML = '<span>Send</span> <span>✉️</span>';
    }
}

/**
 * Display live chat messages
 */
function displayLiveMessages() {
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.innerHTML = '';

    liveMessages.forEach(msg => {
        const messageEl = createMessageElement(msg);
        messagesContainer.appendChild(messageEl);
    });

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Load all conversations
 */
async function loadConversations() {
    try {
        const data = await APIClient.getConversations();
        conversations = data.conversations || [];

        if (conversations.length === 0) {
            document.getElementById('sessionSelector').innerHTML =
                '<option value="">No conversations found</option>';
            document.getElementById('sessionBadge').textContent = 'No sessions';
            return;
        }

        // Populate selector
        const selector = document.getElementById('sessionSelector');
        selector.innerHTML = conversations.map(conv =>
            `<option value="${conv.session_id}">
                ${conv.session_id} - ${conv.message_count} messages (${formatTimestamp(conv.created_at)})
            </option>`
        ).join('');

        // Load the latest conversation by default
        currentSessionId = conversations[0].session_id;
        selector.value = currentSessionId;
        await loadSelectedConversation();

        document.getElementById('sessionBadge').textContent = `${conversations.length} sessions`;

    } catch (error) {
        console.error('Error loading conversations:', error);
        showError('Failed to load conversations');
    }
}

/**
 * Load the selected conversation
 */
async function loadSelectedConversation() {
    const selector = document.getElementById('sessionSelector');
    currentSessionId = selector.value;

    if (!currentSessionId) return;

    try {
        const conversation = await APIClient.getConversation(currentSessionId);
        displayConversation(conversation);
    } catch (error) {
        console.error('Error loading conversation:', error);
        showError('Failed to load conversation');
    }
}

/**
 * Display conversation messages
 */
function displayConversation(conversation) {
    const messagesContainer = document.getElementById('chatMessages');

    if (!conversation || !conversation.messages || conversation.messages.length === 0) {
        messagesContainer.innerHTML = `
            <div class="text-center text-muted">
                <p>No messages in this conversation</p>
            </div>
        `;
        return;
    }

    messagesContainer.innerHTML = '';

    conversation.messages.forEach(msg => {
        const messageEl = createMessageElement(msg);
        messagesContainer.appendChild(messageEl);
    });

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Create a message element
 */
function createMessageElement(message) {
    const div = document.createElement('div');

    // Handle system messages
    if (message.role === 'system') {
        div.className = 'chat-message';
        div.innerHTML = `
            <div style="text-align: center; width: 100%; padding: var(--spacing-sm); background: var(--glass-bg); border-radius: var(--radius-md); margin: var(--spacing-sm) 0;">
                <span style="color: var(--info);">${escapeHtml(message.content)}</span>
            </div>
        `;
        return div;
    }

    div.className = `chat-message ${message.role}`;

    const isUser = message.role === 'user';
    const avatarClass = isUser ? 'user-avatar' : 'assistant-avatar';
    const avatarText = isUser ? 'U' : 'AI';

    // Remove <think> tags from assistant messages
    let content = message.content || '';
    if (!isUser) {
        content = content.replace(/<think>[\s\S]*?<\/think>/g, '');
    }

    // Convert line breaks to <br> and escape HTML
    content = escapeHtml(content).replace(/\n/g, '<br>');

    // Basic markdown: **bold** -> <strong>
    content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Build visualization gallery HTML if present
    let vizHtml = '';
    if (message.visualizations && message.visualizations.length > 0) {
        vizHtml = '<div class="chat-viz-gallery">';
        message.visualizations.forEach((vizUrl, index) => {
            vizHtml += `<div class="chat-viz-item">
                <img src="${vizUrl}" alt="Visualization ${index + 1}" onclick="openImageModal('${vizUrl}')" />
            </div>`;
        });
        vizHtml += '</div>';
    }

    div.innerHTML = `
        <div class="chat-avatar ${avatarClass}">${avatarText}</div>
        <div class="chat-bubble">
            <div>${content}</div>
            ${vizHtml}
            <div class="chat-timestamp">${formatTimestamp(message.timestamp)}</div>
        </div>
    `;

    return div;
}

/**
 * Open image in modal for full-size viewing
 */
function openImageModal(imageUrl) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('imageModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'imageModal';
        modal.className = 'image-modal';
        modal.innerHTML = `
            <div class="image-modal-content">
                <span class="image-modal-close" onclick="closeImageModal()">&times;</span>
                <img id="modalImage" src="" alt="Full size visualization" />
            </div>
        `;
        document.body.appendChild(modal);
    }

    document.getElementById('modalImage').src = imageUrl;
    modal.style.display = 'flex';
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Show error message
 */
function showError(message) {
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.innerHTML = `
        <div class="text-center">
            <p style="color: var(--error);">⚠️ ${escapeHtml(message)}</p>
        </div>
    `;
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', init);
