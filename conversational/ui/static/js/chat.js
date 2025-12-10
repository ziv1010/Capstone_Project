/**
 * Chat Interface Logic
 * Handles conversation history display and live chat
 */

let currentSessionId = null;
let conversations = [];
let chatMode = 'live'; // Default to live
let liveMessages = [];

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

    // Add welcome message
    liveMessages = [{
        role: 'assistant',
        content: '👋 Welcome! I\'m your AI pipeline assistant. Ask me anything about your data or request analysis tasks.',
        timestamp: new Date().toISOString()
    }];
    displayLiveMessages();

    // Load conversations in background for history mode
    await loadConversations();
}

/**
 * Toggle between history and live chat modes
 */
function toggleChatMode() {
    const toggleBtn = document.getElementById('modeToggle');
    const sessionCard = document.getElementById('sessionSelectorCard');
    const inputContainer = document.getElementById('chatInputContainer');

    if (chatMode === 'history') {
        // Switch to live mode
        chatMode = 'live';
        toggleBtn.innerHTML = '📜 History';
        toggleBtn.classList.add('btn-primary');
        toggleBtn.classList.remove('btn-secondary');
        sessionCard.style.display = 'none';
        inputContainer.style.display = 'block';

        // Clear and show welcome message
        liveMessages = [{
            role: 'assistant',
            content: '👋 Welcome! I\'m your AI pipeline assistant. Ask me anything about your data or request analysis tasks.',
            timestamp: new Date().toISOString()
        }];
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

        // Add assistant response
        const assistantMsg = {
            role: 'assistant',
            content: response.response,
            timestamp: new Date().toISOString(),
            metadata: response.metadata
        };
        liveMessages.push(assistantMsg);
        displayLiveMessages();

        // Show info if a task was created
        if (response.task_id) {
            const infoMsg = {
                role: 'system',
                content: `🚀 Task ${response.task_id} created. Monitor progress in the Status page.`,
                timestamp: new Date().toISOString()
            };
            liveMessages.push(infoMsg);
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

    div.innerHTML = `
        <div class="chat-avatar ${avatarClass}">${avatarText}</div>
        <div class="chat-bubble">
            <div>${content}</div>
            <div class="chat-timestamp">${formatTimestamp(message.timestamp)}</div>
        </div>
    `;

    return div;
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
