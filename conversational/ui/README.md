<div align="center">

# 🌐 Web UI & API Server
### *FastAPI Backend & Interactive Frontend*

</div>

The interface layer providing a modern chat experience, real-time monitoring, and artifact browsing.

---

## 🚀 Running the Server

### Option 1: Main Server (Recommended)
This launches the full application stack.
```bash
cd conversational
python server.py
```
> **http://localhost:8000**

### Option 2: UI API Server
Standalone UI server mode.
```bash
python ui/api.py
```
> **http://localhost:8008**

---

## 📁 Structure

| Path | Purpose |
|---|---|
| `api.py` | **FastAPI Application** |
| `static/index.html` | **Chat Interface** |
| `static/logs.html` | **Log Viewer** |
| `static/outputs.html` | **Artifact Browser** |
| `static/status.html` | **Pipeline Monitor** |
| `static/css/main.css` | **Styling** (Dark Mode) |

---

## 🔌 API Endpoints

### 🖥️ Pages

| Route | Description |
|---|---|
| `/` | 💬 Chat Interface |
| `/logs` | 📜 Pipeline Logs |
| `/outputs` | 📂 Output Browser |
| `/status` | ⚡ Task Status |

### ⚡ REST API

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat/send` | `POST` | Send message to agent |
| `/api/conversations` | `GET` | List sessions |
| `/api/stages` | `GET` | List pipeline stages |
| `/api/stages/{name}/outputs` | `GET` | Get stage artifacts |

### 🔄 WebSocket

| Endpoint | Description |
|---|---|
| `/ws/task-progress` | Real-time stage status updates |
| `/ws/logs` | Live log streaming |

---

## 💡 Key Features

- **⚡ Real-time Sidebar**: Shows pipeline stage status (Planning → Running → Done).
- **📝 Markdown Chat**: Full syntax highlighting for code blocks and tables.
- **🔔 Toast Notifications**: Context-aware alerts for success/error.
- **💾 Auto-Persistence**: Chat history is saved automatically.
- **📱 Responsive**: Works on Desktop, Tablet, and Mobile.

---

## 🎨 Customization

Want to change the theme? Edit `static/css/main.css`:

```css
:root {
  --primary: #6366f1;   /* Indigo */
  --secondary: #8b5cf6; /* Violet */
  --bg-dark: #050510;   /* Deep Space */
  --text-main: #f8fafc; /* Slate */
}
```
