#!/usr/bin/env python3
"""
Enhanced API Server for Conversational AI Pipeline UI

Provides comprehensive REST and WebSocket endpoints for:
- Conversation history
- Real-time debugging logs
- Stage outputs (JSON and images)
- Task progress tracking
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
    STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    STAGE6_OUT_DIR, CONVERSATION_STATE_DIR, OUTPUT_ROOT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui_api")

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")

manager = ConnectionManager()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, handling wrapped format."""
    if not path.exists():
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle wrapped format
        if isinstance(data, dict) and "data" in data and "_meta" in data:
            return data["data"]
        return data
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def get_all_conversations() -> List[Dict[str, Any]]:
    """Get list of all conversation sessions."""
    conversations = []
    
    if not CONVERSATION_STATE_DIR.exists():
        return conversations
    
    for file in sorted(CONVERSATION_STATE_DIR.glob("session_*.json"), reverse=True):
        data = load_json_file(file)
        if data:
            # Extract metadata
            messages = data.get("messages", [])
            conversations.append({
                "session_id": data.get("session_id"),
                "created_at": data.get("created_at"),
                "last_updated": data.get("last_updated"),
                "message_count": len(messages),
                "first_message": messages[0].get("content", "")[:100] if messages else "",
                "filename": file.name
            })
    
    return conversations

def get_conversation(session_id: str) -> Optional[Dict[str, Any]]:
    """Get full conversation by session ID."""
    conv_file = CONVERSATION_STATE_DIR / f"{session_id}.json"
    return load_json_file(conv_file)

def get_stage_outputs(stage: str) -> List[Dict[str, Any]]:
    """Get all output files for a stage."""
    stage_dirs = {
        "stage1": SUMMARIES_DIR,
        "stage2": STAGE2_OUT_DIR,
        "stage3": STAGE3_OUT_DIR,
        "stage3b": STAGE3B_OUT_DIR,
        "stage3_5a": STAGE3_5A_OUT_DIR,
        "stage3_5b": STAGE3_5B_OUT_DIR,
        "stage4": STAGE4_OUT_DIR,
        "stage5": STAGE5_OUT_DIR,
        "stage6": STAGE6_OUT_DIR,
    }
    
    stage_dir = stage_dirs.get(stage)
    if not stage_dir or not stage_dir.exists():
        return []
    
    outputs = []
    
    # Get JSON files
    for json_file in stage_dir.glob("*.json"):
        stat = json_file.stat()
        outputs.append({
            "filename": json_file.name,
            "type": "json",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(json_file.relative_to(OUTPUT_ROOT))
        })
    
    # Get parquet files
    for parquet_file in stage_dir.glob("*.parquet"):
        stat = parquet_file.stat()
        outputs.append({
            "filename": parquet_file.name,
            "type": "parquet",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(parquet_file.relative_to(OUTPUT_ROOT))
        })
    
    # Get image files (for stage5)
    for img_file in stage_dir.glob("*.png"):
        stat = img_file.stat()
        outputs.append({
            "filename": img_file.name,
            "type": "image",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(img_file.relative_to(OUTPUT_ROOT))
        })
    
    return sorted(outputs, key=lambda x: x["modified"], reverse=True)

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Enhanced UI API Server...")
    yield
    logger.info("👋 Shutting down Enhanced UI API Server...")

app = FastAPI(
    title="Conversational AI Pipeline UI API",
    description="REST and WebSocket API for pipeline visualization",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================================
# HTML PAGE ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Serve the main chat interface."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "UI not found. Please create ui/static/index.html"}

@app.get("/logs")
async def logs_page():
    """Serve the logs viewer page."""
    html_path = static_dir / "logs.html"
    if html_path.exists():
        return FileResponse(html_path)
    return FileResponse(static_dir / "index.html")

@app.get("/outputs")
async def outputs_page():
    """Serve the outputs browser page."""
    html_path = static_dir / "outputs.html"
    if html_path.exists():
        return FileResponse(html_path)
    return FileResponse(static_dir / "index.html")

@app.get("/status")
async def status_page():
    """Serve the task status page."""
    html_path = static_dir / "status.html"
    if html_path.exists():
        return FileResponse(html_path)
    return FileResponse(static_dir / "index.html")

# ============================================================================
# CONVERSATION API ENDPOINTS
# ============================================================================

@app.get("/api/conversations")
async def list_conversations():
    """Get list of all conversation sessions."""
    try:
        conversations = get_all_conversations()
        return {
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{session_id}")
async def get_conversation_detail(session_id: str):
    """Get full conversation history for a session."""
    try:
        conversation = get_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {session_id} not found")
        
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/latest")
async def get_latest_conversation():
    """Get the most recent conversation."""
    try:
        conversations = get_all_conversations()
        if not conversations:
            return {"message": "No conversations found"}
        
        latest = conversations[0]
        session_id = latest["session_id"]
        return get_conversation(session_id)
    except Exception as e:
        logger.error(f"Error getting latest conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/send")
async def send_chat_message(request: dict):
    """Send a message to the conversational pipeline and get response."""
    try:
        message = request.get("message", "").strip()
        session_id = request.get("session_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Import here to avoid circular dependencies
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from code.master_orchestrator import ConversationalOrchestrator
        from fastapi.concurrency import run_in_threadpool
        
        # Create or reuse orchestrator with session_id
        # Note: Initialization might check files but is generally fast enough
        orchestrator = ConversationalOrchestrator(session_id=session_id)
        
        # Process the message in a thread pool to avoid blocking the event loop
        # This allows other endpoints (status, logs) to respond while the pipeline runs
        result = await run_in_threadpool(orchestrator.process_user_input, message)
        
        return {
            "success": True,
            "response": result.get("response", "No response generated"),
            "session_id": orchestrator.conversation.session_id,
            "action": result.get("action"),
            "task_id": result.get("task_id"),
            "metadata": result.get("metadata", {})
        }
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LOGS API ENDPOINTS
# ============================================================================

@app.get("/api/logs/recent")
async def get_recent_logs(lines: int = 100):
    """Get recent log lines from the running process."""
    try:
        # Read from the terminal output or log file
        # For now, return placeholder - will implement actual log reading
        return {
            "logs": [
                {"level": "INFO", "timestamp": datetime.now().isoformat(), "message": "Log streaming coming soon..."}
            ],
            "count": 1
        }
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await websocket.accept()
    
    # Register connection
    with logs_manager.lock:
        logs_manager.connections.append(websocket)
        count = len(logs_manager.connections)
    
    logger.info(f"WebSocket connected. Total connections: {count}")
    
    try:
        # Send initial recent logs
        log_file = PROJECT_ROOT / "output" / "pipeline.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                # Get last 100 lines
                lines = f.readlines()[-100:]
                for line in lines:
                    try:
                        # Parse log line
                        # Format: 2025-12-10 03:00:00,000 - logger - LEVEL - message
                        parts = line.strip().split(" - ", 3)
                        if len(parts) >= 4:
                            timestamp, _, level, message = parts
                        else:
                            timestamp = datetime.now().isoformat()
                            level = "INFO"
                            message = line.strip()
                            
                        await websocket.send_json({
                            "type": "log",
                            "level": level,
                            "timestamp": timestamp,
                            "message": message
                        })
                    except Exception:
                        pass

        # Keep connection open and send heartbeats
        while True:
            await asyncio.sleep(1)
            # In a real implementation, we would tail the file here
            # For now, we rely on the log_manager broadcast from the API process logs
            # To support external process logs, we need a file watcher
            
            # Simple file watcher logic
            if log_file.exists():
                stat = log_file.stat()
                current_size = getattr(websocket, "_last_size", 0)
                if stat.st_size > current_size:
                    with open(log_file, "r") as f:
                        f.seek(current_size)
                        new_lines = f.readlines()
                        for line in new_lines:
                            try:
                                parts = line.strip().split(" - ", 3)
                                if len(parts) >= 4:
                                    timestamp, _, level, message = parts
                                else:
                                    timestamp = datetime.now().isoformat()
                                    level = "INFO"
                                    message = line.strip()
                                    
                                await websocket.send_json({
                                    "type": "log",
                                    "level": level,
                                    "timestamp": timestamp,
                                    "message": message
                                })
                            except Exception:
                                pass
                    websocket._last_size = stat.st_size
            
    except WebSocketDisconnect:
        with logs_manager.lock:
            if websocket in logs_manager.connections:
                logs_manager.connections.remove(websocket)
                count = len(logs_manager.connections)
        logger.info(f"WebSocket disconnected. Total connections: {count}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        with logs_manager.lock:
            if websocket in logs_manager.connections:
                logs_manager.connections.remove(websocket)

# ============================================================================
# STAGE OUTPUTS API ENDPOINTS
# ============================================================================

@app.get("/api/stages")
async def list_stages():
    """Get list of all available stages."""
    stages = [
        {"name": "stage1", "title": "Data Analysis", "output_dir": str(SUMMARIES_DIR)},
        {"name": "stage2", "title": "Task Proposals", "output_dir": str(STAGE2_OUT_DIR)},
        {"name": "stage3", "title": "Execution Planning", "output_dir": str(STAGE3_OUT_DIR)},
        {"name": "stage3b", "title": "Data Preparation", "output_dir": str(STAGE3B_OUT_DIR)},
        {"name": "stage3_5a", "title": "Method Proposals", "output_dir": str(STAGE3_5A_OUT_DIR)},
        {"name": "stage3_5b", "title": "Benchmarking", "output_dir": str(STAGE3_5B_OUT_DIR)},
        {"name": "stage4", "title": "Execution", "output_dir": str(STAGE4_OUT_DIR)},
        {"name": "stage5", "title": "Visualization", "output_dir": str(STAGE5_OUT_DIR)},
        {"name": "stage6", "title": "Report Generation", "output_dir": str(STAGE6_OUT_DIR)},
    ]
    return {"stages": stages}

@app.get("/api/stages/{stage_name}/outputs")
async def get_stage_output_files(stage_name: str):
    """Get all output files for a specific stage."""
    try:
        outputs = get_stage_outputs(stage_name)
        return {
            "stage": stage_name,
            "outputs": outputs,
            "count": len(outputs)
        }
    except Exception as e:
        logger.error(f"Error getting outputs for {stage_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/{file_path:path}")
async def get_file(file_path: str):
    """Get a specific file from the output directory."""
    try:
        # Security: ensure file is within OUTPUT_ROOT
        full_path = OUTPUT_ROOT / file_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if path is actually within OUTPUT_ROOT (prevent directory traversal)
        try:
            full_path.resolve().relative_to(OUTPUT_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Return file based on type
        if full_path.suffix == '.json':
            data = load_json_file(full_path)
            return JSONResponse(content=data)
        elif full_path.suffix in ['.png', '.jpg', '.jpeg', '.gif']:
            return FileResponse(full_path)
        elif full_path.suffix == '.parquet':
            return FileResponse(full_path, media_type='application/octet-stream')
        else:
            return FileResponse(full_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TASK PROGRESS API ENDPOINTS
# ============================================================================

@app.get("/api/tasks/status")
async def get_all_tasks_status():
    """Get status of all tasks."""
    # This will read from output directories to infer task status
    tasks = []
    
    # Check STAGE4_OUT_DIR for execution results
    if STAGE4_OUT_DIR.exists():
        for result_file in STAGE4_OUT_DIR.glob("execution_result_*.json"):
            data = load_json_file(result_file)
            if data:
                task_id = result_file.stem.replace("execution_result_", "")
                tasks.append({
                    "task_id": task_id,
                    "status": data.get("status", "unknown"),
                    "last_stage": "stage4",
                    "metrics": data.get("metrics", {})
                })
    
    return {
        "tasks": tasks,
        "count": len(tasks)
    }

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get detailed status for a specific task."""
    # Determine which stages have completed for this task
    stage_status = {}
    
    # Add PLAN- prefix if not present
    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
    
    # Check each stage for outputs
    checks = [
        ("stage3", STAGE3_OUT_DIR / f"{plan_id}.json"),
        ("stage3b", STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"),
        ("stage3_5a", STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"),
        ("stage3_5b", STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"),
        ("stage4", STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"),
        ("stage5", STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"),
    ]
    
    current_stage = None
    for stage_name, path in checks:
        if path.exists():
            stage_status[stage_name] = "completed"
        else:
            if current_stage is None:
                current_stage = stage_name
            stage_status[stage_name] = "pending"
    
    return {
        "task_id": task_id,
        "current_stage": current_stage or "completed",
        "stages": stage_status
    }

@app.websocket("/ws/task-progress")
async def websocket_task_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time task progress updates."""
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(2)
            # Send current status of all tasks
            tasks_status = await get_all_tasks_status()
            await websocket.send_json({
                "type": "task_update",
                "data": tasks_status
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting Enhanced UI API Server")
    logger.info("=" * 60)
    logger.info("Chat Interface:    http://localhost:8007/")
    logger.info("Logs Viewer:       http://localhost:8007/logs")
    logger.info("Outputs Browser:   http://localhost:8007/outputs")
    logger.info("Task Status:       http://localhost:8007/status")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")
