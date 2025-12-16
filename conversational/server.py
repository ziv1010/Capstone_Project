#!/usr/bin/env python3
"""
UI Server for Conversational AI Pipeline

Provides a web interface for:
- Chatting with the AI assistant
- Viewing pipeline progress in real-time
- Inspecting stage outputs and model thoughts
- EDA (Exploratory Data Analysis) capabilities
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from code.config import (
    SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
    STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    EDA_OUT_DIR, DATA_DIR,
    logger as pipeline_logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui_server")

# ============================================================================
# GLOBAL STATE
# ============================================================================

class PipelineTracker:
    """Tracks pipeline execution state for the UI."""
    
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.is_running: bool = False
        self.current_stage: Optional[str] = None
        self.stage_status: Dict[str, str] = {}
        self.errors: List[str] = []
        self.lock = threading.Lock()
    
    def start_pipeline(self, task_id: str):
        with self.lock:
            self.current_task_id = task_id
            self.is_running = True
            self.current_stage = "stage1"
            self.stage_status = {}
            self.errors = []
    
    def update_stage(self, stage: str, status: str):
        with self.lock:
            self.stage_status[stage] = status
            if status == "running":
                self.current_stage = stage
    
    def finish_pipeline(self, success: bool = True):
        with self.lock:
            self.is_running = False
            if not success:
                self.errors.append("Pipeline execution failed")
    
    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "task_id": self.current_task_id,
                "is_running": self.is_running,
                "current_stage": self.current_stage,
                "stage_status": self.stage_status.copy(),
                "errors": self.errors.copy()
            }

tracker = PipelineTracker()

# ============================================================================
# API MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    pipeline_started: bool = False
    task_id: Optional[str] = None
    visualizations: List[str] = []  # URLs for inline display
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stage_output_path(stage: str, task_id: str = None) -> Optional[Path]:
    """Get the output file path for a stage."""
    if not task_id:
        task_id = tracker.current_task_id
    
    if not task_id:
        return None
    
    # Ensure task_id has PLAN- prefix for stages that need it
    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
    
    paths = {
        "stage1": SUMMARIES_DIR,  # Multiple files
        "stage2": STAGE2_OUT_DIR / "task_proposals.json",
        "stage3": STAGE3_OUT_DIR / f"{plan_id}.json",
        "stage3b": STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet",
        "stage3_5a": STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json",
        "stage3_5b": STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json",
        "stage4": STAGE4_OUT_DIR / f"execution_result_{plan_id}.json",
        "stage5": STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json",
    }
    
    return paths.get(stage)

def load_stage_output(stage: str, task_id: str = None) -> Optional[Dict[str, Any]]:
    """Load the output for a specific stage."""
    path = get_stage_output_path(stage, task_id)
    
    if path is None:
        return None
    
    # Stage 1 has multiple summary files
    if stage == "stage1":
        summaries = []
        if path.exists():
            for f in path.glob("*.summary.json"):
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        # Handle wrapped format
                        if "data" in data:
                            summaries.append(data["data"])
                        else:
                            summaries.append(data)
                except Exception as e:
                    logger.error(f"Failed to load {f}: {e}")
        if summaries:
            return {"summaries": summaries, "count": len(summaries)}
        return None
    
    # Other stages have single JSON files
    if isinstance(path, Path) and path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle wrapped format
                if "data" in data and "_meta" in data:
                    return data["data"]
                return data
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    
    return None

def infer_stage_status(stage: str, task_id: str = None) -> str:
    """Infer the status of a stage based on file existence."""
    output = load_stage_output(stage, task_id)
    
    if output is not None:
        return "completed"
    
    # Check if this is the current running stage
    state = tracker.get_state()
    if state["is_running"] and state["current_stage"] == stage:
        return "running"
    
    return "pending"

def get_all_stages_status(task_id: str = None) -> Dict[str, Dict[str, Any]]:
    """Get status for all stages."""
    stages = ["stage1", "stage2", "stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]
    result = {}
    
    for stage in stages:
        status = infer_stage_status(stage, task_id)
        result[stage] = {
            "stage_name": stage,
            "status": status,
            "has_output": status == "completed"
        }
    
    return result

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline_background(task_id: str):
    """Run the pipeline in a background thread."""
    from code.master_orchestrator import run_forecasting_pipeline
    
    logger.info(f"Starting background pipeline for task {task_id}")
    tracker.start_pipeline(task_id)
    
    try:
        # Run the pipeline
        state = run_forecasting_pipeline(task_id)
        
        # Update tracker with final state
        if state:
            for stage_name, stage_state in state.stages.items():
                tracker.update_stage(stage_name, stage_state.status.value)
        
        tracker.finish_pipeline(success=True)
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        tracker.finish_pipeline(success=False)

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UI Server...")
    yield
    logger.info("Shutting down UI Server...")

app = FastAPI(title="Conversational AI Pipeline UI", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("ui/static/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat messages (legacy endpoint)."""
    return await chat_send(request, background_tasks)

@app.post("/api/chat/send")
async def chat_send(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat messages."""
    import uuid
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from code.master_orchestrator import ConversationalOrchestrator
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    def process_in_thread(message: str):
        """Run orchestrator in separate thread to not block server."""
        orchestrator = ConversationalOrchestrator()
        return orchestrator.process_user_input(message)
    
    try:
        # Run the blocking orchestrator call in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, process_in_thread, request.message)
        
        response_text = result.get("response", "I couldn't process that request.")
        pipeline_started = False
        task_id = result.get("task_id")
        
        # Extract visualization paths from response
        visualizations = []
        if result.get("visualizations"):
            for viz_path in result["visualizations"]:
                # Convert file path to API URL
                if isinstance(viz_path, str) and viz_path.endswith('.png'):
                    filename = Path(viz_path).name
                    visualizations.append(f"/api/eda/image/{filename}")
        
        # Also check for recent EDA visualizations in the response text
        if not visualizations and ".png" in response_text:
            import re
            # Find PNG paths in response
            png_paths = re.findall(r'([^\s]+\.png)', response_text)
            for p in png_paths:
                if EDA_OUT_DIR.name in p or 'eda' in p.lower():
                    filename = Path(p).name
                    if (EDA_OUT_DIR / filename).exists():
                        visualizations.append(f"/api/eda/image/{filename}")

        # Check if pipeline needs to run
        if result.get("action") == "run_pipeline" and task_id:
            pipeline_started = True
            response_text += f"\n\n🚀 Starting pipeline execution for {task_id}..."
            background_tasks.add_task(run_pipeline_background, task_id)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            pipeline_started=pipeline_started,
            task_id=task_id,
            visualizations=visualizations,
            metadata={"action": result.get("action")}
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            session_id=session_id
        )

@app.get("/api/state")
async def get_state():
    """Get the current pipeline state."""
    state = tracker.get_state()
    stages = get_all_stages_status(state.get("task_id"))
    
    return {
        "task_id": state["task_id"],
        "is_running": state["is_running"],
        "current_stage": state["current_stage"],
        "stages": stages,
        "errors": state["errors"]
    }

@app.get("/api/stage/{stage_name}")
async def get_stage_details(stage_name: str):
    """Get detailed output for a specific stage."""
    valid_stages = ["stage1", "stage2", "stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]
    
    if stage_name not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage_name}")
    
    task_id = tracker.current_task_id
    status = infer_stage_status(stage_name, task_id)
    output = load_stage_output(stage_name, task_id)
    
    return {
        "stage_name": stage_name,
        "status": status,
        "output": output,
        "has_output": output is not None
    }

@app.get("/api/tasks")
async def get_available_tasks():
    """Get list of available tasks from stage 2 output."""
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    
    if not proposals_path.exists():
        return {"tasks": [], "message": "No tasks available. Run stage 1 and 2 first."}
    
    try:
        with open(proposals_path, 'r') as f:
            data = json.load(f)
            
        # Handle wrapped format
        if "data" in data:
            data = data["data"]
        
        proposals = data.get("proposals", [])
        tasks = []
        for p in proposals:
            tasks.append({
                "id": p.get("id"),
                "title": p.get("title"),
                "category": p.get("category"),
                "target_column": p.get("target_column"),
                "feasibility_score": p.get("feasibility_score")
            })
        
        return {"tasks": tasks}
        
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        return {"tasks": [], "error": str(e)}

@app.get("/api/visualizations/{task_id}")
async def get_visualizations(task_id: str):
    """Get visualization files for a task."""
    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
    viz_dir = STAGE5_OUT_DIR
    
    # Look for visualization report
    report_path = viz_dir / f"visualization_report_{plan_id}.json"
    
    if not report_path.exists():
        return {"visualizations": [], "message": "No visualizations available for this task."}
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        if "data" in data:
            data = data["data"]
        
        return {
            "visualizations": data.get("visualizations", []),
            "insights": data.get("insights", []),
            "summary": data.get("summary", ""),
            "task_answer": data.get("task_answer", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to load visualizations: {e}")
        return {"visualizations": [], "error": str(e)}


# ============================================================================
# CONVERSATION HISTORY ENDPOINTS
# ============================================================================

@app.get("/api/conversations")
async def get_conversations():
    """List all conversation sessions."""
    try:
        from code.config import OUTPUT_ROOT
        conversations_dir = OUTPUT_ROOT / "conversations"
        
        if not conversations_dir.exists():
            return {"conversations": []}
        
        sessions = []
        for f in conversations_dir.glob("*.json"):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    sessions.append({
                        "session_id": f.stem,
                        "created_at": data.get("created_at", ""),
                        "message_count": len(data.get("messages", []))
                    })
            except Exception:
                pass
        
        # Sort by creation time, newest first
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"conversations": sessions}
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        return {"conversations": [], "error": str(e)}


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Get a specific conversation by session ID."""
    try:
        from code.config import OUTPUT_ROOT
        conversations_dir = OUTPUT_ROOT / "conversations"
        session_file = conversations_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        with open(session_file, 'r') as f:
            return json.load(f)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LOGS ENDPOINTS  
# ============================================================================

@app.get("/logs")
async def logs_page():
    """Serve the logs page."""
    logs_path = Path("ui/static/logs.html")
    if logs_path.exists():
        return FileResponse(logs_path)
    raise HTTPException(status_code=404, detail="Logs page not found")


@app.get("/outputs")
async def outputs_page():
    """Serve the outputs page."""
    outputs_path = Path("ui/static/outputs.html")
    if outputs_path.exists():
        return FileResponse(outputs_path)
    raise HTTPException(status_code=404, detail="Outputs page not found")


@app.get("/status")
async def status_page():
    """Serve the status page."""
    status_path = Path("ui/static/status.html")
    if status_path.exists():
        return FileResponse(status_path)
    raise HTTPException(status_code=404, detail="Status page not found")


@app.get("/api/logs/recent")
async def get_recent_logs(lines: int = 100):
    """Get recent log entries."""
    try:
        from code.config import OUTPUT_ROOT
        log_file = OUTPUT_ROOT / "pipeline.log"
        
        if not log_file.exists():
            return {"logs": [], "message": "No log file found"}
        
        # Read last N lines
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {"logs": [line.strip() for line in recent]}
        
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return {"logs": [], "error": str(e)}


# ============================================================================
# STAGES ENDPOINTS
# ============================================================================

@app.get("/api/stages")
async def get_stages():
    """Get list of all pipeline stages with their status."""
    stages = [
        {"name": "stage1", "title": "Data Profiling", "order": 1},
        {"name": "stage2", "title": "Task Proposal", "order": 2},
        {"name": "stage3", "title": "Execution Planning", "order": 3},
        {"name": "stage3b", "title": "Data Preparation", "order": 4},
        {"name": "stage3_5a", "title": "Method Proposal", "order": 5},
        {"name": "stage3_5b", "title": "Benchmarking", "order": 6},
        {"name": "stage4", "title": "Execution", "order": 7},
        {"name": "stage5", "title": "Visualization", "order": 8},
        {"name": "stage6", "title": "Final Report", "order": 9},
        {"name": "eda", "title": "EDA Output", "order": 10},
    ]
    
    # Add status for each stage
    for stage in stages:
        stage["status"] = infer_stage_status(stage["name"])
    
    return {"stages": stages}


@app.get("/api/stages/{stage_name}/outputs")
async def get_stage_outputs(stage_name: str):
    """Get list of output files for a specific stage."""
    from code.config import OUTPUT_ROOT
    
    # Map stage names to their output directories
    stage_dirs = {
        "stage1": OUTPUT_ROOT / "summaries",
        "stage2": OUTPUT_ROOT / "stage2_out",
        "stage3": OUTPUT_ROOT / "stage3_out",
        "stage3b": OUTPUT_ROOT / "stage3b_data_prep",
        "stage3_5a": OUTPUT_ROOT / "stage3_5a_method_proposal",
        "stage3_5b": OUTPUT_ROOT / "stage3_5b_benchmarking",
        "stage4": OUTPUT_ROOT / "stage4_out",
        "stage5": OUTPUT_ROOT / "stage5_out",
        "stage6": OUTPUT_ROOT / "stage6_out",
        "eda": OUTPUT_ROOT / "eda_out",
    }
    
    stage_dir = stage_dirs.get(stage_name)
    if not stage_dir or not stage_dir.exists():
        return {"stage_name": stage_name, "outputs": []}
    
    outputs = []
    for f in stage_dir.iterdir():
        if f.is_file():
            file_type = "json" if f.suffix == ".json" else "parquet" if f.suffix == ".parquet" else "image" if f.suffix in [".png", ".jpg"] else "other"
            outputs.append({
                "filename": f.name,
                "path": str(f.relative_to(OUTPUT_ROOT)),
                "type": file_type,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    
    # Sort by modification time, newest first
    outputs.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"stage_name": stage_name, "outputs": outputs}


@app.get("/api/tasks/status")
async def get_all_tasks_status():
    """Get status of all tasks."""
    state = tracker.get_state()
    stages = get_all_stages_status(state.get("task_id"))
    
    return {
        "current_task": state.get("task_id"),
        "is_running": state.get("is_running"),
        "stages": stages
    }


@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get status for a specific task."""
    stages = get_all_stages_status(task_id)
    return {
        "task_id": task_id,
        "stages": stages
    }


# ============================================================================
# EDA ENDPOINTS
# ============================================================================

class EDARequest(BaseModel):
    query: str

@app.post("/api/eda")
async def run_eda_query(request: EDARequest):
    """Run an EDA query and return results."""
    try:
        from code.eda_agent import run_eda
        
        logger.info(f"Running EDA query: {request.query[:50]}...")
        response = run_eda(request.query)
        
        return {
            "success": True,
            "answer": response.answer,
            "visualizations": [
                {
                    "filepath": v.filepath,
                    "title": v.title,
                    "plot_type": v.plot_type
                } for v in response.visualizations
            ] if response.visualizations else [],
            "new_datasets": response.new_datasets_detected
        }
        
    except Exception as e:
        logger.error(f"EDA query failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/eda/datasets")
async def get_eda_datasets():
    """List all available datasets for EDA."""
    try:
        from tools.eda_tools import list_all_datasets
        result = list_all_datasets.invoke({})
        return {"success": True, "datasets": result}
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/eda/visualizations")
async def get_eda_visualizations():
    """Get list of EDA visualizations."""
    try:
        if not EDA_OUT_DIR.exists():
            return {"visualizations": []}
        
        viz_files = []
        for f in EDA_OUT_DIR.glob("*.png"):
            viz_files.append({
                "filename": f.name,
                "path": f"/api/eda/image/{f.name}",
                "size_kb": f.stat().st_size // 1024,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
            })
        
        # Sort by creation time, newest first
        viz_files.sort(key=lambda x: x["created"], reverse=True)
        
        return {"visualizations": viz_files}
        
    except Exception as e:
        logger.error(f"Failed to list EDA visualizations: {e}")
        return {"visualizations": [], "error": str(e)}


@app.get("/api/eda/image/{filename}")
async def get_eda_image(filename: str):
    """Serve an EDA visualization image."""
    image_path = EDA_OUT_DIR / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/png")


# ============================================================================
# FILE SERVING ENDPOINTS
# ============================================================================

@app.get("/api/files/{filepath:path}")
async def get_file(filepath: str):
    """Serve output files (JSON content or file download)."""
    from code.config import OUTPUT_ROOT
    
    file_path = OUTPUT_ROOT / filepath
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Ensure file is within OUTPUT_ROOT (security check)
    try:
        file_path.resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # JSON files: return parsed content
    if file_path.suffix == ".json":
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}")
    
    # Images: serve as response
    if file_path.suffix in [".png", ".jpg", ".jpeg"]:
        media_type = "image/png" if file_path.suffix == ".png" else "image/jpeg"
        return FileResponse(file_path, media_type=media_type)
    
    # Other files: download
    return FileResponse(file_path, filename=file_path.name)


# ============================================================================
# DATA REFRESH ENDPOINT
# ============================================================================

@app.post("/api/data/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """
    Scan for new datasets and auto-summarize them.
    This runs Stage 1 (data profiling) for any datasets without summaries.
    """
    try:
        from code.utils import list_data_files, list_summary_files, profile_csv
        from code.config import DATA_DIR, SUMMARIES_DIR, DataPassingManager
        
        # Get all data files
        all_files = list_data_files(DATA_DIR)
        
        # Get existing summaries
        summary_files = list_summary_files(SUMMARIES_DIR)
        summarized_stems = {
            Path(sf).stem.replace('.summary', '') 
            for sf in summary_files
        }
        
        # Find new files
        new_files = []
        for f in all_files:
            stem = Path(f).stem
            if stem not in summarized_stems:
                new_files.append(f)
        
        if not new_files:
            return {
                "success": True,
                "message": "All datasets are already summarized",
                "new_datasets": [],
                "total_datasets": len(all_files)
            }
        
        # Summarize new datasets
        summarized = []
        errors = []
        
        for filename in new_files:
            try:
                dataset_path = DATA_DIR / filename
                logger.info(f"Auto-summarizing new dataset: {filename}")
                
                # Profile the dataset
                summary = profile_csv(dataset_path)
                
                # Save the summary
                summary_dict = summary.model_dump()
                output_name = f"{dataset_path.stem}.summary.json"
                
                DataPassingManager.save_artifact(
                    data=summary_dict,
                    output_dir=SUMMARIES_DIR,
                    filename=output_name,
                    metadata={"stage": "stage1", "type": "dataset_summary", "source": "auto_refresh"}
                )
                
                summarized.append({
                    "filename": filename,
                    "rows": summary.n_rows,
                    "columns": summary.n_cols,
                    "summary_path": str(SUMMARIES_DIR / output_name)
                })
                logger.info(f"Successfully summarized: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to summarize {filename}: {e}")
                errors.append({"filename": filename, "error": str(e)})
        
        return {
            "success": True,
            "message": f"Summarized {len(summarized)} new dataset(s)",
            "new_datasets": summarized,
            "errors": errors if errors else None,
            "total_datasets": len(all_files)
        }
        
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")
