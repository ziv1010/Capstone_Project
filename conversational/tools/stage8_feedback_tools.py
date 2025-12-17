"""
Stage 8 Tools: Feedback Loop & Auto-Remediation

Tools for analyzing guardrails failures and triggering fixes.
Agent dynamically generates remediation strategies - no hardcoded fixes.
"""

import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE7_OUT_DIR, STAGE8_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR,
    DataPassingManager, logger
)


# ============================================================================
# GUARDRAILS ANALYSIS
# ============================================================================

@tool
def load_guardrails_for_feedback(plan_id: str) -> str:
    """
    Load guardrails report and analyze failure reasons.
    
    Returns detailed analysis of each test result including:
    - Pass/Fail/Warning status
    - Specific metrics that caused failure
    - Recommendations for remediation
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
    
    Returns:
        Detailed guardrails analysis for feedback loop
    """
    try:
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
        report_path = STAGE7_OUT_DIR / f"{task_id}_guardrails_report.json"
        
        if not report_path.exists():
            return f"Guardrails report not found at: {report_path}"
        
        report = DataPassingManager.load_artifact(report_path)
        
        result = ["=== GUARDRAILS ANALYSIS FOR FEEDBACK LOOP ===\n"]
        
        # Overall score
        validity_score = report.get("validity_score", 0)
        validity_label = report.get("validity_label", "UNKNOWN")
        
        result.append(f"Overall Validity: {validity_score:.1f}% ({validity_label})")
        
        if validity_score >= 75:
            result.append("\n✅ Model validity is HIGH - no remediation needed.")
            return "\n".join(result)
        
        result.append(f"\n⚠️ Validity score is {validity_label} - analyzing failures...\n")
        
        # Analyze each test
        tests = report.get("tests", {})
        failures = []
        warnings = []
        
        for test_name, test_data in tests.items():
            status = (test_data.get("status") or "").upper()
            details = test_data.get("details", {})
            reason = details.get("reason", "No reason provided")
            
            result.append(f"\n### {test_name}")
            result.append(f"Status: {status}")
            result.append(f"Reason: {reason}")
            
            if status == "FAIL":
                failures.append((test_name, details))
                result.append("→ REQUIRES REMEDIATION")
            elif status == "WARNING":
                warnings.append((test_name, details))
                result.append("→ Consider improvement")
        
        # Remediation recommendations
        result.append("\n\n=== REMEDIATION OPPORTUNITIES ===")
        
        for test_name, details in failures:
            result.append(f"\n{test_name}:")
            
            if "correlation" in test_name.lower():
                corr = details.get("prediction_actual_correlation", 0)
                result.append(f"  - Low correlation ({corr:.3f})")
                result.append("  - Try: Add lag features, try different algorithms, feature selection")
                
            elif "propensity" in test_name.lower():
                extreme = details.get("extreme_total_pct", 0)
                result.append(f"  - High extreme propensity ({extreme*100:.1f}%)")
                result.append("  - Try: Resample data, stratified training, add balancing features")
                
            elif "ipw" in test_name.lower():
                mae_cv = details.get("mae_cv", 0)
                result.append(f"  - High MAE variation ({mae_cv:.3f})")
                result.append("  - Try: Robust regression, quantile-based features, separate models per range")
                
            elif "residual" in test_name.lower():
                outlier_pct = details.get("outlier_pct", 0)
                skew = details.get("skewness", 0)
                result.append(f"  - Outliers: {outlier_pct:.1f}%, Skew: {skew:.2f}")
                result.append("  - Try: Log transform target, remove outliers, robust scaling")
        
        result.append(f"\n\nTotal failures: {len(failures)}")
        result.append(f"Total warnings: {len(warnings)}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error loading guardrails: {e}"


# ============================================================================
# REMEDIATION CODE EXECUTION
# ============================================================================

@tool
def execute_remediation_code(plan_id: str, remediation_code: str, description: str) -> str:
    """
    Execute dynamically generated remediation code.
    
    The code can:
    - Modify the prepared data (df)
    - Apply transformations
    - Remove outliers
    - Add rebalancing
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        remediation_code: Python code to execute for remediation
        description: Brief description of what the remediation does
    
    Returns:
        Result of code execution
    """
    try:
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        
        if not prepared_path.exists():
            return f"Prepared data not found at: {prepared_path}"
        
        df = pd.read_parquet(prepared_path)
        original_shape = df.shape
        
        result = [f"=== EXECUTING REMEDIATION CODE ===\n"]
        result.append(f"Description: {description}\n")
        result.append("Code:")
        result.append("```python")
        result.append(remediation_code)
        result.append("```\n")
        
        # Create execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
        }
        
        # Execute the code
        try:
            exec(remediation_code, exec_globals)
            df = exec_globals.get('df', df)
        except Exception as e:
            result.append(f"❌ EXECUTION ERROR: {e}")
            result.append(f"\nTraceback:\n{traceback.format_exc()}")
            return "\n".join(result)
        
        # Save modified data
        df.to_parquet(prepared_path, index=False)
        
        result.append(f"✅ SUCCESS: Remediation applied")
        result.append(f"\nOriginal shape: {original_shape}")
        result.append(f"New shape: {df.shape}")
        result.append(f"\nData saved to: {prepared_path}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error executing remediation code: {e}\n{traceback.format_exc()}"


# ============================================================================
# RE-RUN TRIGGERS
# ============================================================================

@tool
def update_method_constraints(plan_id: str, constraints: str) -> str:
    """
    Update constraints for method proposal stage (Stage 3.5A).
    
    This influences the next re-run by adding constraints like:
    - Prefer robust algorithms
    - Use ensemble methods
    - Focus on specific feature types
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        constraints: New constraints to apply for method selection
    
    Returns:
        Confirmation of constraint update
    """
    try:
        constraint_file = STAGE8_OUT_DIR / f"{plan_id}_method_constraints.json"
        
        constraint_data = {
            "plan_id": plan_id,
            "constraints": constraints,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        DataPassingManager.save_artifact(constraint_data, STAGE8_OUT_DIR, f"{plan_id}_method_constraints.json")
        
        result = [f"=== METHOD CONSTRAINTS UPDATED ===\n"]
        result.append(f"Plan ID: {plan_id}")
        result.append(f"Constraints: {constraints}")
        result.append(f"\nSaved to: {constraint_file}")
        result.append("\nThese constraints will be applied when Stage 3.5A is re-run.")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error updating constraints: {e}"


@tool
def mark_for_rerun(plan_id: str, stages_to_rerun: str, reason: str) -> str:
    """
    Mark specific stages for re-execution.
    
    After remediation, this marks which stages need to be re-run
    to apply the fixes.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        stages_to_rerun: Comma-separated stage names (e.g., "stage3_5a,stage3_5b,stage4")
        reason: Reason for re-running
    
    Returns:
        Confirmation of stages marked for re-run
    """
    try:
        rerun_file = STAGE8_OUT_DIR / f"{plan_id}_rerun_request.json"
        
        stages = [s.strip() for s in stages_to_rerun.split(",")]
        
        rerun_data = {
            "plan_id": plan_id,
            "stages_to_rerun": stages,
            "reason": reason,
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "pending"
        }
        
        DataPassingManager.save_artifact(rerun_data, STAGE8_OUT_DIR, f"{plan_id}_rerun_request.json")
        
        result = [f"=== STAGES MARKED FOR RE-RUN ===\n"]
        result.append(f"Plan ID: {plan_id}")
        result.append(f"Stages: {stages}")
        result.append(f"Reason: {reason}")
        result.append(f"\nSaved to: {rerun_file}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error marking for rerun: {e}"


@tool
def save_feedback_report(
    plan_id: str,
    original_score: float,
    issues_found: str,
    remediations_applied: str,
    stages_to_rerun: str,
    expected_improvement: str
) -> str:
    """
    Save the feedback loop report.
    
    Documents what issues were found, what fixes were applied,
    and what stages are queued for re-run.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        original_score: Original validity score from guardrails
        issues_found: Description of issues identified
        remediations_applied: Description of remediations applied
        stages_to_rerun: Which stages will be re-run
        expected_improvement: Expected improvement after remediation
    
    Returns:
        Path to saved feedback report
    """
    try:
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
        
        report = {
            "task_id": task_id,
            "plan_id": plan_id,
            "original_validity_score": original_score,
            "issues_found": issues_found,
            "remediations_applied": remediations_applied,
            "stages_to_rerun": stages_to_rerun,
            "expected_improvement": expected_improvement,
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "awaiting_rerun"
        }
        
        output_path = DataPassingManager.save_artifact(
            report,
            STAGE8_OUT_DIR,
            f"{task_id}_feedback_report.json"
        )
        
        logger.info(f"Feedback report saved to {output_path}")
        
        result = [f"=== FEEDBACK REPORT SAVED ===\n"]
        result.append(f"Task ID: {task_id}")
        result.append(f"Original Score: {original_score:.1f}%")
        result.append(f"Issues: {issues_found}")
        result.append(f"Remediations: {remediations_applied}")
        result.append(f"Stages to Re-run: {stages_to_rerun}")
        result.append(f"Expected Improvement: {expected_improvement}")
        result.append(f"\nReport saved to: {output_path}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error saving feedback report: {e}"


# ============================================================================
# AUTOMATIC RE-RUN WITH NEW TASK VERSION
# ============================================================================

@tool
def clone_task_for_rerun(plan_id: str) -> str:
    """
    Clone a task with a new version ID for re-running.
    
    Creates a new task ID (e.g., TSK-005 → TSK-005-R1, TSK-005-R1 → TSK-005-R2)
    and copies necessary files so the original is preserved.
    
    Args:
        plan_id: Original plan ID (e.g., PLAN-TSK-005)
    
    Returns:
        New plan ID for the re-run version
    """
    try:
        import shutil
        import re
        from code.config import STAGE3B_OUT_DIR, STAGE3_OUT_DIR, STAGE3_5A_OUT_DIR
        
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
        
        # Determine new version number
        # Pattern: TSK-005, TSK-005-R1, TSK-005-R2, etc.
        match = re.match(r"(TSK-\d+)(-R(\d+))?", task_id)
        if match:
            base_id = match.group(1)
            current_version = int(match.group(3)) if match.group(3) else 0
            new_version = current_version + 1
            new_task_id = f"{base_id}-R{new_version}"
        else:
            new_task_id = f"{task_id}-R1"
        
        new_plan_id = f"PLAN-{new_task_id}"
        
        result = [f"=== CLONING TASK FOR RE-RUN ===\n"]
        result.append(f"Original: {task_id} → New: {new_task_id}")
        
        # Copy prepared data with new plan ID
        original_prepared = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        new_prepared = STAGE3B_OUT_DIR / f"prepared_{new_plan_id}.parquet"
        
        if original_prepared.exists():
            shutil.copy2(original_prepared, new_prepared)
            result.append(f"✅ Copied prepared data to: {new_prepared.name}")
        
        # Copy Stage 3 plan if exists
        original_plan = STAGE3_OUT_DIR / f"{plan_id}_plan.json"
        new_plan = STAGE3_OUT_DIR / f"{new_plan_id}_plan.json"
        
        if original_plan.exists():
            # Load, update task_id, and save
            plan_data = DataPassingManager.load_artifact(original_plan)
            plan_data["task_id"] = new_task_id
            plan_data["original_task_id"] = task_id
            plan_data["is_rerun"] = True
            DataPassingManager.save_artifact(plan_data, STAGE3_OUT_DIR, f"{new_plan_id}_plan.json")
            result.append(f"✅ Copied plan to: {new_plan.name}")
        
        # Copy method constraints if they exist
        original_constraints = STAGE8_OUT_DIR / f"{plan_id}_method_constraints.json"
        new_constraints = STAGE8_OUT_DIR / f"{new_plan_id}_method_constraints.json"
        
        if original_constraints.exists():
            shutil.copy2(original_constraints, new_constraints)
            result.append(f"✅ Copied method constraints")
        
        # Save clone metadata
        clone_data = {
            "original_task_id": task_id,
            "original_plan_id": plan_id,
            "new_task_id": new_task_id,
            "new_plan_id": new_plan_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        DataPassingManager.save_artifact(clone_data, STAGE8_OUT_DIR, f"{new_task_id}_clone_info.json")
        
        result.append(f"\n✅ Task cloned successfully")
        result.append(f"New Plan ID: {new_plan_id}")
        result.append(f"New Task ID: {new_task_id}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error cloning task: {e}\n{traceback.format_exc()}"


@tool
def create_rerun_task(
    new_plan_id: str, 
    original_task_description: str,
    improvement_description: str,
    target_column: str
) -> str:
    """
    Create a new task entry for re-run with improvements.
    
    Instead of auto-running stages, this adds a new task to the task_proposals.json
    so the user can run it via chat by typing "run <task_id>".
    
    Args:
        new_plan_id: New plan ID from clone_task_for_rerun (e.g., PLAN-TSK-005-R1)
        original_task_description: Original task description
        improvement_description: Description of improvements/remediations applied
        target_column: Target column for the task
    
    Returns:
        Instructions for user to run the new task
    """
    try:
        from code.config import STAGE2_OUT_DIR
        import json
        
        new_task_id = new_plan_id.replace("PLAN-", "") if new_plan_id.startswith("PLAN-") else new_plan_id
        
        # Load existing task proposals - PRESERVE ORIGINAL STRUCTURE
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        
        if proposals_path.exists():
            # Load raw JSON to preserve structure
            with open(proposals_path, 'r') as f:
                raw_data = json.load(f)
            
            # Handle DataPassingManager format (with _meta and data)
            if isinstance(raw_data, dict) and "data" in raw_data:
                proposals = raw_data["data"]
            else:
                proposals = raw_data
            
            # Extract tasks array
            if isinstance(proposals, dict) and "tasks" in proposals:
                tasks = proposals["tasks"]
            elif isinstance(proposals, list):
                tasks = proposals
            else:
                tasks = []
        else:
            tasks = []
            proposals = {"tasks": tasks}
        
        # Create new task entry
        new_task = {
            "id": new_task_id,
            "task_id": new_task_id,
            "title": f"[RERUN] {original_task_description[:50]}...",
            "description": f"RERUN WITH IMPROVEMENTS:\n{improvement_description}\n\nOriginal: {original_task_description}",
            "target_column": target_column,
            "complexity": "medium",
            "is_rerun": True,
            "original_plan_id": new_plan_id.replace(f"-R{new_plan_id.split('-R')[-1]}", "") if "-R" in new_plan_id else new_plan_id,
            "improvements": improvement_description,
            "status": "ready_to_run"
        }
        
        # Check if task already exists to avoid duplicates
        existing_ids = [t.get("id", "") or t.get("task_id", "") for t in tasks]
        if new_task_id in existing_ids:
            logger.info(f"Task {new_task_id} already exists in proposals, not adding duplicate")
        else:
            # APPEND to existing tasks
            tasks.append(new_task)
            logger.info(f"Appended task {new_task_id} to proposals (now {len(tasks)} tasks)")
        
        # Update the proposals dict
        if isinstance(proposals, dict):
            proposals["tasks"] = tasks
            proposals["timestamp"] = pd.Timestamp.now().isoformat()
        else:
            proposals = {"tasks": tasks, "timestamp": pd.Timestamp.now().isoformat()}
        
        # Save using DataPassingManager to maintain format
        DataPassingManager.save_artifact(proposals, STAGE2_OUT_DIR, "task_proposals.json")
        
        result = [f"=== NEW TASK CREATED FOR RE-RUN ===\n"]
        result.append(f"Task ID: {new_task_id}")
        result.append(f"Target: {target_column}")
        result.append(f"\nImprovements Applied:")
        result.append(improvement_description)
        result.append(f"\n✅ Task added to task_proposals.json")
        result.append(f"\n" + "="*50)
        result.append(f"📌 TO RUN THIS TASK:")
        result.append(f'   Type in chat: "run {new_task_id}"')
        result.append(f"   or: \"execute task {new_task_id}\"")
        result.append(f"="*50)
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error creating rerun task: {e}\n{traceback.format_exc()}"


# Export tools
STAGE8_FEEDBACK_TOOLS = [
    load_guardrails_for_feedback,
    execute_remediation_code,
    update_method_constraints,
    mark_for_rerun,
    save_feedback_report,
    clone_task_for_rerun,
    create_rerun_task,  # Replaced trigger_auto_rerun
]

