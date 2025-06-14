"""Revision node for analysis updates.

This module contains the node function for revising the PRD analysis based on user feedback.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

from core import (
    EstimationState,
    llm,
    logger,
    console,
    log_token_usage,
    update_conversation_memory
)
from file_ops import save_analysis_to_file, load_analysis_from_file

def revise_analysis(state: EstimationState) -> EstimationState:
    """Revise the analysis based on user feedback.
    
    This function:
    1. Gets the current analysis and revision requests
    2. Generates a revised analysis using the LLM
    3. Validates and saves the revised analysis
    4. Updates the state with the new analysis
    
    Args:
        state: The current state containing the analysis and revision requests
        
    Returns:
        Updated state containing:
        - prd_analysis: The revised analysis
        - current_analysis_file: Path to the new analysis file
        - revision_history: Updated history of revisions
    """
    try:
        # Get the current analysis
        current_analysis = state.get("prd_analysis", {})
        if not current_analysis:
            raise ValueError("No analysis found in state")
        
        # Get revision requests
        revision_requests = state.get("revision_requests", [])
        if not revision_requests:
            logger.info("No revision requests found, defaulting to accept")
            return state
        
        # Get the latest revision request
        latest_request = revision_requests[-1]
        user_feedback = latest_request.get("feedback", "")
        
        # Create a prompt for the LLM to revise the analysis
        revision_prompt = f"""You are an expert project analyst. Revise the following analysis based on the user's feedback.
Maintain the same structure and format, but update the content as requested.

Current Analysis:
{json.dumps(current_analysis, indent=2)}

User Feedback:
{user_feedback}

Instructions:
1. Process all feedback as updates to the analysis
2. Keep the same JSON structure with these sections:
   - Project Overview and Objectives
   - Key Components/Modules
   - Timeline and Milestones
   - Critical Assumptions
   - Dependencies and Constraints
   - Resource Requirements
   - Risk Factors
3. For each section, provide a list of items
4. Maintain any existing metadata
5. Ensure all updates are clear and actionable

Return the complete revised analysis as a JSON object."""
        
        # Get revised analysis from LLM
        revision_response = llm.invoke(revision_prompt)
        
        try:
            # Parse the response as JSON
            revised_analysis = json.loads(revision_response)
            
            # Validate the structure
            required_sections = [
                "Project Overview and Objectives",
                "Key Components/Modules",
                "Timeline and Milestones",
                "Critical Assumptions",
                "Dependencies and Constraints",
                "Resource Requirements",
                "Risk Factors"
            ]
            
            # Check if all required sections are present
            missing_sections = [section for section in required_sections 
                              if section not in revised_analysis]
            if missing_sections:
                raise ValueError(f"Missing required sections: {missing_sections}")
            
            # Ensure each section is a list
            for section in required_sections:
                if not isinstance(revised_analysis[section], list):
                    revised_analysis[section] = [str(revised_analysis[section])]
            
            # Preserve metadata if present
            if "_metadata" in current_analysis:
                revised_analysis["_metadata"] = current_analysis["_metadata"]
                revised_analysis["_metadata"]["last_revision"] = datetime.now().isoformat()
                revised_analysis["_metadata"]["revision_feedback"] = user_feedback
            
            # Save the revised analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_analysis_to_file(revised_analysis, timestamp)
            
            # Update state
            state["prd_analysis"] = revised_analysis
            state["current_analysis_file"] = filepath
            state["revision_history"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "revision_completed",
                "feedback": user_feedback,
                "file": filepath
            })
            
            # Log success
            logger.info("Analysis revised successfully")
            logger.info(f"Revised analysis saved to: {filepath}")
            
            # Update conversation memory
            update_conversation_memory(state, "assistant", "Analysis has been revised based on your feedback.")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse revised analysis as JSON: {e}")
            raise ValueError("Invalid JSON response from LLM")
        
    except Exception as e:
        logger.error(f"Error in revision process: {e}")
        # Try to recover by loading the last known good analysis
        try:
            if state.get("current_analysis_file"):
                recovered_analysis = load_analysis_from_file(state["current_analysis_file"])
                state["prd_analysis"] = recovered_analysis
                logger.info("Recovered last known good analysis")
        except Exception as recovery_error:
            logger.error(f"Failed to recover analysis: {recovery_error}")
        
        state["revision_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "revision_failed",
            "error": str(e)
        })
    
    return state 