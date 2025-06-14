"""Review node for analysis review and feedback.

This module contains the node function for reviewing and providing feedback on PRD analysis.
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

def review_analysis(state: EstimationState) -> EstimationState:
    """Review the analysis and get user feedback.
    
    This function:
    1. Displays the current analysis to the user
    2. Gets user feedback on the analysis
    3. Interprets the feedback to determine next steps
    4. Updates the state with feedback and next action
    
    Args:
        state: The current state containing the analysis to review
        
    Returns:
        Updated state containing:
        - review_feedback: The user's feedback
        - revision_requests: Any specific revision requests
        - revision_history: Updated history of revisions
    """
    try:
        # Initialize revision history if not present
        if "revision_history" not in state:
            state["revision_history"] = []
        
        # Get the current analysis
        analysis = state.get("prd_analysis", {})
        if not analysis:
            raise ValueError("No analysis found in state")
        
        # Display the analysis to the user
        console.print("\n[bold blue]Current Analysis:[/bold blue]")
        for section, items in analysis.items():
            if not section.startswith("_"):  # Skip metadata
                console.print(f"\n[bold green]{section}:[/bold green]")
                for item in items:
                    console.print(f"• {item}")
        
        # Get user feedback
        console.print("\n[bold yellow]Please review the analysis above and provide feedback.[/bold yellow]")
        console.print("You can:")
        console.print("• Accept the analysis as is")
        console.print("• Request updates to any sections (additions, removals, or modifications)")
        user_feedback = input("\nYour feedback: ").strip()
        
        # Create a prompt to interpret the feedback
        interpretation_prompt = f"""Based on the user's feedback below, determine if they want to accept the analysis or make updates.
Return ONLY one of these exact strings: "accept" or "update"

User feedback: {user_feedback}"""
        
        # Get interpretation from LLM
        interpretation_response = llm.invoke(interpretation_prompt)
        interpretation = interpretation_response.content.strip().lower()
        
        # Log the interpretation
        logger.info(f"Feedback interpretation: {interpretation}")
        
        # Store the interpreted feedback in state
        state["review_feedback"] = interpretation
        
        # Handle the interpretation
        if interpretation == "accept":
            logger.info("User accepted the analysis")
            state["revision_requests"] = []  # Clear any pending requests
            state["revision_history"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "accept",
                "feedback": user_feedback
            })
        else:
            # Default to update if not explicitly accept
            logger.info("User requested updates to the analysis")
            state["revision_requests"] = [{
                "timestamp": datetime.now().isoformat(),
                "feedback": user_feedback,
                "status": "pending"
            }]
            state["revision_history"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "update_requested",
                "feedback": user_feedback
            })
        
        # Update conversation memory
        update_conversation_memory(state, "user", user_feedback)
        update_conversation_memory(state, "assistant", f"Interpreted feedback as: {interpretation}")
        
    except Exception as e:
        logger.error(f"Error in review process: {e}")
        # Default to accept on error to maintain flow
        state["review_feedback"] = "Error occurred during review. Defaulting to accept."
        state["revision_requests"] = []
        state["revision_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "error_default_accept",
            "error": str(e)
        })
    
    return state 