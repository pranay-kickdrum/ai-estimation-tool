"""Summary node for analysis summarization.

This module contains the node function for creating a final summary of the PRD analysis.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from core import (
    EstimationState,
    llm,
    logger,
    console,
    log_token_usage,
    update_conversation_memory
)
from file_ops import save_analysis_to_file

def summarize_analysis(state: EstimationState) -> EstimationState:
    """Create a comprehensive summary of the accepted analysis.
    
    This function:
    1. Takes the final accepted analysis
    2. Generates a comprehensive summary including:
       - Key points from each section
       - Summary of changes made during revisions
       - Final recommendations
       - Total items and sections analyzed
       - Time taken for analysis
    3. Saves the summary to a file
    4. Updates the state with the summary
    
    Args:
        state: The current state containing the accepted analysis
        
    Returns:
        Updated state containing:
        - analysis_summary: The generated summary
        - summary_file: Path to the saved summary file
    """
    try:
        # Get the final analysis
        analysis = state.get("prd_analysis", {})
        if not analysis:
            raise ValueError("No analysis found in state")
        
        # Get revision history for context
        revision_history = state.get("revision_history", [])
        
        # Create a prompt for generating the summary
        summary_prompt = f"""Create a comprehensive summary of the following PRD analysis.
Include key insights, recommendations, and metrics.

Analysis:
{json.dumps(analysis, indent=2)}

Revision History:
{json.dumps(revision_history, indent=2)}

Instructions:
1. Create a structured summary with these sections:
   - Executive Summary
   - Key Findings by Section
   - Revision History Summary
   - Final Recommendations
   - Analysis Metrics
2. For each section, provide clear, actionable insights
3. Include specific metrics where relevant
4. Highlight any critical points or risks
5. Format the output as a JSON object with these sections as keys

Return the complete summary as a JSON object."""
        
        # Get summary from LLM and extract content
        summary_response = llm.invoke(summary_prompt)
        summary_content = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
        try:
            # Parse the response content as JSON
            summary = json.loads(summary_content)
            
            # Add metadata
            summary["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "analysis_file": state.get("current_analysis_file", ""),
                "total_revisions": len(revision_history),
                "total_items": sum(len(v) for k, v in analysis.items() 
                                 if not k.startswith("_")),
                "sections_analyzed": [k for k in analysis.keys() 
                                    if not k.startswith("_")]
            }
            
            # Save the summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_analysis_to_file(summary, timestamp, prefix="summary_")
            
            # Update state
            state["analysis_summary"] = summary
            state["summary_file"] = filepath
            
            # Display the summary
            console.print("\n[bold blue]Analysis Summary:[/bold blue]")
            
            # Create a table for metrics
            metrics_table = Table(title="Analysis Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            
            metrics_table.add_row("Total Items", str(summary["_metadata"]["total_items"]))
            metrics_table.add_row("Sections Analyzed", str(len(summary["_metadata"]["sections_analyzed"])))
            metrics_table.add_row("Total Revisions", str(summary["_metadata"]["total_revisions"]))
            
            console.print(metrics_table)
            
            # Display each section of the summary
            for section, content in summary.items():
                if not section.startswith("_"):
                    console.print(f"\n[bold green]{section}:[/bold green]")
                    if isinstance(content, list):
                        for item in content:
                            console.print(f"• {item}")
                    else:
                        console.print(content)
            
            # Log success
            logger.info("Analysis summary generated successfully")
            logger.info(f"Summary saved to: {filepath}")
            
            # Update conversation memory
            update_conversation_memory(state, "assistant", "Analysis summary has been generated.")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary as JSON: {e}")
            logger.error(f"Raw response content: {summary_content}")
            raise ValueError("Invalid JSON response from LLM")
        
    except Exception as e:
        logger.error(f"Error in summary generation: {e}")
        state["analysis_summary"] = {
            "error": str(e),
            "status": "failed",
            "_metadata": {
                "error_timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__
            }
        }
    
    return state 