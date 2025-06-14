"""Clarification node for generating requirement clarification questions.

This module contains the node function for generating specific, actionable questions
to clarify requirements in the PRD analysis.
"""

import os
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
from file_ops import save_clarification_questions

def generate_clarification_questions(state: EstimationState) -> EstimationState:
    """Generate specific, actionable questions for requirement clarification.
    
    This function analyzes the accepted PRD analysis to identify areas that need
    clarification for accurate estimation. It generates focused questions based on
    technical requirements, deployment scenarios, integration details, and scope boundaries.
    
    Args:
        state: Current state containing the accepted analysis
        
    Returns:
        Updated state with:
        - clarification_questions: Generated questions and summary
        - clarification_file: Path to saved questions file
        - clarification_responses: Empty list for future responses
        - clarification_history: Initial history entry
        - Updated messages and conversation memory
    """
    try:
        # Get the accepted analysis
        analysis = state["prd_analysis"]
        if not analysis:
            raise ValueError("No analysis results found in state")
        
        # Create a prompt for generating clarification questions
        prompt = f"""As a Senior Technical Analyst at Kickdrum, analyze this PRD analysis and generate specific, 
        actionable questions that stakeholders need to answer for accurate project estimation.
        
        Current Analysis:
        {json.dumps(analysis, indent=2)}
        
        Generate questions following these rules:
        1. Reference specific sections/requirements
        2. One focused question per unclear point
        3. Technical and specific - avoid generic questions
        4. Answerable in 1-2 concrete sentences
        5. Focus on items that could change effort by >4 hours
        
        Priority Areas to Examine:
        - Deployment & Infrastructure
        - Integration & APIs
        - Scope Boundaries
        - Technical Specifications
        - Data & Security
        - User Experience
        
        Return a JSON object with this structure:
        {{
            "clarification_questions": [
                {{
                    "reference": "Exact requirement ID, section title, or quoted text",
                    "question": "Specific, direct question",
                    "category": "technical|business|integration|deployment|data|security",
                    "effort_variance": "Estimated hour range difference",
                    "assumptions_if_unresolved": "What assumptions will be made if unanswered",
                    "priority": "high|medium|low",
                    "status": "pending",  # Add status field for tracking
                    "question_id": "Q1",  # Add unique ID for each question
                    "created_at": "timestamp"  # Add creation timestamp
                }}
            ],
            "summary": {{
                "total_questions": "Number of questions",
                "high_priority_count": "Number of high-priority questions",
                "main_risk_areas": ["List of primary areas with unclear requirements"],
                "estimation_confidence": "low|medium|high",
                "generated_at": "timestamp"  # Add generation timestamp
            }}
        }}
        
        Return ONLY the JSON object, no additional text."""
        
        # Get questions from LLM
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Log token usage
        log_token_usage(prompt, response.content, "generate_clarification_questions")
        
        # Clean and parse the response
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            clarification_data = json.loads(content)
            
            # Validate the structure
            required_keys = ["clarification_questions", "summary"]
            for key in required_keys:
                if key not in clarification_data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Add metadata to questions
            timestamp = datetime.now().isoformat()
            for i, q in enumerate(clarification_data["clarification_questions"], 1):
                q["question_id"] = f"Q{i}"
                q["status"] = "pending"
                q["created_at"] = timestamp
            
            # Add metadata to summary
            clarification_data["summary"]["generated_at"] = timestamp
            
            # Display the questions with pretty printing
            console.print("\n[bold blue]Requirement Clarification Questions[/bold blue]\n")
            console.print(Panel(
                Text("Review and respond to the following questions to clarify project requirements and reduce estimation uncertainty.", style="white"),
                border_style="blue"
            ))
            
            # Group questions by category
            questions_by_category = {}
            for q in clarification_data["clarification_questions"]:
                category = q["category"]
                if category not in questions_by_category:
                    questions_by_category[category] = []
                questions_by_category[category].append(q)
            
            # Display questions by category
            for category, questions in questions_by_category.items():
                console.print(f"\n[bold]{category.title()} Questions[/bold]")
                for i, q in enumerate(questions, 1):
                    priority_color = {
                        "high": "red",
                        "medium": "yellow",
                        "low": "green"
                    }.get(q["priority"], "white")
                    
                    status_color = {
                        "pending": "yellow",
                        "answered": "green",
                        "updated": "blue"
                    }.get(q["status"], "white")
                    
                    # Format the question panel with better spacing and alignment
                    question_text = Text.assemble(
                        (f"ID: {q['question_id']}\n", "bold"),
                        (f"Reference: {q['reference']}\n", "bold"),
                        (f"Question: {q['question']}\n", "bold"),
                        (f"Effort Variance: {q['effort_variance']}\n", "bold"),
                        (f"Assumptions if Unresolved: {q['assumptions_if_unresolved']}\n", "bold"),
                        (f"Priority: {q['priority'].upper()}\n", f"bold {priority_color}"),
                        (f"Status: {q['status'].upper()}", f"bold {status_color}")
                    )
                    
                    console.print(Panel(
                        question_text,
                        border_style=priority_color,
                        padding=(1, 2),
                        width=100  # Set a fixed width for better alignment
                    ))
            
            # Display summary
            summary = clarification_data["summary"]
            console.print("\n[bold]Summary[/bold]")
            console.print(Panel(
                Text.assemble(
                    (f"Total Questions: {summary['total_questions']}\n", "bold"),
                    (f"High Priority Questions: {summary['high_priority_count']}\n", "bold"),
                    (f"Main Risk Areas: {', '.join(summary['main_risk_areas'])}\n", "bold"),
                    (f"Estimation Confidence: {summary['estimation_confidence'].upper()}\n", "bold"),
                    (f"Generated: {summary['generated_at']}", "bold")
                ),
                border_style="blue",
                padding=(1, 2),
                width=100
            ))
            
            # Save the questions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_clarification_questions(clarification_data["clarification_questions"], timestamp)
            
            # Create initial history entry
            history_entry = {
                "timestamp": timestamp,
                "action": "generated",
                "questions_count": len(clarification_data["clarification_questions"]),
                "high_priority_count": summary["high_priority_count"],
                "estimation_confidence": summary["estimation_confidence"]
            }
            
            # Update state with all clarification-related fields
            return {
                **state,
                "clarification_questions": clarification_data,
                "clarification_file": filepath,
                "clarification_responses": [],  # Initialize empty responses list
                "clarification_history": [history_entry],  # Initialize history with first entry
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": f"Generated {summary['total_questions']} clarification questions. Results saved to: {filepath}"}
                ],
                "conversation_memory": update_conversation_memory(
                    state,
                    "assistant",
                    f"Generated {summary['total_questions']} clarification questions for requirement analysis"
                )
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse clarification questions: {e}")
            logger.error(f"Raw response: {content}")
            raise ValueError("Failed to generate valid clarification questions")
            
    except Exception as e:
        logger.error(f"Error generating clarification questions: {e}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                {"role": "system", "content": f"Error generating clarification questions: {str(e)}"}
            ]
        } 