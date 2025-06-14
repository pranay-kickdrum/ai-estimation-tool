"""Graph node functions for the chatbot."""
from typing import Dict, Any, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.run_helpers import traceable
from ..config import (
    MODEL_NAME, TEMPERATURE, NODE_CONFIGS,
    SYSTEM_PROMPT, PARAGRAPH_GENERATION_PROMPT,
    REVIEW_PROMPT, BULLET_POINT_PROMPT
)
from ..utils.logging_utils import log_token_usage
from ..utils.formatting import display_conversation_context

# Initialize the language model
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE
)

@traceable(
    project_name="content_generation",
    tags=["generation", "paragraph"],
    metadata=NODE_CONFIGS["generate_paragraph"]
)
def generate_paragraph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a paragraph based on the topic."""
    topic = state.get("topic", "")
    if not topic:
        return {"paragraph": "No topic provided."}
    
    # Get conversation context
    messages = state.get("messages", [])
    # Handle both dictionary and message object formats
    context = "\n".join([
        f"{msg.type if hasattr(msg, 'type') else msg['role']}: {msg.content if hasattr(msg, 'content') else msg['content']}"
        for msg in messages
    ])
    
    # Create the prompt
    prompt = PARAGRAPH_GENERATION_PROMPT.format(topic=topic)
    if context:
        prompt += f"\n\nContext from previous conversation:\n{context}"
    
    # Generate the paragraph
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    paragraph = response.content
    
    # Log token usage
    log_token_usage(prompt, paragraph, "generate_paragraph")
    
    return {"paragraph": paragraph}

@traceable(
    project_name="content_review",
    tags=["review", "feedback"],
    metadata=NODE_CONFIGS["review_paragraph"]
)
def review_paragraph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Review the generated paragraph and provide feedback."""
    paragraph = state.get("paragraph", "")
    if not paragraph:
        return {"suggestion": "No paragraph to review."}
    
    # Get user feedback if any
    user_feedback = state.get("user_feedback", "")
    feedback_context = f"\nUser feedback: {user_feedback}" if user_feedback else ""
    
    # Create the prompt
    prompt = REVIEW_PROMPT.format(paragraph=paragraph) + feedback_context
    
    # Generate review
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    suggestion = response.content
    
    # Log token usage
    log_token_usage(prompt, suggestion, "review_paragraph")
    
    return {"suggestion": suggestion}

@traceable(
    project_name="content_extraction",
    tags=["extraction", "bullet_points"],
    metadata=NODE_CONFIGS["bullet_point_generator"]
)
def bullet_point_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate bullet points from the paragraph."""
    paragraph = state.get("paragraph", "")
    if not paragraph:
        return {"bullets": ["No paragraph available for bullet points."]}
    
    # Create the prompt
    prompt = BULLET_POINT_PROMPT.format(paragraph=paragraph)
    
    # Generate bullet points
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    bullet_text = response.content
    
    # Parse bullet points
    bullets = [
        line.strip("- ").strip()
        for line in bullet_text.split("\n")
        if line.strip().startswith("-") or line.strip().startswith("•")
    ]
    
    # Log token usage
    log_token_usage(prompt, bullet_text, "bullet_point_generator")
    
    return {"bullets": bullets}

def should_continue(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if the conversation should continue."""
    user_feedback = state.get("user_feedback", "").lower()
    
    # Check for explicit end signals
    if any(signal in user_feedback for signal in ["end", "stop", "done", "finish"]):
        return {"next": "end", "messages": state.get("messages", [])}
    
    # Check for revision requests
    if any(signal in user_feedback for signal in ["revise", "change", "modify", "edit"]):
        return {"next": "generate", "paragraph": None, "bullets": None}
    
    # Default to continuing if we have a paragraph
    return {"next": "generate" if state.get("paragraph") else "end", **state} 