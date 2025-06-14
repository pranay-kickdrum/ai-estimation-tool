"""Graph state definitions for the chatbot."""
from typing import TypedDict, Literal, Optional, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """State definition for the graph."""
    topic: Optional[str]
    paragraph: Optional[str]
    bullets: Optional[list[str]]
    user_feedback: Optional[Literal["accept", "suggest"]]
    suggestion: Optional[str]
    messages: Annotated[list, add_messages]
    conversation_memory: Optional[dict]  # Store the conversation memory state 