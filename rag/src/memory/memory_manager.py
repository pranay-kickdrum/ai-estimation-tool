"""Memory management module for the chatbot."""
from typing import List, Dict, Any
import tiktoken
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

logger = logging.getLogger("rich")

class CustomMemory:
    """Custom memory implementation using trim_messages for efficient message management."""
    
    def __init__(self, max_tokens: int = 2000, max_messages: int = 5):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        logger.info(f"Initialized CustomMemory with max_tokens={max_tokens}, max_messages={max_messages}")
    
    def _count_tokens(self, message: Any) -> int:
        """Count tokens in a message or text string."""
        if hasattr(message, 'content'):
            # Handle LangChain message objects
            text = message.content
        elif isinstance(message, str):
            # Handle plain strings
            text = message
        else:
            # Handle other cases by converting to string
            text = str(message)
        
        return len(self.tokenizer.encode(text))
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to memory."""
        if role == "user":
            message = HumanMessage(content=content)
        elif role == "assistant":
            message = AIMessage(content=content)
        else:
            message = SystemMessage(content=content)
        
        self.messages.append({"role": role, "content": content, "message": message})
        self._trim_messages()
    
    def _trim_messages(self) -> None:
        """Trim messages to stay within token and message limits."""
        # First trim by message count
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Convert to LangChain message format
        lc_messages = [msg["message"] for msg in self.messages]
        
        try:
            # Then trim by token count using langchain_core's trim_messages
            trimmed_messages = trim_messages(
                lc_messages,
                max_tokens=self.max_tokens,
                token_counter=self._count_tokens
            )
            
            # Update our messages list with trimmed messages
            self.messages = [
                {"role": msg.type, "content": msg.content, "message": msg}
                for msg in trimmed_messages
            ]
        except Exception as e:
            logger.error(f"Error trimming messages: {e}")
            # Fallback: just keep the last N messages
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in memory."""
        return self.messages
    
    def get_context(self) -> str:
        """Get formatted context from messages."""
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages
        ])
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = []

def create_memory() -> CustomMemory:
    """Create a new memory instance."""
    return CustomMemory()

def update_conversation_memory(state: Dict[str, Any], role: str, content: str) -> Dict[str, Any]:
    """Update the conversation memory with a new message."""
    # Get existing memory or create new one
    memory = state.get("conversation_memory")
    if not isinstance(memory, CustomMemory):
        memory = create_memory()
        state["conversation_memory"] = memory
    
    # Add new message
    memory.add_message(role, content)
    
    # Update state with new messages
    state["messages"] = memory.get_messages()
    
    return state 