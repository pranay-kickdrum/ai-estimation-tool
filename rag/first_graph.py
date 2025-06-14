from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional, List, Dict, Any
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
import os
from dotenv import load_dotenv
import tiktoken
import logging
from datetime import datetime
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langsmith.run_helpers import traceable
import textwrap
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
# from langchain.callbacks.tracers.langchain import wait_for_tracer

# Set up rich console
console = Console()

# Set up logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=False,
            show_path=False
        ),
        logging.FileHandler(f'conversation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("rich")

# Load environment variables
load_dotenv()

# Set up LangSmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-estimation-tool")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

if not LANGCHAIN_API_KEY:
    logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled.")
    client = None
    tracer = None
else:
    client = Client(
        api_url=LANGCHAIN_ENDPOINT,
        api_key=LANGCHAIN_API_KEY,
    )
    tracer = LangChainTracer(
        project_name=LANGCHAIN_PROJECT,
        client=client,
    )
    logger.info(f"LangSmith tracing enabled for project: {LANGCHAIN_PROJECT}")

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Add these constants at the top with other imports
MAX_CONTEXT_TOKENS = 2000  # Maximum tokens to keep in context
MAX_MEMORY_ITEMS = 5      # Maximum number of recent interactions to keep in full

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def log_token_usage(prompt: str, response: str, node_name: str):
    """Log token usage for an LLM invocation."""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
    total_tokens = prompt_tokens + response_tokens
    
    logger.info(f"Token usage in {node_name}:")
    logger.info(f"  Prompt tokens: {prompt_tokens}")
    logger.info(f"  Response tokens: {response_tokens}")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Estimated cost: ${(total_tokens/1000) * 0.03:.4f}")  # GPT-4 pricing

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

memory = MemorySaver()

def get_callback_manager():
    """Get callback manager with LangSmith tracing if available."""
    if tracer:
        return CallbackManager([tracer])
    return None

# Initialize LLM with tracing
llm = init_chat_model(
    "openai:gpt-4.1",
    callbacks=get_callback_manager()
)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

### Define state
class GraphState(TypedDict):
    topic: Optional[str]
    paragraph: Optional[str]
    bullets: Optional[list[str]]
    user_feedback: Optional[Literal["accept", "suggest"]]
    suggestion: Optional[str]
    messages: Annotated[list, add_messages]
    conversation_memory: Optional[dict]  # Store the conversation memory state

def format_message(role: str, content: str) -> str:
    """Format a message in a chat-like style."""
    role_colors = {
        "user": "blue",
        "assistant": "green",
        "system": "yellow"
    }
    color = role_colors.get(role.lower(), "white")
    
    # Wrap the content for better readability
    wrapped_content = textwrap.fill(content, width=80)
    
    # Create a panel for the message
    panel = Panel(
        Text(wrapped_content, style=color),
        title=f"[{color}]{role.upper()}[/{color}]",
        border_style=color,
        padding=(1, 2)
    )
    return panel

def log_memory_usage(messages: List[Dict[str, Any]], operation: str):
    """Log memory usage statistics in a clean format."""
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    utilization = (total_tokens/MAX_CONTEXT_TOKENS)*100
    
    # Create a memory usage panel
    memory_panel = Panel(
        Text.assemble(
            f"Messages: {len(messages)}/{MAX_MEMORY_ITEMS}\n",
            f"Tokens: {total_tokens}/{MAX_CONTEXT_TOKENS}\n",
            f"Utilization: {utilization:.1f}%",
            style="dim"
        ),
        title=f"[dim]Memory Usage ({operation})[/dim]",
        border_style="dim",
        padding=(1, 2)
    )
    console.print(memory_panel)

class CustomMemory:
    """Custom memory implementation using trim_messages for efficient message management."""
    
    def __init__(self, max_tokens: int = MAX_CONTEXT_TOKENS, max_messages: int = MAX_MEMORY_ITEMS):
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
        log_memory_usage(self.messages, "before_trim")
        self._trim_messages()
        log_memory_usage(self.messages, "after_trim")
    
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
    return CustomMemory(max_tokens=MAX_CONTEXT_TOKENS, max_messages=MAX_MEMORY_ITEMS)

def update_conversation_memory(state: GraphState, role: str, content: str) -> dict:
    """Update the conversation memory with a new message."""
    # Initialize memory if not exists
    if "conversation_memory" not in state or not state["conversation_memory"]:
        memory = create_memory()
    else:
        # Recreate memory from state
        memory = create_memory()
        for msg in state["conversation_memory"].get("messages", []):
            memory.add_message(msg["role"], msg["content"])
    
    # Add new message
    memory.add_message(role, content)
    
    # Return updated memory state
    return {
        "messages": memory.get_messages(),
        "output": content
    }

def get_conversation_context(state: GraphState) -> str:
    """Get the conversation context from memory with pretty printing."""
    if not state.get("conversation_memory"):
        return ""
    
    messages = state["conversation_memory"].get("messages", [])
    log_memory_usage(messages, "context_retrieval")
    
    # Format messages into a plain text string for context
    return "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    ])

def display_conversation_context(messages: List[Dict[str, Any]]):
    """Display the conversation context with pretty printing."""
    for msg in messages:
        console.print(format_message(msg["role"], msg["content"]))

@traceable(
    project_name=LANGCHAIN_PROJECT,
    tags=["generation", "paragraph"],
    metadata={
        "function_type": "content_generator",
        "version": "1.0.0",
        "importance": "high",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_memory_items": MAX_MEMORY_ITEMS
    }
)
def generate_paragraph(state: GraphState) -> GraphState:
    # If we have a suggestion, use it as the topic
    topic = state.get("suggestion") or state["topic"]
    
    # If we already have a paragraph and no suggestion, return the current state
    if state.get("paragraph") and not state.get("suggestion"):
        return state
    
    # Get pruned conversation context
    context = get_conversation_context(state)
    
    # Display current conversation context
    if state.get("conversation_memory"):
        display_conversation_context(state["conversation_memory"].get("messages", []))
    
    # Log context length before generation
    context_tokens = count_tokens(context)
    console.print(Panel(
        Text(f"Generating paragraph about: {topic}", style="bold blue"),
        title="[bold blue]GENERATION[/bold blue]",
        border_style="blue"
    ))
    
    prompt = f"""Generate a concise, informative paragraph about '{topic}'. 
    The paragraph should be well-structured and provide key insights about the topic.
    Focus on accuracy and clarity.
    
    Recent context (last {MAX_MEMORY_ITEMS} interactions):
    {context}"""
    
    # Update memory with user request
    conversation_memory = update_conversation_memory(
        state,
        "user",
        f"Generate a paragraph about {topic}"
    )
    
    response = llm_with_tools.invoke([{"role": "user", "content": prompt}])
    
    # Log token usage in a clean format
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response.content)
    total_tokens = prompt_tokens + response_tokens
    
    token_panel = Panel(
        Text.assemble(
            f"Prompt tokens: {prompt_tokens}\n",
            f"Response tokens: {response_tokens}\n",
            f"Total tokens: {total_tokens}\n",
            f"Estimated cost: ${(total_tokens/1000) * 0.03:.4f}",
            style="dim"
        ),
        title="[dim]Token Usage[/dim]",
        border_style="dim",
        padding=(1, 2)
    )
    console.print(token_panel)
    
    # Update memory with assistant response
    conversation_memory = update_conversation_memory(
        {**state, "conversation_memory": conversation_memory},
        "assistant",
        response.content
    )
    
    return {
        **state,
        "paragraph": response.content,
        "topic": topic,
        "messages": state.get("messages", []) + [
            {"role": "user", "content": f"Generate a paragraph about {topic}"},
            {"role": "assistant", "content": response.content}
        ],
        "conversation_memory": conversation_memory
    }

@traceable(
    project_name=LANGCHAIN_PROJECT,
    tags=["review", "feedback"],
    metadata={
        "function_type": "feedback_processor",
        "version": "1.0.0",
        "importance": "high",
        "max_context_tokens": MAX_CONTEXT_TOKENS
    }
)
def review_paragraph(state: GraphState) -> GraphState:
    # Display the current paragraph with pretty printing
    console.print(Panel(
        Text(state["paragraph"], style="green"),
        title="[green]CURRENT PARAGRAPH[/green]",
        border_style="green"
    ))
    
    # Get conversation context
    context = get_conversation_context(state)
    
    user_response = input("\nHow would you like to proceed with this paragraph?\nYou can provide feedback for improvements or let me know if you'd like to move forward: ").strip()
    
    # Update memory with user response
    conversation_memory = update_conversation_memory(
        state,
        "user",
        user_response
    )
    
    interpretation_prompt = f"""Given the following user response about a paragraph, determine if they want to:
    1. Suggest improvements or changes (return 'suggest')
    2. Accept and move forward (return 'accept')
    
    User response: "{user_response}"
    
    If the user wants suggestions, also provide specific improvement suggestions based on their feedback.
    Return in format: 'suggest: [suggestions]' or 'accept'"""
    
    # Log token count before interpretation
    logger.info(f"\nInterpreting user response: {user_response[:50]}...")
    logger.info(f"Context length: {count_tokens(context)} tokens")
    
    interpretation = llm_with_tools.invoke([{"role": "user", "content": interpretation_prompt}]).content.strip().lower()
    
    # Log token usage
    log_token_usage(interpretation_prompt, interpretation, "review_paragraph")
    
    # Update memory with interpretation
    conversation_memory = update_conversation_memory(
        {**state, "conversation_memory": conversation_memory},
        "assistant",
        interpretation
    )
    
    if interpretation.startswith("suggest:"):
        suggestion = interpretation.split(":", 1)[1].strip()
        
        # Update memory with suggestion
        conversation_memory = update_conversation_memory(
            {**state, "conversation_memory": conversation_memory},
            "user",
            f"I'd like the following improvements: {suggestion}"
        )
        
        # Clear the current paragraph to force regeneration
        return {
            **state,
            "user_feedback": "suggest",
            "suggestion": suggestion,
            "paragraph": None,  # Clear the current paragraph
            "messages": state.get("messages", []) + [
                {"role": "user", "content": user_response},
                {"role": "assistant", "content": interpretation},
                {"role": "user", "content": f"I'd like the following improvements: {suggestion}"}
            ],
            "conversation_memory": conversation_memory
        }
    
    # Interpret as acceptance
    conversation_memory = update_conversation_memory(
        {**state, "conversation_memory": conversation_memory},
        "user",
        "The paragraph looks good, let's proceed."
    )
    
    return {
        **state,
        "user_feedback": "accept",
        "messages": state.get("messages", []) + [
            {"role": "user", "content": user_response},
            {"role": "assistant", "content": interpretation},
            {"role": "user", "content": "The paragraph looks good, let's proceed."}
        ],
        "conversation_memory": conversation_memory
    }

@traceable(
    project_name=LANGCHAIN_PROJECT,
    tags=["generation", "bullets"],
    metadata={
        "function_type": "content_generator",
        "version": "1.0.0",
        "importance": "medium"
    }
)
def bullet_point_generator(state: GraphState) -> GraphState:
    paragraph = state["paragraph"]
    prompt = f"""Based on the following paragraph, generate 3-5 key bullet points that capture the main ideas:
    
    {paragraph}
    
    Format the response as a list of concise bullet points."""
    
    messages = state.get("messages", []) + [{"role": "user", "content": prompt}]
    response = llm_with_tools.invoke(messages)
    # Split the response into bullet points, handling different formats
    bullets = [line.strip().lstrip('•-*').strip() for line in response.content.split('\n') if line.strip()]
    return {
        **state,
        "bullets": bullets,
        "messages": messages + [{"role": "assistant", "content": response.content}]
    }

### Final Node: Pretty printer
def pretty_print(state: GraphState) -> GraphState:
    """Print the final output with pretty formatting."""
    # Print the paragraph in a panel
    console.print(Panel(
        Text(state["paragraph"], style="green"),
        title="[green]FINAL PARAGRAPH[/green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Format bullet points with proper spacing
    bullet_text = "\n".join([
        f"• {bullet}" for bullet in state["bullets"]
    ])
    
    # Print bullet points in a panel
    console.print(Panel(
        Text(bullet_text, style="blue"),
        title="[blue]KEY POINTS[/blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    return state

### Router based on review feedback
def router(state: GraphState) -> Literal["generate_paragraph", "bullet_point_generator"]:
    if state["user_feedback"] == "suggest":
        return "generate_paragraph"
    elif state["user_feedback"] == "accept":
        return "bullet_point_generator"
    else:
        raise ValueError(f"Invalid user feedback: {state['user_feedback']}")

### Build the graph
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("generate_paragraph", generate_paragraph)
builder.add_node("review", review_paragraph)
builder.add_node("bullet_point_generator", bullet_point_generator)
builder.add_node("pretty_print", pretty_print)
builder.add_node("tools", ToolNode(tools=tools))

# Define edges
builder.set_entry_point("generate_paragraph")
builder.add_edge("generate_paragraph", "review")
builder.add_conditional_edges(
    "review",
    router,
    {
        "generate_paragraph": "generate_paragraph",
        "bullet_point_generator": "bullet_point_generator"
    }
)
builder.add_edge("bullet_point_generator", "pretty_print")
builder.add_edge("pretty_print", END)

# Add tool edges
builder.add_conditional_edges(
    "generate_paragraph",
    tools_condition,
)
builder.add_conditional_edges(
    "bullet_point_generator",
    tools_condition,
)
builder.add_edge("tools", "generate_paragraph")
builder.add_edge("tools", "bullet_point_generator")

graph = builder.compile(checkpointer=memory)

def stream_graph_updates(initial_input: dict, thread_id: str):
    """Stream updates from the graph with pretty printing."""
    config = {"thread_id": thread_id}
    seen_messages = set()  # Track messages we've already displayed
    
    for event in graph.stream(initial_input, config):
        for value in event.values():
            if "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                # Create a unique identifier for the message
                if isinstance(last_message, dict):
                    message_id = f"{last_message.get('role')}:{last_message.get('content')}"
                    if (message_id not in seen_messages and 
                        last_message.get("role") == "assistant" and 
                        "content" in last_message and
                        not any(x in last_message["content"].lower() for x in 
                               ["token usage", "interpreting", "generating"])):
                        console.print(format_message("assistant", last_message["content"]))
                        seen_messages.add(message_id)
                elif hasattr(last_message, "content"):
                    message_id = f"{last_message.type}:{last_message.content}"
                    if (message_id not in seen_messages and
                        not any(x in last_message.content.lower() for x in 
                               ["token usage", "interpreting", "generating"])):
                        console.print(format_message("assistant", last_message.content))
                        seen_messages.add(message_id)

def main():
    print("Welcome to the Paragraph and Bullet Point Generator!")
    if tracer:
        print("LangSmith tracing is enabled. You can monitor the conversation at:")
        print(f"https://smith.langchain.com/projects/{LANGCHAIN_PROJECT}")
    print("This tool will help you generate a paragraph and bullet points about any topic.")
    print("Type 'quit' at any time to exit.")
    
    # Initialize memory
    memory = create_memory()
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        if topic.lower() == 'quit':
            print("Goodbye!")
            break
            
        thread_id = str(uuid.uuid4())
        initial_input = {
            "topic": topic,
            "messages": [{"role": "user", "content": f"I want to learn about {topic}"}],
            "conversation_memory": {"messages": memory.get_messages()}
        }
        
        try:
            # Print user's topic for context
            print(f"\nGenerating content about: {topic}")
            stream_graph_updates(initial_input, thread_id)
            
            # Update memory with the conversation
            if "conversation_memory" in initial_input:
                memory = create_memory()
                for msg in initial_input["conversation_memory"]["messages"]:
                    memory.add_message(msg["role"], msg["content"])
            
            # Ask if user wants to generate another
            again = input("\nWould you like to generate another? (yes/no): ").strip().lower()
            if again != 'yes':
                print("Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()