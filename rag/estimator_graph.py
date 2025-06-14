"""Main graph module for the estimation tool.

This module defines the graph structure and workflow for analyzing PRDs
and generating estimates.
"""

from pprint import pprint
from typing import Dict, Any, List, TypedDict, Literal, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
import tiktoken
import textwrap
import json

from core import (
    EstimationState,
    llm,
    logger,
    console,
    log_token_usage,
    update_conversation_memory,
    get_conversation_context,
    display_conversation_context,
    DATA_PATH
)
from file_ops import save_analysis_to_file
from nodes import (
    analyze_prd,
    review_analysis,
    revise_analysis,
    summarize_analysis,
    generate_clarification_questions
)
from IPython.display import Image, display
# Constants for memory management
MAX_CONTEXT_TOKENS = 2000  # Maximum tokens to keep in context
MAX_MEMORY_ITEMS = 5      # Maximum number of recent interactions to keep in full

tokenizer = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

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

def update_conversation_memory(state: EstimationState, role: str, content: str) -> dict:
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

def get_conversation_context(state: EstimationState) -> str:
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
        )
        # Commenting out file logging for now
        # logging.FileHandler(f'estimation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("rich")

# Load environment variables
load_dotenv()

# Get data directory from environment variable
DATA_PATH = os.getenv("DATA_PATH")
if not DATA_PATH:
    raise EnvironmentError("DATA_PATH environment variable is not set. Please set it to the path of your data directory.")

# Convert to absolute path if relative
DATA_PATH = os.path.abspath(DATA_PATH)
if not os.path.exists(DATA_PATH):
    raise EnvironmentError(f"Data directory not found at: {DATA_PATH}")

logger.info(f"Using data directory: {DATA_PATH}")

# Initialize LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

llm = init_chat_model(
    "openai:gpt-4",
    temperature=0.7
)

# Define the state structure

def validate_and_merge_chunks(chunks: List[Dict[str, Any]], 
                            min_chunk_size: int = 800,  # Minimum size in tokens
                            max_chunk_size: int = 1600,  # Maximum size in tokens
                            target_chunk_size: int = 1200) -> List[Dict[str, Any]]:
    """Validate chunks and merge/split them to maintain target size."""
    if not chunks:
        return []
    
    # Group chunks by section
    section_groups = {}
    for chunk in chunks:
        section = chunk["section"]
        if section not in section_groups:
            section_groups[section] = []
        section_groups[section].append(chunk)
    
    # Process each section
    merged_chunks = []
    for section, section_chunks in section_groups.items():
        # Sort chunks by their chunk number
        section_chunks.sort(key=lambda x: x["chunk_num"])
        
        # First pass: Split any chunks that are too large
        temp_chunks = []
        for chunk in section_chunks:
            chunk_size = len(chunk["chunk"]) // 4  # Approximate tokens
            
            if chunk_size > max_chunk_size:
                # Split large chunk into smaller pieces
                text = chunk["chunk"]
                while text:
                    # Find a good split point near target size
                    split_point = min(len(text), target_chunk_size * 4)  # Convert tokens to chars
                    if split_point < len(text):
                        # Try to find a sentence boundary
                        last_period = text[:split_point].rfind('. ')
                        if last_period > split_point * 0.7:  # If we found a good break point
                            split_point = last_period + 1
                        else:
                            # Try to find a paragraph break
                            last_break = text[:split_point].rfind('\n\n')
                            if last_break > split_point * 0.7:
                                split_point = last_break + 2
                    
                    # Create new chunk
                    new_chunk = {
                        "chunk": text[:split_point].strip(),
                        "section": section,
                        "section_path": chunk["section_path"],
                        "chunk_num": len(temp_chunks) + 1,
                        "total_chunks": 0,  # Will update later
                        "metadata": {
                            "headers": chunk["metadata"]["headers"],
                            "section_headers": chunk["metadata"]["section_headers"],
                            "key_topics": chunk["metadata"]["key_topics"],
                            "section_start": text[:50] + "..." if text else "",
                            "section_end": text[:split_point][-50:] + "..." if text else "",
                            "approximate_tokens": split_point // 4,
                            "chunking_method": chunk["metadata"]["chunking_method"] + "_split"
                        }
                    }
                    temp_chunks.append(new_chunk)
                    text = text[split_point:].strip()
            else:
                temp_chunks.append(chunk)
        
        # Second pass: Merge small chunks
        current_chunk = None
        final_chunks = []
        
        for chunk in temp_chunks:
            chunk_size = len(chunk["chunk"]) // 4  # Approximate tokens
            
            if current_chunk is None:
                current_chunk = chunk
            elif chunk_size < min_chunk_size:
                # Try to merge with current chunk
                current_size = len(current_chunk["chunk"]) // 4
                if current_size + chunk_size <= max_chunk_size:
                    # Merge chunks
                    current_chunk["chunk"] += "\n\n" + chunk["chunk"]
                    current_chunk["metadata"]["approximate_tokens"] = len(current_chunk["chunk"]) // 4
                    current_chunk["metadata"]["section_end"] = chunk["chunk"][-50:] + "..." if chunk["chunk"] else ""
                else:
                    # Current chunk would be too large, start new one
                    final_chunks.append(current_chunk)
                    current_chunk = chunk
            else:
                # Current chunk is a good size, add it and start new one
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Add the last chunk if it exists
        if current_chunk:
            final_chunks.append(current_chunk)
        
        # Update chunk numbers and total chunks for this section
        total_chunks = len(final_chunks)
        for i, chunk in enumerate(final_chunks, 1):
            chunk["chunk_num"] = i
            chunk["total_chunks"] = total_chunks
            merged_chunks.append(chunk)
    
    # Sort merged chunks by section and chunk number
    merged_chunks.sort(key=lambda x: (x["section"], x["chunk_num"]))
    
    # Log chunk size statistics
    chunk_sizes = [len(c["chunk"]) // 4 for c in merged_chunks]  # Convert to tokens
    logger.info(f"Chunk size statistics after validation:")
    logger.info(f"• Min: {min(chunk_sizes):,} tokens")
    logger.info(f"• Max: {max(chunk_sizes):,} tokens")
    logger.info(f"• Average: {sum(chunk_sizes) / len(chunk_sizes):,.0f} tokens")
    logger.info(f"• Total chunks: {len(merged_chunks)}")
    
    return merged_chunks

def router(state: EstimationState) -> Literal["revise_analysis", "summarize_analysis"]:
    """Route based on review feedback."""
    if state["review_feedback"] == "accept":
        return "summarize_analysis"
    else:
        return "revise_analysis"  # Handle both "update" and other feedback types

def format_value(value: Any) -> str:
    """Format a value (list, dict, or string) into a readable string."""
    if isinstance(value, list):
        return "\n".join(f"• {item}" for item in value)
    elif isinstance(value, dict):
        return "\n".join(f"• {k}: {v}" for k, v in value.items())
    else:
        return str(value)

def pretty_print_analysis(state: EstimationState) -> EstimationState:
    """Format and display the PRD analysis results in a readable way."""
    try:
        analysis = state["prd_analysis"]
        if not analysis:
            raise ValueError("No analysis results found in state")
        
        # Save analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = save_analysis_to_file(analysis, timestamp)
        
        # Create a pretty printer
        pp = pprint.PrettyPrinter(indent=2, width=100, sort_dicts=False)
        
        # Print all sections with clear headers
        console.print("\n[bold]PRD Analysis Results[/bold]\n")
        
        for section_name in [
            "Project Overview and Objectives",
            "Key Components/Modules",
            "Timeline and Milestones",
            "Critical Assumptions",
            "Dependencies and Constraints",
            "Resource Requirements",
            "Risk Factors"
        ]:
            if section_name in analysis and analysis[section_name]:
                console.print(f"\n[bold]{section_name}[/bold]")
                console.print(Panel(
                    Text(pp.pformat(analysis[section_name]), style="white"),
                    border_style="white"
                ))
        
        # Add a summary panel with file location
        console.print(Panel(
            Text(f"Analysis complete! Results saved to:\n{output_path}\n\nReview the sections above for detailed information.", style="bold green"),
            title="[bold green]Analysis Summary[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        logger.error(f"Error in pretty printing analysis: {e}")
        console.print(Panel(
            Text(f"Error formatting analysis results: {str(e)}", style="bold red"),
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
    
    return state

def visualize_graph_mermaid(graph, output_path: str = None):
    """Visualize the graph using Mermaid and optionally save to file."""
    try:
        # Get the graph and draw it
        mermaid_graph = graph.get_graph()
        png_data = mermaid_graph.draw_mermaid_png()
        
        if output_path:
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(png_data)
            logger.info(f"Graph visualization saved to: {output_path}")
            
            # Display the graph path in a panel
            console.print(Panel(
                Text(f"Graph visualization saved to:\n{output_path}", style="green"),
                title="[green]Graph Visualization[/green]",
                border_style="green"
            ))
        
        # splay in notebook if running in IPython
        try:
            display(Image(png_data))
        except Exception as e:
            logger.debug(f"Not running in IPython environment: {e}")
            
    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}")
        raise

def create_estimation_graph():
    """Create the estimation graph with all nodes and edges.
    
    The graph defines the workflow for analyzing PRDs:
    1. analyze_prd: Initial analysis of the PRD
    2. review_analysis: User review and feedback
    3. revise_analysis: Update analysis based on feedback
    4. summarize_analysis: Create final summary
    5. generate_clarification_questions: Generate questions for clarification
    
    Returns:
        Compiled graph ready for execution
    """
    # Create graph with system recursion limit
    builder = StateGraph(
        EstimationState,
        config_schema={
            "recursion_limit": 25,  # Match system recursion limit
            "max_iterations": 10    # Allow multiple revisions
        }
    )
    
    # Add nodes from the nodes package
    builder.add_node("analyze_prd", analyze_prd)
    builder.add_node("review_analysis", review_analysis)
    builder.add_node("revise_analysis", revise_analysis)
    builder.add_node("summarize_analysis", summarize_analysis)
    builder.add_node("generate_clarification_questions", generate_clarification_questions)
    
    # Set entry point
    builder.set_entry_point("analyze_prd")
    
    # Add edges
    builder.add_edge("analyze_prd", "review_analysis")
    
    # Add conditional edges based on review feedback
    builder.add_conditional_edges(
        "review_analysis",
        router,
        {
            "revise_analysis": "revise_analysis",
            "summarize_analysis": "summarize_analysis"
        }
    )
    
    # Add edge from revise back to review
    builder.add_edge("revise_analysis", "review_analysis")
    
    # Add edge from summarize to clarification questions
    builder.add_edge("summarize_analysis", "generate_clarification_questions")
    
    # Add final edge to end
    builder.add_edge("generate_clarification_questions", END)
    
    # Create the graph
    graph = builder.compile()
    
    # Visualize the graph using Mermaid
    try:
        # Generate a timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"estimation_graph_{timestamp}.png"
        
        # Visualize and save the graph
        visualize_graph_mermaid(graph, output_path)
    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}")
    
    return graph

def find_prd_files(data_dir: str = None) -> List[str]:
    """Find all supported document files in the data directory.
    
    Args:
        data_dir: Optional directory to search for files. If None, uses DATA_PATH.
        
    Returns:
        List of paths to supported document files (.md, .txt, .pdf)
    """
    if data_dir is None:
        data_dir = DATA_PATH
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")
    
    supported_files = []
    for file in os.listdir(data_dir):
        if file.endswith(('.md', '.txt', '.pdf')):
            supported_files.append(os.path.join(data_dir, file))
    
    return sorted(supported_files)  # Sort files for consistent ordering

def read_prd_file(file_path: str) -> str:
    """Read and validate a PRD file.
    
    Args:
        file_path: Path to the PRD file
        
    Returns:
        Content of the PRD file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or too large
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PRD file not found: {file_path}")
    
    # Check file size (limit to 10MB)
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"PRD file is empty: {file_path}")
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise ValueError(f"PRD file too large ({file_size/1024/1024:.1f}MB): {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"PRD file is empty after reading: {file_path}")
            return content
    except UnicodeDecodeError:
        raise ValueError(f"PRD file is not valid UTF-8 text: {file_path}")

def main():
    """Main function to run the estimation graph."""
    console.print("\n[bold blue]Welcome to the AI Estimation Tool![/bold blue]")
    console.print(f"[dim]Using data directory: {DATA_PATH}[/dim]")
    
    try:
        # Find document files
        doc_files = find_prd_files()
        if not doc_files:
            console.print("\n[red]No supported files found in the data directory.[/red]")
            console.print(f"Please add document files (.md, .txt, or .pdf) to: {DATA_PATH}")
            return
        
        # Display available files
        console.print("\n[bold]Available document files:[/bold]")
        for i, file_path in enumerate(doc_files, 1):
            # Show relative path if file is in a subdirectory
            rel_path = os.path.relpath(file_path, DATA_PATH)
            console.print(f"{i}. {rel_path}")
        
        # Let user select a file
        while True:
            try:
                selection = input("\nEnter the number of the document to analyze (or 'q' to quit): ").strip()
                if selection.lower() == 'q':
                    return
                
                file_index = int(selection) - 1
                if 0 <= file_index < len(doc_files):
                    selected_file = doc_files[file_index]
                    break
                else:
                    console.print("[red]Invalid selection. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")
        
        # Read the selected file
        rel_path = os.path.relpath(selected_file, DATA_PATH)
        console.print(f"\n[bold]Reading document: {rel_path}[/bold]")
        doc_content = read_prd_file(selected_file)  # We'll keep the function name for now
        
        # Initialize the graph
        console.print("\n[bold]Initializing analysis graph...[/bold]")
        graph = create_estimation_graph()
        
        # Initialize state with document content
        initial_state = {
            "prd_doc": doc_content,
            "prd_analysis": None,
            "messages": [],
            "conversation_memory": None,
            "revision_history": []
        }
        
        # Run the graph
        console.print("\n[bold]Starting analysis...[/bold]")
        for event in graph.stream(initial_state):
            # Process events and update state
            pass
            
    except FileNotFoundError as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
    except ValueError as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {str(e)}[/red]")
        logger.exception("Unexpected error in main function")

if __name__ == "__main__":
    main() 