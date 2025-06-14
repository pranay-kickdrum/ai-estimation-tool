"""Logging utilities for the chatbot."""
import logging
from datetime import datetime
import tiktoken
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict, Any

# Initialize rich console
console = Console()

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

def setup_logging() -> logging.Logger:
    """Set up logging with rich formatting."""
    # Create logs directory if it doesn't exist
    log_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_time=False,
                show_path=False
            ),
            logging.FileHandler(log_filename)
        ]
    )
    
    # Create logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)
    
    # Ensure the logger propagates to root
    logger.propagate = True
    
    return logger

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def log_token_usage(prompt: str, response: str, node_name: str):
    """Log token usage for an LLM invocation."""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
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

def log_memory_usage(messages: List[Dict[str, Any]], operation: str):
    """Log memory usage statistics in a clean format."""
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    utilization = (total_tokens/2000)*100  # Using default max_tokens
    
    # Create a memory usage panel
    memory_panel = Panel(
        Text.assemble(
            f"Messages: {len(messages)}/5\n",  # Using default max_messages
            f"Tokens: {total_tokens}/2000\n",
            f"Utilization: {utilization:.1f}%",
            style="dim"
        ),
        title=f"[dim]Memory Usage ({operation})[/dim]",
        border_style="dim",
        padding=(1, 2)
    )
    console.print(memory_panel) 