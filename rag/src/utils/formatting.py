"""Formatting utilities for the chatbot."""
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict, Any

console = Console()

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

def display_conversation_context(messages: List[Dict[str, Any]]):
    """Display the conversation context with pretty printing."""
    for msg in messages:
        console.print(format_message(msg["role"], msg["content"]))

def pretty_print_final_output(paragraph: str, bullets: List[str]):
    """Print the final output with pretty formatting."""
    # Print the paragraph in a panel
    console.print(Panel(
        Text(paragraph, style="green"),
        title="[green]FINAL PARAGRAPH[/green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Format bullet points with proper spacing
    bullet_text = "\n".join([
        f"• {bullet}" for bullet in bullets
    ])
    
    # Print bullet points in a panel
    console.print(Panel(
        Text(bullet_text, style="blue"),
        title="[blue]KEY POINTS[/blue]",
        border_style="blue",
        padding=(1, 2)
    )) 