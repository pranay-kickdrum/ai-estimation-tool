"""Main entry point for the chatbot."""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from rich.panel import Panel
from src.graph.graph_state import GraphState
from src.graph.graph_nodes import (
    generate_paragraph,
    review_paragraph,
    bullet_point_generator,
    should_continue
)
from src.memory.memory_manager import CustomMemory, update_conversation_memory
from src.utils.logging_utils import setup_logging, console
from src.utils.formatting import display_conversation_context, pretty_print_final_output
from src.config import MAX_CONTEXT_TOKENS, MAX_MEMORY_ITEMS

# Set up logging
logger = setup_logging()

def create_graph() -> StateGraph:
    """Create the conversation graph."""
    logger.info("Creating conversation graph...")
    
    # Initialize the graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("generate", generate_paragraph)
    graph.add_node("review", review_paragraph)
    graph.add_node("bullet_points", bullet_point_generator)
    graph.add_node("should_continue", should_continue)
    
    # Add edges
    graph.add_edge("generate", "review")
    graph.add_edge("review", "bullet_points")
    graph.add_edge("bullet_points", "should_continue")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "should_continue",
        lambda x: x["next"],
        {
            "generate": "generate",
            "end": END
        }
    )
    
    # Set the entry point
    graph.set_entry_point("generate")
    
    logger.info("Graph created successfully!")
    return graph

def process_user_input(topic: str, user_feedback: str = "") -> Dict[str, Any]:
    """Process user input and run the graph."""
    logger.info(f"\nProcessing input - Topic: {topic}")
    if user_feedback:
        logger.info(f"User feedback: {user_feedback}")
    
    # Initialize memory
    memory = CustomMemory(
        max_tokens=MAX_CONTEXT_TOKENS,
        max_messages=MAX_MEMORY_ITEMS
    )
    
    # Create initial state
    initial_state = {
        "topic": topic,
        "user_feedback": user_feedback,
        "messages": [],
        "conversation_memory": memory
    }
    
    # Add initial user message to memory
    if user_feedback:
        initial_state = update_conversation_memory(initial_state, "user", user_feedback)
    else:
        initial_message = f"Generate content about: {topic}"
        logger.info(f"\nInitial message: {initial_message}")
        initial_state = update_conversation_memory(initial_state, "user", initial_message)
    
    # Create and run the graph
    logger.info("\nCreating and running the graph...")
    graph = create_graph()
    app = graph.compile()
    
    # Run the graph
    logger.info("\nStarting graph execution...")
    for state in app.stream(initial_state):
        # Update conversation memory with assistant's response if available
        if "paragraph" in state and state["paragraph"]:
            logger.info("\n[bold green]Generated Paragraph:[/bold green]")
            console.print(Panel(state["paragraph"], title="Paragraph", border_style="green"))
            state = update_conversation_memory(state, "assistant", state["paragraph"])
        
        if "suggestion" in state and state["suggestion"]:
            logger.info("\n[bold yellow]Review:[/bold yellow]")
            console.print(Panel(state["suggestion"], title="Review", border_style="yellow"))
            state = update_conversation_memory(state, "assistant", state["suggestion"])
        
        if "bullets" in state and state["bullets"]:
            logger.info("\n[bold blue]Bullet Points:[/bold blue]")
            bullet_text = "\n".join([f"• {bullet}" for bullet in state["bullets"]])
            console.print(Panel(bullet_text, title="Bullet Points", border_style="blue"))
            state = update_conversation_memory(state, "assistant", "Generated bullet points: " + bullet_text)
        
        # Display conversation context
        if state.get("messages"):
            logger.info("\n[bold]Conversation Context:[/bold]")
            display_conversation_context(state["messages"])
    
    logger.info("\nGraph execution completed!")
    return state

def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold blue]Welcome to the Content Generation Chatbot![/bold blue]\n"
        "Enter 'end' to finish the conversation.",
        border_style="blue"
    ))
    
    # Get initial topic
    topic = input("\nEnter a topic to generate content about: ")
    
    # Process initial topic
    state = process_user_input(topic)
    
    # Main conversation loop
    while True:
        # Get user feedback
        user_feedback = input("\nEnter your feedback (or 'end' to finish): ")
        
        if user_feedback.lower() == "end":
            break
        
        # Process feedback
        state = process_user_input(topic, user_feedback)
    
    # Display final output
    if state.get("paragraph") and state.get("bullets"):
        console.print("\n[bold green]Final Output:[/bold green]")
        pretty_print_final_output(state["paragraph"], state["bullets"])

if __name__ == "__main__":
    main() 