from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from IPython.display import Image, display
import json
import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
# from langgraph.graph import visualize_graph

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
        logging.FileHandler(f'estimation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("rich")

# Load environment variables
load_dotenv()

# Initialize LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

llm = init_chat_model(
    "openai:gpt-4",
    temperature=0.7
)

# Define the state structure
class EstimationState(TypedDict):
    # Input documents
    prd_doc: Optional[str]  # PRD document content
    
    # PRD Analysis
    prd_analysis: Optional[Dict[str, Any]]  # Analysis results including key components, timeline, assumptions, etc.
    
    # Conversation tracking
    messages: Annotated[list, add_messages]
    conversation_memory: Optional[dict]

def split_into_chunks(text: str, max_tokens: int = 4000) -> List[str]:
    """Split text into chunks of roughly max_tokens size.
    
    Uses a simple character count approximation (4 chars ≈ 1 token).
    Tries to split at paragraph boundaries when possible.
    """
    # Approximate tokens (roughly 4 chars per token)
    max_chars = max_tokens * 4
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If a single paragraph is too long, split it into sentences
        if len(para) > max_chars:
            sentences = para.split('. ')
            for sentence in sentences:
                sentence = sentence.strip() + '. '  # Add back the period
                if current_length + len(sentence) > max_chars:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            # If adding this paragraph would exceed max length, save current chunk
            if current_length + len(para) > max_chars:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def analyze_chunk(chunk: str, chunk_num: int, total_chunks: int, llm, system_message: SystemMessage) -> Dict[str, Any]:
    """Analyze a single chunk of the PRD document.
    
    Args:
        chunk: The text chunk to analyze
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        llm: The language model to use
        system_message: System message for the analysis
    
    Returns:
        Dict containing the analysis results for this chunk
    """
    try:
        logger.info(f"Processing chunk {chunk_num}/{total_chunks}")
        
        # Create messages for this chunk
        messages = [
            system_message,
            HumanMessage(content=f"Please analyze this section of the PRD document (Part {chunk_num}/{total_chunks}):\n\n{chunk}")
        ]
        
        # Get analysis from LLM
        response = llm.invoke(messages)
        
        # Parse the response
        try:
            chunk_analysis = json.loads(response.content)
            return chunk_analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for chunk {chunk_num}: {e}")
            return {}
            
    except Exception as e:
        logger.error(f"Error analyzing chunk {chunk_num}: {e}")
        return {}

def analyze_prd(state: EstimationState) -> EstimationState:
    """Analyze the PRD document and extract key information using parallel processing."""
    try:
        # Get the PRD content
        prd_content = state["prd_doc"]
        if not prd_content:
            raise ValueError("No PRD content found in state")
        
        # Preprocess the PRD content
        chunks = split_into_chunks(prd_content, max_tokens=3000)
        logger.info(f"Split PRD into {len(chunks)} chunks for analysis")
        
        # Initialize analysis results
        analysis_results = {
            "Project Overview and Objectives": [],
            "Key Components/Modules": [],
            "Timeline and Milestones": [],
            "Critical Assumptions": [],
            "Dependencies and Constraints": [],
            "Resource Requirements": [],
            "Risk Factors": []
        }
        
        # Create a system message to guide the analysis
        system_message = SystemMessage(content="""You are an expert project analyst. Analyze the provided PRD section and extract the following key information:
1. Project Overview and Objectives
2. Key Components/Modules
3. Timeline and Milestones
4. Critical Assumptions
5. Dependencies and Constraints
6. Resource Requirements
7. Risk Factors

For each section, extract only the most relevant information. If a section is not present in this chunk, return an empty list for that section.
Format the output as a structured dictionary with these sections as keys and lists as values.
Keep your analysis concise and focused on the most important points.""")
        
        # Create a partial function with the common arguments
        analyze_chunk_partial = partial(
            analyze_chunk,
            total_chunks=len(chunks),
            llm=llm,
            system_message=system_message
        )
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(analyze_chunk_partial, chunk, i+1): i+1 
                for i, chunk in enumerate(chunks)
            }
            
            # Process completed chunks as they finish
            for future in as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    chunk_analysis = future.result()
                    # Merge results
                    for key in analysis_results:
                        if key in chunk_analysis and chunk_analysis[key]:
                            analysis_results[key].extend(chunk_analysis[key])
                except Exception as e:
                    logger.error(f"Chunk {chunk_num} generated an exception: {e}")
        
        # Deduplicate and clean up results
        for key in analysis_results:
            analysis_results[key] = list(set(analysis_results[key]))  # Remove duplicates
            analysis_results[key].sort()  # Sort for consistency
        
        state["prd_analysis"] = analysis_results
        
        # Log the analysis
        logger.info("PRD Analysis completed")
        logger.info(Panel(
            Text(str(state["prd_analysis"]), style="green"),
            title="[green]PRD Analysis Results[/green]",
            border_style="green"
        ))
        
    except Exception as e:
        logger.error(f"Error in PRD analysis: {e}")
        state["prd_analysis"] = {
            "error": str(e),
            "status": "failed"
        }
    
    return state

def save_analysis_to_file(analysis: Dict[str, Any], timestamp: str = None) -> str:
    """Save the analysis results to a JSON file.
    
    Args:
        analysis: The analysis results dictionary
        timestamp: Optional timestamp to use in filename. If None, generates new timestamp.
    
    Returns:
        str: Path to the saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    output_path = os.path.join(output_dir, f"prd_analysis_{timestamp}.json")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to: {output_path}")
    return output_path

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
        
        # Display in notebook if running in IPython
        try:
            display(Image(png_data))
        except Exception as e:
            logger.debug(f"Not running in IPython environment: {e}")
            
    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}")
        raise

def create_estimation_graph():
    # Create graph with system recursion limit
    builder = StateGraph(
        EstimationState,
        config_schema={
            "recursion_limit": 25,  # Match system recursion limit
            "max_iterations": 1     # Only need one iteration for PRD analysis
        }
    )
    
    # Add nodes
    builder.add_node("analyze_prd", analyze_prd)
    builder.add_node("pretty_print_analysis", pretty_print_analysis)
    
    # Set entry point
    builder.set_entry_point("analyze_prd")
    
    # Simple linear flow with direct edges
    builder.add_edge("analyze_prd", "pretty_print_analysis")
    builder.add_edge("pretty_print_analysis", END)
    
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

def main():
    """Main function to run the estimation graph."""
    print("Welcome to the AI Estimation Tool!")
    print("Processing PRD from data directory...")
    
    # Initialize the graph (this will also generate the visualization)
    graph = create_estimation_graph()
    
    try:
        # Read PRD from data directory
        prd_path = "data/ambra-storage-migration.md"
        with open(prd_path, 'r') as f:
            prd_content = f.read()
        
        # Initialize state with PRD content
        initial_state = {
            "prd_doc": prd_content,  # PRD content loaded from file
            "prd_analysis": None,    # Will be populated by analyze_prd node
            "messages": [],
            "conversation_memory": None
        }
        
        # Run the graph
        for event in graph.stream(initial_state):
            # Process events and update state
            pass
            
    except FileNotFoundError:
        print(f"\nError: Could not find PRD file at {prd_path}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 