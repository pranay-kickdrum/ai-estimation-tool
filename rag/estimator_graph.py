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
from pathlib import Path
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
        total_chunks = len(chunks)
        logger.info(f"Split PRD into {total_chunks} chunks for analysis")
        
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
            total_chunks=total_chunks,
            llm=llm,
            system_message=system_message
        )
        
        # Track progress
        completed_chunks = 0
        failed_chunks = 0
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, total_chunks)) as executor:
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
                    if chunk_analysis:  # Only merge if we got valid results
                        # Merge results
                        for key in analysis_results:
                            if key in chunk_analysis and chunk_analysis[key]:
                                analysis_results[key].extend(chunk_analysis[key])
                        completed_chunks += 1
                    else:
                        failed_chunks += 1
                        
                    # Log progress
                    logger.info(f"Progress: {completed_chunks + failed_chunks}/{total_chunks} chunks processed "
                              f"({completed_chunks} successful, {failed_chunks} failed)")
                    
                except Exception as e:
                    failed_chunks += 1
                    logger.error(f"Chunk {chunk_num} generated an exception: {e}")
                    logger.info(f"Progress: {completed_chunks + failed_chunks}/{total_chunks} chunks processed "
                              f"({completed_chunks} successful, {failed_chunks} failed)")
        
        # Log final processing status
        logger.info(f"All chunks processed. Final status: {completed_chunks} successful, {failed_chunks} failed")
        
        if completed_chunks == 0:
            raise ValueError("No chunks were successfully processed")
        
        # Deduplicate and clean up results
        for key in analysis_results:
            analysis_results[key] = list(set(analysis_results[key]))  # Remove duplicates
            analysis_results[key].sort()  # Sort for consistency
        
        # Add processing metadata to results
        analysis_results["_metadata"] = {
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "failed_chunks": failed_chunks,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Update state with analysis results
        state["prd_analysis"] = analysis_results
        
        # Log the analysis completion
        logger.info("PRD Analysis completed successfully")
        logger.info(Panel(
            Text(f"Analysis completed with {completed_chunks}/{total_chunks} chunks processed successfully.\n"
                 f"Results include {sum(len(v) for k, v in analysis_results.items() if not k.startswith('_'))} total items.",
                 style="green"),
            title="[green]PRD Analysis Results[/green]",
            border_style="green"
        ))
        
    except Exception as e:
        logger.error(f"Error in PRD analysis: {e}")
        state["prd_analysis"] = {
            "error": str(e),
            "status": "failed",
            "_metadata": {
                "error_timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__
            }
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
            "prd_doc": doc_content,  # We'll keep the state key name for now
            "prd_analysis": None,
            "messages": [],
            "conversation_memory": None
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