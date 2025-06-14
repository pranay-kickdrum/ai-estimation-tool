from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
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
import re
import tiktoken
import textwrap

# Set up rich console
console = Console()

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Constants for memory management
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

class EstimationState(TypedDict):
    """State management for the PRD analysis process.
    
    This TypedDict defines the structure of the state that flows through the analysis graph.
    The state maintains:
    1. Input document content
    2. Analysis results and their file location
    3. Review and revision tracking
    4. Conversation memory
    
    Attributes:
        prd_doc: The content of the PRD document being analyzed
        prd_analysis: The current analysis results, including key components, timeline, etc.
        current_analysis_file: Path to the JSON file containing the current analysis
        review_feedback: User's feedback type (accept/revise/add/remove)
        revision_requests: List of specific revision requests from the user
        revision_history: History of changes made during revisions
        messages: List of conversation messages
        conversation_memory: State of the conversation memory
    """
    # Input documents
    prd_doc: Optional[str]  # PRD document content
    
    # PRD Analysis
    prd_analysis: Optional[Dict[str, Any]]  # Analysis results including key components, timeline, assumptions, etc.
    current_analysis_file: Optional[str]  # Path to the current analysis file
    
    # Review and revision tracking
    review_feedback: Optional[Literal["accept", "revise", "add", "remove"]]
    revision_requests: Optional[List[str]]  # List of specific revision requests
    revision_history: Optional[List[Dict[str, Any]]]  # Track changes made
    
    # Conversation tracking
    messages: Annotated[list, add_messages]
    conversation_memory: Optional[dict]  # Store the conversation memory state

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

def split_into_semantic_chunks(text: str, max_tokens: int = 3000) -> List[Dict[str, Any]]:
    """Split text into semantic chunks using LangChain's experimental text splitters.
    
    Args:
        text: The document text to split
        max_tokens: Maximum tokens per chunk (approximate)
        
    Returns:
        List of dictionaries containing chunk information and metadata
    """
    # First, split by markdown headers to preserve document structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # Use experimental markdown splitter for better header detection
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,  # Return complete sections
        strip_headers=False  # Keep headers in the content
    )
    
    # Split the document into sections based on headers
    md_splits = markdown_splitter.split_text(text)
    
    # Initialize embeddings for semantic chunking
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Using the latest embedding model
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Use semantic chunker for intelligent content splitting
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        buffer_size=8,  # Reduced from 12 to consider less context
        add_start_index=True,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.95,  # Reduced from 0.97 to allow more splits
        sentence_split_regex=r'(?<=[.?!])\s+(?=[A-Z])'  # Split on sentence boundaries
    )
    
    # Use token splitter as fallback with smaller chunks
    token_splitter = TokenTextSplitter(
        chunk_size=1000,  # Reduced from 1200 to allow for better merging
        chunk_overlap=100,  # Overlap for context
        encoding_name="cl100k_base"  # GPT-4 tokenizer
    )
    
    chunks = []
    section_chunks = {}  # Track chunks per section
    
    for split in md_splits:
        # Get the section name from headers
        section_name = "Introduction"
        section_headers = {}
        for header_level in range(1, 5):
            header_key = f"Header {header_level}"
            if header_key in split.metadata and split.metadata[header_key]:
                section_headers[header_level] = split.metadata[header_key]
                if header_level == 1:
                    section_name = split.metadata[header_key]
        
        # Build section path
        section_path = " > ".join(
            section_headers[level] 
            for level in sorted(section_headers.keys())
        ) if section_headers else section_name
        
        # Try semantic splitting first
        try:
            logger.info(f"Attempting semantic splitting for section: {section_path}")
            section_chunks_list = semantic_splitter.split_text(split.page_content)
            if not section_chunks_list:
                raise ValueError("Semantic splitter returned empty list")
            
            # Log the number of chunks from semantic splitting
            logger.info(f"Semantic splitting produced {len(section_chunks_list)} chunks for section {section_path}")
            
            # Convert to our chunk format
            formatted_chunks = []
            for i, chunk_text in enumerate(section_chunks_list, 1):
                chunk = {
                    "chunk": chunk_text,
                    "section": section_name,
                    "section_path": section_path,
                    "chunk_num": i,
                    "total_chunks": len(section_chunks_list),
                    "metadata": {
                        "headers": split.metadata,
                        "section_headers": section_headers,
                        "key_topics": [],
                        "section_start": chunk_text[:50] + "..." if chunk_text else "",
                        "section_end": chunk_text[-50:] + "..." if chunk_text else "",
                        "approximate_tokens": len(chunk_text) // 4,
                        "chunking_method": "semantic"
                    }
                }
                formatted_chunks.append(chunk)
            section_chunks_list = formatted_chunks
            
        except Exception as e:
            logger.warning(f"Semantic splitting failed for section {section_path}, falling back to token splitting: {e}")
            # Fall back to token splitting
            try:
                logger.info(f"Attempting token splitting for section: {section_path}")
                section_chunks_list = token_splitter.split_text(split.page_content)
                if not section_chunks_list:
                    raise ValueError("Token splitter returned empty list")
                
                # Log the number of chunks from token splitting
                logger.info(f"Token splitting produced {len(section_chunks_list)} chunks for section {section_path}")
                
                # Convert to our chunk format
                formatted_chunks = []
                for i, chunk_text in enumerate(section_chunks_list, 1):
                    chunk = {
                        "chunk": chunk_text,
                        "section": section_name,
                        "section_path": section_path,
                        "chunk_num": i,
                        "total_chunks": len(section_chunks_list),
                        "metadata": {
                            "headers": split.metadata,
                            "section_headers": section_headers,
                            "key_topics": [],
                            "section_start": chunk_text[:50] + "..." if chunk_text else "",
                            "section_end": chunk_text[-50:] + "..." if chunk_text else "",
                            "approximate_tokens": len(chunk_text) // 4,
                            "chunking_method": "token"
                        }
                    }
                    formatted_chunks.append(chunk)
                section_chunks_list = formatted_chunks
                
            except Exception as e2:
                logger.error(f"Both semantic and token splitting failed for section {section_path}: {e2}")
                # Create a single chunk as last resort
                section_chunks_list = [{
                    "chunk": split.page_content,
                    "section": section_name,
                    "section_path": section_path,
                    "chunk_num": 1,
                    "total_chunks": 1,
                    "metadata": {
                        "headers": split.metadata,
                        "section_headers": section_headers,
                        "key_topics": [],
                        "section_start": split.page_content[:50] + "..." if split.page_content else "",
                        "section_end": split.page_content[-50:] + "..." if split.page_content else "",
                        "approximate_tokens": len(split.page_content) // 4,
                        "chunking_method": "fallback"
                    }
                }]
        
        # Log chunk sizes before validation
        chunk_sizes = [len(c["chunk"]) // 4 for c in section_chunks_list]
        logger.info(f"Chunk sizes before validation for section {section_path}:")
        logger.info(f"• Min: {min(chunk_sizes):,} tokens")
        logger.info(f"• Max: {max(chunk_sizes):,} tokens")
        logger.info(f"• Average: {sum(chunk_sizes) / len(chunk_sizes):,.0f} tokens")
        
        # Validate and merge chunks
        section_chunks_list = validate_and_merge_chunks(section_chunks_list)
        
        # Log chunk sizes after validation
        chunk_sizes = [len(c["chunk"]) // 4 for c in section_chunks_list]
        logger.info(f"Chunk sizes after validation for section {section_path}:")
        logger.info(f"• Min: {min(chunk_sizes):,} tokens")
        logger.info(f"• Max: {max(chunk_sizes):,} tokens")
        logger.info(f"• Average: {sum(chunk_sizes) / len(chunk_sizes):,.0f} tokens")
        logger.info(f"• Total chunks: {len(section_chunks_list)}")
        
        # Update section chunk count
        if section_name not in section_chunks:
            section_chunks[section_name] = 0
        section_chunks[section_name] += len(section_chunks_list)
        
        # Add chunks to the main list
        chunks.extend(section_chunks_list)
    
    # Update total_chunks for all chunks in each section
    for chunk in chunks:
        chunk["total_chunks"] = section_chunks[chunk["section"]]
    
    # Log chunking statistics with validation info
    chunking_stats = {
        "total_chunks": len(chunks),
        "sections": len(section_chunks),
        "chunks_per_section": {
            section: count for section, count in section_chunks.items()
        },
        "chunking_methods": {
            "semantic": len([c for c in chunks if c["metadata"]["chunking_method"] == "semantic"]),
            "token": len([c for c in chunks if c["metadata"]["chunking_method"] == "token"])
        },
        "validation": {
            "min_chunk_size": 800,
            "max_chunk_size": 1600,
            "target_chunk_size": 1200,
            "merged_chunks": len(chunks) - len(validate_and_merge_chunks(chunks))
        }
    }
    logger.info("Chunking statistics:")
    logger.info(json.dumps(chunking_stats, indent=2))
    
    return chunks

def analyze_chunk(chunk_data: Dict[str, Any], llm, system_message: SystemMessage) -> Dict[str, Any]:
    """Analyze a single chunk of the document.
    
    Args:
        chunk_data: Dictionary containing chunk information and content
        llm: The language model to use
        system_message: System message for the analysis
    
    Returns:
        Dict containing the analysis results for this chunk
    """
    try:
        chunk = chunk_data["chunk"]
        section = chunk_data["section"]
        section_path = chunk_data["section_path"]
        chunk_num = chunk_data["chunk_num"]
        total_chunks = chunk_data["total_chunks"]
        metadata = chunk_data["metadata"]
        
        logger.info(f"Processing {section_path} (Chunk {chunk_num}/{total_chunks})")
        if metadata["key_topics"]:
            logger.info(f"Key topics in chunk: {', '.join(metadata['key_topics'])}")
        
        # Create messages for this chunk with explicit JSON formatting instructions
        messages = [
            system_message,
            HumanMessage(content=f"""Please analyze this section of the document and return a JSON object with the following structure:

{{
    "Project Overview and Objectives": [],
    "Key Components/Modules": [],
    "Timeline and Milestones": [],
    "Critical Assumptions": [],
    "Dependencies and Constraints": [],
    "Resource Requirements": [],
    "Risk Factors": []
}}

Section: {section_path}
Part: {chunk_num} of {total_chunks}
Key Topics: {', '.join(metadata['key_topics']) if metadata['key_topics'] else 'None specified'}

Content:
{chunk}

Note: 
1. This is part {chunk_num} of {total_chunks} for the {section_path} section.
2. Please analyze it in the context of the overall document structure and the identified key topics.
3. Return ONLY the JSON object, with no additional text or explanation.
4. If a section has no relevant information, return an empty list for that section.
5. Each item in the lists should be a string describing a single point.""")
        ]
        
        # Get analysis from LLM
        response = llm.invoke(messages)
        
        # Clean and parse the response
        try:
            # Extract JSON from the response, handling potential markdown code blocks
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse the JSON
            chunk_analysis = json.loads(content)
            
            # Validate the structure
            required_keys = [
                "Project Overview and Objectives",
                "Key Components/Modules",
                "Timeline and Milestones",
                "Critical Assumptions",
                "Dependencies and Constraints",
                "Resource Requirements",
                "Risk Factors"
            ]
            
            # Ensure all required keys exist
            for key in required_keys:
                if key not in chunk_analysis:
                    chunk_analysis[key] = []
                elif not isinstance(chunk_analysis[key], list):
                    chunk_analysis[key] = [str(chunk_analysis[key])]
                else:
                    # Convert all items to strings
                    chunk_analysis[key] = [str(item) for item in chunk_analysis[key]]
            
            # Add section metadata
            chunk_analysis["_metadata"] = {
                "section": section,
                "section_path": section_path,
                "headers": metadata["headers"],
                "section_headers": metadata["section_headers"],
                "key_topics": metadata["key_topics"],
                "chunk_num": chunk_num,
                "total_chunks": total_chunks,
                "chunking_method": metadata["chunking_method"]
            }
            
            return chunk_analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for {section_path} chunk {chunk_num}: {e}")
            logger.error(f"Raw response: {response.content}")
            # Return empty analysis with error metadata
            return {
                "Project Overview and Objectives": [],
                "Key Components/Modules": [],
                "Timeline and Milestones": [],
                "Critical Assumptions": [],
                "Dependencies and Constraints": [],
                "Resource Requirements": [],
                "Risk Factors": [],
                "_metadata": {
                    "section": section,
                    "section_path": section_path,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content[:200] + "..." if len(response.content) > 200 else response.content
                }
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {section} chunk {chunk_num}: {e}")
        return {
            "Project Overview and Objectives": [],
            "Key Components/Modules": [],
            "Timeline and Milestones": [],
            "Critical Assumptions": [],
            "Dependencies and Constraints": [],
            "Resource Requirements": [],
            "Risk Factors": [],
            "_metadata": {
                "section": section,
                "error": str(e)
            }
        }

def analyze_prd(state: EstimationState) -> EstimationState:
    """Analyze the document and extract key information using parallel processing.
    
    This function is the entry point for the analysis process. It:
    1. Processes the PRD document into semantic chunks
    2. Analyzes each chunk in parallel
    3. Combines the results into a complete analysis
    4. Saves the analysis to a file
    5. Updates the state with both the analysis and file location
    
    The analysis is saved to a timestamped JSON file in the analysis_output directory,
    and the file path is stored in state["current_analysis_file"] for later use by
    the review and revision nodes.
    
    Args:
        state: The current state containing the PRD document
        
    Returns:
        Updated state containing:
        - prd_analysis: The complete analysis results
        - current_analysis_file: Path to the saved analysis file
        - Additional metadata and processing information
        
    Example state flow:
        Input state:
            {
                "prd_doc": "Project requirements...",
                "prd_analysis": None,
                "current_analysis_file": None
            }
        Output state:
            {
                "prd_doc": "Project requirements...",
                "prd_analysis": {"Project Overview": [...], ...},
                "current_analysis_file": "analysis_output/prd_analysis_20240314_123456.json"
            }
    """
    try:
        # Get the document content
        doc_content = state["prd_doc"]
        if not doc_content:
            raise ValueError("No document content found in state")
        
        # Log analysis configuration at INFO level
        logger.info("Starting Document Analysis")
        logger.info("Analysis Configuration:")
        logger.info("• Using OpenAI Embeddings (text-embedding-3-small)")
        logger.info("• Semantic Chunking with buffer size: 10 sentences")
        logger.info("• Breakpoint threshold: 95th percentile")
        logger.info("• Fallback to token-based splitting if needed")
        logger.info("• Parallel processing with ThreadPoolExecutor")
        
        # Log document statistics at INFO level
        doc_stats = {
            "total_characters": len(doc_content),
            "total_words": len(doc_content.split()),
            "total_lines": len(doc_content.splitlines()),
            "estimated_tokens": len(doc_content) // 4  # Rough estimate
        }
        logger.info("Document Statistics:")
        logger.info(f"• Characters: {doc_stats['total_characters']:,}")
        logger.info(f"• Words: {doc_stats['total_words']:,}")
        logger.info(f"• Lines: {doc_stats['total_lines']:,}")
        logger.info(f"• Estimated Tokens: {doc_stats['estimated_tokens']:,}")
        
        # Log detailed document stats at DEBUG level
        logger.debug("Detailed Document Statistics:")
        logger.debug(f"• Average word length: {sum(len(word) for word in doc_content.split()) / doc_stats['total_words']:.1f} characters")
        logger.debug(f"• Average line length: {doc_stats['total_characters'] / doc_stats['total_lines']:.1f} characters")
        logger.debug(f"• Words per line: {doc_stats['total_words'] / doc_stats['total_lines']:.1f}")
        
        # Split into semantic chunks using LangChain
        logger.info("Splitting document into semantic chunks...")
        chunks = split_into_semantic_chunks(doc_content, max_tokens=3000)
        unique_sections = len(set(c["section"] for c in chunks))
        logger.info(f"Split document into {len(chunks)} semantic chunks across {unique_sections} sections")
        
        # Log section structure at INFO level
        section_structure = {}
        for chunk in chunks:
            section = chunk["section"]
            if section not in section_structure:
                section_structure[section] = {
                    "total_chunks": chunk["total_chunks"],
                    "headers": chunk["metadata"]["headers"]
                }
        
        logger.info("Document Structure:")
        for section, info in section_structure.items():
            logger.info(f"• {section} ({info['total_chunks']} chunks)")
            # Log header hierarchy at DEBUG level
            headers = info["headers"]
            for level in range(1, 5):
                header_key = f"Header {level}"
                if header_key in headers and headers[header_key]:
                    logger.debug(f"  {'  ' * (level-1)}└─ {headers[header_key]}")
        
        # Calculate and log chunk statistics
        total_chars = sum(len(c["chunk"]) for c in chunks)
        total_tokens = total_chars // 4  # Approximate tokens
        avg_chars = total_chars / len(chunks)
        avg_tokens = total_tokens / len(chunks)
        
        logger.info("\n=== Chunk Size Analysis ===")
        logger.info(f"Total Document Size:")
        logger.info(f"• Characters: {total_chars:,}")
        logger.info(f"• Estimated Tokens: {total_tokens:,}")
        
        logger.info(f"\nChunk Distribution:")
        logger.info(f"• Number of Chunks: {len(chunks):,}")
        logger.info(f"• Average Chunk Size: {avg_chars:.0f} characters")
        logger.info(f"• Average Tokens per Chunk: {avg_tokens:.0f}")
        logger.info(f"• Target Tokens per Chunk: 1,200")
        logger.info(f"• Current vs Target: {avg_tokens:.0f} vs 1,200 tokens")
        
        # Log section distribution
        logger.info(f"\nChunks per Section:")
        section_stats = {}
        for chunk in chunks:
            section = chunk["section"]
            if section not in section_stats:
                section_stats[section] = {
                    "chunks": 0,
                    "total_chars": 0,
                    "total_tokens": 0
                }
            section_stats[section]["chunks"] += 1
            section_stats[section]["total_chars"] += len(chunk["chunk"])
            section_stats[section]["total_tokens"] += len(chunk["chunk"]) // 4
        
        for section, stats in section_stats.items():
            avg_section_tokens = stats["total_tokens"] / stats["chunks"] if stats["chunks"] > 0 else 0
            logger.info(f"• {section}:")
            logger.info(f"  - Chunks: {stats['chunks']}")
            logger.info(f"  - Total Tokens: {stats['total_tokens']:,}")
            logger.info(f"  - Avg Tokens per Chunk: {avg_section_tokens:.0f}")
        
        # Log chunking methods used
        chunking_methods = {}
        for chunk in chunks:
            method = chunk["metadata"]["chunking_method"]
            if method not in chunking_methods:
                chunking_methods[method] = 0
            chunking_methods[method] += 1
        
        logger.info(f"\nChunking Methods Used:")
        for method, count in chunking_methods.items():
            logger.info(f"• {method.title()} Chunking: {count} chunks")
        
        # Log size distribution
        chunk_sizes = [len(c["chunk"]) // 4 for c in chunks]  # Convert to tokens
        logger.info(f"\nChunk Size Distribution (in tokens):")
        logger.info(f"• Smallest Chunk: {min(chunk_sizes):,} tokens")
        logger.info(f"• Largest Chunk: {max(chunk_sizes):,} tokens")
        logger.info(f"• Median Chunk: {sorted(chunk_sizes)[len(chunk_sizes)//2]:,} tokens")
        
        # Calculate percentage of chunks within target range
        target_min = 800  # Minimum target size
        target_max = 1600  # Maximum target size
        chunks_in_range = sum(1 for size in chunk_sizes if target_min <= size <= target_max)
        percent_in_range = (chunks_in_range / len(chunks)) * 100
        
        logger.info(f"\nTarget Size Analysis:")
        logger.info(f"• Target Range: {target_min:,} to {target_max:,} tokens")
        logger.info(f"• Chunks in Target Range: {chunks_in_range} of {len(chunks)} ({percent_in_range:.1f}%)")
        
        # Update chunking_stats with new information
        chunking_stats = {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "chunks_in_target_range": chunks_in_range,
            "percent_in_target_range": percent_in_range,
            "chunking_methods": chunking_methods,
            "section_stats": section_stats
        }
        
        logger.info("Starting analysis of chunks...")
        
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
        system_message = SystemMessage(content="""You are an expert project analyst. Analyze the provided document section and extract the following key information:
1. Project Overview and Objectives
2. Key Components/Modules
3. Timeline and Milestones
4. Critical Assumptions
5. Dependencies and Constraints
6. Resource Requirements
7. Risk Factors

For each section, extract only the most relevant information. If a section is not present in this chunk, return an empty list for that section.
Format the output as a structured dictionary with these sections as keys and lists as values.
Keep your analysis concise and focused on the most important points.
Consider the context of which section of the document you're analyzing.""")
        
        # Create a partial function with the common arguments
        analyze_chunk_partial = partial(
            analyze_chunk,
            llm=llm,
            system_message=system_message
        )
        
        # Track progress
        completed_chunks = 0
        failed_chunks = 0
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(analyze_chunk_partial, chunk): chunk 
                for chunk in chunks
            }
            
            # Process completed chunks as they finish
            for future in as_completed(future_to_chunk):
                chunk_data = future_to_chunk[future]
                try:
                    chunk_analysis = future.result()
                    if chunk_analysis:  # Only merge if we got valid results
                        # Merge results
                        for key in analysis_results:
                            if key in chunk_analysis and chunk_analysis[key]:
                                # Add section context to items if not already present
                                items = chunk_analysis[key]
                                section = chunk_data["section"]
                                for item in items:
                                    if not any(section in str(x) for x in analysis_results[key]):
                                        analysis_results[key].append(f"[{section}] {item}")
                                    else:
                                        analysis_results[key].append(item)
                        completed_chunks += 1
                        logger.debug(f"Successfully analyzed chunk {chunk_data['chunk_num']}/{chunk_data['total_chunks']} from section '{chunk_data['section']}'")
                    else:
                        failed_chunks += 1
                        logger.warning(f"Empty analysis result for chunk {chunk_data['chunk_num']}/{chunk_data['total_chunks']} from section '{chunk_data['section']}'")
                        
                    # Log progress at INFO level
                    logger.info(f"Progress: {completed_chunks + failed_chunks}/{len(chunks)} chunks processed "
                              f"({completed_chunks} successful, {failed_chunks} failed)")
                    
                except Exception as e:
                    failed_chunks += 1
                    logger.error(f"Chunk from section '{chunk_data['section']}' generated an exception: {e}")
                    logger.info(f"Progress: {completed_chunks + failed_chunks}/{len(chunks)} chunks processed "
                              f"({completed_chunks} successful, {failed_chunks} failed)")
        
        # Log final processing status
        if completed_chunks == 0:
            logger.error("No chunks were successfully processed")
            raise ValueError("No chunks were successfully processed")
        
        logger.info(f"All chunks processed. Final status: {completed_chunks} successful, {failed_chunks} failed")
        
        # Deduplicate and clean up results
        for key in analysis_results:
            # Remove duplicates while preserving order
            seen = set()
            analysis_results[key] = [
                x for x in analysis_results[key] 
                if not (x in seen or seen.add(x))
            ]
            analysis_results[key].sort()  # Sort for consistency
            logger.debug(f"Section '{key}' contains {len(analysis_results[key])} unique items")
        
        # Add processing metadata to results
        analysis_results["_metadata"] = {
            "total_chunks": len(chunks),
            "completed_chunks": completed_chunks,
            "failed_chunks": failed_chunks,
            "sections_analyzed": list(set(c["section"] for c in chunks)),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Update state with analysis results and file location
        state["prd_analysis"] = analysis_results
        
        # Save initial analysis to file and store path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = save_analysis_to_file(analysis_results, timestamp)
        state["current_analysis_file"] = filepath
        
        # Log completion with file location
        logger.info("Document Analysis completed successfully")
        logger.info(f"Results include {sum(len(v) for k, v in analysis_results.items() if not k.startswith('_'))} total items")
        logger.info(f"Analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error in document analysis: {e}")
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
    
    This function handles saving the analysis results to a JSON file in the analysis_output directory.
    Each analysis is saved with a timestamp to maintain a history of revisions.
    
    Args:
        analysis: The analysis results dictionary containing all sections and metadata
        timestamp: Optional timestamp to use in filename. If None, generates new timestamp.
    
    Returns:
        str: Path to the saved file
        
    Example:
        >>> analysis = {"Project Overview": ["item1", "item2"]}
        >>> filepath = save_analysis_to_file(analysis)
        >>> print(filepath)
        'analysis_output/prd_analysis_20240314_123456.json'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    output_path = os.path.join(output_dir, f"prd_analysis_{timestamp}.json")
    
    # Save to file with pretty printing
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to: {output_path}")
    return output_path

def load_analysis_from_file(filepath: str) -> Dict[str, Any]:
    """Load analysis results from a JSON file.
    
    This function handles loading the analysis results from a previously saved JSON file.
    It includes error handling for file operations and JSON parsing.
    
    Args:
        filepath: Path to the analysis file to load
        
    Returns:
        Dict containing the analysis results with all sections and metadata
        
    Raises:
        ValueError: If the file doesn't exist or contains invalid JSON
        FileNotFoundError: If the specified file cannot be found
        
    Example:
        >>> analysis = load_analysis_from_file("analysis_output/prd_analysis_20240314_123456.json")
        >>> print(analysis.keys())
        ['Project Overview', 'Key Components', ...]
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in analysis file {filepath}: {e}")
        raise ValueError(f"Invalid analysis file format: {e}")
    except Exception as e:
        logger.error(f"Error loading analysis from {filepath}: {e}")
        raise

def review_analysis(state: EstimationState) -> EstimationState:
    """Review the PRD analysis and get user feedback."""
    try:
        # Get analysis from state
        analysis = state["prd_analysis"]
        if not analysis:
            raise ValueError("No analysis results found in state")
        
        # Display current analysis with pretty printing
        console.print("\n[bold blue]Current Analysis Results[/bold blue]")
        
        # Define sections to display
        sections = [
            "Project Overview and Objectives",
            "Key Components/Modules",
            "Timeline and Milestones",
            "Critical Assumptions",
            "Dependencies and Constraints",
            "Resource Requirements",
            "Risk Factors"
        ]
        
        # Display each section
        for section_name in sections:
            if section_name in analysis and analysis[section_name]:
                console.print(f"\n[bold]{section_name}[/bold]")
                console.print(Panel(
                    Text("\n".join(f"• {item}" for item in analysis[section_name]), style="white"),
                    border_style="white"
                ))
        
        # Get conversation context
        context = get_conversation_context(state)
        
        # Get user feedback with natural language input
        console.print("\n[bold yellow]Please review the analysis above and provide your feedback.[/bold yellow]")
        console.print("[dim]You can:[/dim]")
        console.print("• Accept the analysis as is")
        console.print("• Request updates to any sections (additions, removals, or modifications)")
        console.print("\n[dim]Please describe what changes you'd like to make in your own words.[/dim]")
        
        user_response = input("\nYour feedback: ").strip()
        
        # Update conversation memory with user's feedback
        conversation_memory = update_conversation_memory(
            state,
            "user",
            f"User provided feedback on analysis: {user_response}"
        )
        
        # Simplified interpretation prompt
        interpretation_prompt = f"""Given the following user response about PRD analysis, determine if they want to:
        1. Accept the analysis as is
        2. Make updates to the analysis
        
        Return ONLY one of these exact strings:
        - "accept" if the user wants to accept the analysis as is
        - "update" if the user wants to make any changes
        
        User response: "{user_response}"
        
        Return ONLY the action string, nothing else."""
        
        interpretation = llm.invoke([{"role": "user", "content": interpretation_prompt}]).content.strip().lower()
        
        # Log token usage
        log_token_usage(interpretation_prompt, interpretation, "review_analysis")
        
        # Update memory with interpretation
        conversation_memory = update_conversation_memory(
            {**state, "conversation_memory": conversation_memory},
            "assistant",
            interpretation
        )
        
        # Parse the interpretation
        if interpretation == "accept":
            logger.info("User accepted the analysis")
            return {
                **state,
                "review_feedback": "accept",
                "messages": state.get("messages", []) + [
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": "Analysis accepted"}
                ],
                "conversation_memory": conversation_memory
            }
        else:
            # Store the update request
            revision_requests = state.get("revision_requests", [])
            revision_requests.append(user_response)  # Store the full feedback
            
            logger.info(f"User requested updates: {user_response}")
            return {
                **state,
                "review_feedback": "update",
                "revision_requests": revision_requests,
                "messages": state.get("messages", []) + [
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": "Updates requested"}
                ],
                "conversation_memory": conversation_memory
            }
            
    except Exception as e:
        logger.error(f"Error in review_analysis: {e}")
        # On error, default to accept to allow the process to continue
        return {
            **state,
            "review_feedback": "accept",
            "messages": state.get("messages", []) + [
                {"role": "system", "content": f"Error in review, defaulting to accept: {str(e)}"}
            ]
        }

def revise_analysis(state: EstimationState) -> EstimationState:
    """Revise the PRD analysis based on user feedback."""
    try:
        # Get revision requests
        revision_requests = state.get("revision_requests", [])
        
        if not revision_requests:
            logger.info("No revision requests found, proceeding to summarize")
            return {
                **state,
                "review_feedback": "accept"  # Force accept if no revisions
            }
        
        # Load current analysis from file
        current_file = state.get("current_analysis_file")
        if not current_file or not os.path.exists(current_file):
            raise ValueError("No analysis file available for revision")
        
        current_analysis = load_analysis_from_file(current_file)
        
        # Get conversation context
        context = get_conversation_context(state)
        
        # Create a prompt for revision that handles mixed feedback
        revision_prompt = f"""Please update the PRD analysis based on the following feedback.
        The feedback may include additions, removals, or modifications to any section.
        Return ONLY a valid JSON object with the same structure as the current analysis.
        Do not include any other text or explanation.
        
        Current Analysis:
        {json.dumps(current_analysis, indent=2)}
        
        User Feedback:
        {json.dumps(revision_requests, indent=2)}
        
        Recent Context:
        {context}
        
        Instructions:
        1. Process all feedback as updates to the analysis
        2. Add new items where requested
        3. Remove items that should be removed
        4. Modify existing items if needed
        5. Keep the same JSON structure
        6. Return ONLY the JSON object with the updated analysis
        
        Return ONLY the JSON object with the updated analysis."""
        
        # Get revision from LLM
        response = llm.invoke([{"role": "user", "content": revision_prompt}])
        
        # Log token usage
        log_token_usage(revision_prompt, response.content, "revise_analysis")
        
        # Clean the response content
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse the revised analysis
        try:
            revised_analysis = json.loads(content)
            
            # Validate the structure
            required_keys = [
                "Project Overview and Objectives",
                "Key Components/Modules",
                "Timeline and Milestones",
                "Critical Assumptions",
                "Dependencies and Constraints",
                "Resource Requirements",
                "Risk Factors"
            ]
            
            # Ensure all required keys exist
            for key in required_keys:
                if key not in revised_analysis:
                    revised_analysis[key] = []
                elif not isinstance(revised_analysis[key], list):
                    revised_analysis[key] = [str(revised_analysis[key])]
                else:
                    # Convert all items to strings
                    revised_analysis[key] = [str(item) for item in revised_analysis[key]]
            
            # Update revision history
            revision_history = state.get("revision_history", [])
            revision_history.append({
                "timestamp": datetime.now().isoformat(),
                "requests": revision_requests,
                "changes": {
                    "before": current_analysis,
                    "after": revised_analysis
                }
            })
            
            # Save revised analysis to new file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filepath = save_analysis_to_file(revised_analysis, timestamp)
            
            # Update memory with revision
            conversation_memory = update_conversation_memory(
                state,
                "assistant",
                f"Updated analysis based on user feedback"
            )
            
            logger.info("Successfully updated analysis")
            return {
                **state,
                "prd_analysis": revised_analysis,
                "current_analysis_file": new_filepath,
                "revision_history": revision_history,
                "revision_requests": [],  # Clear requests after processing
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": "Analysis has been updated based on your feedback."}
                ],
                "conversation_memory": conversation_memory
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse updated analysis: {e}")
            logger.error(f"Raw response: {content}")
            # On parsing error, keep the current analysis and proceed to review
            return {
                **state,
                "messages": state.get("messages", []) + [
                    {"role": "system", "content": "Failed to parse updated analysis, keeping current version."}
                ]
            }
            
    except Exception as e:
        logger.error(f"Error in revise_analysis: {e}")
        # On error, keep the current analysis and proceed to review
        return {
            **state,
            "messages": state.get("messages", []) + [
                {"role": "system", "content": f"Error in revision, keeping current version: {str(e)}"}
            ]
        }

def summarize_analysis(state: EstimationState) -> EstimationState:
    """Create a final summary of the accepted analysis."""
    try:
        analysis = state["prd_analysis"]
        revision_history = state.get("revision_history", [])
        
        if not analysis:
            raise ValueError("No analysis results found in state")
        
        # Create summary panels
        console.print("\n[bold green]Final Analysis Summary[/bold green]")
        
        # Summary of key points
        console.print("\n[bold]Key Points:[/bold]")
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
                items = analysis[section_name]
                if items:  # Only show sections with content
                    console.print(f"\n[bold]{section_name}[/bold]")
                    console.print(Panel(
                        Text("\n".join(f"• {item}" for item in items[:3]), style="white"),  # Show top 3 items
                        border_style="white"
                    ))
        
        # Summary of revision history
        if revision_history:
            console.print("\n[bold]Revision History:[/bold]")
            for i, revision in enumerate(revision_history, 1):
                console.print(f"\nRevision {i}:")
                console.print(Panel(
                    Text(f"Requests: {json.dumps(revision['requests'], indent=2)}", style="dim"),
                    border_style="dim"
                ))
        
        # Save final analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = save_analysis_to_file(analysis, timestamp)
        
        # Add summary to state
        summary = {
            "timestamp": timestamp,
            "output_path": output_path,
            "total_sections": len([k for k in analysis.keys() if not k.startswith("_")]),
            "total_items": sum(len(v) for k, v in analysis.items() if not k.startswith("_")),
            "revision_count": len(revision_history)
        }
        
        # Update memory with summary
        conversation_memory = update_conversation_memory(
            state,
            "assistant",
            f"Analysis complete! Summary: {json.dumps(summary, indent=2)}"
        )
        
        return {
            **state,
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": f"Analysis complete! Results saved to: {output_path}"}
            ],
            "conversation_memory": conversation_memory
        }
        
    except Exception as e:
        logger.error(f"Error in summarize_analysis: {e}")
        return state

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
            "max_iterations": 10    # Allow multiple revisions
        }
    )
    
    # Add nodes
    builder.add_node("analyze_prd", analyze_prd)
    builder.add_node("review_analysis", review_analysis)
    builder.add_node("revise_analysis", revise_analysis)
    builder.add_node("summarize_analysis", summarize_analysis)
    
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
    
    # Add final edge to end
    builder.add_edge("summarize_analysis", END)
    
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