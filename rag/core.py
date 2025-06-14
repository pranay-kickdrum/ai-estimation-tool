"""Core module for shared dependencies.

This module contains shared dependencies used across the estimation tool,
including state management, LLM configuration, and common utilities.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
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
import tiktoken
import textwrap
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Set up rich console
console = Console()

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Constants for memory management
MAX_CONTEXT_TOKENS = 2000  # Maximum tokens to keep in context
MAX_MEMORY_ITEMS = 5      # Maximum number of recent interactions to keep in full

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

class EstimationState(TypedDict):
    """State management for the PRD analysis process.
    
    This TypedDict defines the structure of the state that flows through the analysis graph.
    The state maintains:
    1. Input document content
    2. Analysis results and their file location
    3. Review and revision tracking
    4. Clarification questions and responses
    5. Conversation memory
    
    Attributes:
        prd_doc: The content of the PRD document being analyzed
        prd_analysis: The current analysis results, including key components, timeline, etc.
        current_analysis_file: Path to the JSON file containing the current analysis
        review_feedback: User's feedback type (accept/revise/add/remove)
        revision_requests: List of specific revision requests from the user
        revision_history: History of changes made during revisions
        clarification_questions: Generated questions for requirement clarification
        clarification_file: Path to the JSON file containing clarification questions
        clarification_responses: User's responses to clarification questions
        clarification_history: History of question updates and responses
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
    
    # Clarification tracking
    clarification_questions: Optional[Dict[str, Any]]  # Generated clarification questions and summary
    clarification_file: Optional[str]  # Path to the clarification questions file
    clarification_responses: Optional[List[Dict[str, Any]]]  # User's responses to questions
    clarification_history: Optional[List[Dict[str, Any]]]  # History of question updates and responses
    
    # Conversation tracking
    messages: Annotated[list, add_messages]
    conversation_memory: Optional[dict]  # Store the conversation memory state

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