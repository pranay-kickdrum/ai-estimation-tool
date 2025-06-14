"""Analysis node for PRD analysis.

This module contains the analyze_prd function which processes PRD documents
and extracts key information using parallel processing and semantic analysis.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

from core import (
    EstimationState,
    llm,
    logger,
    console,
    log_token_usage,
    update_conversation_memory,
    SystemMessage,
    split_into_semantic_chunks,
    analyze_chunk
)
from file_ops import save_analysis_to_file

def analyze_prd(state: EstimationState) -> EstimationState:
    """Analyze the document and extract key information using parallel processing.
    
    This function is the entry point for the analysis process. It:
    1. Processes the PRD document into semantic chunks
    2. Analyzes each chunk in parallel
    3. Combines the results into a complete analysis
    4. Saves the analysis to a file
    5. Updates the state with both the analysis and file location
    
    Args:
        state: The current state containing the PRD document
        
    Returns:
        Updated state containing:
        - prd_analysis: The complete analysis results
        - current_analysis_file: Path to the saved analysis file
        - Additional metadata and processing information
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
        
        # Split into semantic chunks using LangChain
        logger.info("Splitting document into semantic chunks...")
        chunks = split_into_semantic_chunks(doc_content, max_tokens=3000)
        unique_sections = len(set(c["section"] for c in chunks))
        logger.info(f"Split document into {len(chunks)} semantic chunks across {unique_sections} sections")
        
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