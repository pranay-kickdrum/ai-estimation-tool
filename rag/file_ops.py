"""File operations module for the estimation tool.

This module handles all file operations including:
- Saving and loading analysis files
- Managing output directories
- File validation and error handling
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger("rich")

# Constants
DEFAULT_OUTPUT_DIR = "analysis_output"
DEFAULT_CLARIFICATION_DIR = "clarification_output"

def ensure_output_dir(directory: str = DEFAULT_OUTPUT_DIR) -> str:
    """Ensure the output directory exists.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        str: Absolute path to the directory
    """
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def save_analysis_to_file(
    analysis: Dict[str, Any],
    timestamp: Optional[str] = None,
    prefix: str = "prd_analysis_",
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """Save the analysis results to a JSON file.
    
    This function handles saving the analysis results to a JSON file in the specified output directory.
    Each analysis is saved with a timestamp to maintain a history of revisions.
    
    Args:
        analysis: The analysis results dictionary containing all sections and metadata
        timestamp: Optional timestamp to use in filename. If None, generates new timestamp
        prefix: Optional prefix for the filename. Defaults to "prd_analysis_"
        output_dir: Directory to save the file in. Defaults to DEFAULT_OUTPUT_DIR
    
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
    
    # Ensure output directory exists
    abs_output_dir = ensure_output_dir(output_dir)
    
    # Create filename with timestamp
    output_path = os.path.join(abs_output_dir, f"{prefix}{timestamp}.json")
    
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

def save_clarification_questions(
    questions: Dict[str, Any],
    timestamp: Optional[str] = None,
    output_dir: str = DEFAULT_CLARIFICATION_DIR
) -> str:
    """Save clarification questions to a JSON file.
    
    Args:
        questions: Dictionary containing the clarification questions and metadata
        timestamp: Optional timestamp to use in filename. If None, generates new timestamp
        output_dir: Directory to save the file in. Defaults to DEFAULT_CLARIFICATION_DIR
    
    Returns:
        str: Path to the saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    abs_output_dir = ensure_output_dir(output_dir)
    
    # Create filename with timestamp
    output_path = os.path.join(abs_output_dir, f"clarification_questions_{timestamp}.json")
    
    # Save to file with pretty printing
    with open(output_path, 'w') as f:
        json.dump(questions, f, indent=2)
    
    logger.info(f"Clarification questions saved to: {output_path}")
    return output_path

def get_latest_analysis_file(output_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """Get the path to the most recent analysis file.
    
    Args:
        output_dir: Directory to search in. Defaults to DEFAULT_OUTPUT_DIR
        
    Returns:
        Optional[str]: Path to the latest analysis file, or None if no files found
    """
    try:
        abs_output_dir = ensure_output_dir(output_dir)
        files = [f for f in os.listdir(abs_output_dir) if f.endswith('.json')]
        if not files:
            return None
        
        # Sort by modification time, newest first
        latest_file = max(
            files,
            key=lambda f: os.path.getmtime(os.path.join(abs_output_dir, f))
        )
        return os.path.join(abs_output_dir, latest_file)
    except Exception as e:
        logger.error(f"Error finding latest analysis file: {e}")
        return None

def cleanup_old_files(
    max_age_days: int = 30,
    output_dirs: List[str] = [DEFAULT_OUTPUT_DIR, DEFAULT_CLARIFICATION_DIR]
) -> None:
    """Clean up old analysis and clarification files.
    
    Args:
        max_age_days: Maximum age of files to keep in days
        output_dirs: List of directories to clean up
    """
    try:
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        for directory in output_dirs:
            abs_dir = ensure_output_dir(directory)
            for filename in os.listdir(abs_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(abs_dir, filename)
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up old files: {e}") 