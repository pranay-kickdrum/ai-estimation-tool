"""Configuration settings for the chatbot."""
from typing import Dict, Any

# Memory settings
MAX_CONTEXT_TOKENS = 2000  # Maximum tokens in conversation context
MAX_MEMORY_ITEMS = 5      # Maximum number of messages to keep in memory

# Model settings
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.7

# Token pricing (per 1K tokens)
TOKEN_PRICES = {
    "gpt-4": {
        "input": 0.03,    # $0.03 per 1K input tokens
        "output": 0.06    # $0.06 per 1K output tokens
    }
}

# Graph node settings
NODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "generate_paragraph": {
        "function_type": "generation",
        "version": "1.0",
        "importance": "high",
        "max_context_tokens": 2000,
        "max_memory_items": 5
    },
    "review_paragraph": {
        "function_type": "review",
        "version": "1.0",
        "importance": "high",
        "max_context_tokens": 2000,
        "max_memory_items": 5
    },
    "bullet_point_generator": {
        "function_type": "extraction",
        "version": "1.0",
        "importance": "medium",
        "max_context_tokens": 2000,
        "max_memory_items": 5
    }
}

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE_PREFIX = "chatbot"

# Prompt templates
SYSTEM_PROMPT = """You are an AI assistant helping to generate and refine content. 
Your role is to maintain context, provide clear feedback, and ensure high-quality output."""

PARAGRAPH_GENERATION_PROMPT = """Generate a detailed paragraph about {topic}. 
Focus on clarity, accuracy, and engaging content."""

REVIEW_PROMPT = """Review the following paragraph and provide specific feedback:
{paragraph}

Consider:
1. Clarity and coherence
2. Accuracy of information
3. Engagement and readability
4. Areas for improvement"""

BULLET_POINT_PROMPT = """Extract key points from this paragraph:
{paragraph}

Focus on:
1. Main ideas
2. Supporting details
3. Important facts or figures
4. Key takeaways""" 