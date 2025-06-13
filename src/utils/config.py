from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str
    
    # Optional OpenAI Configuration
    OPENAI_MODEL_NAME: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.7
    
    # Database Configuration
    CHROMA_DB_DIR: str = "./data/chroma_db"
    
    # Additional settings
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma"
    DOCUMENT_STORE_DIRECTORY: str = "data/documents"
    
    # Model settings
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(settings.DOCUMENT_STORE_DIRECTORY, exist_ok=True) 