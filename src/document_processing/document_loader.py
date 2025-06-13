from typing import List, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain.schema import Document
from utils.config import settings

class DocumentLoader:
    """Handles loading of different document formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects containing the document content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
            logging.error(f"Unsupported file format: {file_path.suffix}")
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logging.info(f"Loading file: {file_path}")
        
        # Select appropriate loader based on file extension
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in {'.doc', '.docx'}:
            loader = Docx2txtLoader(str(file_path))
        else:
            # Fallback to unstructured loader for other text files
            loader = UnstructuredFileLoader(str(file_path))
        
        # Load and return documents
        return loader.load()
    
    @staticmethod
    def load_documents(directory: str, recursive: bool = False) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to the directory containing documents
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects containing all document contents
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        
        # Get all files in directory
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    logging.info(f"Loading document: {file_path}")
                    documents = DocumentLoader.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {str(e)}")
                    continue
        
        return all_documents 