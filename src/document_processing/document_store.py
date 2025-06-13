from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from utils.config import settings
import logging

class DocumentStore:
    """Handles document storage and retrieval using ChromaDB."""
    
    def __init__(self):
        """Initialize the document store with ChromaDB and OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="prd_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LangChain vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name="prd_documents",
            embedding_function=self.embeddings
        )
    
    def store_documents(self, 
                       documents: List[Document], 
                       metadata: Dict[str, Any],
                       project_id: str) -> None:
        """
        Store documents in the vector store.
        
        Args:
            documents: List of Document objects to store
            metadata: Additional metadata about the documents
            project_id: Unique identifier for the project
        """
        # Add project metadata to each document
        for doc in documents:
            doc.metadata.update({
                "project_id": project_id,
                **metadata
            })
        logging.info(f"Storing {len(documents)} document(s) in vector store for project {project_id}.")
        self.vector_store.add_documents(documents)
        
        # Store metadata separately
        metadata_path = Path(settings.DOCUMENT_STORE_DIRECTORY) / f"{project_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Metadata stored at {metadata_path}")
    
    def search_similar(self, 
                      query: str, 
                      n_results: int = 5,
                      project_id: Optional[str] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            project_id: Optional project ID to filter results
            
        Returns:
            List of similar documents
        """
        # Prepare filter
        filter_dict = {"project_id": project_id} if project_id else None
        
        # Search vector store
        return self.vector_store.similarity_search(
            query,
            k=n_results,
            filter=filter_dict
        )
    
    def get_project_metadata(self, project_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project metadata
        """
        metadata_path = Path(settings.DOCUMENT_STORE_DIRECTORY) / f"{project_id}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found for project: {project_id}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_projects(self) -> List[str]:
        """
        List all stored project IDs.
        
        Returns:
            List of project IDs
        """
        metadata_files = Path(settings.DOCUMENT_STORE_DIRECTORY).glob("*_metadata.json")
        return [f.stem.replace("_metadata", "") for f in metadata_files] 