from typing import List, Dict, Any
import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from utils.config import settings

class DocumentSummarizer:
    """Handles document processing and summarization using AI."""
    
    def __init__(self):
        """Initialize the summarizer with OpenAI model and text splitter."""
        self.llm = ChatOpenAI(
            model_name=settings.DEFAULT_MODEL,
            temperature=0.1,
            api_key=settings.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Define the summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing software project requirements documents (PRDs).
            Your task is to create a comprehensive summary of the document that captures:
            1. Project Overview and Objectives
            2. Key Deliverables
            3. Technical Requirements
            4. Functional Requirements
            5. Constraints and Assumptions
            6. Timeline and Milestones
            7. Stakeholders and Roles
            
            Format your response as a structured JSON with these sections.
            If any section is not present in the document, mark it as "Not specified".
            Be concise but thorough in your analysis."""),
            ("user", "{text}")
        ])
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt
        )
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process a list of documents and generate a comprehensive summary.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            Dictionary containing the processed information and summary
        """
        logging.info(f"Processing {len(documents)} document(s) for summarization.")
        # Combine all document content
        combined_text = "\n\n".join(doc.page_content for doc in documents)
        
        # Split into manageable chunks
        chunks = self.text_splitter.split_text(combined_text)
        
        # Process each chunk and combine results
        summaries = []
        for chunk in chunks:
            try:
                logging.info("Summarizing a chunk of text.")
                summary = self.summary_chain.invoke({"text": chunk})
                summaries.append(summary)
            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")
                continue
        
        # Combine summaries and generate final summary
        combined_summary = "\n\n".join(str(s) for s in summaries)
        logging.info("Combining summaries and generating final summary.")
        final_summary = self.summary_chain.invoke({"text": combined_summary})
        
        # Extract metadata from documents
        metadata = {
            "source_files": list(set(doc.metadata.get("source", "") for doc in documents)),
            "total_pages": sum(doc.metadata.get("page", 0) for doc in documents),
            "processed_chunks": len(chunks)
        }
        
        logging.info(f"Summarization complete. {len(chunks)} chunks processed.")
        return {
            "summary": final_summary,
            "metadata": metadata,
            "raw_chunks": chunks  # Store chunks for vector storage
        }
    
    def extract_key_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract key entities and relationships from the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing extracted entities and relationships
        """
        # TODO: Implement entity extraction using NER and relation extraction
        # This will be useful for the next phase of the project
        pass 