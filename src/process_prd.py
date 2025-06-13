import argparse
import json
from pathlib import Path
from document_processing.document_loader import DocumentLoader
from document_processing.document_store import DocumentStore
from ai.summarizer import DocumentSummarizer
from utils.config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_prd(file_path: str, project_id: str) -> None:
    """
    Process a PRD document and store it for future use.
    
    Args:
        file_path: Path to the PRD document
        project_id: Unique identifier for the project
    """
    logging.info(f"Starting PRD processing for project: {project_id}")
    # Initialize components
    loader = DocumentLoader()
    summarizer = DocumentSummarizer()
    store = DocumentStore()
    
    try:
        # Load document
        logging.info(f"Loading document: {file_path}")
        documents = loader.load_document(file_path)
        logging.info(f"Loaded {len(documents)} document(s)")
        
        # Process and summarize
        logging.info("Processing and summarizing document(s)...")
        result = summarizer.process_documents(documents)
        logging.info("Document(s) processed and summarized.")
        
        # Print summary for review
        print("\nGenerated Summary:")
        print("=" * 80)
        print(result["summary"])
        print("=" * 80)
        
        # Save summary to output folder
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / f"{project_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"summary": result["summary"], "metadata": result["metadata"]}, f, indent=2)
        logging.info(f"Summary saved to {summary_path}")
        
        # Ask for user confirmation
        response = input("\nDo you want to store this processed document in the vector DB? (y/n): ")
        if response.lower() == 'y':
            # Store documents and metadata
            logging.info("Storing document in vector DB...")
            store.store_documents(
                documents=documents,
                metadata=result["metadata"],
                project_id=project_id
            )
            logging.info(f"Document stored successfully with project ID: {project_id}")
        else:
            logging.info("Document processing cancelled by user.")
        logging.info(f"PRD processing completed for project: {project_id}")
            
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process PRD documents for project estimation")
    parser.add_argument("file_path", help="Path to the PRD document (should be in the input folder)")
    parser.add_argument("--project-id", required=True, help="Unique identifier for the project")
    
    args = parser.parse_args()
    # Prepend input folder if not already present
    file_path = args.file_path
    if not file_path.startswith("input/"):
        file_path = str(Path("input") / file_path)
    process_prd(file_path, args.project_id)

if __name__ == "__main__":
    main() 