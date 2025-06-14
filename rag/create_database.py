from dotenv import load_dotenv
load_dotenv()
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai 
import os
import shutil
from langchain_community.document_loaders import CSVLoader

# Load environment variables. Assumes that project contains .env file with API keys
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

# Read paths from environment variables, with defaults
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
DATA_PATH = os.getenv("DATA_PATH", "data/")

# Read chunking parameters from environment variables, with defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


def main():
    print(f"CHROMA_PATH from env: {CHROMA_PATH}")
    print(f"DATA_PATH from env: {DATA_PATH}")
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    pert_docs = load_pert_csv_documents()
    all_docs = documents + pert_docs
    chunks = split_text(all_docs)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    print(f"Starting chunking process for {len(documents)} documents...")
    print(f"Chunking parameters from env: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Log the first 3 chunks for inspection
    for i, document in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Incrementally add new chunks to the existing database, or create if not exists.
    if os.path.exists(CHROMA_PATH):
        print(f"Chroma database exists at {CHROMA_PATH}. Adding new chunks...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
        db.add_documents(chunks)
    else:
        print(f"Chroma database does not exist at {CHROMA_PATH}. Creating new database...")
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
        )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def load_pert_csv_documents():
    pert_dir = os.getenv("PERT_DIR", "pert")
    documents = []
    if not os.path.exists(pert_dir):
        print(f"PERT directory '{pert_dir}' does not exist.")
        return documents
    for filename in os.listdir(pert_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(pert_dir, filename)
            print(f"Loading PERT CSV file: {file_path}")
            csv_loader = CSVLoader(file_path=file_path)
            documents.extend(csv_loader.load())
    print(f"Loaded {len(documents)} documents from PERT CSV files.")
    return documents


if __name__ == "__main__":
    main()