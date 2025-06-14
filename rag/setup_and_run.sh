#!/bin/bash

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

echo "uv found. Setting up the environment..."

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

echo "Environment setup complete."

echo "Running create_database.py..."
.venv/bin/python rag/create_database.py

echo "Running query_data.py..."
.venv/bin/python rag/query_data.py

echo "Running compare_embeddings.py..."
.venv/bin/python rag/compare_embeddings.py

echo "All scripts executed successfully!"

echo "To activate the uv virtual environment, run:"
echo "  source .venv/bin/activate"

echo "To run the scripts, use:"
echo "  python rag/create_database.py"
echo "  python rag/query_data.py"
echo "  python rag/compare_embeddings.py"

echo "All done!" 