#!/bin/bash

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Install dependencies
uv pip install -r requirements.txt

echo "Environment setup complete."

# Source the venv automatically if on macOS/Linux
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
    echo "Sourcing the virtual environment..."
    source .venv/bin/activate
    echo "Virtual environment activated."
    echo "You can now run your scripts with:"
    echo "  python3 rag/create_database.py"
    echo "  python3 rag/query_data.py"
    echo "  python3 rag/compare_embeddings.py"
else
    echo "On Windows, activate with:"
    echo "  .venv\\Scripts\\activate"
    echo "Then run your scripts with:"
    echo "  python3 rag/create_database.py"
    echo "  python3 rag/query_data.py"
    echo "  python3 rag/compare_embeddings.py"
fi 