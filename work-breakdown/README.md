# Module Breakdown Generator

This tool generates high-level module breakdowns for software projects by combining:
1. Historical project data (via RAG - Retrieval Augmented Generation)
2. New PRD (Product Requirements Document) and supporting documentation

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file based on the `env_example` file:
```bash
cp env_example .env
```
4. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the module breakdown generator:

```bash
python main.py
```

### Options

- `--output`: Specify the output file path (default: `module_breakdown.json`)
- `--format`: Specify the output format (`json` or `text`, default: `text`)

Example:
```bash
python main.py --output my_breakdown.json --format json
```

## Project Structure

- `main.py`: Entry point for the application
- `module_breakdown_generator.py`: Core logic for generating module breakdowns
- `openai_client.py`: OpenAI API client setup
- `vector_store_client.py`: Mock vector store client (replace with your actual vector store)
- `prd_data.py`: Mock PRD data (replace with your actual PRD analysis component)

## Customization

### Using Your Own Vector Store

Replace the `MockVectorStore` class in `vector_store_client.py` with your actual vector store implementation. The interface should remain the same:

```python
def query_similar_projects(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
    # Your implementation here
    pass
```

### Using Your Own PRD Analysis

Replace the `MockPRDData` class in `prd_data.py` with your actual PRD analysis component. The interface should remain the same:

```python
def get_requirements_summary(self) -> str:
    # Your implementation here
    pass
```

## Example Output

The tool generates a module breakdown in the following format:

```json
{
  "modules": [
    {
      "name": "Core Infrastructure (IAC)",
      "description": "Terraform scripts for AWS resources",
      "estimated_hours": 120
    },
    {
      "name": "CI/CD Implementation",
      "description": "Jenkins pipelines for deployment",
      "estimated_hours": 80
    }
  ]
}
```

## Next Steps

1. Integrate with your actual vector store for historical data
2. Integrate with your PRD analysis component
3. Extend the module breakdown with more detailed work items
4. Add support for team composition and sprint planning 