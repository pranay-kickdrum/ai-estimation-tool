# AI Project Estimation Tool

An AI-powered tool for processing PRDs and generating software project estimates.

## Features

- Document Processing (PDF, DOC, DOCX)
- AI-powered PRD analysis and summarization
- Vector storage for semantic search
- Project scope understanding and module generation
- Task breakdown and estimation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
ai-estimation-tool/
├── src/
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── document_processor.py
│   │   └── document_store.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── summarizer.py
│   │   └── extractor.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── tests/
├── requirements.txt
└── README.md
```

## Usage

[Documentation coming soon]