import os

env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here

# Optional: Model Configuration
OPENAI_MODEL_NAME=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7

# Optional: Database Configuration
CHROMA_DB_DIR=./data/chroma_db
"""

with open('.env', 'w') as f:
    f.write(env_content)

print("Created .env file. Please edit it to add your OpenAI API key.") 