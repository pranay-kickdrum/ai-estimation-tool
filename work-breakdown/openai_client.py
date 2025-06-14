"""
OpenAI client setup for the module breakdown generator.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Initialize LangChain OpenAI client for more complex chains
llm = ChatOpenAI(
    model="gpt-4o",  # Using GPT-4o for best reasoning capabilities
    temperature=0.2,  # Lower temperature for more deterministic outputs
    api_key=api_key
)

def test_openai_connection():
    """Test the OpenAI connection by sending a simple request."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, are you working?"}
            ]
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # Test the connection when run directly
    success, message = test_openai_connection()
    if success:
        print("✅ OpenAI connection successful!")
        print(f"Response: {message}")
    else:
        print(f"❌ OpenAI connection failed: {message}") 