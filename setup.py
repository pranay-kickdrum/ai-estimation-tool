from setuptools import setup, find_packages

setup(
    name="ai-estimation-tool",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-openai>=0.0.2",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.0",
        "chromadb>=0.4.22",
        "python-magic>=0.4.27",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "unstructured>=0.10.30",
        "openai>=1.0.0",
    ],
) 