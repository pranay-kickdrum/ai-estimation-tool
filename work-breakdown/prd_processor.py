# work-breakdown/prd_processor.py
"""
PRD Processor

This module processes PDF or Markdown PRD files and extracts structured information
using LLMs to classify content into different categories.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PRDSection(BaseModel):
    """A section of the PRD with classified content."""
    title: str = Field(..., description="Title of the section")
    content: List[str] = Field(..., description="List of content items in this section")

class PRDData(BaseModel):
    """Structured data extracted from a PRD."""
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Brief description of the project")
    functional_requirements: List[Dict[str, str]] = Field(..., description="Functional requirements")
    non_functional_requirements: List[Dict[str, str]] = Field(..., description="Non-functional requirements")
    business_goals: List[str] = Field(..., description="Business goals")
    technical_constraints: List[str] = Field(..., description="Technical constraints")
    assumptions: List[str] = Field(..., description="Assumptions")
    stakeholders: List[Dict[str, str]] = Field(..., description="Stakeholders")
    risks: List[Dict[str, str]] = Field(..., description="Risks")

class PRDProcessor:
    """
    Processes PDF or Markdown PRD files and extracts structured information using LLMs.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the PRD processor.
        
        Args:
            model_name: The OpenAI model to use
        """
        self.model_name = model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    
    def extract_text_from_markdown(self, file_path: str) -> str:
        """
        Extract text content from a Markdown file.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Extracted text content
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.md', '.markdown']:
            return self.extract_text_from_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are: .pdf, .md, .markdown")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split the text into manageable chunks for processing.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def extract_project_info(self, text: str) -> Dict[str, str]:
        """
        Extract basic project information from the text.
        
        Args:
            text: The text to process
            
        Returns:
            Dictionary with project name and description
        """
        prompt = """
        Extract the project name and a brief description from the following PRD text.
        
        Text:
        {text}
        
        Respond with a JSON object that follows this structure:
        {{
            "project_name": "Name of the project",
            "project_description": "Brief description of the project"
        }}
        """
        
        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert at analyzing PRDs and extracting key information."},
                {"role": "user", "content": prompt.format(text=text[:4000])}  # Use first 4000 chars for project info
            ],
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    
    def classify_section(self, text: str, section_type: str) -> List[Dict[str, str]]:
        """
        Classify text into a specific section type.
        
        Args:
            text: The text to classify
            section_type: The type of section to extract
            
        Returns:
            List of items for the section
        """
        section_descriptions = {
            "functional_requirements": "Features or capabilities the system must provide. These are specific, measurable actions the system should perform. They define what the system must do in response to inputs or under specific conditions.",
            "non_functional_requirements": "Quality attributes like performance, security, usability, reliability, etc. These define how the system should behave and its constraints.",
            "business_goals": "High-level objectives the project aims to achieve from a business perspective.",
            "technical_constraints": "Limitations or requirements related to technology, infrastructure, or implementation that will impact the project's development.",
            "assumptions": "Things assumed to be true for the project to proceed that may impact estimation or timelines if proven false.",
            "stakeholders": "People or groups with interest in or influence over the project.",
            "risks": "Potential issues that could impact the project's success, including their severity, likelihood, and potential mitigation strategies."
        }
        
        section_formats = {
            "functional_requirements": """
                [
                    {{
                        "id": "FR-1",
                        "description": "The system shall [action verb] [object] [condition]",
                        "priority": "High/Medium/Low"
                    }}
                ]
                
                Examples:
                - "The system shall send a confirmation email when an order is placed."
                - "The system shall validate user credentials before granting access."
                - "The system shall allow users to reset their password via email verification."
                - "The system shall process payments within 3 seconds of submission."
                - "The system shall generate a report when the user selects a date range."
            """,
            "non_functional_requirements": """
                [
                    {{
                        "id": "NFR-1",
                        "description": "The system shall [quality attribute]",
                        "category": "Performance/Security/Usability/Reliability/etc."
                    }}
                ]
                
                Examples:
                - "The system shall respond to user requests within 0.5 seconds under normal load."
                - "The system shall encrypt all user data using AES-256 encryption."
                - "The system shall be available 99.9% of the time."
                - "The system shall support at least 1000 concurrent users."
                - "The system shall be compatible with Chrome, Firefox, and Safari browsers."
            """,
            "business_goals": """
                [
                    "Goal 1 description",
                    "Goal 2 description"
                ]
                
                Examples:
                - "Increase customer retention by 15% within 6 months of launch"
                - "Reduce operational costs by automating manual processes"
                - "Expand market reach to new geographic regions"
                - "Improve customer satisfaction scores by at least 10 points"
            """,
            "technical_constraints": """
                [
                    "Constraint 1 description",
                    "Constraint 2 description"
                ]
                
                Examples (limit to the 10 most important constraints):
                - "Must integrate with existing Oracle database system"
                - "Must be compatible with legacy API version 2.1"
                - "Must operate within the existing AWS infrastructure"
                - "Must comply with GDPR data protection requirements"
                - "Development must use the company's standard technology stack (React, Node.js, MongoDB)"
            """,
            "assumptions": """
                [
                    "Assumption 1 description",
                    "Assumption 2 description"
                ]
                
                Examples (limit to the 10 most critical assumptions that impact estimation or timelines):
                - "The client will provide all necessary API documentation within the first week"
                - "The existing database schema will not require significant modifications"
                - "Third-party payment gateway integration will not change during development"
                - "The development team will have access to production-like test environments"
                - "Stakeholders will be available for weekly review meetings"
            """,
            "stakeholders": """
                [
                    {{
                        "role": "Product Manager",
                        "responsibilities": "Define product requirements"
                    }}
                ]
                
                Examples:
                - "Product Owner: Responsible for defining requirements and prioritizing features"
                - "IT Operations: Responsible for deployment and infrastructure support"
                - "Marketing Department: Provides branding guidelines and promotional requirements"
                - "End Users: Will use the system daily for core business operations"
            """,
            "risks": """
                [
                    {{
                        "description": "Risk description",
                        "impact": "High/Medium/Low",
                        "mitigation": "How to mitigate this risk"
                    }}
                ]
                
                Examples (focus on significant risks):
                - "Integration with legacy systems may take longer than estimated due to poor documentation"
                - "Key technical staff may leave during the project, causing knowledge loss"
                - "Third-party API changes could require significant rework"
                - "Security vulnerabilities may be discovered during penetration testing"
                - "Regulatory requirements may change during development"
            """
        }
        
        prompt = f"""
        Extract all {section_type.replace('_', ' ')} from the following PRD text.
        
        {section_descriptions[section_type]}
        
        Text:
        {{text}}
        
        Respond with a JSON array that follows this structure:
        {section_formats[section_type]}
        
        Be thorough and ensure you don't miss any relevant information. If none are found, return an empty array.
        """
        
        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert at analyzing PRDs and extracting key information."},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        # The result might be wrapped in a key with the section name
        if section_type in result:
            return result[section_type]
        # Or it might be the direct array
        for key in result:
            if isinstance(result[key], list):
                return result[key]
        return []
    
    def process_prd(self, file_path: str) -> PRDData:
        """
        Process a PRD file and extract structured information.
        
        Args:
            file_path: Path to the PRD file (PDF or Markdown)
            
        Returns:
            PRDData object with structured information
        """
        # Extract text from file
        print(f"Extracting text from {file_path}...")
        text = self.extract_text(file_path)
        
        # Split text into chunks
        print("Splitting text into chunks...")
        chunks = self.split_text(text)
        
        # Extract project info
        print("Extracting project information...")
        project_info = self.extract_project_info(chunks[0])
        
        # Process each chunk for different sections
        all_sections = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "business_goals": [],
            "technical_constraints": [],
            "assumptions": [],
            "stakeholders": [],
            "risks": []
        }
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            for section_type in all_sections.keys():
                print(f"  Extracting {section_type.replace('_', ' ')}...")
                section_items = self.classify_section(chunk, section_type)
                
                # For list types, extend the list
                if section_type in ["business_goals", "technical_constraints", "assumptions"]:
                    all_sections[section_type].extend(section_items)
                # For dict types, append unique items
                else:
                    existing_ids = set()
                    if section_type in ["functional_requirements", "non_functional_requirements"]:
                        existing_ids = {item["id"] for item in all_sections[section_type] if "id" in item}
                    
                    for item in section_items:
                        if section_type not in ["functional_requirements", "non_functional_requirements"] or \
                           "id" not in item or item["id"] not in existing_ids:
                            all_sections[section_type].append(item)
        
        # Create PRDData object
        prd_data = PRDData(
            project_name=project_info["project_name"],
            project_description=project_info["project_description"],
            functional_requirements=all_sections["functional_requirements"],
            non_functional_requirements=all_sections["non_functional_requirements"],
            business_goals=all_sections["business_goals"],
            technical_constraints=all_sections["technical_constraints"],
            assumptions=all_sections["assumptions"],
            stakeholders=all_sections["stakeholders"],
            risks=all_sections["risks"]
        )
        
        return prd_data
    
    def get_requirements_summary(self, prd_data: PRDData) -> str:
        """
        Get a text summary of the requirements for use in prompts.
        
        Args:
            prd_data: The PRD data object
            
        Returns:
            String containing a summary of the requirements
        """
        summary = f"Project: {prd_data.project_name}\n\n"
        summary += f"Description: {prd_data.project_description}\n\n"
        
        summary += "Functional Requirements:\n"
        for req in prd_data.functional_requirements:
            priority = req.get("priority", "")
            priority_str = f" (Priority: {priority})" if priority else ""
            summary += f"- {req['description']}{priority_str}\n"
        
        summary += "\nNon-Functional Requirements:\n"
        for req in prd_data.non_functional_requirements:
            category = req.get("category", "")
            category_str = f" (Category: {category})" if category else ""
            summary += f"- {req['description']}{category_str}\n"
        
        summary += "\nBusiness Goals:\n"
        for goal in prd_data.business_goals:
            summary += f"- {goal}\n"
        
        summary += "\nTechnical Constraints:\n"
        for constraint in prd_data.technical_constraints:
            summary += f"- {constraint}\n"
        
        return summary

if __name__ == "__main__":
    # Example usage
    processor = PRDProcessor()
    file_path = "./data/Testmo CRM & Stripe Research Project.md";
    
    if os.path.exists(file_path):
        try:
            prd_data = processor.process_prd(file_path)
            
            # Save to JSON file
            output_path = os.path.splitext(file_path)[0] + "_processed.json"
            with open(output_path, "w") as f:
                f.write(prd_data.json(indent=2))
            
            print(f"Processed PRD saved to {output_path}")
            
            # Print summary
            print("\nPRD Summary:")
            print("===========")
            print(processor.get_requirements_summary(prd_data))
        except ValueError as e:
            print(f"Error: {str(e)}")
    else:
        print(f"File not found: {file_path}")