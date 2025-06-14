# work-breakdown/module_breakdown_generator.py (modified)
"""
Module Breakdown Generator

This module combines historical data from the vector store and PRD data
to generate a high-level module breakdown using OpenAI's LLM.
"""
import json
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from openai_client import client
from vector_store_client import MockVectorStore
from prd_processor import PRDProcessor, PRDData  # Updated import

class ModuleItem(BaseModel):
    """A single module in the module breakdown."""
    name: str = Field(..., description="Name of the module")
    description: str = Field(..., description="Brief description of the module")
    estimated_hours: int = Field(..., description="Estimated hours for this module")

class ModuleBreakdown(BaseModel):
    """The complete module breakdown for a project."""
    modules: List[ModuleItem] = Field(..., description="List of modules in the breakdown")

class ModuleBreakdownGenerator:
    """
    Generates a high-level module breakdown by combining historical data
    and PRD data using OpenAI's LLM.
    """
    
    def __init__(self, prd_data: PRDData = None):
        """
        Initialize the generator with vector store and PRD data.
        
        Args:
            prd_data: Optional PRDData object. If None, you'll need to load it later.
        """
        self.vector_store = MockVectorStore()
        self.prd_data = prd_data
    
    def load_prd_data(self, pdf_path: str) -> PRDData:
        """
        Load PRD data from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PRDData object
        """
        processor = PRDProcessor()
        self.prd_data = processor.process_prd(pdf_path)
        return self.prd_data
    
    def _format_historical_data(self, historical_projects: List[Dict[str, Any]]) -> str:
        """
        Format historical project data for inclusion in the prompt.
        
        Args:
            historical_projects: List of historical projects with module breakdowns
            
        Returns:
            Formatted string of historical data
        """
        if not historical_projects:
            return "No similar historical projects found."
        
        result = "Similar Historical Projects:\n\n"
        for i, project in enumerate(historical_projects):
            result += f"Project {i+1}: {project['project_name']} ({project['project_type']})\n"
            result += "Module Breakdown:\n"
            for module in project['module_breakdown']:
                result += f"- {module['name']} ({module['estimated_hours']} hours): {module['description']}\n"
            result += "\n"
        
        return result
    
    def generate_module_breakdown(self) -> ModuleBreakdown:
        """
        Generate a high-level module breakdown using OpenAI's LLM.
        
        Returns:
            ModuleBreakdown object containing the modules
        """
        if not self.prd_data:
            raise ValueError("PRD data not loaded. Call load_prd_data() first.")
        
        # Get PRD summary
        prd_processor = PRDProcessor()
        prd_summary = prd_processor.get_requirements_summary(self.prd_data)
        
        # Query vector store for similar historical projects
        historical_projects = self.vector_store.query_similar_projects(prd_summary)
        
        # Format historical data for the prompt
        historical_data_text = self._format_historical_data(historical_projects)
        
        # Create the prompt
        system_prompt = """
        You are an expert software architect and project estimator. Your task is to analyze the given PRD (Product Requirements Document) 
        and generate a high-level module breakdown for the project. Use the historical project data provided as reference.
        
        Your module breakdown should:
        1. Cover all functional and non-functional requirements in the PRD
        2. Be organized into logical, cohesive modules
        3. Include a brief description for each module
        4. Include an estimated number of hours for each module
        5. Be consistent with similar historical projects where appropriate
        
        Respond with a JSON object that follows this structure:
        {
            "modules": [
                {
                    "name": "Module Name",
                    "description": "Brief description of the module",
                    "estimated_hours": 100
                },
                ...
            ]
        }
        """
        
        user_prompt = f"""
        # PRD Summary
        {prd_summary}
        
        # Historical Data
        {historical_data_text}
        
        Based on the PRD and historical data, generate a high-level module breakdown for this project.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        # Parse the response
        response_content = response.choices[0].message.content
        module_breakdown_dict = json.loads(response_content)
        
        # Convert to Pydantic model for validation
        module_breakdown = ModuleBreakdown(**module_breakdown_dict)
        
        return module_breakdown

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        generator = ModuleBreakdownGenerator()
        generator.load_prd_data(pdf_path)
        breakdown = generator.generate_module_breakdown()
        
        print("Generated Module Breakdown:")
        print("--------------------------")
        for module in breakdown.modules:
            print(f"{module.name} ({module.estimated_hours} hours)")
            print(f"  {module.description}")
            print()
    else:
        print("Please provide a path to a PRD PDF file.")
        print("Usage: python module_breakdown_generator.py path/to/prd.pdf")