"""
Mock PRD data to simulate the input from the PRD analysis component.
In a real implementation, this would come from the PRD analysis module.
"""
from typing import Dict, List, Any

class MockPRDData:
    """
    A mock class to simulate the PRD data that would be provided by the PRD analysis component.
    Replace this with your actual PRD data structure.
    """
    
    def __init__(self):
        """Initialize with sample PRD data."""
        self.prd_data = {
            "project_name": "Cloud-Native Application Modernization",
            "project_description": "Modernize the legacy monolithic application into a cloud-native microservices architecture",
            "functional_requirements": [
                {
                    "id": "FR-1",
                    "description": "Migrate the application to AWS cloud infrastructure",
                    "priority": "High"
                },
                {
                    "id": "FR-2",
                    "description": "Implement containerization for all services",
                    "priority": "High"
                },
                {
                    "id": "FR-3",
                    "description": "Set up CI/CD pipelines for automated deployment",
                    "priority": "Medium"
                },
                {
                    "id": "FR-4",
                    "description": "Decouple email service from the main application",
                    "priority": "Medium"
                },
                {
                    "id": "FR-5",
                    "description": "Implement separate cronjob service for scheduled tasks",
                    "priority": "Low"
                }
            ],
            "non_functional_requirements": [
                {
                    "id": "NFR-1",
                    "description": "Implement comprehensive logging and monitoring",
                    "category": "Observability"
                },
                {
                    "id": "NFR-2",
                    "description": "Ensure high availability with 99.9% uptime",
                    "category": "Reliability"
                },
                {
                    "id": "NFR-3",
                    "description": "Implement IP whitelisting for Meta API integration",
                    "category": "Security"
                },
                {
                    "id": "NFR-4",
                    "description": "Conduct performance testing to ensure system can handle peak loads",
                    "category": "Performance"
                }
            ],
            "technical_constraints": [
                "Must use Terraform for infrastructure as code",
                "Must use Kubernetes for container orchestration",
                "Must integrate with existing AWS services",
                "Must maintain backward compatibility with legacy APIs"
            ],
            "clarifying_questions": [
                {
                    "question": "What is the expected traffic volume for the application?",
                    "answer": "Peak of 10,000 concurrent users"
                },
                {
                    "question": "Are there any specific compliance requirements?",
                    "answer": "GDPR and SOC2 compliance required"
                }
            ]
        }
    
    def get_prd_data(self) -> Dict[str, Any]:
        """
        Get the PRD data.
        
        Returns:
            Dictionary containing the PRD data
        """
        return self.prd_data
    
    def get_requirements_summary(self) -> str:
        """
        Get a text summary of the requirements for use in prompts.
        
        Returns:
            String containing a summary of the requirements
        """
        prd = self.prd_data
        summary = f"Project: {prd['project_name']}\n\n"
        summary += f"Description: {prd['project_description']}\n\n"
        
        summary += "Functional Requirements:\n"
        for req in prd['functional_requirements']:
            summary += f"- {req['description']} (Priority: {req['priority']})\n"
        
        summary += "\nNon-Functional Requirements:\n"
        for req in prd['non_functional_requirements']:
            summary += f"- {req['description']} (Category: {req['category']})\n"
        
        summary += "\nTechnical Constraints:\n"
        for constraint in prd['technical_constraints']:
            summary += f"- {constraint}\n"
        
        return summary

# Example usage
if __name__ == "__main__":
    prd_data = MockPRDData()
    summary = prd_data.get_requirements_summary()
    print(summary) 