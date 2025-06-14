# work-breakdown/main.py (modified)
"""
Main entry point for the module breakdown generator.

This script provides a simple command-line interface for generating
module breakdowns from PRD and historical data.
"""
import json
import argparse
from module_breakdown_generator import ModuleBreakdownGenerator

def main():
    """Main entry point for the module breakdown generator."""
    parser = argparse.ArgumentParser(description="Generate a module breakdown for a project")
    parser.add_argument(
        "prd_file",
        type=str,
        help="Path to the PRD PDF file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="module_breakdown.json", 
        help="Output file path for the module breakdown (default: module_breakdown.json)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format (json or text)"
    )
    args = parser.parse_args()
    
    print(f"Processing PRD file: {args.prd_file}...")
    generator = ModuleBreakdownGenerator()
    generator.load_prd_data(args.prd_file)
    
    print("Generating module breakdown...")
    breakdown = generator.generate_module_breakdown()
    
    # Output the result
    if args.format == "json":
        # Save to JSON file
        with open(args.output, "w") as f:
            json.dump(breakdown.model_dump(), f, indent=2)
        print(f"Module breakdown saved to {args.output}")
    else:
        # Print to console in text format
        print("\nGenerated Module Breakdown:")
        print("==========================")
        total_hours = 0
        for module in breakdown.modules:
            print(f"{module.name} ({module.estimated_hours} hours)")
            print(f"  {module.description}")
            print()
            total_hours += module.estimated_hours
        
        print(f"Total estimated hours: {total_hours}")
        
        # Also save to JSON file
        with open(args.output, "w") as f:
            json.dump(breakdown.model_dump(), f, indent=2)
        print(f"Module breakdown also saved to {args.output}")

if __name__ == "__main__":
    main()