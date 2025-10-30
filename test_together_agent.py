"""
Test the Together AI agent implementation
"""
import os
from dotenv import load_dotenv
from src.agent import TogetherCodeFixAgent

# Load environment variables from .env file
load_dotenv()

# Simple test case
buggy_code = """
def add_numbers(a, b):
    return a - b  # BUG: Should be addition, not subtraction
"""

test_cases = [
    "assert add_numbers(1, 2) == 3",
    "assert add_numbers(5, 3) == 8",
    "assert add_numbers(0, 0) == 0",
]

def main():
    # Make sure API key is set
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable not set!")
        print("Get your API key from: https://api.together.xyz/settings/api-keys")
        print("Set it with: export TOGETHER_API_KEY='your-key-here'")
        return
    
    print("Together API key found!")
    print(f"Testing Together AI agent with Llama-3.3-70B-Instruct-Turbo")
    print("="*70)
    
    # Create agent
    agent = TogetherCodeFixAgent(
        api_key=api_key,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        max_iterations=5,
        verbose=True
    )
    
    # Fix the code
    fixed_code = agent.fix_code(
        buggy_code=buggy_code,
        error_description="Addition function returns wrong result",
        test_cases=test_cases,
        task_id="test_addition"
    )
    
    print("\n" + "="*70)
    print("FINAL RESULT:")
    print("="*70)
    print(fixed_code)
    print("="*70)

if __name__ == "__main__":
    main()
