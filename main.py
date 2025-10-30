"""
Main script to download CommitPackFT data and run evaluation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.commitpack_loader import (
    download_and_filter_commitpack,
    load_commitpack_python,
    convert_to_bugfix_format
)
from src.local_agent import LocalCodeFixAgent
from src.executor import CodeExecutor
from tqdm import tqdm
import time
import argparse


def download_dataset(max_samples: int = 1000):
    """Download and filter CommitPackFT dataset."""
    print("="*70)
    print("DOWNLOADING COMMITPACKFT DATASET")
    print("="*70)
    
    output_file = download_and_filter_commitpack(
        output_file="data/commitpack_python.jsonl",
        max_samples=max_samples,
        force_download=False
    )
    
    if output_file:
        print(f"\n✓ Dataset ready at: {output_file}")
    else:
        print("\n✗ Download failed, will use sample data")


def evaluate_on_commitpack(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    num_samples: int = 10,
    use_sample: bool = False
):
    """
    Evaluate agent on CommitPackFT dataset.
    
    Args:
        model_name: HuggingFace model name
        num_samples: Number of samples to evaluate
        use_sample: Use sample data instead of full dataset
    """
    
    print("="*70)
    print("EVALUATING AGENT ON COMMITPACKFT")
    print("="*70)
    
    # Load data
    commit_examples = load_commitpack_python(
        file_path="data/commitpack_python.jsonl",
        max_samples=num_samples
    )
    
    # Convert to bug-fix format
    problems = [convert_to_bugfix_format(ex) for ex in commit_examples]
    
    print(f"\nEvaluating on {len(problems)} code change examples")
    print(f"Model: {model_name}\n")
    
    # Initialize agent
    agent = LocalCodeFixAgent(model_name=model_name, max_iterations=3)
    executor = CodeExecutor()
    
    # Evaluate
    results = []
    exact_matches = 0
    syntax_valid = 0
    
    for i, problem in enumerate(tqdm(problems, desc="Processing")):
        print(f"\n{'-'*70}")
        print(f"{i+1}. {problem['task_id']}")
        print(f"   Change: {problem['prompt']}")
        print("-"*70)
        
        start = time.time()
        
        # Try to fix the code
        try:
            fixed_code = agent.fix_code(
                buggy_code=problem['buggy_code'],
                error_description=problem['prompt'],
                test_cases=None  # CommitPack doesn't have test cases
            )
        except Exception as e:
            print(f"   [ERROR] {e}")
            fixed_code = problem['buggy_code']
        
        elapsed = time.time() - start
        
        # Check if fix is valid Python
        is_valid, error = executor.validate_code(fixed_code)
        if is_valid:
            syntax_valid += 1
        
        # Check if matches expected fix
        is_exact_match = fixed_code.strip() == problem['fixed_code'].strip()
        if is_exact_match:
            exact_matches += 1
            print(f"   ✓ EXACT MATCH")
        else:
            print(f"   ~ DIFFERENT FIX {'(valid syntax)' if is_valid else '(syntax error)'}")
        
        print(f"   Time: {elapsed:.2f}s")
        
        results.append({
            'task_id': problem['task_id'],
            'exact_match': is_exact_match,
            'syntax_valid': is_valid,
            'time': elapsed
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Total: {len(results)}")
    print(f"Exact matches: {exact_matches}/{len(results)} ({100*exact_matches/len(results):.1f}%)")
    print(f"Valid syntax: {syntax_valid}/{len(results)} ({100*syntax_valid/len(results):.1f}%)")
    print(f"Avg time: {sum(r['time'] for r in results)/len(results):.2f}s")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Fix Agent Evaluation")
    parser.add_argument("--download", action="store_true", help="Download dataset (WARNING: CommitPackFT is very large)")
    parser.add_argument("--max-download", type=int, default=1000, help="Max samples to download")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Model name")
    parser.add_argument("--use-humanevalfix", action="store_true", help="Use HumanEvalFix instead of CommitPackFT")
    
    args = parser.parse_args()
    
    if args.download:
        print("\n" + "="*70)
        print("WARNING: CommitPackFT is VERY LARGE")
        print("="*70)
        print("Each file is ~500MB. Full dataset is 458 files = ~230GB")
        print("Downloading 1 file will take several minutes and use ~500MB disk space.")
        print("\nRECOMMENDATION: Use --use-humanevalfix flag instead")
        print("  or press Ctrl+C to cancel")
        print("="*70)
        
        import time
        time.sleep(5)  # Give user time to cancel
        
        download_dataset(max_samples=args.max_download)
    
    if args.evaluate:
        if args.use_humanevalfix:
            # Use HumanEvalFix dataset instead
            from src.humanevalfix_loader import load_humanevalfix_dataset
            from test_agent import evaluate_agent
            
            print("\n" + "="*70)
            print("USING HUMANEVALFIX DATASET")
            print("="*70)
            
            problems = load_humanevalfix_dataset()[:args.num_samples]
            
            print(f"\nEvaluating on {len(problems)} HumanEvalFix problems")
            print(f"Model: {args.model}\n")
            
            evaluate_agent(problems, model_name=args.model)
        else:
            evaluate_on_commitpack(
                model_name=args.model,
                num_samples=args.num_samples,
                use_sample=args.sample
            )
    
    if not args.download and not args.evaluate:
        print("Usage:")
        print("  Evaluate on HumanEvalFix: python main.py --evaluate --use-humanevalfix --num-samples 10")
        print("  Evaluate on CommitPack (requires download): python main.py --download --evaluate")
        print("  Download CommitPackFT: python main.py --download --max-download 1000")
        print("\nNote: CommitPackFT is very large (~230GB full dataset)")
        print("      HumanEvalFix is recommended for quick evaluation")
