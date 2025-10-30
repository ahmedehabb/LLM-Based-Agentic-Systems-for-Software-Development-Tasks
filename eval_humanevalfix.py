#!/usr/bin/env python3
"""
Simple standalone evaluation of Together AI Agent on HumanEvalFix.
Uses Together AI's Llama-3.3-70B-Instruct-Turbo via API.
"""
import json
import sys
import os
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import TogetherCodeFixAgent
from src.executor import CodeExecutor


def load_humanevalfix_dataset(limit=None):
    """Load HumanEvalFix dataset from bigcode-evaluation-harness."""
    from datasets import load_dataset
    
    # Load the dataset
    print("Loading HumanEvalFix dataset...")
    dataset = load_dataset("bigcode/humanevalpack", "python")['test']
    
    problems = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        
        # Extract COMPLETE buggy code with function signature
        declaration = item.get('declaration', '')
        buggy_solution = item.get('buggy_solution', '')
        tests = item.get('test', '')
        task_id = item.get('task_id', f'task_{i}')
        
        # Combine declaration + buggy solution for COMPLETE code
        buggy_code = declaration + buggy_solution
        
        # Parse tests into list
        test_list = [line.strip() for line in tests.split('\n') if line.strip() and 'assert' in line]
        
        problems.append({
            'task_id': task_id,
            'buggy_code': buggy_code,  # Now includes function signature!
            'declaration': declaration,
            'buggy_solution': buggy_solution,
            'tests': test_list,
            'full_test_code': tests
        })
    
    print(f"Loaded {len(problems)} problems")
    return problems


def evaluate_agent(agent, problems, verbose=True):
    """Evaluate agent on problems."""
    results = {
        'total': len(problems),
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    executor = CodeExecutor()
    
    print(f"\nEvaluating on {len(problems)} problems...")
    print("=" * 80)
    
    for i, problem in enumerate(tqdm(problems, desc="Evaluating", disable=verbose)):
        task_id = problem['task_id']
        buggy_code = problem['buggy_code']
        tests = problem['tests']
        
        print(f"\n{'='*80}")
        print(f"Problem {i+1}/{len(problems)}: {task_id}")
        print(f"{'='*80}")
        
        # Show buggy code
        print(f"\nBuggy Code:")
        print("─" * 80)
        print(buggy_code[:300] if len(buggy_code) > 300 else buggy_code)
        if len(buggy_code) > 300:
            print("...")
        print("─" * 80)
        
        # Show tests
        print(f"\nTests ({len(tests)} total):")
        for j, test in enumerate(tests[:3], 1):
            print(f"  {j}. {test}")
        if len(tests) > 3:
            print(f"  ... and {len(tests) - 3} more")
        
        # Get fix from agent
        print(f"\nAgent is working...")
        try:
            import time
            start_time = time.time()
            
            fixed_code = agent.fix_code(
                buggy_code=buggy_code,
                test_cases=tests,
                task_id=task_id
            )
            
            elapsed = time.time() - start_time
            
            # Show fixed code
            print(f"\nFixed Code (took {elapsed:.1f}s):")
            print("─" * 80)
            print(fixed_code[:300] if len(fixed_code) > 300 else fixed_code)
            if len(fixed_code) > 300:
                print("...")
            print("─" * 80)
            
            # Test the fixed code
            print(f"\nRunning tests...")
            result = executor.execute_code(fixed_code, tests)
            passed = result.get('tests_passed', 0)
            total = result.get('tests_total', len(tests))
            
            # CRITICAL: If no tests were run, that's a failure!
            if total == 0 or not result.get('success', False):
                success = False
            else:
                success = (passed == total)
            
            if success:
                results['passed'] += 1
                status = f"PASS - All {total} tests passed!"
                print(f"\n{status}")
            else:
                results['failed'] += 1
                # Better error messages
                if total == 0:
                    status = f"FAIL - No tests could be run! Code is likely invalid."
                else:
                    status = f"FAIL - {passed}/{total} tests passed"
                print(f"\n{status}")
                if result.get('error'):
                    print(f"\nError:")
                    print(result['error'][:500])
            
            results['details'].append({
                'task_id': task_id,
                'success': success,
                'tests_passed': passed,
                'tests_total': total,
                'elapsed_time': elapsed,
                'buggy_code': buggy_code,
                'fixed_code': fixed_code,
                'error': result.get('error', None) if not success else None
            })
            
        except Exception as e:
            results['failed'] += 1
            error_msg = str(e)
            print(f"\nERROR: {error_msg[:200]}")
            
            results['details'].append({
                'task_id': task_id,
                'success': False,
                'buggy_code': buggy_code,
                'error': error_msg
            })
        
        print(f"\n{'='*80}")
        print(f"Progress: {results['passed']}/{i+1} passed ({results['passed']/(i+1)*100:.1f}%)")
        print(f"{'='*80}\n")
    
    return results


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    total = results['total']
    passed = results['passed']
    failed = results['failed']
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nTotal Problems: {total}")
    print(f"Passed:         {passed} ({pass_rate:.1f}%)")
    print(f"Failed:         {failed}")
    
    print(f"\nPass@1: {pass_rate:.2f}%")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Together AI Agent on HumanEvalFix")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to evaluate")
    parser.add_argument("--output", type=str, default="humanevalfix_results.json", help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                        help="Together AI model to use")
    parser.add_argument("--api-key", type=str, default=None, 
                        help="Together API key (or set TOGETHER_API_KEY env var)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Together AI Agent - HumanEvalFix Evaluation")
    print("="*80)
    print()
    print("Configuration:")
    print(f"   Problems: {args.limit if args.limit else 'all'}")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.output}")
    
    # Check API key
    api_key = args.api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print()
        print("Error: TOGETHER_API_KEY not set!")
        print("   Get your API key from: https://api.together.xyz/settings/api-keys")
        print("   Set it with: export TOGETHER_API_KEY='your-key-here'")
        print("   Or add to .env file: TOGETHER_API_KEY=your-key-here")
        return
    
    print("   Together API key found")
    
    # Load dataset
    problems = load_humanevalfix_dataset(limit=args.limit)
    
    # Create agent
    print(f"\nInitializing Together AI Agent...")
    agent = TogetherCodeFixAgent(
        api_key=api_key,
        model_name=args.model,
        verbose=True  # Always show agent details during eval
    )
    
    # Evaluate
    results = evaluate_agent(agent, problems, verbose=not args.quiet)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
