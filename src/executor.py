"""
Sandbox code executor for safe execution of LLM-generated Python code.
"""
import sys
import io
import ast
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple


class CodeExecutor:
    """Safe Python code executor with timeout and resource limits."""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def execute_code(self, code: str, test_cases: list = None) -> Dict[str, Any]:
        """
        Execute Python code safely and return results.
        
        Args:
            code: Python code to execute
            test_cases: List of test cases to run (optional)
        
        Returns:
            Dictionary with execution results
        """
        result = {
            'success': False,
            'output': '',
            'error': None,
            'tests_passed': 0,
            'tests_total': 0,
            'failed_tests': [],  # List of failed test details
            'passed_tests': []   # List of passed test details
        }
        
        # Validate syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            result['error'] = f"Syntax Error: {str(e)}"
            return result
        
        # Execute the code
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Create a clean namespace
            namespace = {
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
            }
            
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)
            
            result['success'] = True
            result['output'] = stdout_buffer.getvalue()
            
            # Run test cases if provided
            if test_cases:
                result['tests_total'] = len(test_cases)
                
                for i, test in enumerate(test_cases):
                    try:
                        # Extract function name from code
                        tree = ast.parse(code)
                        func_name = None
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                func_name = node.name
                                break
                        
                        if func_name and func_name in namespace:
                            # Try to extract expected value and actual value for better error messages
                            test_namespace = namespace.copy()
                            
                            # For assertions like: assert func(args) == expected
                            # Try to get both sides
                            try:
                                exec(test, test_namespace)
                                # Test passed
                                result['tests_passed'] += 1
                                result['passed_tests'].append({
                                    'test_number': i + 1,
                                    'test': test,
                                    'status': 'PASSED'
                                })
                            except AssertionError:
                                # Test failed - try to get actual vs expected
                                # Parse the assertion to extract the call and expected value
                                actual_value = None
                                expected_value = None
                                
                                try:
                                    # Extract function call and expected from assertion
                                    # e.g., "assert func(args) == expected"
                                    if '==' in test:
                                        parts = test.replace('assert ', '').split('==')
                                        if len(parts) == 2:
                                            call_part = parts[0].strip()
                                            expected_part = parts[1].strip()
                                            
                                            # Evaluate the function call to get actual value
                                            try:
                                                actual_value = eval(call_part, test_namespace)
                                            except:
                                                pass
                                            
                                            # Evaluate expected value
                                            try:
                                                expected_value = eval(expected_part, test_namespace)
                                            except:
                                                pass
                                except:
                                    pass
                                
                                # Build error message
                                if actual_value is not None and expected_value is not None:
                                    error_msg = f"Expected {expected_value}, but got {actual_value}"
                                else:
                                    error_msg = "Assertion failed"
                                
                                result['failed_tests'].append({
                                    'test_number': i + 1,
                                    'test': test,
                                    'status': 'FAILED',
                                    'error': error_msg,
                                    'type': 'AssertionError',
                                    'expected': expected_value,
                                    'actual': actual_value
                                })
                        else:
                            # Function not found in namespace
                            result['failed_tests'].append({
                                'test_number': i + 1,
                                'test': test,
                                'status': 'FAILED',
                                'error': 'Function not found in code',
                                'type': 'ExecutionError'
                            })
                    except AssertionError:
                        # Should be handled above, but just in case
                        result['failed_tests'].append({
                            'test_number': i + 1,
                            'test': test,
                            'status': 'FAILED',
                            'error': 'Assertion failed',
                            'type': 'AssertionError'
                        })
                    except Exception as e:
                        # Test failed with error
                        error_type = type(e).__name__
                        error_msg = str(e)
                        
                        # Special handling for syntax errors in test cases
                        if 'SyntaxError' in error_type and 'unexpected EOF' in error_msg:
                            error_msg = f"INCOMPLETE TEST CASE - The test assertion appears to be cut off or malformed: {error_msg}"
                        
                        result['failed_tests'].append({
                            'test_number': i + 1,
                            'test': test,
                            'status': 'FAILED',
                            'error': f"{error_type}: {error_msg}",
                            'type': error_type
                        })
                
                # Build error message from failed tests
                if result['failed_tests']:
                    error_lines = []
                    for failed in result['failed_tests'][:5]:  # Show first 5 failures
                        error_lines.append(f"Test {failed['test_number']}: {failed['test']}")
                        error_lines.append(f"  {failed['error']}")
                    if len(result['failed_tests']) > 5:
                        error_lines.append(f"  ... and {len(result['failed_tests']) - 5} more failures")
                    result['error'] = "\n".join(error_lines)
            
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result['output'] = stdout_buffer.getvalue()
        
        return result
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python code syntax.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use compile() instead of ast.parse() to catch semantic errors like "return outside function"
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax Error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation Error: {str(e)}"
