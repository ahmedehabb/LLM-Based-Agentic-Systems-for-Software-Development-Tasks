"""
Code fixing agent using Together AI with Llama-3.3-70B-Instruct-Turbo
Uses function calling for run_code tool.
"""
from typing import List, Dict, Optional
import time
import json
import os
from together import Together
from .executor import CodeExecutor


class AgentObserver:
    """Tracks and logs agent actions for observability."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs = []
        self.start_time = None
        
    def start_session(self, task_id: str):
        """Start a new observation session."""
        self.start_time = time.time()
        self.log("SESSION_START", f"Starting task: {task_id}", level="INFO")
    
    def log(self, action: str, message: str, level: str = "INFO", data: dict = None):
        """Log an action with timestamp."""
        timestamp = time.time()
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        log_entry = {
            'timestamp': timestamp,
            'elapsed': elapsed,
            'action': action,
            'level': level,
            'message': message,
            'data': data or {}
        }
        self.logs.append(log_entry)
        
        if self.verbose:
            prefix = {
                'INFO': '  [INFO]',
                'TOOL': '  [TOOL]',
                'AGENT': '  [AGENT]',
                'LLM': '  [LLM]',
                'ERROR': '  [ERROR]',
                'SUCCESS': '  [SUCCESS]'
            }.get(level, '  [LOG]')
            
            print(f"{prefix} {message}")
    
    def get_summary(self) -> Dict:
        """Get summary of the session."""
        return {
            'total_logs': len(self.logs),
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'actions': [log['action'] for log in self.logs],
            'errors': [log for log in self.logs if log['level'] == 'ERROR']
        }


# Global references for tool dependencies
_test_cases = []
_observer = None
_previous_code = None  # Track previous code submission


def run_code_tool(code: str, reason: str) -> Dict:
    """
    Execute Python code in sandbox and return results.
    This is the actual function that gets called when LLM invokes the tool.
    
    Args:
        code: Complete executable Python code
        reason: Brief explanation of what bug was fixed
    
    Returns:
        Dict with success status and test results
    """
    global _test_cases, _observer, _previous_code
    
    if not code:
        return {'success': False, 'error': 'No code provided'}
    
    # Check if this is the same code as before (detect loops)
    if _previous_code is not None and code.strip() == _previous_code.strip():
        warning_msg = "WARNING: You submitted the EXACT SAME code as your previous attempt!"
        warning_msg += "\n\nYou're stuck in a loop! The tests are failing, which means your fix didn't work."
        warning_msg += "\n\nTry a DIFFERENT approach:"
        warning_msg += "\n  - Re-read the test failures carefully"
        warning_msg += "\n  - Think about what ELSE could cause those failures"
        warning_msg += "\n  - Try a completely different fix"
        
        if _observer:
            _observer.log("DUPLICATE_CODE", "Same code submitted twice!", level="ERROR")
        
        return {
            'success': False,
            'error': warning_msg,
            'duplicate_submission': True
        }
    
    # Store this code for next time
    _previous_code = code
    
    if _observer:
        _observer.log("TOOL_CALLED", "LLM CALLED run_code TOOL!", level="TOOL")
        _observer.log("TOOL_EXECUTE", f"Running code ({len(code)} chars)...", level="TOOL")
        print("\n" + "="*70)
        print("TOOL CALL DETECTED!")
        print(f"   Tool: run_code")
        print(f"   Reason: {reason}")
        print(f"   Code length: {len(code)} chars")
        print(f"\nCode received by tool:")
        print("─" * 70)
        print(code)
        print("─" * 70)
        print("="*70 + "\n")
    
    executor = CodeExecutor()
    
    # Validate syntax
    is_valid, syntax_error = executor.validate_code(code)
    
    if not is_valid:
        error_msg = f"Syntax Error:\n{syntax_error}"
        if _observer:
            _observer.log("SYNTAX_ERROR", syntax_error, level="ERROR")
            print(f"\n{error_msg}\n")
        return {'success': False, 'error': error_msg}
    
    if _observer:
        print("Syntax valid - running tests...\n")
        print(f"DEBUG: _test_cases has {len(_test_cases)} tests")
        if _test_cases:
            print(f"   First test: {_test_cases[0]}")
    
    # Run tests
    if _test_cases:
        result = executor.execute_code(code, _test_cases)
        passed = result.get('tests_passed', 0)
        total = result.get('tests_total', len(_test_cases))
        failed_tests = result.get('failed_tests', [])
        
        if passed == total and passed > 0:
            msg = f"SUCCESS! All {total} tests passed!"
            if _observer:
                _observer.log("ALL_PASS", msg, level="SUCCESS")
            return {
                'success': True,
                'tests_passed': passed,
                'tests_total': total,
                'message': msg
            }
        else:
            # Build detailed failure message
            failure_details = []
            
            if passed > 0:
                failure_details.append(f"{passed} test(s) passed")
            
            failure_details.append(f"\n{total - passed} test(s) FAILED:")
            
            for failed in failed_tests[:3]:
                failure_details.append(f"\n  Test #{failed['test_number']}: {failed['test']}")
                
                # Show actual vs expected if available
                if 'actual' in failed and 'expected' in failed and failed['actual'] is not None:
                    failure_details.append(f"  Expected: {failed['expected']}")
                    failure_details.append(f"  Got:      {failed['actual']}")
                    failure_details.append(f"  Values don't match!")
                else:
                    failure_details.append(f"  {failed['error']}")
            
            if len(failed_tests) > 3:
                failure_details.append(f"\n  ... and {len(failed_tests) - 3} more test(s) failed")
            
            failure_details.append(f"\nDEBUGGING HINT:")
            failure_details.append(f"  - Compare the tests that PASS vs those that FAIL")
            failure_details.append(f"  - Look for patterns in the inputs/outputs")
            
            error_msg = "\n".join(failure_details)
            
            if _observer:
                _observer.log("TESTS_FAILED", f"{passed}/{total} passed", level="ERROR")
            
            return {
                'success': False,
                'tests_passed': passed,
                'tests_total': total,
                'failed_count': len(failed_tests),
                'failed_tests': [{'test': f['test'], 'error': f['error']} for f in failed_tests],
                'error': error_msg
            }
    else:
        # No tests - just validate it runs
        result = executor.execute_code(code, ["pass"])
        if result.get('error'):
            msg = f"Runtime Error:\n{result['error']}"
            if _observer:
                _observer.log("RUNTIME_ERROR", result['error'], level="ERROR")
            return {'success': False, 'error': msg}
        else:
            msg = "Code runs without errors!"
            if _observer:
                _observer.log("NO_ERROR", msg, level="SUCCESS")
            return {'success': True, 'message': msg}


class TogetherCodeFixAgent:
    """Code fixing agent using Together AI with Llama-3.3-70B and function calling"""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 max_iterations: int = 10,
                 verbose: bool = True):
        api_keys_raw = api_key or os.getenv("TOGETHER_API_KEY")
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.observer = AgentObserver(verbose=verbose)
        
        if not api_keys_raw:
            raise ValueError("Together API key required! Set TOGETHER_API_KEY env var or pass api_key parameter")
        
        # Parse comma-separated API keys
        self.api_keys = [key.strip() for key in api_keys_raw.split(',') if key.strip()]
        
        if not self.api_keys:
            raise ValueError("No valid API keys found!")
        
        # Create multiple clients for round-robin usage
        self.clients = [Together(api_key=key) for key in self.api_keys]
        self.current_client_index = 0
        
        if self.verbose and len(self.api_keys) > 1:
            print(f"Loaded {len(self.api_keys)} API keys for round-robin usage")
        
        # For backward compatibility
        self.client = self.clients[0]
        self.api_key = self.api_keys[0]
        
        # Define the run_code tool for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "Execute Python code in a sandbox and test it against predefined test cases. This is your ONLY way to verify fixes. Call this with your fixed code to see if tests pass. Returns test results with pass/fail status and error details if any tests fail.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Complete, executable Python code with all imports, function definitions, and the fix applied. Must be syntactically valid Python."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief one-sentence explanation of what bug you identified and what change you made to fix it. Example: 'Added abs() to calculate absolute distance between elements'"
                            }
                        },
                        "required": ["code", "reason"]
                    }
                }
            }
        ]
    
    def get_next_client(self):
        """Get next client in round-robin fashion."""
        client = self.clients[self.current_client_index]
        client_num = self.current_client_index + 1
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client, client_num
    
    def fix_code(self, 
                 buggy_code: str, 
                 error_description: str = "", 
                 test_cases: list = None, 
                 task_id: str = "unknown") -> str:
        """
        Fix buggy Python code using Together AI with Llama-3.3-70B.
        
        Args:
            buggy_code: The buggy code to fix
            error_description: Description of the bug
            test_cases: List of test cases
            task_id: Identifier for this task
        
        Returns:
            Fixed code
        """
        self.observer.start_session(task_id)
        self.observer.log("FIX_START", f"Starting code fix with Together AI ({self.model_name})", level="AGENT")
        
        print(f"\n{'='*70}")
        print(f"TASK: {task_id}")
        print(f"DESCRIPTION: {error_description}")
        print(f"MODEL: {self.model_name}")
        print(f"{'='*70}")
        
        # Set global references for the tool
        global _test_cases, _observer, _previous_code
        _test_cases = test_cases or []
        _observer = self.observer
        _previous_code = None  # Reset for new session
        
        # DEBUG: Verify test cases were set
        if self.verbose:
            print(f"\nDEBUG: Setting _test_cases globally")
            print(f"   _test_cases now has: {len(_test_cases)} tests")
            if _test_cases:
                print(f"   First test: {_test_cases[0]}")
        
        # Build system message
        system_message = """You are an expert Python debugging agent with access to a code execution sandbox.

YOUR MISSION: Fix buggy Python code by analyzing it, identifying bugs, and testing fixes iteratively.

AVAILABLE TOOL:
- run_code(code: str, reason: str): Executes your code against predefined test cases
  Returns: {"success": bool, "tests_passed": int, "tests_total": int, "error": str}

WORKFLOW:
1. READ the buggy code carefully
2. ANALYZE what tests expect vs what code does
3. IDENTIFY the bug(s)
4. CALL run_code with your fixed code + brief explanation
5. READ the test results:
   - All pass -> STOP (you're done!)
   - Some fail -> Analyze failures, try DIFFERENT fix, go to step 4
6. ITERATE until all tests pass (max 10 attempts)

CRITICAL RULES:
- ALWAYS call run_code - never just explain the fix
- Each attempt must try something DIFFERENT (don't repeat same code)
- Read error messages carefully - they tell you what's wrong
- When all tests pass, STOP immediately (don't call again)
- Keep your 'reason' brief (1 sentence) - save tokens for code

COMMON BUG PATTERNS & FIXES:
1. Missing absolute value:
   Bug: distance = elem1 - elem2
   Fix: distance = abs(elem1 - elem2)

2. Wrong comparison operator:
   Bug: if distance < threshold (when should be <=)
   Fix: if distance <= threshold

3. Off-by-one in loops:
   Bug: for i in range(len(arr) - 1)
   Fix: for i in range(len(arr))

4. Not handling spaces/whitespace:
   Bug: for c in string: process(c)
   Fix: for c in string: if c != ' ': process(c)

5. Adding instead of just returning:
   Bug: return value + 1.0
   Fix: return value

EXAMPLE FIX:
Buggy: def has_close(nums, threshold): return any(n1-n2 < threshold for n1 in nums for n2 in nums)
Fixed: def has_close(nums, threshold): return any(abs(n1-n2) < threshold for i, n1 in enumerate(nums) for j, n2 in enumerate(nums) if i != j)
Reason: Added abs() and excluded same-element comparisons

NOW: Call run_code with your first fix attempt!"""
        
        # Build user message with ALL test cases (no truncation for 70B model)
        test_info = ""
        if test_cases:
            test_info = "\n\nTEST CASES (Your code must pass ALL of these):\n"
            for i, t in enumerate(test_cases, 1):
                test_info += f"  {i}. {t}\n"
        
        user_message = f"""BUGGY CODE TO FIX:

```python
{buggy_code}
```
{test_info}

TASK: Fix the bug(s) in this code so it passes ALL test cases.

ACTION: Call run_code NOW with:
  - code: Your complete fixed version
  - reason: One sentence explaining what you changed

START NOW - Don't explain, just call the function!"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Run agent loop
        self.observer.log("AGENT_RUN", "Running Together AI agent...", level="AGENT")
        
        tool_was_called = False
        final_code = None
        consecutive_failures = 0  # Track failures for temperature adjustment
        
        for iteration_count in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Iteration {iteration_count + 1}/{self.max_iterations}")
                print(f"   Message history size: {len(messages)} messages")
                if len(messages) > 10:
                    print(f"   WARNING: Large message history may slow down API calls!")
                print(f"{'='*70}")
            
            # Call Together AI
            max_retries = 3
            retry_delay = 2  # seconds
            
            for retry_attempt in range(max_retries):
                try:
                    # Get next client in round-robin
                    current_client, client_num = self.get_next_client()
                    
                    # DETAILED: Show what we're sending to LLM
                    if self.verbose and iteration_count == 0 and retry_attempt == 0:
                        print(f"\nSENDING TO LLM (first iteration):")
                        print(f"{'='*70}")
                        for idx, msg in enumerate(messages):
                            role = msg.get('role', 'unknown')
                            print(f"\n[Message {idx}] Role: {role}")
                            if role == 'system':
                                content = msg.get('content', '')
                                print(f"Content:\n{content[:500]}...")
                            elif role == 'user':
                                content = msg.get('content', '')
                                print(f"Content:\n{content[:800]}...")
                            elif role == 'tool':
                                print(f"Tool: {msg.get('name')}")
                                print(f"Result: {msg.get('content')[:200]}...")
                        print(f"{'='*70}\n")
                    
                    # Minimal output - just show we're calling
                    if self.verbose and retry_attempt == 0:
                        if len(self.clients) > 1:
                            print(f"Calling API (key {client_num}/{len(self.clients)})...", end='', flush=True)
                        else:
                            print(f"Calling API...", end='', flush=True)
                    elif self.verbose and retry_attempt > 0:
                        print(f"Retry {retry_attempt + 1}/{max_retries} with key {client_num}...", end='', flush=True)
                    
                    start_time = time.time()

                    # Adaptive temperature: lower if stuck, higher if making progress
                    if iteration_count < 3:
                        temperature = 0.7  # Start creative
                    elif consecutive_failures >= 3:
                        temperature = 0.3  # Get more deterministic if stuck
                    else:
                        temperature = 0.5  # Middle ground
                    
                    # Direct API call with built-in retry mechanism
                    api_retry_count = 3
                    api_retry_delay = 1
                    response = None
                    
                    for api_attempt in range(api_retry_count):
                        try:
                            response = current_client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                tools=self.tools,
                                tool_choice="auto",
                                temperature=temperature
                            )
                            break  # Success!
                        except Exception as api_error:
                            if api_attempt < api_retry_count - 1:
                                print(f"\nWARNING: API attempt {api_attempt + 1} failed: {str(api_error)[:100]}")
                                print(f"   Retrying in {api_retry_delay}s...")
                                time.sleep(api_retry_delay)
                                api_retry_delay *= 2  # Exponential backoff
                            else:
                                # Last attempt failed, raise the error
                                raise api_error
                    
                    elapsed = time.time() - start_time
                    
                    if self.verbose:
                        print(f" ✓ ({elapsed:.1f}s)")
                    
                    # Success! Break out of retry loop
                    break
                    
                except Exception as e:
                    error_msg = f"Together AI API Error (attempt {retry_attempt + 1}/{max_retries}): {str(e)}"
                    self.observer.log("API_ERROR", error_msg, level="ERROR")
                    print(f"\n{error_msg}")
                    
                    # Show more details about the error
                    if self.verbose:
                        print(f"\nError details:")
                        print(f"   Type: {type(e).__name__}")
                        print(f"   Message: {str(e)}")
                        if hasattr(e, 'status_code'):
                            print(f"   Status code: {e.status_code}")
                        
                        # Check common issues
                        error_str = str(e).lower()
                        if 'timeout' in error_str:
                            print(f"\nAPI timeout - the call took too long")
                            print(f"   Switching to next API key on retry...")
                        elif 'rate limit' in error_str or '429' in error_str:
                            print(f"\nRate limit hit - waiting before retry...")
                        elif 'api key' in error_str or '401' in error_str or '403' in error_str:
                            print(f"\nAPI key issue - check your TOGETHER_API_KEY")
                        elif 'connection' in error_str:
                            print(f"\nConnection issue - check your internet connection")
                    
                    # If this was the last retry, give up
                    if retry_attempt == max_retries - 1:
                        print(f"\nAll {max_retries} retry attempts failed - giving up")
                        break
                    
                    # Wait before retrying
                    if self.verbose:
                        print(f"\nWaiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            else:
                # All retries failed
                break
            
            # Process the response (only if we got here successfully)
            assistant_message = response.choices[0].message
            
            if self.verbose:
                print(f"\nLLM Response:")
                print(f"   Role: {assistant_message.role}")
                if assistant_message.content:
                    print(f"   Content ({len(assistant_message.content)} chars):")
                    print(f"   >>> {assistant_message.content[:300]}...")
                else:
                    print(f"   Content: <empty>")
                if assistant_message.tool_calls:
                    print(f"   Tool calls: {len(assistant_message.tool_calls)}")
                    for tc in assistant_message.tool_calls:
                        print(f"      - {tc.function.name}")
                        # Show tool call arguments
                        try:
                            args = json.loads(tc.function.arguments)
                            print(f"        Args: {list(args.keys())}")
                            if 'reason' in args:
                                print(f"        Reason: {args['reason'][:100]}...")
                            if 'code' in args:
                                print(f"        Code length: {len(args['code'])} chars")
                        except:
                            pass
                else:
                    print(f"   Tool calls: None [WARNING]")
            
            # Add assistant message to history
            messages.append(assistant_message)
            
            if self.verbose:
                print(f"\nAdded assistant message to history (now {len(messages)} messages)")
            
            # Check if model called tools
            if assistant_message.tool_calls:
                    tool_was_called = True
                    
                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if self.verbose:
                            print(f"\nTOOL CALL: {function_name}")
                            print(f"   Arguments: {list(function_args.keys())}")
                        
                        # Execute the tool
                        if function_name == "run_code":
                            tool_result = run_code_tool(
                                code=function_args.get("code", ""),
                                reason=function_args.get("reason", "No reason provided")
                            )
                            
                            # Store code if we got it
                            if function_args.get("code"):
                                final_code = function_args.get("code")
                            
                            # Add tool result to messages
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(tool_result)
                            }
                            messages.append(tool_message)
                            
                            if self.verbose:
                                print(f"\nAdded tool result to history (now {len(messages)} messages)")
                                result_str = json.dumps(tool_result)
                                print(f"   Tool result preview: {result_str[:150]}...")
                            
                            if self.verbose:
                                print(f"\nTOOL RESULT:")
                                print(f"   Success: {tool_result.get('success', False)}")
                                if tool_result.get('success'):
                                    print(f"   {tool_result.get('message', 'Tests passed!')}")
                                else:
                                    error_preview = tool_result.get('error', 'Unknown error')
                                    print(f"   Error: {error_preview}...")
                                    print(f"   Tests passed: {tool_result.get('tests_passed', 0)}/{tool_result.get('tests_total', 0)}")
                            
                            # Check if tests passed - if so, STOP!
                            if tool_result.get('success'):
                                if self.verbose:
                                    print(f"\nTests passed! Stopping agent loop.")
                                self.observer.log("FIX_SUCCESS", "Code fix completed - all tests passed", level="SUCCESS")
                                
                                summary = self.observer.get_summary()
                                print(f"\nFix complete in {summary['total_time']:.2f}s")
                                
                                return final_code if final_code else buggy_code
                            else:
                                # Test failed - increment failure counter
                                consecutive_failures += 1
            
            else:
                # Model didn't call tool - might be done or needs prompting
                if self.verbose:
                    print(f"\nWARNING: Model did NOT call any tools in iteration {iteration_count + 1}")
                    if assistant_message.content:
                        print(f"   Model said: {assistant_message.content}...")
                
                if iteration_count == 0:
                    # First iteration without tool call - force it
                    if self.verbose:
                        print(f"\nFORCING tool call with user prompt...")
                    
                    messages.append({
                        "role": "user",
                        "content": "STOP! You did NOT call run_code! Call it NOW with your best guess for the fix. Just call the function - no more thinking!"
                    })
                    continue
                else:
                    # Later iteration - model might think it's done
                    if self.verbose:
                        print(f"\nModel stopped calling tools - ending loop")
                    break
        
        # If we get here, either max iterations reached or something went wrong
        if final_code and len(final_code) > 20:
            self.observer.log("FIX_PARTIAL", "Returning last submitted code (tests may not pass)", level="ERROR")
            print(f"\nMax iterations reached - returning last code attempt")
            return final_code
        
        self.observer.log("FIX_FAILED", "Could not fix code", level="ERROR")
        print(f"\nFailed to fix code")
        return buggy_code

