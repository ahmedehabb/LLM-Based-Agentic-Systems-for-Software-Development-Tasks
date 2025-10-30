# AI Code Fixing Agent

This project is an automated code fixing agent that uses Together AI's Llama 3.3 70B model to fix bugs in Python code. It takes buggy code and test cases, then iteratively tries different fixes until all tests pass.

## What it does

The agent works like a debugging assistant. You give it broken Python code and some test cases, and it figures out what's wrong and fixes it. It uses function calling to actually run the code in a sandbox and see if the tests pass, then adjusts its approach based on the results.

## How it works

The system has three main parts:

1. The agent (src/agent.py) - This is the brain. It talks to the Llama model, sends code fixes to be tested, and analyzes the results. It keeps trying different approaches until tests pass or it runs out of attempts.

2. The executor (src/executor.py) - This runs the code safely in a sandbox. It validates syntax, runs test cases, and reports back what passed and what failed with detailed error messages.

3. The evaluation script (eval_humanevalfix.py) - This tests the agent on the HumanEvalFix dataset, which has 164 buggy Python functions that need fixing.

## Key features

Multi-API key support: You can add multiple Together API keys separated by commas in your .env file. The agent rotates between them to speed things up when running lots of evaluations.

Adaptive temperature: The agent starts creative (0.7) then gets more focused (0.5) and eventually very deterministic (0.3) if it gets stuck. This helps it explore different solutions first, then zero in on the right one.

Retry mechanism: API calls retry up to 3 times with exponential backoff if they fail. This handles temporary network issues or rate limits.

Smart error messages: When tests fail, the executor shows you exactly what the expected value was versus what the code actually returned. Makes debugging way easier.

Loop detection: If the agent submits the exact same code twice, it gets a warning to try a different approach. Prevents getting stuck.

## Project structure

```
src/
  agent.py       - Main AI agent with function calling logic
  executor.py    - Safe code execution sandbox
  
eval_humanevalfix.py  - Evaluation script for HumanEvalFix dataset
requirements.txt      - Python dependencies
.env                  - API keys (you need to create this)

Results:
humanevalfix_results.json  - Full results from latest evaluation run
```

## How to run it

First, install dependencies:

```bash
pip install -r requirements.txt
```

Note: This project uses the HumanEvalFix dataset from the `datasets` library. You don't need to clone the bigcode-evaluation-harness repository.

Create a .env file with your Together API key:

```
TOGETHER_API_KEY=your_key_here
```

For multiple keys (faster evaluation):

```
TOGETHER_API_KEY=key1,key2
```

Run evaluation on the full dataset:

```bash
python eval_humanevalfix.py
```

Or limit to just a few problems for testing:

```bash
python eval_humanevalfix.py --limit 5
```

## Results

The agent was evaluated on 100 problems from HumanEvalFix and got:

- Total: 100 problems
- Passed: 81 (81% pass rate)
- Failed: 19

Results are saved in humanevalfix_results.json with full details including:
- Which tests passed/failed
- How long each fix took
- The buggy code and fixed code
- Error messages for failures

Most fixes happen in 1-2 iterations and take 1-5 seconds per problem. The agent is pretty good at spotting common bugs like missing absolute values, wrong operators, or off-by-one errors.

## Common bugs it fixes

The agent is particularly good at catching:

- Missing abs() calls when calculating distances
- Wrong comparison operators (< vs <=)
- Off-by-one errors in loops
- Not handling spaces or whitespace properly
- Wrong return values (adding when should just return)
- Comparing wrong variables or using wrong indices

## Implementation notes

The agent uses function calling with a single tool called run_code. The LLM calls this tool with complete fixed code and a brief explanation. The tool executes it against test cases and returns results.

Messages are tracked in a conversation history. First iteration shows you everything being sent to the LLM for debugging. After that it just shows high-level progress.

The system prompt includes few-shot examples of common bug patterns and their fixes to help guide the model toward better solutions faster.

If tests keep failing after 3 attempts, temperature drops to 0.3 to make the model more focused and deterministic.

API calls are made directly without threading (earlier versions used ThreadPoolExecutor but it caused hanging issues).

## Requirements

- Python 3.9 or higher
- Together API key (get from https://api.together.xyz)
- About 4GB RAM for running evaluations
- Internet connection for API calls

## Notes

The executor uses RestrictedPython for safe code execution. It has timeouts and limited namespace to prevent dangerous operations.

Test cases from HumanEvalFix sometimes have formatting issues (incomplete assertions). The executor detects these and reports them clearly.

The agent has a max of 10 iterations per problem. Most problems are solved in 1-3 iterations.

Round-robin key usage is logged so you can see which key is handling each request.
