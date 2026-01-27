"""
Code for generating vague PR issues for rollout one.
Messy code but providing for reproduction and transparency.

Usage:
    python create_rollout_one_prompts.py
"""

import os
import json
import time

from jinja2 import Template

from sera.utils import pp_query, pp_regex

PROMPT = """
I want a set of prompts that will tell an AI agent to improve, optimize, and fix bugs in a codebase.
These prompts should be general. Some possible directions are:
- Inconsistent method signatures
- Shared function behavior drift
- Mutated shared state
- Conflicting library versions
- Incorrect method overrides
- Missing required implementations
- Suppressed critical exceptions
- Inconsistent error types
- Mismatched data shapes
- Unhandled enum/schema updates
- Async/sync mismatches
- Concurrent shared-resource modification
- Incorrect variable assignment
- Missing validation checks
- Incomplete functionality
- API interaction mismatch
- Faulty logic flow
- Inefficient performance
- Security vulnerability
- UI state misuse
- Misconfigured settings
- Distributed interaction failure
- Resource leakage
- Concurrency race conditions
- Data inconsistency
- Browser incompatibility
- Visual layout issues
- Real-time timing errors
- Resource overuse
- Hardware access mistake
After exhausting these, you can also come up with your own. This is a list of prompts so far:
{{prompts}}
Write one more short prompt that encourages a fix in a new, broad direction. The direction should generalize to any codebase, so avoid niche topics. The prompt should also specify that the fix could be in the start function OR a function related to it.
Only change whats in <pr_description> in the previous prompts. Assume the exact same jinja inputs. Write your answer in <output> tags. 
"""

def call(prompts,):
    synth_pr = pp_regex(pp_query(base_url="https://api.anthropic.com/v1/", model="claude-sonnet-4-5-20250929", system="You are a helpful software assistant",
                            prompt=PROMPT,
                            api_key=os.getenv("ANTHROPIC_API_KEY"),
                            args={"prompts": prompts}))
    if synth_pr:
        synth_pr[0] = synth_pr[0].replace("\\n", "\n").replace("\"", "").strip()
        print(synth_pr[0])
        return synth_pr[0]

def main():
    first_prompt = """
    <uploaded_files>
    {{working_dir}}
    </uploaded_files>
    I've uploaded a python code repository in the directory {{working_dir}}. Consider the following PR description:

    <pr_description>
    Possible bug in the library related to {{start_fn}} in {{start_fn_file}}.
    When I call {{start_fn}}() my behavior is not what is expected. The issue may be in {{start_fn}} or a function downstream/upstream of it.
    </pr_description>

    Can you help me implement the necessary changes to the repository so that the issues described in the <pr_description> are fixed?
    I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
    Your task is to make the minimal changes to non-tests files in the {{working_dir}} directory to ensure the issues in <pr_description> are fixed.
    Follow these steps to resolve the issue:
    1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
    2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
    3. Edit the sourcecode of the repo to resolve the issue
    4. Rerun your reproduce script and confirm that the error is fixed!
    5. Think about edgecases and make sure your fix handles them as well
    Your thinking should be thorough and so it's fine if it's very long.
    """
    initial_issue_prompts = [
        first_prompt
    ]
    for i in range(50):
        while True:
            try:
                next_prompt = call(initial_issue_prompts)
                break
            except Exception as e:
                time.sleep(10)
        initial_issue_prompts.append(next_prompt)

    with open("initial_issue_prompts.json", "w") as f:
        json.dump(initial_issue_prompts, f, indent=4)

main()