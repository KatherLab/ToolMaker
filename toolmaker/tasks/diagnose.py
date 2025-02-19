from collections.abc import Sequence

from pydantic import BaseModel, Field

from toolmaker.agent import Agent, AgentState, Runtime
from toolmaker.llm import LLMCall, typed_call
from toolmaker.runtime.code import FunctionCallResult
from toolmaker.tasks.assess import FunctionExecutionAssessment
from toolmaker.utils import truncate_str
from toolmaker.utils.logging import tlog


class GatheredInformation(BaseModel):
    diagnosis: str = Field(description="A diagnosis of the issue that occurred.")
    plan: str = Field(description="A plan to fix the issue.")


@tlog.state_fn
def diagnose(
    state: AgentState,
    /,
    output: FunctionCallResult,
    code: str,
    problem_summaries: Sequence[str],
    assessment: FunctionExecutionAssessment,
    completion: LLMCall,
    runtime: Runtime,
) -> AgentState[GatheredInformation]:
    user_prompt = f"""
Your initial code implementation did not work. This was attempt number {len(problem_summaries)} to fix the problem.

Here is a summary of the previous problems, and your attempts to fix them. Keep this in mind as we proceed, and avoid repeating the same mistakes.
<summaries>
{"\n".join(f"<summary number={i}>{summary}</summary>" for i, summary in enumerate(problem_summaries))}
</summaries>

The current version of your code (after {len(problem_summaries)} attempts) is below.
IMPORTANT: this is the most up-to-date version of your code, so focus on it when diagnosing the problem.
```python
{code}
```

Upon executing this updated function, I received another error.
As a diligent software engineer AI, your task is now to diagnose the issue and fix the function.
You can't see, draw, or interact with a browser, but you can read and write files, and you can run commands, and you can think.
You will be provided with the stdout and stderr from the function execution.
First, use your tools (e.g. running commands, listing directories, reading files, etc.) to gather information about the issue, in order to diagnose it.
Specifically, try to find out the root cause of the issue. Often, this requires reading relevant code files in the repository to understand how the problem occured, and if any assumptions you made in your implementation of the function are incorrect.
Then, formulate a plan to fix the issue, and finally respond with that plan.

NOTE: The plan you write should be the immediate plan to modify the function to fix the issue. 
After you provide the plan, you will then be asked to provide the code to implement the plan, and I will execute that code. 
I will then give you the output of the code execution, and you will be asked to provide a new plan to fix the new issue. 
Therefore, if after exploring the codebase you still don't know what's wrong, your plan should be to modify the function to provide more logging to help you diagnose the problem next time it is executed.

IMPORTANT: While you are able to interact with the environment (writing files, running commands, etc.), any changes you make will be lost when the function is executed again, as the environment will be reset. Therefore, use this opportunity only to gather information about the issue, and not to fix it.
HINT: After gathering information, you may decide to use a slightly different approach to fix the issue -- if this is the case, include this in your plan! 
HINT: Always prefer importing code from the repository, rather than implementing it yourself. The information you gather may contain code that you can import to fix the issue.

Output (stdout and stderr) of the function execution:
<output>
{truncate_str(output.stdout, max_length=20000)}
</output>

Initial assessment why the function call was not successful:
<assessment>
{assessment.reasoning}
</assessment>

As mentioned above, your immediate task is to diagnose the issue, and formulate a plan to fix it.
"""

    return Agent(
        typed_call(completion, GatheredInformation),
    ).run(
        state >> dict(role="user", content=user_prompt),
        runtime=runtime,
    )
