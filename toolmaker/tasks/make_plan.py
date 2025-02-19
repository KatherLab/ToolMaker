from typing import cast

from toolmaker.actions import ACTIONS
from toolmaker.actions.io import WriteFile
from toolmaker.agent import Agent, AgentState, Runtime, completion_step
from toolmaker.definition import ToolDefinition, get_local_install_path
from toolmaker.llm import LLM, LLM_MODEL, LLM_MODEL_REASONING
from toolmaker.runtime.client import HTTPRuntimeClient
from toolmaker.tasks.install import SYSTEM_PROMPT, InstalledRepository
from toolmaker.utils.logging import tlog
from toolmaker.utils.papers import get_paper_summary_prompt


@tlog.state_fn
def explore_repository(
    llm: LLM,
    runtime: Runtime,
    definition: ToolDefinition,
    installed_repository: InstalledRepository | None = None,
    include_paper_summary: bool = False,
) -> AgentState[str]:
    user_prompt = f"""# Background
The repository `{definition.repo.name}` is fully set up and installed at `{get_local_install_path(definition.repo)!s}`.
We need to wrap a specific functionality from this repository into a standalone python function, that can be called independently. 
This function will be called `{definition.name}`, and it is described as follows:
<description>
{definition.description}
</description>

The function will have the following arguments:
<arguments>
{"\n".join(f"<argument>{arg!r}</argument>" for arg in definition.arguments)}
</arguments>

# High-level approach
In order to implement this function, you will follow these steps:
1. Explore the repository to gather all relevant information needed to write the plan.
2. Write a plan for the body/implementation of the function. This plan should be in the form of very high-level pseudo-code, that describes how the function will work.
3. Write the function, based on the plan.

# Task
Right now, you are at step 1: Explore the repository to gather all relevant information needed to write the plan.
This step is very important, and you must be thorough because you will rely on this information later when implementing the function.
To gather all relevant information needed for the implementation, explore the repository, but only look at relevant files.
Use the tools at your disposal to read files, list directories, search, etc.
HINT 1: If the repository contains a README file, that is often a good starting point. Note that there may be zero or more README files. Always check for README files, and prefer to follow the instructions therein.
HINT 2: If the repository provides a command line interface, prefer to invoke that via subprocess, rather than calling the underlying python functions. Only as a last resort, wrap python functions.
HINT 3: Do NOT attempt to read image files, audio files, etc.
Do not unnecessarily read files that are not relevant for the task.
**However, make sure to read ALL files (e.g. documentation, code, configuration files, etc.) that are necessary in order to implement the function. It should be possible to implement the function based only on the plan and the files you read!**
**You should read relevant code files in order to understand how the functionality you are wrapping is implemented. If you are planning to wrap specific functions, be sure to read the relevant code in order to understand what the input and output arguments/formats are. This is especially relevant if the function you are wrapping produces output files that you will need to read.**
Do NOT write the function yet.
Your task is specifically to explore the repository to gather information.

Once you have gathered ALL relevant information, respond with a one-paragraph summary of what you found.

Remember, the function should do the following:
<description>
{definition.description}
</description>

As such, the signature of the function will be:
```python
{definition!s}
```
    """

    if include_paper_summary:
        user_prompt += f"\n\n{get_paper_summary_prompt(definition)}\n\n"

    system_prompt = f"{SYSTEM_PROMPT}\n\nYou have already installed the {definition.repo.name} repository and its dependencies at `{get_local_install_path(definition.repo)!s}`."

    return Agent(
        llm[LLM_MODEL],
        # Do not give this agent the ability to write files, as we don't want it to implement the function yet.
        actions=tuple(action for action in ACTIONS.values() if action is not WriteFile),
    ).run(
        AgentState(response=None)
        >> dict(role="system", content=system_prompt)
        >> dict(role="user", content=user_prompt),
        runtime=runtime,
    )


@tlog.state_fn
def make_plan(
    llm: LLM,
    runtime: HTTPRuntimeClient,
    definition: ToolDefinition,
    installed_repository: InstalledRepository | None = None,
    include_paper_summary: bool = False,
) -> AgentState[str]:
    state = explore_repository(
        llm=llm,
        runtime=runtime,
        definition=definition,
        installed_repository=installed_repository,
        include_paper_summary=include_paper_summary,
    )

    user_prompt = f"""
    Using the information you gathered previously, your task is now to write an outline (plan) for the body/implementation of the function. 
    This plan should be in the form of very high-level pseudo-code, that describes how the function will work.
    It should be a numbered list of steps, each of which describes what you will do in that step.
    Respond with just this list of steps, nothing else.
    Remember, the function should do the following: `{definition.description}`
    
    As such, the signature of the function will be:
    ```python
    {definition!s}
    ```
    """

    return cast(
        AgentState[str],
        completion_step(
            state >> dict(role="user", content=user_prompt),
            llm[LLM_MODEL_REASONING],
        ),
    )
