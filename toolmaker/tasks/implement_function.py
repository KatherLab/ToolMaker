from functools import partial
from typing import cast

from toolarena.definition import Repository, ToolDefinition

from toolmaker.agent import AgentState, completion_step
from toolmaker.definition import get_local_install_path
from toolmaker.llm import LLMCall
from toolmaker.utils.llm import process_llm_code_output
from toolmaker.utils.logging import tlog


def environment_variables_prompt(repo: Repository) -> str:
    return (
        f"""IMPORTANT: the following environment variables are set in your system environment: {", ".join(f"`{k}`" for k in repo.env.keys())}.
These environment variables are automatically available within your python function (e.g. via `os.environ`).
However, if you want to run subcommands, you need to pass them explicitly from os.environ to the subprocess.
"""
        if repo.env
        else ""
    )


def coding_instructions(definition: ToolDefinition) -> str:
    return f"""
You **must** output a valid, standalone python function that is callable without any modification by a user.
The requirements for the code are:
1. Import the required modules/libraries.
2. You are only allowed to write a single python function. It must start with 'def ...' and end with 'return ...'.
3. You are not allowed to output free texts, test code for the function or anything outside of the function definition.
4. The function needs to be a standalone function that can be called independently.
5. Make sure all required imports are included in the function.
6. The function must perform the task you are given. As a reminder, the task is: `{definition.description}`.
7. Make sure the function accepts all required parameters as inputs.
8. The function must have type hints and a docstring.
9. The function must be named exactly `{definition.name}`.
10. The function must be a valid python function, that can be executed by a python interpreter.

{environment_variables_prompt(definition.repo)}

Additional instructions:
* Write the function in such a way that it can easily be debugged later. This means that you should include a lot of print statements for logging purposes. Especially for long-running tasks, it is important to print the progress periodically.
* When catching exceptions in the code (with `try` and `except`), make sure to output the entire stack trace to stderr, so that it can be used to diagnose any issues, e.g. using `traceback.format_exc()`.
* When running commands and scripts (e.g. using subprocesses), make sure to stream the stdout and stderr to the parent process, so that it can be used to diagnose any issues.
  Use the utility function `run_and_stream_command` provided by the `subprocess_utils` module. It accepts the same arguments as `subprocess.Popen`, and returns a tuple `(return_code, output)` (return_code is an integer, output is a string containing stdout and stderr combined).
  The `run_and_stream_command` automatically handles the streaming of stdout and stderr to the parent process. Be sure to appropriately set the `cwd` argument.
  Example usage:
  ```python
  from subprocess_utils import run_and_stream_command  # you must import this
  return_code, output = run_and_stream_command("echo hello && echo world", shell=True, env={{"MY_VAR": "my_value"}}, cwd="/workspace/my_project")  # shell=True is the default
  ```
* Make sure that you do not run interactive commands. If some python function that you are calling itself runs interactive commands, try and find a way to avoid calling that function. If, as a last resort, you cannot avoid calling that function, mock/patch the external interactive function to ensure that it does not run interactive commands.
* Always prefer to import existing functions into the function you are writing, or run existing scripts/modules (e.g. via the subprocess functionality descibed above), instead of writing your own implementations. Only if this does not work, or there is no existing function that can be imported, write your own implementation.
"""


@partial(tlog.state_fn, name="update_code")
def implement_function(
    state: AgentState,
    /,
    definition: ToolDefinition,
    plan: str,
    completion: LLMCall[str],
) -> AgentState[str]:
    user_prompt = f"""Now that you have identified the plan for the implementation, you need to write the actual implementation of the function.
This needs to be a standalone python function, that can be called independently. 
This function will be called `{definition.name}`, and it is described as follows: `{definition.description}`.
The function will have the following arguments:
{"\n".join(("- " + repr(arg)) for arg in definition.arguments)}

As such, the signature of the function will be:
```python
{definition!s}
```

Your task is now to write the Python function.
To do so, follow the plan you identified earlier for the implementation:
<plan>
{plan}
</plan>

{coding_instructions(definition)}

Remember, you should use the repository `{definition.repo.name}` (installed at `{get_local_install_path(definition.repo)!s}`) to complete the task.
Finally, ensure your function is ready-to-use without any modifications by a user. In many cases, wrapping an existing function, script or module in a subprocess is enough.
Respond with the code of the function only, without any other text.
"""

    return cast(
        AgentState[str],
        completion_step(
            state >> dict(role="user", content=user_prompt),
            completion,
            map_result=process_llm_code_output,  # type: ignore
        ),
    )
