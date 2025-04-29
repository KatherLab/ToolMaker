import shutil
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from toolarena.definition import Repository, ToolDefinition

import toolmaker
from toolmaker.runtime.client import Mounts
from toolmaker.utils.env import substitute_env_vars
from toolmaker.utils.io import rmdir
from toolmaker.utils.papers import get_paper_summary_prompt

WORKSPACES_DIR = (
    Path(toolmaker.__file__).resolve().parent.parent.parent
    / "openhands_toolmaker"
    / "workspaces"
)


def environment_variables_prompt(repo: Repository) -> str:
    return (
        f"""IMPORTANT: the following environment variables are set in your system environment: {", ".join(f"`{k}`" for k in repo.env.keys())}.
These environment variables are automatically available in your system and will be available within the Python function you implement.
However, if you decide to run the python function yourself, you will need to pass them explicitly to the subprocess.
"""
        if repo.env
        else ""
    )


def create_openhands_prompt(
    definition: ToolDefinition, include_paper_summary: bool = False
) -> str:
    prompt = f"""
Your task is to create a tool from the repository {definition.repo.name} which implements the function `{definition.name}` to perform the following task: `{definition.description}`.
While you may perform any necessary installations, configurations, downloads or setups, your deliverables are the following two files:
1. A bash script, named `/workspace/install.sh` that will install all necessary dependencies for the tool to run.
2. A Python file, named `/workspace/code.py` that will contain the code for the tool.

# Part 1: Install the repository
Clone and locally set up the {definition.repo.name} repository from GitHub.
Follow these steps:
1. Git clone the repository {definition.repo.info()}.
2. Check the README (find it if it is not in the root directory) and closely follow the recommended instructions to set up the entire repository correctly for the user.
3. Follow the instructions in the README to correctly set up the repository for the user. Perform any necessary installations, configurations, downloads or setups as described. If the repository is in Python, prefer using `pip` as opposed to conda, virtualenv, or similar. Install the repository and its dependencies globally.
4. Make sure that you complete every step, so that a user could directly use this repository without the need to do further setups, installations or downloads. This includes downloading any necessary models. However, do NOT download any datasets.
If you encounter any issues, try to solve them.

{environment_variables_prompt(definition.repo)}

# Part 2: Implement the tool function
You need to implement a standalone python function, that can be called independently. 
This function will be called `{definition.name}`, and it is described as follows: `{definition.description}`.
The function will have the following arguments:
{"\n".join((f"- {arg_name} ({arg.type}): {arg.description}") for arg_name, arg in definition.arguments.items())}

As such, the signature of the function will be:
```python
{definition!s}
```
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

Remember, you should use the repository `{definition.repo.name}` to complete the task.
Finally, ensure your function is ready-to-use without any modifications by a user. In many cases, wrapping an existing function, script or module in a subprocess is enough.
Note: It may be useful to run the function with the following example invocation to test it:
```python3
from code import {definition.name}
{definition.name}({", ".join(f"{k}={v!r}" for k, v in definition.example.arguments.items())})
```

# IMPORTANT:
- The only two files that you need to produce are `/workspace/install.sh` and `/workspace/code.py` (though you may create other files as well, or install additional dependencies in the process).
- You may use any tools at your disposal to complete the task.
- From within a fresh environment (i.e. a fresh Docker image of python:3.12) that contains the `/workspace` directory which is empty except for your `install.sh` and `code.py` files, it should be possible to run the `install.sh` script, and then run the `code.py` file, without any additional prior installations or dependencies.
- The `code.py` file should NOT contain any imports at the top of the file. The first line of the file should be the function signature (of the `{definition.name}` function). In the body of the function, you may import any necessary modules.
"""
    if include_paper_summary:
        prompt += f"\n\n{get_paper_summary_prompt(definition)}"
    return prompt


def create_openhands_config(environment_variables: dict) -> str:
    return f"""
[core]
debug = true

[sandbox]
#timeout = 120
enable_gpu = true
runtime_startup_env_vars = {{{", ".join(f"{k} = {v!r}" for k, v in environment_variables.items())}}}
"""


def create_openhands_task(
    task: Annotated[
        str, typer.Argument(help="YAML file containing the task definition")
    ],
    task_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for the task",
        ),
    ],
    include_paper_summary: Annotated[
        bool,
        typer.Option(
            "--paper-summary",
            help="Whether to include the paper summary in the prompt",
        ),
    ] = False,
) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    definition = ToolDefinition.from_yaml(task)

    env = {k: substitute_env_vars(v) for k, v in definition.repo.env.items()}

    rmdir(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("prompt.txt").write_text(
        create_openhands_prompt(definition, include_paper_summary=include_paper_summary)
    )
    task_dir.joinpath("openhands_config.toml").write_text(create_openhands_config(env))
    task_dir.joinpath(".env").write_text("\n".join(f"{k}={v}" for k, v in env.items()))

    # Copy definition
    shutil.copy(task, task_dir / "task_definition.yaml")

    # Mounts
    input_mount_dir = task_dir / "input"
    input_mount_dir.mkdir(parents=True, exist_ok=True)
    output_mount_dir = task_dir / "output"
    output_mount_dir.mkdir(parents=True, exist_ok=True)
    mounts = Mounts(
        input=input_mount_dir,
        output=output_mount_dir,
        input_mapping=definition.example.mount,
    )
    mounts.reset()  # copies data

    # Create workspace directory
    task_dir.joinpath("workspace").mkdir(parents=True, exist_ok=True)

    # Add a file to indicate that the task has been created
    task_dir.joinpath(".created").touch()

    logger.info(f"Created task {task_dir}")


if __name__ == "__main__":
    typer.run(create_openhands_task)
    typer.run(create_openhands_task)
