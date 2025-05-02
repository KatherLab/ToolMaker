import os
from typing import Self

import yaml
from pydantic import BaseModel, Field
from toolarena.definition import Repository, ToolDefinition

from toolmaker.agent import Agent, AgentState, Runtime
from toolmaker.definition import get_local_install_path
from toolmaker.llm import LLM, LLM_MODEL, typed_call
from toolmaker.utils.logging import tlog
from toolmaker.utils.papers import get_paper_summary_prompt
from toolmaker.utils.paths import LOCAL_WORKSPACE_DIR


class InstalledRepository(BaseModel):
    path: str = Field(description="The path to the cloned and installed repository.")
    summary: str = Field(
        description="A one-paragraph summary of what you did and what you accomplished. Be sure to include important paths and files to things you installed or downloaded."
    )

    @classmethod
    def from_yaml(cls, yaml_file: os.PathLike | str) -> Self:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


SYSTEM_PROMPT = f"""You're a diligent software engineer AI. You can't see, draw, or interact with
a browser, but you can read and write files, and you can run commands, and you can think.
The user will specify a task for you to complete. You likely need to run several actions
in order to complete the task. You will only be able to execute a single action at a time.

Use the tools (actions) that are at your disposal. 
Each time you invoke a tool, provide a one-sentence summary of why you are invoking it
and what you expect to accomplish by invoking it.

Your workspace directory and current working directory is `{LOCAL_WORKSPACE_DIR!s}`.

You will continue the process of invoking tools until you have completed the task."""


def environment_variables_prompt(repo: Repository) -> str:
    return (
        f"""IMPORTANT: the following environment variables are set in your system environment: {", ".join(f"`{k}`" for k in repo.env.keys())}.
These environment variables are automatically available when you run the `run_bash_command` tool, and don't need to be set in the `env` field of the tool call.
"""
        if repo.env
        else ""
    )


@tlog.state_fn
def install_repository(
    llm: LLM,
    runtime: Runtime,
    definition: ToolDefinition,
    max_steps: int = 20,
    include_paper_summary: bool = False,
) -> AgentState[InstalledRepository]:
    install_path = str(get_local_install_path(definition.repo))

    user_prompt = f"""Clone and locally set up the {definition.repo.name} repository from GitHub.
Follow these steps:
1. Git clone the repository {definition.repo.info()} into the directory `{install_path}`.
2. Check the README (find it if it is not in the root directory) and closely follow the recommended instructions to set up the entire repository correctly for the user.
3. Follow the instructions in the README to correctly set up the repository for the user. Perform any necessary installations, configurations, downloads or setups as described. If the repository is in Python, prefer using `pip` as opposed to conda, virtualenv, or similar. Install the repository and its dependencies globally. Do not use Docker or similar container tools (even if the README suggests it); instead, install the repository and its dependencies globally.
4. Make sure that you complete every step, so that a user could directly use this repository without the need to do further setups, installations or downloads. This includes downloading any necessary pretrained models. However, do NOT download any datasets.
If you encounter any issues, try to solve them.

{environment_variables_prompt(definition.repo)}

You should set up the repository in such a way that it can be used to implement the following task later on:
<intended_task>
{definition.xml_summary}
</intended_task>
IMPORTANT: Your task right now is to only set up the repository, NOT implement this task.

{get_paper_summary_prompt(definition) if include_paper_summary else ""}

Continue calling tools until you are done and have installed and set up the repository.
Once you are done, provide a brief summary of what you did and what you accomplished, as well as the absolute path to the cloned and installed repository."""

    state = Agent(typed_call(llm[LLM_MODEL], InstalledRepository)).run(
        AgentState(response=None)
        >> dict(role="system", content=SYSTEM_PROMPT)
        >> dict(role="user", content=user_prompt),
        runtime=runtime,
        max_steps=max_steps,
    )

    if state.response.path != install_path:
        raise RuntimeError(
            f"Agent did not install the repository to the expected path: {install_path}"
        )

    return state
