import os
from collections.abc import Sequence

from openai import BaseModel
from pydantic import Field

from toolmaker.actions.actions import Action, Observation, register_action
from toolmaker.utils.bash import run_bash_command


class BashCommandOutput(Observation):
    """The output of a bash command."""

    return_code: int = Field(..., description="The return code of the bash command.")


class EnvironmentVariable(BaseModel):
    """An environment variable."""

    key: str = Field(..., description="The name of the environment variable.")
    value: str = Field(..., description="The value of the environment variable.")


@register_action
class RunBashCommand(Action):
    """Run a bash command, return the output.

    The command must be non-interactive.
    Do not use `sudo` or `su` to run the command.

    Some environment variables may be automatically set by the system (e.g. HF_TOKEN).
    You may use these environment variables in your command without explicitly setting them in the `env` field.

    Always prefer to run a single command at a time because the command output will be truncated if it is too long, thus
    potentially losing important information.
    """

    action = "run_bash_command"
    command: str = Field(..., description="The bash command to run.")
    env: Sequence[EnvironmentVariable] = Field(
        default_factory=(),
        description="Additional environment variables to set before running the command (only visible to the bash command). Do NOT set environment variables here that you want to use from the system instead.",
    )

    async def __call__(self) -> BashCommandOutput:
        out = await run_bash_command(
            self.command, env=dict(os.environ) | {e.key: e.value for e in self.env}
        )
        return BashCommandOutput(content=out.output, return_code=out.return_code)

    bash_side_effect = True

    def bash(self) -> str:
        return self.command


# @register_action
# class SetEnvironmentVariable(Action):
#     """Set an environment variable."""

#     action = "set_environment_variable"
#     key: str = Field(..., description="The name of the environment variable to set.")
#     value: str = Field(..., description="The value of the environment variable to set.")

#     def __call__(self) -> None:
#         os.environ[self.key] = self.value
#         return Observation(content="Environment variable set.")

#     bash_side_effect = True

#     def bash(self) -> str:
#         return f"export {self.key}={shlex.quote(self.value)}"
