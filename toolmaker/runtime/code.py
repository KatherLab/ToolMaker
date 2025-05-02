from collections.abc import Mapping
from typing import Any, Literal, Self, Sequence

from pydantic import BaseModel, model_validator
from toolarena.definition import ArgumentValue

from toolmaker.utils.env import substitute_env_vars


class FunctionCall(BaseModel):
    code: str
    name: str
    args: Sequence[ArgumentValue]

    @model_validator(mode="after")
    def check_unique_arg_names(self) -> Self:
        arg_names = [arg.name for arg in self.args]
        if len(arg_names) != len(set(arg_names)):
            raise ValueError("Argument names must be unique")
        return self

    def substitute_env_vars(self, env: Mapping[str, str] | None = None) -> Self:
        return self.model_copy(
            deep=True,
            update=dict(
                args=[
                    v.model_copy(
                        deep=True,
                        update=dict(
                            value=substitute_env_vars(v.value, env)
                            if isinstance(v.value, str)
                            else v.value
                        ),
                    )
                    for v in self.args
                ]
            ),
        )


class FunctionCallResult(BaseModel):
    status: Literal["success", "error"]
    result: Any | None = None
    stdout: str | None = None
