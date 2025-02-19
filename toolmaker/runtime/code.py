from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel

from toolmaker.definition import ArgumentValue
from toolmaker.utils.env import substitute_env_vars


class FunctionCall(BaseModel):
    code: str
    name: str
    args: Mapping[str, ArgumentValue]

    def substitute_env_vars(self, env: Mapping[str, str] | None = None) -> Self:
        return self.model_copy(
            deep=True,
            update=dict(
                args={
                    k: substitute_env_vars(v, env) if isinstance(v, str) else v
                    for k, v in self.args.items()
                }
            ),
        )


class FunctionCallResult(BaseModel):
    status: Literal["success", "error"]
    result: Any | None = None
    stdout: str | None = None
