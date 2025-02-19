"""This module defines the base classes for actions and observations."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Self, cast

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel, Field

from toolmaker.utils import truncate_str
from toolmaker.utils.llm import pydantic_to_function_schema


class Observation(BaseModel):
    content: Any = Field(..., description="The content of the observation.")


def truncate_observation[T: Observation](observation: T, max_length: int = 15000) -> T:
    """Truncate the content of an observation if it exceeds the maximum length to avoid excessive token usage."""
    content = (
        observation.content
        if isinstance(observation.content, str)
        else json.dumps(observation.model_dump(include={"content"})["content"])
    )
    if len(content) <= max_length:
        return observation

    content = truncate_str(content, max_length)
    return cast(T, observation.model_copy(update={"content": content}))


class Action[T: Observation](BaseModel):
    reasoning: str = Field(
        "",
        description="Short one-sentence explanation why this action is chosen and what it should achieve.",
    )
    action: ClassVar[str]

    def __call__(self) -> T:
        raise NotImplementedError("Action must implement __call__")

    @classmethod
    def to_function_schema(cls: type[Self]) -> ChatCompletionToolParam:
        """Convert the action to the OpenAI function schema for tool calling."""
        return pydantic_to_function_schema(cls, name=cls.action)

    def __repr__(self) -> str:
        # Ensure the `reasoning` field is always last in the repr
        fields = sorted(
            self.model_fields.keys(),
            key=lambda key: (key == "reasoning", key),
        )
        return f"{self.__class__.__name__}({', '.join(f'{field}={getattr(self, field)!r}' for field in fields)})"

    def model_dump(self, *args, **kwargs) -> dict:
        result = super().model_dump(*args, **kwargs)
        result["action"] = self.action
        return result

    bash_side_effect: ClassVar[bool] = True  # whether the action has a side effect

    def bash(self) -> str:
        """Return the bash command representation of the action."""
        raise NotImplementedError


ACTIONS: dict[str, type[Action]] = {}


def register_action[T: type[Action]](action: T) -> T:
    ACTIONS[action.action] = action
    return action


def observation_type_for_action[T: Observation](
    action: type[Action[T]],
) -> type[T]:
    """Given an action type, return the type of the observation it returns."""
    return action.__call__.__annotations__.get("return", Observation)
