from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, overload

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from toolmaker.actions import (
    ACTIONS,
    Action,
    FunctionCallErrorObservation,
    Observation,
    truncate_observation,
)
from toolmaker.actions.io import ReadFile
from toolmaker.agent.state import AgentState
from toolmaker.llm import LLMCall, TypedLLMCall, typed_call
from toolmaker.utils.logging import tlog


class AgentExhaustedError(Exception):
    """Raised when the agent has exhausted its maximum number of steps."""


class Runtime(Protocol):
    """A runtime that can execute actions and return observations."""

    def execute[TObs: Observation](
        self, action: Action[TObs]
    ) -> TObs | FunctionCallErrorObservation: ...


class Agent[T: (BaseModel | str)]:
    def __init__(
        self,
        completion: TypedLLMCall[T],
        actions: Sequence[type[Action]] = tuple(ACTIONS.values()),
    ):
        self.completion = completion
        self.actions: Mapping[str, type[Action]] = {
            action.action: action for action in actions
        }
        self.tools: Sequence[ChatCompletionToolParam] = [
            action.to_function_schema() for action in self.actions.values()
        ]

    @classmethod
    def _may_truncate(cls, action: Action, observation: Observation) -> bool:
        # Never truncate markdown files
        return not (
            isinstance(action, ReadFile)
            and Path(action.path).suffix in (".md", ".py", ".ipynb")
        )

    def step(
        self, state: AgentState[T | None], /, runtime: Runtime
    ) -> AgentState[T | None]:
        """Run a single step of the agent.

        Args:
            state: The initial state of the agent.
            runtime: The runtime to use to execute the agent's actions.

        Returns:
            The result of the agent step. The `.messages` field will contain the new messages, not including the initial
            messages passed to the `step` method via the `messages` argument.
        """
        message = self.completion(messages=state.messages, tools=self.tools)

        if not message.tool_calls:
            # This fixes a bug in the OpenAI API. When submitting an assistant
            # message with an empty tool call list in the next request, the API
            # will throw an error. To fix this, we set the tool call list to None.
            message.tool_calls = None

        state >>= cast(ChatCompletionMessageParam, message.model_dump())

        for call in message.tool_calls or []:
            action_type: type[Action] = self.actions[call.function.name]
            action: Action = action_type.model_validate_json(call.function.arguments)

            # Execute the action
            with tlog.context("action", content=action, bash=action.bash()) as ctx:
                observation = runtime.execute(action)
                if self._may_truncate(action, observation):
                    observation = truncate_observation(observation)
                ctx.content = observation

            state >>= dict(
                role="tool",
                content=observation.model_dump_json(),
                tool_call_id=call.id,
            )

            state = state.append_action(action, observation)

        return state.with_response(message.parsed)

    def run(
        self,
        state: AgentState = AgentState(response=None),
        *,
        runtime: Runtime,
        max_steps: int = 40,
    ) -> AgentState[T]:
        """Run the agent until it either returns a response or reaches the maximum number of steps.

        Returns:
            The result of the agent run. The `.messages` field will contain the new messages, not including the initial
            messages passed to the `run` method via the `messages` argument.
        """

        state = state.with_response(None)
        for step in range(max_steps):
            with tlog.context("step", step=step) as ctx:
                state = self.step(state, runtime)
                ctx.content = state.response

            # If the agent has returned a response, we can stop
            if state.response is not None:
                return cast(AgentState[T], state)
        raise AgentExhaustedError(
            f"Agent has exhausted its maximum number of steps ({max_steps})."
        )


@overload
def completion_step[T: BaseModel | str](
    state: AgentState,
    completion: TypedLLMCall[T],
    *,
    map_result: Callable[[T | None], T | None] = lambda x: x,
) -> AgentState[T | None]: ...


@overload
def completion_step[T: BaseModel | str](
    state: AgentState,
    completion: LLMCall[T],
    *,
    map_result: Callable[[T | None], T | None] = lambda x: x,
    response_format: type[T],
) -> AgentState[T | None]: ...


def completion_step[T: BaseModel | str](
    state: AgentState,
    completion: TypedLLMCall[T] | LLMCall[T],
    *,
    map_result: Callable[[T | None], T | None] = lambda x: x,
    response_format: type[T] | None = None,
) -> AgentState[T | None]:
    if response_format is not None:
        completion = typed_call(completion, response_format)  # type: ignore
    response = completion(messages=state.messages)
    state >>= cast(
        ChatCompletionAssistantMessageParam,
        dict(role="assistant", content=response.parsed),
    )
    return state.with_response(map_result(response.parsed))
