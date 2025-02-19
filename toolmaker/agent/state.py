from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Self

from openai.types.chat import ChatCompletionMessageParam

from toolmaker.actions import Action, Observation
from toolmaker.utils import remove_newlines
from toolmaker.utils.logging import WithResponse

type ActionAndObservation[TObs: Observation] = tuple[Action[TObs], TObs]


@dataclasses.dataclass(frozen=True)
class AgentState[T](WithResponse[T]):
    response: T
    actions: Sequence[ActionAndObservation] = ()
    messages: Sequence[ChatCompletionMessageParam] = ()

    def append_message(self, message: ChatCompletionMessageParam) -> Self:
        return dataclasses.replace(self, messages=(*self.messages, message))

    def append_messages(self, messages: Sequence[ChatCompletionMessageParam]) -> Self:
        return dataclasses.replace(self, messages=(*self.messages, *messages))

    __rshift__ = append_message

    def append_action[TObs: Observation](
        self, action: Action[TObs], observation: TObs
    ) -> Self:
        return dataclasses.replace(
            self,
            actions=(*self.actions, (action, observation)),
        )

    def reset_actions(self) -> Self:
        return dataclasses.replace(self, actions=())

    def with_response[TResponse](self, response: TResponse) -> AgentState[TResponse]:
        return dataclasses.replace(self, response=response)  # type: ignore

    def bash(self) -> str:
        return "\n".join(
            f"""# Step {i}: {remove_newlines(action.reasoning)}
{'' if action.bash_side_effect else '# '}{action.bash()}
# observation: {remove_newlines(repr(observation))}
"""
            for i, (action, observation) in enumerate(self.actions)
        )
