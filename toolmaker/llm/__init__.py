from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Protocol, Self, cast, overload

import litellm
from loguru import logger
from openai.lib._parsing import parse_chat_completion
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from pydantic import BaseModel, ValidationError

from toolmaker.llm.completions import completion_factory
from toolmaker.utils.logging import tlog

MAX_COST: Final[float] = 5.0  # dollars
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "gpt-4o")
LLM_MODEL_REASONING: Final[str] = os.getenv("LLM_MODEL_REASONING", "o1-mini")
LLM_MODEL_SUMMARY: Final[str] = os.getenv(
    "LLM_MODEL_SUMMARY", "gpt-4o"
)  # for paper summary


class LLMCall[T: BaseModel | str](Protocol):
    """This is essentially a union of `client.beta.chat.completions.parse` and `client.chat.completions.create`.

    Use `LLMCall[str]` as an equivalent to `client.chat.completions.create`.
    Use `LLMCall[BaseModel]` as an equivalent to `client.beta.chat.completions.parse`.

    In both cases, the return value is a `ParsedChatCompletionMessage[T]`. The `parsed` field will be `None` if `T` is `str`.
    """

    def __call__(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        response_format: type[T] | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ParsedChatCompletionMessage[T]: ...


class TypedLLMCall[T: BaseModel | str](Protocol):
    """Partial application of `LLMCall` that is fixed to use a particular response format, equivalent to `partial(llm, response_format=response_format)`."""

    def __call__(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ParsedChatCompletionMessage[T]: ...


def typed_call[T: BaseModel](
    llm: LLMCall[T], response_format: type[T]
) -> TypedLLMCall[T]:
    """Return a partial application of `llm` that is fixed to use the given response format, equivalent to `partial(llm, response_format=response_format)`."""
    return cast(TypedLLMCall[T], partial(llm, response_format=response_format))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Usage) -> Self:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def from_litellm(cls, usage: litellm.Usage) -> Self:
        return Usage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            total_tokens=usage.total_tokens or 0,
        )

    __add__ = add


@dataclass
class LLM:
    tokens: dict[str, Usage] = field(default_factory=partial(defaultdict, Usage))
    cost: dict[str, float] = field(default_factory=partial(defaultdict, float))

    @property
    def total_prompt_tokens(self) -> int:
        return sum(usage.prompt_tokens for usage in self.tokens.values())

    @property
    def total_completion_tokens(self) -> int:
        return sum(usage.completion_tokens for usage in self.tokens.values())

    @property
    def total_cost(self) -> float:
        return sum(self.cost.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cost={self.total_cost}, prompt_tokens={self.total_prompt_tokens}, completion_tokens={self.total_completion_tokens})"

    @overload
    def completion[T: BaseModel](
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        response_format: type[T],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ParsedChatCompletionMessage[T]: ...

    @overload
    def completion(
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        response_format: None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ParsedChatCompletionMessage[str]: ...

    def completion[T: BaseModel | str](
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        response_format: type[T] | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ParsedChatCompletionMessage[T]:
        is_typed = response_format is not None
        prompt_tokens = litellm.token_counter(
            model=model,
            messages=messages,
            tools=tools,
        )
        prompt_cost = litellm.completion_cost(model=model, messages=messages)
        with tlog.context(
            "llm_call",
            model=model,
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            content=messages,
        ) as ctx:
            if prompt_cost + self.total_cost > MAX_COST:
                raise RuntimeError(
                    f"Cost limit exceeded: {prompt_cost + self.total_cost:.3f} > {MAX_COST:.3f}"
                )
            # print("MESSAGES:", json.dumps(list(messages)))
            # print("TOOLS:", json.dumps(tools))

            response = completion_factory(model)(
                model=model,
                messages=list(messages),
                tools=tools,
                response_format=response_format if is_typed else None,
            )
            message = response.choices[0].message
            completion_cost = litellm.completion_cost(response)
            usage = Usage.from_litellm(response.usage)
            self.tokens[model] += usage
            self.cost[model] += completion_cost
            logger.debug(
                f"Called {model} with {usage.total_tokens} tokens for ${completion_cost:.3f} (total cost: ${self.total_cost:.3f})"
            )
            ctx.content = message.model_dump()
            ctx.update(
                cost=completion_cost,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                overall_cost=self.total_cost,
                overall_prompt_tokens=self.total_prompt_tokens,
                overall_completion_tokens=self.total_completion_tokens,
            )

        # Parse (if typed), and return
        if is_typed:
            message.refusal = None
            try:
                return (
                    parse_chat_completion(
                        response_format=cast(type[T], response_format),
                        chat_completion=response,
                        input_tools=tools or [],
                    )
                    .choices[0]
                    .message
                )
            except ValidationError as e:
                raise RuntimeError(
                    f"Validation error: {e}\n\nResponse: {message.content}"
                ) from e
        else:
            return ParsedChatCompletionMessage(
                **message.model_dump(), parsed=message.content
            )

    def for_model(self, model: str) -> LLMCall:
        """Return a partial application of `completion` that is fixed to use the given model."""
        return partial(self.completion, model=model)

    __getitem__ = for_model
