from collections.abc import Sequence
from typing import Protocol

import litellm
import tenacity
import os
import functools
from litellm.types.utils import ModelResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from pydantic import BaseModel

from toolmaker.utils.llm import pydantic_to_function_schema


class ChatCompletion(Protocol):
    """Basically the signature of `litellm.completion`."""

    def __call__(
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> ModelResponse: ...


litellm_completion_retry_on_api_error: ChatCompletion = tenacity.retry(
    litellm.completion,
    retry=tenacity.retry_if_exception_type(litellm.APIError),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    after=tenacity.after_log(logger, logger.level("WARNING").no),
)

litellm_completion_retry_on_rate_limit: ChatCompletion = tenacity.retry(
    litellm.completion,
    retry=tenacity.retry_if_exception_type(litellm.RateLimitError),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(60),
    after=tenacity.after_log(logger, logger.level("WARNING").no),
)


def _make_finish_tool(response_format: type[BaseModel]) -> type[BaseModel]:
    class Finish(BaseModel):
        """This tool ends the conversation turn. Use it to respond to the user (in the format requested by the user) once you are ready to do so (i.e. after calling other tools, if applicable)."""

        response: response_format

    return Finish


def claude_completion(
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam] | None = None,
    response_format: type[BaseModel] | None = None,
) -> ModelResponse:
    FINISH_TOOL_NAME = "__finish__"
    if not tools:
        # return litellm.completion(
        #     model=model, messages=messages, response_format=response_format
        # )
        tools = []
    # Claude doesn't support specifying response_format and tools at the same time,
    # so we need to add a tool that will return the response in the correct format.
    logger.debug(f"Using Claude completion (with an extra {FINISH_TOOL_NAME} tool)")
    finish_tool = _make_finish_tool(
        response_format if response_format is not None else str
    )
    tools = [*tools, pydantic_to_function_schema(finish_tool, name=FINISH_TOOL_NAME)]
    response = litellm_completion_retry_on_rate_limit(
        model=model, messages=messages, tools=tools, tool_choice="required"
    )
    finish_call: ChatCompletionMessageToolCall | None = None
    for call in response.choices[0].message.tool_calls or ():
        if call.function.name == FINISH_TOOL_NAME:
            finish_call = call
            break
    if finish_call:
        assert all(
            call.function.name == FINISH_TOOL_NAME
            for call in response.choices[0].message.tool_calls
        ), "Expected only finish calls in the response"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        parsed = finish_tool.model_validate_json(finish_call.function.arguments)
        response.choices[0].message.content = (
            parsed.response.model_dump_json()
            if isinstance(parsed.response, BaseModel)
            else parsed.response
        )
    return response


def completion_factory(model: str) -> ChatCompletion:
    if model == "o3-mini":
        # Sometimes o3-mini is flaky, so we retry a few times
        return litellm_completion_retry_on_api_error
    elif "claude" in model:
        return claude_completion
    elif "ollama" in model:
        return functools.partial(litellm.completion, api_base=os.getenv("LOCAL_LLM_ENDPOINT"))
    return litellm.completion
