from typing import cast

from loguru import logger
from openai.types.chat import ChatCompletionAssistantMessageParam
from pydantic import BaseModel, Field
from toolarena.definition import ToolDefinition

from toolmaker.agent import AgentState, completion_step
from toolmaker.llm import LLMCall
from toolmaker.runtime.code import FunctionCallResult
from toolmaker.utils import truncate_str
from toolmaker.utils.logging import tlog


class FunctionExecutionAssessment(BaseModel):
    successful: bool = Field(
        description="Whether the function call was successful, and the task is complete."
    )
    reasoning: str = Field(description="The reasoning for the assessment.")


@tlog.state_fn
def is_successful_execution(
    state: AgentState,
    /,
    output: FunctionCallResult,
    definition: ToolDefinition,
    completion: LLMCall,
) -> AgentState[FunctionExecutionAssessment]:
    state >>= dict(
        role="user",
        content=f"""I executed the function you wrote.
Based on the output and returned result, assess whether the function call was successful or not.
Specifically, you should assess whether the function performed the task it was supposed to perform.
Also make sure that the returned result is plausible and matches the stdout/stderr output logs, if applicable.
As a reminder, the task is the following:
<task_description>
{definition.description}
</task_description>

Description of expected result:
<expected_result_description>
{definition.description_of_returns()}
</expected_result_description>

Returned result:
<result>
{truncate_str(repr(output.result), max_length=10000)}
</result>

Output (stdout and stderr) of the function execution:
<output>
{truncate_str(output.stdout, max_length=10000)}
</output>

**IMPORTANT: You must also ensure that the returned result itself is correct. This includes ensuring that the result dict contains the correct keys and values, and that the values have the correct types and shapes! If any of these are incorrect, the function call is NOT successful! If this is the case, include this in your reasoning.**
""",
    )

    if output.status == "error":
        assessment = FunctionExecutionAssessment(
            successful=False,
            reasoning="The function call failed with a non-zero exit code.",
        )
        state >>= cast(
            ChatCompletionAssistantMessageParam,
            dict(role="assistant", content=assessment.model_dump()),
        )
        return state.with_response(assessment)

    logger.debug("Status code is successful; using LLM to check output.")

    return cast(
        AgentState[FunctionExecutionAssessment],
        completion_step(state, completion, response_format=FunctionExecutionAssessment),
    )
