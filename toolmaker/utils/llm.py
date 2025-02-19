from openai import pydantic_function_tool
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel


def pydantic_to_function_schema(
    cls: type[BaseModel], name: str | None = None, description: str | None = None
) -> ChatCompletionToolParam:
    """Convert a BaseModel to the OpenAI function schema for tool calling."""
    tool = pydantic_function_tool(cls, name=name, description=description)

    # Remove default values from parameters
    for parameter in tool["function"]["parameters"]["properties"].values():  # type: ignore
        if "default" in parameter:
            del parameter["default"]

    # Set all additionalProperties to false
    for definition in tool["function"]["parameters"].get("$defs", {}).values():
        definition["additionalProperties"] = False
    return tool


def process_llm_code_output(code: str) -> str:
    """Process the output of an LLM call to remove any markdown code blocks if it is enclosed in them."""
    code = code.strip()
    if code.startswith("```python") and code.endswith("```"):
        code = code[len("```python") : -len("```")]
    elif code.startswith("```") and code.endswith("```"):
        code = code[len("```") : -len("```")]
    return code
