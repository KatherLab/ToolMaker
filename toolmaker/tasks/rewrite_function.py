from functools import partial
from typing import cast

from toolarena.definition import ToolDefinition

from toolmaker.agent import AgentState, completion_step
from toolmaker.llm import LLMCall
from toolmaker.tasks.diagnose import GatheredInformation
from toolmaker.tasks.implement_function import coding_instructions
from toolmaker.utils.llm import process_llm_code_output
from toolmaker.utils.logging import tlog


@partial(tlog.state_fn, name="update_code")
def rewrite_function(
    state: AgentState,
    /,
    definition: ToolDefinition,
    code_draft: str,
    diagnosis: GatheredInformation,
    completion: LLMCall,
) -> AgentState[str]:
    user_prompt = f"""Now that you have identified the problem as well as a plan to fix the function, you need to write the updated implementation of the function.
Remember, the function is called `{definition.name}`, and it is described as follows: `{definition.description}`.
The function will have the following arguments:
<arguments>
{"\n".join(f"<argument>{arg!r}</argument>" for arg in definition.arguments)}
</arguments>

As such, the signature of the function will be:
```python
{definition!s}
```

Your task is now to write the Python function.
To do so, use the information you gathered above to fix the function.

{coding_instructions(definition)}

As a reminder, the current draft of the function is:
```python
{code_draft}
```

Remember, your diagnosis is:
<diagnosis>
{diagnosis.diagnosis}
</diagnosis>

And your plan to fix the issue is:
<plan>
{diagnosis.plan}
</plan>

Respond with the updated function code only, without any other text.
"""

    return cast(
        AgentState[str],
        completion_step(
            state >> dict(role="user", content=user_prompt),
            completion,
            map_result=process_llm_code_output,  # type: ignore
        ),
    )
