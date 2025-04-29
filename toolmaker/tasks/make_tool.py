from collections.abc import Callable, MutableSequence
from typing import cast

from toolarena.definition import ToolDefinition

from toolmaker.agent import AgentState, completion_step
from toolmaker.llm import LLM, LLM_MODEL, LLM_MODEL_REASONING
from toolmaker.runtime.client import HTTPRuntimeClient
from toolmaker.runtime.code import FunctionCall
from toolmaker.tasks.assess import is_successful_execution
from toolmaker.tasks.diagnose import diagnose
from toolmaker.tasks.implement_function import implement_function
from toolmaker.tasks.install import InstalledRepository
from toolmaker.tasks.make_plan import make_plan
from toolmaker.tasks.rewrite_function import rewrite_function
from toolmaker.utils.logging import log_and_reraise, tlog


@tlog.state_fn
def summarize_problem(state: AgentState, /, llm: LLM) -> AgentState[str]:
    user_prompt = """
Provide a one-paragraph summary of the most recent problem that occured, your diagnosis of it, and how you attempted to fix it with this code change. 
Be specific. 
Include any file paths and other details that are relevant to the problem/solution. 
Your summary should contain all information needed to implement the fix, and include the key insights/observations made for diagnosing the problem.
Begin your response with "The problem was..."
"""
    return cast(
        AgentState[str],
        completion_step(
            state >> dict(role="user", content=user_prompt), llm[LLM_MODEL]
        ),
    )


@tlog.state_fn
@log_and_reraise
def make_tool(
    definition: ToolDefinition,
    *,
    llm: LLM,
    reset_runtime: Callable[[], HTTPRuntimeClient],
    installed_repository: InstalledRepository | None = None,
    max_iterations: int = 30,
    include_paper_summary: bool = False,
) -> AgentState[FunctionCall]:
    # Load the checkpoint
    runtime = reset_runtime()

    # Make initial plan
    state: AgentState = make_plan(
        llm,
        runtime,
        definition,
        installed_repository=installed_repository,
        include_paper_summary=include_paper_summary,
    )
    plan: str = state.response

    # Initial implementation
    state = implement_function(
        state, definition, plan=plan, completion=llm[LLM_MODEL_REASONING]
    )
    code: str = state.response

    # stores summaries of the previous attempts
    problem_summaries: MutableSequence[str] = []

    # Keep reference to the state to reset it later
    state_checkpoint = state

    try:
        for iteration in range(max_iterations):
            with tlog.context("iteration", iteration=iteration):
                # Reset conversation state
                state = state_checkpoint
                # Reset the runtime to checkpoint
                runtime = reset_runtime()

                state >>= dict(
                    role="user",
                    content="I reset the environment to the freshly installed repository, and will now execute the updated function you wrote.",
                )
                # Execute the function
                function = FunctionCall(
                    name=definition.name, code=code, args=definition.example.arguments
                )
                output = runtime.run_function(function)
                tlog("function_execution_result", output)

                assessment = is_successful_execution(
                    state, output, definition, llm[LLM_MODEL]
                ).response
                if assessment.successful:
                    return state.with_response(function)

                # Gather information and formulate a plan to fix the function
                state = diagnose(
                    state,
                    output=output,
                    code=code,
                    problem_summaries=problem_summaries,
                    completion=llm[LLM_MODEL],
                    runtime=runtime,
                    assessment=assessment,
                )

                # Write the updated function
                state = rewrite_function(
                    state,
                    definition,
                    code_draft=code,
                    diagnosis=state.response,
                    completion=llm[LLM_MODEL],  # TODO: use reasoning model?
                )
                code = state.response

                # Summarize the problem
                summary = summarize_problem(state, llm).response
                problem_summaries.append(summary)

        raise RuntimeError("Max iterations reached")
    finally:
        runtime.stop()
        runtime.stop()
        runtime.stop()
