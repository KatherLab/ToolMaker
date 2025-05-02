from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from toolarena.definition import ToolDefinition

from toolmaker.llm import LLM, LLM_MODEL_SUMMARY
from toolmaker.utils.paths import PAPER_SUMMARIES_DIR, PAPERS_DIR


def summarize_paper_for_task(definition: ToolDefinition, paper: str, llm: LLM) -> str:
    logger.debug(f"Summarizing paper {paper} for task {definition.name}")
    paper_text = PAPERS_DIR.joinpath(f"{paper}.txt").read_text()
    system_prompt = """
    You are a helpful research assistant that summarizes papers.
    You will be given a task description (enclosed in <task_description> tags) and a paper (enclosed in <paper> tags).
    Your job is to summarize the paper in a few paragraphs, focusing on including all relevant information from the paper that is relevant to the task.
    Your summary should explain in detail how the method described in the paper works.
    Specifically, your summary should include useful information needed to perform the task given in the <task_description> tag.
    As the last paragraph of your summary, explain how the paper relates to the task.
    Respond with the summary, and nothing else.
    """.strip()

    user_prompt = f"""
    <task_description>
    {definition.xml_summary}
    </task_description>

    <paper>
    {paper_text}
    </paper>
    """.strip()

    response = llm.completion(
        model=LLM_MODEL_SUMMARY,
        messages=[
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ],
    )

    return f"""
<paper_summary id="{paper}">
{response.content}
</paper_summary>
""".strip()


def summarize_papers_for_task(
    task: Annotated[str, typer.Argument(help="The task to summarize the papers for.")],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="The file to write the summaries to."),
    ] = None,
) -> str:
    llm = LLM()
    definition = ToolDefinition.from_yaml(task)
    summaries = "\n\n".join(
        summarize_paper_for_task(definition, paper, llm) for paper in definition.papers
    )
    output_file = Path(
        output_file
        if output_file
        else PAPER_SUMMARIES_DIR.joinpath(f"{definition.name}.txt")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(summaries)

    logger.info(f"Summaries written to {output_file}")
    logger.info(f"LLM: {llm}")

    return summaries


if __name__ == "__main__":
    typer.run(summarize_papers_for_task)
