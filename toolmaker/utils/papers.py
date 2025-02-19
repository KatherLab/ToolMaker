from toolmaker.definition import ToolDefinition


def get_paper_summary_prompt(definition: ToolDefinition) -> str:
    return f"""
For context, here are summaries of the paper(s) underlying the `{definition.name}` repository:
<paper_summaries>
{definition.get_paper_summary()}
</paper_summaries>
"""
