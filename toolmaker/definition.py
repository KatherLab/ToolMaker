from pathlib import Path

from toolarena import ToolDefinition
from toolarena.definition import Repository

from toolmaker.utils.paths import LOCAL_WORKSPACE_DIR, TOOLS_DIR


def get_paper_summary(definition: ToolDefinition) -> str:
    summary_file = TOOLS_DIR / "paper_summaries" / f"{definition.name}.txt"
    return summary_file.read_text()


def get_local_install_path(repo: Repository) -> Path:
    """Returns the path to the local installation of the repository (inside the container)."""
    return LOCAL_WORKSPACE_DIR / repo.name_without_owner


__all__ = ["get_paper_summary", "get_local_install_path"]
