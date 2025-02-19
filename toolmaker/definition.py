import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field

from toolmaker.utils.paths import LOCAL_WORKSPACE_DIR, TOOLS_DIR

type ArgumentType = Literal["str", "int", "float", "bool", "list", "dict"]
type ArgumentValue = str | int | float | bool | list | dict | None


class ToolArgument(BaseModel):
    description: str
    type: ArgumentType


class ToolInvocation(BaseModel):
    arguments: Mapping[str, ArgumentValue]
    mount: Mapping[str, str] = Field(default_factory=dict)


class Repository(BaseModel):
    name: str
    url: str
    env: Mapping[str, str] = Field(default_factory=dict)
    commit: str | None = None
    branch: str | None = None

    @property
    def name_without_owner(self) -> str:
        return self.name.split("/")[-1]

    def info(self) -> str:
        s = f"{self.name} from {self.url}"
        if self.commit:
            s += f" (at commit: {self.commit})"
        if self.branch:
            s += f" (at branch: {self.branch})"
        return s


# class Paper(BaseModel):
#     name: str
#     url: str

#     @property
#     def text(self) -> str:
#         paper_path = BENCHMARK_DIR / "papers" / f"{self.name}.txt"
#         if not paper_path.is_file():
#             raise FileNotFoundError(f"Paper {self.name} not found at {paper_path}")
#         with open(paper_path, "r") as f:
#             return f.read()


class ToolDefinition(BaseModel):
    name: str
    category: str | None = None  # e.g. "pathology", "radiology", "genomics_proteomics"
    repo: Repository
    description: str
    arguments: Mapping[str, ToolArgument]
    returns: Mapping[str, ToolArgument] = Field(default_factory=dict)
    example: ToolInvocation
    test_cases: Mapping[str, ToolInvocation] = Field(default_factory=dict)
    papers: Sequence[str]  # paths to `*.txt` files in `papers/`
    note: str | None = (
        None  # additional information about this task (will not be shown to the model)
    )

    def _arg_str(self) -> str:
        return ", ".join(
            f"{name}: {arg.type} = {self.example.arguments[name]!r}"
            for name, arg in self.arguments.items()
        )

    def description_of_returns(self) -> str:
        if self.returns:
            return f"""dict with the following structure:
{{
{"\n".join(f"  {key!r}: {arg.type}  # {arg.description}" for key, arg in self.returns.items())}
}}"""
        return "empty dict"

    @property
    def python_signature(self) -> str:
        indent = " " * 4
        return f"""def {self.name}({self._arg_str()}) -> dict:
{indent}\"\"\"
{indent}{self.description.replace("\n", f"\n{indent}")}
{indent}
{indent}Args:
{"\n".join(f"{indent}    {name}: {arg.description}" for name, arg in self.arguments.items())}
{indent}
{indent}Returns:
{indent}    {self.description_of_returns().replace("\n", f"\n{indent}    ")}
{indent}\"\"\"
"""

    def __str__(self) -> str:
        return self.python_signature

    @classmethod
    def from_yaml(cls, path: os.PathLike | str) -> Self:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @property
    def xml_summary(self) -> str:
        return f"""<description>
{self.description}
</description>
<arguments>
{"\n".join(f"{name} ({arg.type}): {arg.description} (example: {self.example.arguments[name]!r})" for name, arg in self.arguments.items())}
</arguments>
<returns>
{self.description_of_returns()}
</returns>"""

    def get_paper_summary(self) -> str:
        summary_file = TOOLS_DIR / "paper_summaries" / f"{self.name}.txt"
        return summary_file.read_text()


def get_local_install_path(repo: Repository) -> Path:
    """Returns the path to the local installation of the repository (inside the container)."""
    return LOCAL_WORKSPACE_DIR / repo.name_without_owner
