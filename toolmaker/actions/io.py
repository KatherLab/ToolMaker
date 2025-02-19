import shlex
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from toolmaker.actions.actions import Action, Observation, register_action
from toolmaker.actions.errors import FunctionCallError
from toolmaker.utils.jupyter import read_notebook
from toolmaker.utils.paths import LOCAL_MOUNT_DIR, LOCAL_WORKSPACE_DIR

_IGNORED_FOLDERS: set[str] = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".vscode",
    ".idea",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
}
ALLOWED_RECURSIVE_LISTING_DIRS: set[str] = {
    str(LOCAL_WORKSPACE_DIR),
    str(LOCAL_MOUNT_DIR),
}


class FileWriteObservation(Observation):
    """Observation for writing content to a file"""

    content: None = None
    filename: str = Field(..., description="The path of the file written to.")


class FileReadObservation(Observation):
    """Observation for reading content from a file"""

    content: str = Field(..., description="The content of the file.")


class ListDirectoryObservation(Observation):
    """Observation for listing the contents of a directory"""

    content: Sequence[str] = Field(..., description="The contents of the directory.")


class FindFilesObservation(Observation):
    """Observation for finding files containing a search term"""

    content: Sequence[str] | Mapping[str, Sequence[str]] = Field(
        ...,
        description="The names of the files containing the search term, or a dict of the file names to the line contents containing the search term.",
    )


@register_action
class WriteFile(Action):
    """Write content to a file given a path."""

    action = "write_file"
    path: str = Field(
        ...,
        description="The path of the file to write to. Use absolute paths. If the file does not exist, it will be created (including parent directories if necessary).",
    )
    content: str = Field(..., description="The content to write to the file.")
    description: str = Field(
        "",
        description="A one-sentence description of the content written to the file.",
    )

    def __call__(self) -> FileWriteObservation:
        path = Path(self.path)

        # prevent the model from overwriting existing files; filename however shall be as specific as possible
        # if path.exists():
        #     _v = 1
        #     while path.exists():
        #         path = path.with_name(f"{path.stem}_v{_v}{path.suffix}")
        #         _v += 1
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", encoding="utf-8") as f:
            f.write(self.content)

        return FileWriteObservation(filename=str(path))

    bash_side_effect = True

    def bash(self) -> str:
        return f"echo {shlex.quote(self.content)} > {shlex.quote(self.path)}"


@register_action
class ReadFile(Action):
    """Read content from a file given a path."""

    action = "read_file"
    path: str = Field(..., description="The path of the file to read from.")

    def __call__(self) -> FileReadObservation:
        path = Path(self.path)
        try:
            if path.suffix == ".ipynb":
                content = read_notebook(path)
            else:
                content = path.read_text(encoding="utf-8")
            return FileReadObservation(content=content)
        except FileNotFoundError:
            raise FunctionCallError(
                f"File not found: {self.path}. Perhaps I should list the contents of the parent directory using the {ListDirectory.action!r} action."
            )
        except IsADirectoryError:
            raise FunctionCallError(
                f"File is a directory: {self.path}. Perhaps I should list the contents of the directory instead using the {ListDirectory.action!r} action."
            )
        except UnicodeDecodeError:
            raise FunctionCallError(
                f"File is not a text file and thus cannot be read: {self.path}."
            )

    bash_side_effect = False

    def bash(self) -> str:
        return f"cat {shlex.quote(self.path)}"


def _list_directory(path: Path, recursive: bool = False) -> Iterator[str]:
    # List files first, then directories
    files: list[Path] = []
    dirs: list[Path] = []
    for file in path.iterdir():
        (dirs if file.is_dir() else files).append(file)
    files = sorted(files)
    dirs = sorted(dirs)

    # Yield files first, then directories
    yield from (str(f.absolute()) for f in files)
    yield from (str(d.absolute()) + "/" for d in dirs)
    if recursive:
        for d in dirs:
            if d.name not in _IGNORED_FOLDERS:
                yield from _list_directory(d, recursive=True)


@register_action
class ListDirectory(Action):
    """Get the contents of the directory at the given path."""

    action = "list_directory"
    path: str = Field(..., description="The path of the directory to list.")
    recursive: bool = Field(
        False,
        description="Whether to list the contents recursively. Only recurse if absolutely necessary to avoid listing too many files.",
    )

    def __call__(self) -> ListDirectoryObservation:
        if self.recursive and (
            not any(
                str(Path(self.path).resolve()).startswith(p)
                for p in ALLOWED_RECURSIVE_LISTING_DIRS
            )
        ):
            raise FunctionCallError(
                f"Recursive listing of directories is only allowed in certain directories: {ALLOWED_RECURSIVE_LISTING_DIRS}."
            )
        try:
            files = _list_directory(
                Path(self.path).resolve(),
                recursive=self.recursive,
            )
            return ListDirectoryObservation(content=list(files))
        except FileNotFoundError:
            raise FunctionCallError(f"Directory not found: {self.path}")
        except NotADirectoryError:
            raise FunctionCallError(
                f"Not a directory: {self.path}. Perhaps I should read the file instead using the {ReadFile.action!r} action."
            )
        except PermissionError as e:
            raise FunctionCallError(f"Permission denied: {e!s}.") from e

    bash_side_effect = False

    def bash(self) -> str:
        return f"ls {'-R' if self.recursive else ''} {shlex.quote(self.path)}"


# @register_action  # FIXME: removed this for now because it could generate overly long messages
class FindFiles(Action):
    """Search the given directory recursively for files whose content contains the given string.

    Note: the search_term is matched against the file content, not the filename.
    """

    action = "find_files"
    path: str = Field(..., description="The directory to search.")
    search_term: str = Field(
        ..., description="The string to search for in the file content."
    )
    show_context: bool = Field(
        False,
        description="Whether to show the content of the line containing the search term in the results (use `False` to save tokens).",
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.path == "":
            self.path = "."
        return self

    def _get_command(self) -> str:
        return f"find {shlex.quote(str(self.path))} -type f ! -path '*/.*' -exec grep -IH -- {shlex.quote(self.search_term)} {{}} +"

    def __call__(self) -> FindFilesObservation:
        import subprocess

        if not Path(self.path).exists():
            raise FunctionCallError(f"Directory not found: {self.path}")

        cmd = self._get_command()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and result.stdout != "":
            raise FunctionCallError(f"Error executing command: {cmd}: {result}")
        results = defaultdict(list)  # map of file path to content
        for line in result.stdout.splitlines():
            if ":" in line:
                filename, context = line.split(":", maxsplit=1)
                if filename.startswith("./"):
                    filename = filename[2:]
                results[filename].append(context)
        return FindFilesObservation(
            content=results if self.show_context else list(results.keys())
        )

    bash_side_effect = False

    def bash(self) -> str:
        return self._get_command()
