import functools
from pathlib import Path

import nbconvert
import nbformat


def read_notebook(path: Path) -> str:
    """Removes all cell outputs and metadata from a notebook."""
    exporter = nbconvert.Exporter(
        preprocessors=[
            nbconvert.preprocessors.ClearOutputPreprocessor(),
            nbconvert.preprocessors.ClearMetadataPreprocessor(),
        ]
    )
    node, _ = exporter.from_filename(str(path))
    return nbformat.writes(node)


if __name__ == "__main__":
    import typer

    @typer.run
    @functools.wraps(read_notebook)
    def main(*args, **kwargs):
        print(read_notebook(*args, **kwargs))
