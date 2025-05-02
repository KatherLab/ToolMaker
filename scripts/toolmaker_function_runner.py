"""This is a standalone script that can be used to run a ToolMaker function.

Usage:
python toolmaker_function_runner.py <info_path>

Where <info_path> is the path to the function info file. The file should be a JSON file with the following structure:
{
    "code": <code_of_function>,
    "path": <path_to_function_file>,      # This script will write the code to this file
    "name": <name_of_function>,           # The name of the function to call
    "args": <arguments_of_function>,      # The keyword arguments to pass to the function, as a list of {"name": <name_of_argument>, "value": <value_of_argument>}
    "output_path": <path_to_output_file>  # The path to the output file. The function must produce a JSON serializable object.
}
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


def load_symbol_from_file(file_path: os.PathLike, symbol_name: str) -> Any:
    """Load a symbol from a file."""
    spec = importlib.util.spec_from_file_location("module.name", str(file_path))
    module = importlib.util.module_from_spec(spec) if spec else None
    if not spec or not spec.loader or not module:
        raise ValueError(f"Failed to load module from {file_path}")
    spec.loader.exec_module(module)
    func = getattr(module, symbol_name, None)
    if not func:
        raise ValueError(f"Symbol {symbol_name} not found in {file_path}")
    return func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a ToolMaker function.")
    parser.add_argument(
        "info_path", type=Path, help="The path to the function info file."
    )
    args = parser.parse_args()

    with args.info_path.open("r") as f:
        info = json.load(f)
    with open(info["path"], "w") as f:
        f.write(info["code"])
    function = load_symbol_from_file(info["path"], info["name"])
    result = function(**{arg["name"]: arg["value"] for arg in info["args"]})
    with open(info["output_path"], "w") as f:
        json.dump({"result": result}, f)
