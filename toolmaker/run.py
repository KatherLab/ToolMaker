import hashlib
import json
import time
from pathlib import Path

import yaml
from toolarena.definition import ToolDefinition, ToolInvocation

from toolmaker.runtime.client import DockerRuntimeClient, Mounts
from toolmaker.runtime.code import FunctionCall, FunctionCallResult
from toolmaker.utils.env import substitute_env_vars
from toolmaker.utils.io import friendly_name, rmdir
from toolmaker.utils.logging import logger
from toolmaker.utils.paths import TOOLS_DIR


class ToolRunResult(FunctionCallResult):
    output_path: str
    elapsed: float = 0.0

    @property
    def resolved_output_path(self) -> Path:
        return TOOLS_DIR / self.output_path


def _get_run_dir(
    tool: str, invocation: ToolInvocation, installed: str | None = None
) -> Path:
    tool_folder = TOOLS_DIR / "tools" / tool
    if not installed:
        installed = json.load(tool_folder.joinpath("install.json").open())["installed"]
    name = hashlib.sha256(
        f"{tool}:{installed}:{json.dumps(invocation.arguments, sort_keys=True)}:{json.dumps(invocation.mount, sort_keys=True)}".encode()
    ).hexdigest()
    return tool_folder / "run" / name


def is_run_cached(
    tool: str, invocation: ToolInvocation, installed: str | None = None
) -> bool:
    try:
        return (
            _get_run_dir(tool, invocation, installed).joinpath("result.json").exists()
        )
    except FileNotFoundError:
        return False


def run_tool(
    tool: str,
    invocation: ToolInvocation,
    installed: str | None = None,
    prefix: str | None = None,
    must_be_cached: bool = False,
) -> ToolRunResult:
    if prefix:
        tool = f"{prefix}/{tool}"
    tool_folder = TOOLS_DIR / "tools" / tool
    definition = ToolDefinition.from_yaml(tool_folder / "task.yaml")
    if not installed:
        installed = definition.name
        if prefix:
            installed = f"{prefix}/{installed}"

    # Check if the result is already cached
    run_folder = _get_run_dir(tool, invocation, installed)
    result_file = run_folder / "result.json"
    if result_file.exists():
        logger.info(f"Result already cached at {result_file}. Returning cached result.")
        with result_file.open("r") as f:
            return ToolRunResult.model_validate_json(f.read())
    if must_be_cached:
        raise RuntimeError(f"Result is not cached at {result_file}")

    run_folder.mkdir(parents=True, exist_ok=True)
    code = tool_folder.joinpath("code.py").read_text()

    with run_folder.joinpath("invocation.yaml").open("w") as f:
        yaml.dump(invocation.model_dump(), f)

    # Create the tool
    mounts = Mounts(
        input=run_folder / "input",
        output=run_folder / "output",
        input_mapping=invocation.mount,
    )
    mounts.reset()

    runtime = DockerRuntimeClient.load_checkpoint(
        f"run-{friendly_name(tool)}",
        tag=f"installed-{friendly_name(installed)}",
        reuse_existing=False,
        mounts=mounts,
        env={k: substitute_env_vars(v) for k, v in definition.repo.env.items()},
        # docker image must already exist
        build=False,
        allow_build=False,
    )

    t0 = time.time()
    result = runtime.run_function(
        FunctionCall(code=code, name=definition.name, args=invocation.arguments)
    )
    elapsed = time.time() - t0
    logger.info(f"Tool {tool} ran in {elapsed:.2f} seconds")

    call_result = ToolRunResult(
        **result.model_dump(),
        output_path=str(run_folder.joinpath("output").relative_to(TOOLS_DIR)),
        elapsed=elapsed,
    )

    with result_file.open("w") as f:
        f.write(call_result.model_dump_json(indent=2))

    runtime.stop()
    rmdir(run_folder / "input")

    return call_result
