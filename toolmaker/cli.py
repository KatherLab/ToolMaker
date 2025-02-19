import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from toolmaker.definition import ToolDefinition
from toolmaker.llm import LLM
from toolmaker.run import run_tool
from toolmaker.runtime.client import (
    INSTALLED_DOCKERFILE,
    DockerRuntimeClient,
    HTTPRuntimeClient,
    Mounts,
    build_image,
)
from toolmaker.tasks.install import InstalledRepository, install_repository
from toolmaker.tasks.make_tool import make_tool
from toolmaker.utils.env import substitute_env_vars
from toolmaker.utils.io import chown_dir_using_docker, friendly_name, rmdir
from toolmaker.utils.logging import logger, tlog
from toolmaker.utils.paths import BENCHMARK_DIR, TOOLS_DIR

app = typer.Typer()


@app.command()
def install(
    task: Annotated[
        str, typer.Argument(help="Path to the YAML file containing the task definition")
    ],
    name: Annotated[
        Optional[str],
        typer.Option(
            help="A name for this experiment run (by default uses the name of the task)"
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option(help="The maximum number of steps to take")
    ] = 30,
    build: Annotated[
        bool, typer.Option(help="Whether to build the Docker image from scratch")
    ] = False,
    prefix: Annotated[
        Optional[str],
        typer.Option(help="The prefix to use for the name"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite if a tool with the same name already exists"
        ),
    ] = False,
    include_paper_summary: Annotated[
        bool,
        typer.Option(
            "--paper-summary",
            help="Whether to include the paper summary in the prompt",
        ),
    ] = False,
) -> None:
    """Install a repository."""
    definition = ToolDefinition.from_yaml(task)
    run_name = name or definition.name
    if prefix:
        run_name = f"{prefix}/{run_name}"

    install_folder = TOOLS_DIR / "install" / run_name
    if install_folder.exists():
        if not force:
            raise RuntimeError(
                f"Tool folder {install_folder} already exists. Use --force to overwrite."
            )
        logger.warning(f"Tool folder {install_folder} already exists. Deleting it.")
        rmdir(install_folder)

    install_folder.mkdir(parents=True, exist_ok=True)
    with install_folder.joinpath("repository.yaml").open("w") as f:
        yaml.dump(definition.repo.model_dump(mode="json"), f)
    with install_folder.joinpath("task_definition.yaml").open("w") as f:
        yaml.dump(definition.model_dump(mode="json"), f)

    llm = LLM()
    runtime = DockerRuntimeClient.create(
        f"install-{friendly_name(run_name)}",
        reuse_existing=False,
        build=build,
        env={k: substitute_env_vars(v) for k, v in definition.repo.env.items()},
    )
    with tlog.log_to(install_folder / "logs.jsonl"):
        installed_state = install_repository(
            llm,
            runtime,
            definition,
            max_steps=max_steps,
            include_paper_summary=include_paper_summary,
        )
        tlog("installed_repository_bash", installed_state.bash())
        with install_folder.joinpath("install.sh").open("w") as f:
            f.write(installed_state.bash())
        with install_folder.joinpath("installed_repository.yaml").open("w") as f:
            yaml.dump(installed_state.response.model_dump(mode="json"), f)
        runtime.save_checkpoint(tag=f"installed-{friendly_name(run_name)}")

    runtime.stop()


@app.command("create")
def create_tool(
    task: Annotated[
        str, typer.Argument(help="Path to the YAML file containing the task definition")
    ],
    name: Annotated[
        Optional[str], typer.Option(help="The name of the newly created tool")
    ] = None,
    installed: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the experiment run of the installed repository. If unspecified, will use the name of the repository."
        ),
    ] = None,
    prefix: Annotated[
        Optional[str],
        typer.Option(help="The prefix to use for the tool name and install name"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite if a tool with the same name already exists"
        ),
    ] = False,
    include_paper_summary: Annotated[
        bool,
        typer.Option(
            "--paper-summary",
            help="Whether to include the paper summary in the prompt",
        ),
    ] = False,
) -> None:
    """Create a tool using an installed repository."""
    run_name = name or Path(task).stem
    if prefix:
        run_name = f"{prefix}/{run_name}"

    tool_folder = TOOLS_DIR / "tools" / run_name
    if tool_folder.exists():
        if not force:
            raise RuntimeError(
                f"Tool folder {tool_folder} already exists. Use --force to overwrite."
            )
        logger.warning(f"Tool folder {tool_folder} already exists. Deleting it.")
        rmdir(tool_folder)
    tool_folder.mkdir(parents=True, exist_ok=True)

    # Copy YAML files
    task_file = tool_folder / "task_definition.yaml"
    shutil.copy(task, task_file)

    # Load the task definition
    definition = ToolDefinition.from_yaml(task_file)
    installed_name = installed or run_name

    installed_repository = (
        InstalledRepository.from_yaml(installed_repository_file)
        if (
            installed_repository_file := (
                TOOLS_DIR / "install" / installed_name / "installed_repository.yaml"
            )
        ).exists()
        else None
    )

    # Copy install script
    shutil.copy(
        TOOLS_DIR / "install" / installed_name / "install.sh",
        tool_folder / "install.sh",
    )
    tool_folder.joinpath("install.json").write_text(
        json.dumps(
            {
                "installed": installed_name,
            }
        )
    )

    mounts = Mounts(
        input=tool_folder / "input",
        output=tool_folder / "output",
        input_mapping=definition.example.mount,
    )

    def reset_runtime() -> HTTPRuntimeClient:
        mounts.reset()
        return DockerRuntimeClient.load_checkpoint(
            friendly_name(run_name),
            tag=f"installed-{friendly_name(installed_name)}",
            reuse_existing=False,
            mounts=mounts,
            env={k: substitute_env_vars(v) for k, v in definition.repo.env.items()},
        )

    # Create the tool
    with tlog.log_to(tool_folder / "logs.jsonl"):
        # Create the tool
        llm = LLM()
        state = make_tool(
            definition,
            llm=llm,
            reset_runtime=reset_runtime,
            installed_repository=installed_repository,
            include_paper_summary=include_paper_summary,
        )
        code = state.response.code
        tlog("tool_code", code)

        # Save code
        code_file = tool_folder / "code.py"
        with code_file.open("w") as f:
            f.write(code)
        logger.info(f"Written code to {code_file}")


@app.command()
def run(
    tool: Annotated[str, typer.Argument(help="The name of the tool to run")],
    name: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the test case to run. If unspecified, will run all test cases."
        ),
    ] = None,
    installed: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the experiment run of the installed repository. Leave empty to use the name of the run that was used to create the tool."
        ),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(
            help="Whether to reload the tool definition from the YAML definition"
        ),
    ] = True,
    prefix: Annotated[
        Optional[str],
        typer.Option(help="The prefix to use for the tool name and install name"),
    ] = None,
) -> None:
    """Run a tool."""
    if prefix:
        tool = f"{prefix}/{tool}"
    definition = ToolDefinition.from_yaml(
        TOOLS_DIR / "tools" / tool / "task_definition.yaml"
    )
    if reload:
        definition = ToolDefinition.from_yaml(
            BENCHMARK_DIR / "tasks" / f"{definition.name}.yaml"
        )

    if installed is None:
        installed = json.loads(
            (TOOLS_DIR / "tools" / tool / "install.json").read_text()
        )["installed"]
    elif prefix:
        installed = f"{prefix}/{installed}"

    # Run the invocation(s)
    names = definition.test_cases.keys() if name is None else [name]
    for name in names:
        logger.info(f"Running invocation {name} for tool {tool}")
        invocation = definition.test_cases[name]
        result = run_tool(tool, invocation, installed=installed)

        print(f"Status: {result.status}")
        print(f"Output path: {result.output_path}")
        print(f"Result:\n{result.result!r}")
        print(f"Stdout:\n{result.stdout}")


@app.command(name="import")
def import_tool(
    path: Annotated[str, typer.Argument(help="The path to the tool folder to import")],
    name: Annotated[
        Optional[str], typer.Option(help="The name of the tool to import")
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite if a tool with the same name already exists"
        ),
    ] = False,
) -> None:
    """Import a tool."""
    run_name = name or Path(path).stem
    installed_name = friendly_name(run_name)
    tool_folder = TOOLS_DIR / "tools" / run_name
    if tool_folder.exists():
        if force:
            logger.warning(f"Tool folder {tool_folder} already exists. Deleting it.")
            rmdir(tool_folder)
        else:
            raise ValueError(
                f"Tool folder {tool_folder} already exists (use --force to overwrite)."
            )
    chown_dir_using_docker(path)
    shutil.copytree(path, tool_folder)

    # Construct .env file
    env_file = tool_folder / ".env"
    definition = ToolDefinition.from_yaml(tool_folder / "task_definition.yaml")
    env_file.write_text(
        "\n".join(
            f"{k}={shlex.quote(substitute_env_vars(v))}"
            for k, v in definition.repo.env.items()
        )
    )

    tool_folder.joinpath("install.json").write_text(
        json.dumps(
            {
                "installed": installed_name,
            }
        )
    )

    build_image(
        tag=f"installed-{installed_name}",
        context=tool_folder,
        dockerfile=INSTALLED_DOCKERFILE,
    )


@app.command(name="rmdir")
def rmdir_command(
    dir: Annotated[str, typer.Argument(help="The directory to remove")],
) -> None:
    """Remove a directory."""
    rmdir(dir)


@app.command(name="chown")
def chown_command(
    path: Annotated[
        str,
        typer.Argument(help="The path to the file or directory to change ownership of"),
    ],
    user: Annotated[
        int, typer.Option(help="The user to change ownership to")
    ] = os.getuid(),
    group: Annotated[
        int, typer.Option(help="The group to change ownership to")
    ] = os.getgid(),
) -> None:
    """Change ownership of a file or directory."""
    chown_dir_using_docker(path, user, group)
