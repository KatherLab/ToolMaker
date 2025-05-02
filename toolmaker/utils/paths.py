import os
from pathlib import Path

# Paths on the host machine
ROOT_DIR = Path(__file__).parent.parent.parent
BENCHMARK_DIR = Path(os.getenv("BENCHMARK_DIR", ROOT_DIR / "benchmark")).resolve()
TASKS_DIR = BENCHMARK_DIR / "tasks"
TOOLS_DIR = Path(os.getenv("TOOLS_DIR", ROOT_DIR / "tool_output")).resolve()
PAPERS_DIR = BENCHMARK_DIR / "papers"
PAPER_SUMMARIES_DIR = TOOLS_DIR / "paper_summaries"
TOOL_DOCKERFILE = ROOT_DIR / "docker" / "tool.Dockerfile"


# Paths in the container
LOCAL_WORKSPACE_DIR = Path("/workspace")
LOCAL_MOUNT_DIR = Path("/mount")
LOCAL_TOOLMAKER_DIR = Path(os.environ.get("TOOLMAKER_DIR", "/toolmaker"))
LOCAL_TOOLMAKER_RUNNER_PATH = LOCAL_TOOLMAKER_DIR.joinpath(
    "toolmaker_function_runner.py"
)
LOCAL_TOOLMAKER_FUNCTION_DIR = LOCAL_TOOLMAKER_DIR.joinpath("toolmaker_function")
