from __future__ import annotations

import functools
import inspect
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from pytest_lazy_fixtures import lf
from toolmaker.definition import ToolDefinition, ToolInvocation
from toolmaker.run import ToolRunResult, is_run_cached, run_tool
from toolmaker.utils.paths import TASKS_DIR

type ToolFixture = Callable[[ToolInvocation], ToolRunResult]
type InvocationFixture = Callable[[ToolFixture], ToolRunResult]

TEST_TOOL_PATTERN = re.compile(r"^test_(?P<tool>[a-zA-Z0-9_]+).py$")
TESTS_DATA_DIR = Path(__file__).parent / "data"

PREFIXES = (None, "toolmaker", "openhands")


def _tool_fixture(
    name: str, module: str, prefix: str | None = None
) -> Callable[[], ToolFixture]:
    def fixture() -> ToolFixture:
        return functools.partial(run_tool, name, prefix=prefix, must_be_cached=True)

    fixture.__name__ = name
    fixture.__doc__ = f"Tool fixture for {name}."
    fixture.__module__ = module

    return pytest.fixture(
        fixture,
        scope="module",
        # params=[pytest.param(None, marks=pytest.mark.tool(name=name))],
        # ids=[""],
    )


def _invocation_fixture(
    tool_name: str,
    invocation_name: str,
    invocation: ToolInvocation,
    module: str,
    prefix: str | None = None,
) -> InvocationFixture:
    def fixture(tool: ToolFixture) -> ToolRunResult:
        return tool(invocation)

    full_tool_name = f"{prefix}/{tool_name}" if prefix else tool_name

    fixture.__name__ = invocation_name
    fixture.__doc__ = f"Test case {invocation_name} for tool {tool_name}."
    fixture.__module__ = module

    marks = [
        pytest.mark.tool_invocation(
            prefix=prefix, tool=tool_name, invocation=invocation_name
        )
    ]
    if is_run_cached(full_tool_name, invocation):
        marks.append(pytest.mark.cached)

    return pytest.fixture(
        fixture,
        scope="module",
        params=[pytest.param(None, marks=marks)],
        ids=[full_tool_name],
    )


def parametrize_invocation(
    *invocations: str | InvocationFixture,
) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        "invocation",
        [lf(getattr(invocation, "__name__", invocation)) for invocation in invocations],
    )


def get_fixtures(tool_name: str) -> Mapping[str, ToolFixture | InvocationFixture]:
    prefix = os.getenv("TOOLMAKER_BENCHMARK_PREFIX", None)
    module = f"tests.test_{tool_name}"
    fixtures = {}
    fixtures["tool"] = _tool_fixture(tool_name, module=module, prefix=prefix)
    definition = ToolDefinition.from_yaml(TASKS_DIR / f"{tool_name}.yaml")
    for invocation_name, invocation in definition.test_cases.items():
        fixtures[invocation_name] = _invocation_fixture(
            tool_name, invocation_name, invocation, module=module, prefix=prefix
        )
    return fixtures


def initialize() -> None:
    """Initialize the fixtures for the current test module.

    This function should be called from each test module, where the name of the test file is `test_<tool_name>.py`.
    It will populate the global namespace with the fixtures for the tool.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        caller_filename = os.path.basename(caller_frame.f_code.co_filename)
        tool_name = TEST_TOOL_PATTERN.match(caller_filename).group("tool")
        caller_frame.f_globals.update(get_fixtures(tool_name))
    finally:
        del frame  # Break reference cycle
