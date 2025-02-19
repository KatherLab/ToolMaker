import pytest
from toolmaker.utils.bash import run_bash_command


async def test_run_command_success():
    """Test that run_command successfully executes a simple command."""
    result = await run_bash_command("echo hello")

    assert result.return_code == 0
    assert result.output.strip() == "hello"


async def test_run_command_failure():
    """Test that run_command properly handles non-zero return codes."""
    result = await run_bash_command("python -c 'exit(1)'")

    assert result.return_code == 1


async def test_stdout_delay():
    result = await run_bash_command(
        """python -c "import time; time.sleep(1); print('hello')" """
    )

    assert result.return_code == 0
    assert result.output.strip() == "hello"


@pytest.mark.parametrize(
    "command",
    ["python -c 'input()'", "uv run --with huggingface_hub[cli] huggingface-cli login"],
)
async def test_run_command_that_reads_from_stdin(command):
    """Test that run_command properly handles commands that read from stdin."""
    result = await run_bash_command(command)

    assert result.return_code != 0
    assert (
        "The EOFError may be caused because the script is waiting for user input"
        in result.output
    )


async def test_run_command_shell():
    result = await run_bash_command("echo hello && echo world")

    assert result.return_code == 0
    assert result.output.strip() == "hello\nworld"


async def test_run_command_with_explicit_env_var():
    result = await run_bash_command("echo $MY_VAR", env={"MY_VAR": "123"})

    assert result.return_code == 0
    assert result.output.strip() == "123"
