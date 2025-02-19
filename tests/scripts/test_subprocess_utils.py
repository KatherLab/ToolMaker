import pytest
from scripts.subprocess_utils import run_and_stream_command


def test_run_command_success():
    """Test that run_command successfully executes a simple command."""
    return_code, output = run_and_stream_command("echo hello")

    assert return_code == 0
    assert output.strip() == "hello"


def test_run_command_failure():
    """Test that run_command properly handles non-zero return codes."""
    return_code, output = run_and_stream_command("python -c 'exit(1)'")

    assert return_code == 1


def test_stdout_delay():
    return_code, output = run_and_stream_command(
        """python -c "import time; time.sleep(1); print('hello')" """
    )

    assert return_code == 0
    assert output.strip() == "hello"


@pytest.mark.parametrize(
    "command",
    ["python -c 'input()'", "uv run --with huggingface_hub[cli] huggingface-cli login"],
)
def test_run_command_that_reads_from_stdin(command):
    """Test that run_command properly handles commands that read from stdin."""
    return_code, output = run_and_stream_command(command)

    assert return_code != 0
    assert (
        "The EOFError may be caused because the script is waiting for user input"
        in output
    )


def test_run_command_shell():
    return_code, output = run_and_stream_command("echo hello && echo world", shell=True)

    assert return_code == 0
    assert output.strip() == "hello\nworld"
