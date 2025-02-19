import asyncio
import functools
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, cast

import pyte
from loguru import logger


class BashOutput(NamedTuple):
    """The output of a bash command."""

    return_code: int
    output: str


if hasattr(sys.stdout, "buffer"):
    _write_stdout = sys.stdout.buffer.write
else:
    _write_stdout = lambda x: sys.stdout.write(x.decode("utf-8"))  # noqa: E731


async def _record_stream(
    stream: asyncio.StreamReader,
    block_size: int = 64,
    screen_width: int = 2048,
    screen_height: int = 256,
    screen_history: int = 100,
    stream_to_stdout: bool = False,
) -> str:
    if stream_to_stdout:
        print("Streaming to stdout...\n")
    # We will capture the output in a pyte screen buffer (a virtual terminal emulator).
    # This allows us to capture the output and also limit the number of tokens we will need to send back to the LLM (because some commands, e.g. pip, apt, overwrite parts of its ouput).
    screen = pyte.HistoryScreen(screen_width, screen_height, history=screen_history)
    screen.set_mode(pyte.modes.LNM)
    bs = pyte.ByteStream(screen)
    while len(block := await stream.read(block_size)):
        bs.feed(block.replace(b"\n", b"\r\n"))
        if stream_to_stdout:
            _write_stdout(block)

    if screen.history.top or screen.history.bottom:
        logger.warning(
            f"Screen buffer overflowed (more than {screen.lines} lines), some output may be lost; maybe increase SCREEN_HEIGHT"
        )

    output = [x.rstrip() for x in screen.display]
    # Ensure there is a maximum of two newlines in a row
    output = functools.reduce(
        lambda acc, x: acc + [x] if x or acc and acc[-1] else acc, output, []
    )
    return "\n".join(output)


async def run_bash_command(
    command: str,
    cwd: str | os.PathLike | None = None,
    block_size: int = 32,
    screen_width: int = 2048,
    screen_height: int = 256,
    screen_history: int = 100,
    stream_to_stdout: bool = True,
    env: Mapping[str, str] | None = None,
) -> BashOutput:
    """Run a bash command and return the output."""
    logger.info(f"Running command: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        text=False,
        cwd=str(Path(cwd).resolve().absolute()) if cwd else None,
        env=env,
        start_new_session=True,  # this is to make sure an error is thrown if the command attempts to read from stdin
    )
    output = await _record_stream(
        cast(asyncio.StreamReader, process.stdout),
        block_size=block_size,
        screen_width=screen_width,
        screen_height=screen_height,
        screen_history=screen_history,
        stream_to_stdout=stream_to_stdout,
    )
    if (lines := output.strip().splitlines()) and lines[-1].startswith("EOFError"):
        output += (
            "\nNOTE: The EOFError may be caused because the script is waiting for user input, "
            "which is not supported. Make sure that you do not run commands that require user input!\n"
        )
    return_code = await process.wait()
    return BashOutput(output=output, return_code=return_code)
