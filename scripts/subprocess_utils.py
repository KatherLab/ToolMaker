import os
import subprocess
from collections.abc import Mapping


def run_and_stream_command(
    args,
    *,
    bufsize: int = 16,
    stdin=None,
    stdout=None,
    stderr=None,
    shell=True,
    text=False,
    env: Mapping[str, str] = {},
    start_new_session=True,
    **kwargs,
):
    assert not text, "text=True is not supported"
    assert start_new_session, "start_new_session=False is not supported"

    output = bytearray()
    buffer = bytearray()

    # Start the subprocess
    with subprocess.Popen(
        args,
        **kwargs,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=16,
        text=False,
        shell=shell,
        env={**dict(os.environ), **env},
        start_new_session=True,
    ) as process:  # type: ignore
        try:
            # Stream combined stdout and stderr
            while True:
                chunk = process.stdout.read(32)
                if not chunk:
                    break
                output.extend(chunk)
                buffer.extend(chunk)

                try:
                    # Decode and print valid UTF-8 sequences
                    text = buffer.decode("utf-8")
                    print(text, end="")
                    buffer.clear()  # Clear buffer if decoding was successful
                except UnicodeDecodeError:
                    # Keep partial data in the buffer
                    pass
        except Exception:
            process.kill()
            raise
    # Wait for the process to finish and get the return code
    return_code = process.wait()

    # Decode any remaining data in the buffer
    decoded_output = output.decode("utf-8")
    if (lines := decoded_output.strip().splitlines()) and lines[-1].startswith(
        "EOFError"
    ):
        decoded_output += (
            "\nNOTE: The EOFError may be caused because the script is waiting for user input, "
            "which is not supported. Make sure that you do not run commands that require user input!\n"
        )
    return return_code, decoded_output
