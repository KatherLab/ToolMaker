import os
import shutil
from pathlib import Path

from loguru import logger


def friendly_name(name: str) -> str:
    return name.replace("/", "__")


def join_paths(parent: os.PathLike | str, *children: os.PathLike | str) -> Path:
    """Join paths and ensure that the resulting path is relative to the parent."""
    path = Path(parent)
    for child in children:
        path = path / Path(child)
    if not path.resolve().is_relative_to(Path(parent).resolve()):
        raise ValueError(f"Path {path} is not relative to {parent}")

    return path


def chown_dir_using_docker(
    dir: os.PathLike | str, uid: int = os.getuid(), gid: int = os.getgid()
) -> None:
    import docker  # type: ignore

    client = docker.from_env()
    logger.debug(f"Chowning directory {dir} to {uid}:{gid} using docker")
    client.containers.run(
        "busybox",
        ["chown", "-R", f"{uid}:{gid}", "/mnt/mydir"],
        volumes={str(Path(dir).resolve()): {"bind": "/mnt/mydir", "mode": "rw"}},
        auto_remove=True,
    )


def rmdir(dir: os.PathLike | str) -> None:
    try:
        if Path(dir).exists():
            shutil.rmtree(dir)
    except PermissionError:
        logger.warning(
            f"Failed to remove directory {dir}. "
            "This may happen because docker wrote files with permissions that are not writable by the current user. "
            "We will try to chown the directory to the current user."
        )
        chown_dir_using_docker(dir)
        shutil.rmtree(dir)
    logger.debug(f"Removed directory {dir}.")
