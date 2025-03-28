"""This is the client that runs inside the Docker container."""

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, Sequence

import docker  # type: ignore
import requests
import tenacity
from docker.models.containers import Container  # type: ignore
from docker.models.images import Image  # type: ignore
from docker.types import DeviceRequest, Mount  # type: ignore
from loguru import logger

from toolmaker.actions import (
    ACTIONS,
    Action,
    FunctionCallErrorObservation,
    Observation,
    observation_type_for_action,
)
from toolmaker.runtime.code import FunctionCall, FunctionCallResult
from toolmaker.utils.io import join_paths, rmdir
from toolmaker.utils.paths import BENCHMARK_DIR, ROOT_DIR

type MountMapping = Mapping[str, str]  # host -> container

TOOLMAKER_RUNTIME_IMAGE = "toolmaker-runtime"
RUNTIME_BUILD_CONTEXT = ROOT_DIR
RUNTIME_DOCKERFILE = RUNTIME_BUILD_CONTEXT / "docker" / "runtime.Dockerfile"
INSTALLED_DOCKERFILE = RUNTIME_BUILD_CONTEXT / "docker" / "installed.Dockerfile"


class RuntimeClient(Protocol):
    def execute[TObs: Observation](
        self, action: Action[TObs]
    ) -> TObs | FunctionCallErrorObservation: ...


@dataclass(frozen=True)
class HTTPRuntimeClient(RuntimeClient):
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def execute[TObs: Observation](
        self, action: Action[TObs]
    ) -> TObs | FunctionCallErrorObservation:
        response = requests.post(
            f"{self.url}/execute/{action.action}", json=action.model_dump()
        )
        observation_type = observation_type_for_action(ACTIONS[action.action])
        if response.status_code == 500:
            observation_type = FunctionCallErrorObservation
        return observation_type.model_validate_json(response.text)

    def run_function(self, function: FunctionCall) -> FunctionCallResult:
        function = function.substitute_env_vars()
        response = requests.post(f"{self.url}/run", json=function.model_dump())
        return FunctionCallResult.model_validate_json(response.text)

    def is_alive(self) -> bool:
        try:
            response = requests.get(f"{self.url}/alive")
            return (
                response.status_code == 200
                and response.json().get("status", None) == "ok"
            )
        except requests.exceptions.ConnectionError:
            return False

    def wait_for_alive(self, timeout: float | None = 10.0) -> None:
        if timeout is None:
            timeout = float("inf")

        if not tenacity.retry(
            retry=tenacity.retry_if_result(False.__eq__),
            stop=tenacity.stop_after_delay(timeout),
            wait=tenacity.wait_fixed(1),
        )(self.is_alive)():
            raise RuntimeError(
                f"Runtime client did not become ready after {timeout} seconds"
            )


@dataclass(frozen=True)
class Mounts:
    input: Path | None = None  # folder to mount as input
    output: Path | None = None  # folder to mount as output
    data_dir: Path | None = BENCHMARK_DIR / "data"  # folder to copy input data from
    input_mapping: MountMapping | None = None  # mapping of input data to mount

    def to_docker(self) -> Sequence[Mount]:
        mounts = []
        if self.input is not None:
            mounts.append(
                Mount(
                    target="/mount/input",
                    source=str(Path(self.input).resolve()),
                    type="bind",
                    read_only=False,  # TODO: make read-only?
                )
            )
        if self.output is not None:
            mounts.append(
                Mount(
                    target="/mount/output",
                    source=str(Path(self.output).resolve()),
                    type="bind",
                    read_only=False,
                )
            )
        return mounts

    def reset(self) -> None:
        """Setup the input and output mounts by copying data."""
        if self.output:
            rmdir(self.output)
            self.output.mkdir(parents=True, exist_ok=True)

        if self.input:
            rmdir(self.input)
            self.input.mkdir(parents=True, exist_ok=True)
            if not self.input_mapping:
                logger.debug("Not creating any input mounts...")
                return
            if not self.data_dir:
                raise ValueError("data_dir is required")

            for src, dst in self.input_mapping.items():
                src_path = join_paths(self.data_dir, src)
                dst_path = join_paths(self.input, dst)
                logger.debug(f"Copying {src_path} to {dst_path}")
                if not src_path.exists():
                    raise FileNotFoundError(f"Input data not found: {src_path}")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)


def get_docker() -> docker.DockerClient:
    return docker.from_env(timeout=480)  # increase timeout to avoid timeout errors


def build_image(
    repository: str = TOOLMAKER_RUNTIME_IMAGE,
    *,
    tag: str,
    context: Path | str = RUNTIME_BUILD_CONTEXT,
    dockerfile: Path | str = RUNTIME_DOCKERFILE,
) -> Image:
    """Build the image from docker/runtime.Dockerfile using BuildKit."""
    logger.debug(f"Building image {repository}:{tag} using Docker BuildKit")
    image, output = get_docker().images.build(
        path=str(context),
        dockerfile=str(dockerfile),
        tag=f"{repository}:{tag}",
        buildargs={"DOCKER_BUILDKIT": "1"},
    )
    for record in output:
        try:
            print(record["stream"], end="")
        except Exception:
            logger.debug(f"Error printing build output: {record}")
            break
    logger.info(f"Built image {repository}:{tag} using Docker BuildKit")
    return image


@dataclass(frozen=True)
class DockerRuntimeClient(HTTPRuntimeClient):
    name: str  # the name of the container
    repository: str  # the docker repository of the image (i.e. name of the image)
    tag: str  # the docker tag of the image

    @classmethod
    def _start_container(
        cls,
        image: str,
        name: str,
        port: int,
        tag: str = "latest",
        mounts: Mounts | None = None,
        gpus: Sequence[str] | None = None,
        env: Mapping[str, str] = {},
    ) -> Container:
        device_requests = []
        if gpus is None:
            gpus = [
                gpu for gpu in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if gpu
            ]
        if gpus:
            device_requests.append(
                DeviceRequest(device_ids=gpus, capabilities=[["gpu"]])
            )
            if "CUDA_VISIBLE_DEVICES" not in env:
                env = dict(env)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpus)))
        logger.debug(f"Starting container with GPUs {gpus} on port {port}")

        # Start a container with the supplied name
        container = get_docker().containers.run(
            image=f"{image}:{tag}",
            name=name,
            detach=True,
            ports={8000: port},
            tty=True,
            mounts=(mounts or Mounts()).to_docker(),
            mem_limit="100g",
            shm_size="10g",
            device_requests=device_requests,
            environment=env,
        )
        logger.info(
            f"Started runtime client {container.name} on port {port} from image {image}:{tag}"
        )
        return container

    @classmethod
    def from_container(cls, container: Container, port: int | None = None) -> Self:
        repository, tag = container.image.tags[0].split(":")
        return cls(
            host="localhost",
            port=port or int(container.ports["8000/tcp"][0]["HostPort"]),
            name=container.name,
            repository=repository,
            tag=tag,
        )

    @classmethod
    def create(
        cls,
        name: str,
        port: int = int(os.getenv("TOOLMAKER_RUNTIME_PORT", "8000")),
        reuse_existing: bool = True,
        build: bool = False,
        image: str = TOOLMAKER_RUNTIME_IMAGE,
        tag: str = "latest",
        timeout: float | None = 10.0,  # max wait time for the runtime to become ready
        mounts: Mounts | None = None,
        gpus: Sequence[str] | None = None,
        env: Mapping[str, str] = {},
        allow_build: bool = True,
    ) -> Self:
        """Create a new runtime client by building the image and starting the container."""
        client = get_docker()

        try:
            container = client.containers.get(name)
            logger.info(f"Found existing container {name}")
            if reuse_existing and mounts is None:
                for tag in container.image.tags:
                    container_image, container_tag = tag.split(":")
                    if container_image == image and container_tag == tag:
                        return cls.from_container(container)
            container.remove(force=True)
            logger.info(f"Removed existing container {name}")
        except docker.errors.NotFound:
            pass

        if build or not client.images.list(f"{image}:{tag}"):
            if not allow_build:
                raise RuntimeError(
                    f"Image {image}:{tag} not found and build is not allowed"
                )
            build_image(image, tag=tag)

        client = cls.from_container(
            cls._start_container(
                image, name, port=port, tag=tag, mounts=mounts, gpus=gpus, env=env
            ),
            port=port,
        )
        client.wait_for_alive(timeout=timeout)
        return client

    def stop(self):
        container = get_docker().containers.get(self.name)
        container.stop()
        container.remove(force=True)

    def save_checkpoint(self, tag: str) -> None:
        logger.info(
            f"Saving checkpoint {self.repository}:{tag} for container {self.name}"
        )
        container = get_docker().containers.get(self.name)
        container.commit(repository=self.repository, tag=tag)

    @classmethod
    def load_checkpoint(
        cls,
        name: str,
        tag: str,
        image: str = TOOLMAKER_RUNTIME_IMAGE,
        reuse_existing: bool = False,
        mounts: Mounts | None = None,
        gpus: Sequence[str] | None = None,
        env: Mapping[str, str] = {},
        build: bool = False,  # force build
        allow_build: bool = True,
    ) -> Self:
        logger.info(f"Loading checkpoint {image}:{tag} as {name}")
        return cls.create(
            name,
            reuse_existing=reuse_existing,
            build=build,
            allow_build=allow_build,
            image=image,
            tag=tag,
            mounts=mounts,
            gpus=gpus,
            env=env,
        )
