"""This is the client that runs inside the Docker container."""

from collections.abc import Mapping
from typing import Protocol, Self, Sequence

from loguru import logger
from toolarena.runtime import DockerRuntimeClient as ToolArenaDockerRuntimeClient
from toolarena.runtime import Mounts, get_docker
from typing_extensions import deprecated

from toolmaker.actions import (
    ACTIONS,
    Action,
    FunctionCallErrorObservation,
    Observation,
    observation_type_for_action,
)
from toolmaker.runtime.code import FunctionCall, FunctionCallResult

type MountMapping = Mapping[str, str]  # host -> container

TOOL_CHECKPOINT_IMAGE_NAME = "toolmaker-tool"


class RuntimeClient(Protocol):
    def execute[TObs: Observation](
        self, action: Action[TObs]
    ) -> TObs | FunctionCallErrorObservation: ...


class DockerRuntimeClient(ToolArenaDockerRuntimeClient, RuntimeClient):
    def execute[TObs: Observation](
        self, action: Action[TObs]
    ) -> TObs | FunctionCallErrorObservation:
        response = self.http_client.post(
            f"{self.url}/execute/{action.action}", json=action.model_dump()
        )
        observation_type = observation_type_for_action(ACTIONS[action.action])
        if response.status_code == 500:
            observation_type = FunctionCallErrorObservation
        return observation_type.model_validate_json(response.text)

    def run_function(self, function: FunctionCall) -> FunctionCallResult:
        function = function.substitute_env_vars()
        response = self.http_client.post(f"{self.url}/run", json=function.model_dump())
        return FunctionCallResult.model_validate_json(response.text)

    @deprecated("Use `toolarena.runtime.build_image` instead")
    def save_checkpoint(
        self, tag: str, image: str = TOOL_CHECKPOINT_IMAGE_NAME
    ) -> None:
        logger.info(f"Saving checkpoint {image}:{tag} for container {self.name}")
        container = get_docker().containers.get(self.name)
        container.commit(repository=image, tag=tag)

    @classmethod
    def load_checkpoint(
        cls,
        name: str,
        tag: str,
        image: str = TOOL_CHECKPOINT_IMAGE_NAME,
        mounts: Mounts | None = None,
        gpus: Sequence[str] | None = None,
        env: Mapping[str, str] = {},
    ) -> Self:
        logger.info(f"Loading checkpoint {image}:{tag} as {name}")
        return cls.create(
            name,
            image=f"{image}:{tag}",
            mounts=mounts,
            gpus=gpus,
            env=env,
        )
