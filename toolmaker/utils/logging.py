from __future__ import annotations

import datetime
import functools
import json
import traceback
import traceback as tb
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Self, overload

from loguru import logger
from openai._types import NOT_GIVEN, NotGiven
from pydantic import BaseModel

type JsonCompatible = Mapping | Sequence | str | None
type Loggable = BaseModel | JsonCompatible


def loggable_to_json_compatible(x: Loggable) -> JsonCompatible:
    return x.model_dump() if isinstance(x, BaseModel) else x


class WithResponse[T](Protocol):
    response: T


@dataclass
class ToolLogger:
    log_files: list[Path] = field(default_factory=list)

    def log(
        self,
        name: str,
        /,
        content: Loggable | NotGiven = NOT_GIVEN,
        *,
        type: str = "event",
        metadata: Mapping[str, Loggable] = {},
    ):
        logger.debug(
            f"{type}:{name}: {repr(content) if not isinstance(content, NotGiven) else ''}"
        )
        message = (
            json.dumps(
                dict(
                    time=datetime.datetime.now().strftime("%H:%M:%S"),
                    type=type,
                    name=name,
                    **(
                        dict(content=loggable_to_json_compatible(content))
                        if not isinstance(content, NotGiven)
                        else {}
                    ),
                    metadata=metadata,
                )
            )
            + "\n"
        )
        for log_file in self.log_files:
            with open(log_file, "a") as f:
                f.write(message)

    __call__ = log

    @overload
    def fn[**P, R: Loggable](
        elf,
        func: Callable[P, R],
        /,
        *,
        log_result: Literal[True] = True,
        map_result: Callable[[R], Loggable] = lambda x: x,
        name: str | None = None,
    ) -> Callable[P, R]: ...

    @overload
    def fn[**P, R](
        elf,
        func: Callable[P, R],
        /,
        *,
        log_result: Literal[True] = True,
        map_result: Callable[[R], Loggable],
        name: str | None = None,
    ) -> Callable[P, R]: ...

    @overload
    def fn[**P, R](
        self,
        func: Callable[P, R],
        /,
        *,
        log_result: Literal[False],
        name: str | None = None,
    ) -> Callable[P, R]: ...

    def fn[**P, R](
        self,
        func: Callable[P, R],
        /,
        *,
        log_result: bool = True,
        map_result: Callable[[R], Loggable] = lambda x: x,  # type: ignore
        name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap a function in a context manager that logs the function call and result."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with self.context(name or func.__name__) as ctx:
                result = func(*args, **kwargs)
                if log_result:
                    ctx.content = map_result(result)
                return result

        return wrapper

    def state_fn[**P, R: WithResponse](
        self, func: Callable[P, R], /, *, name: str | None = None
    ) -> Callable[P, R]:
        """Special case of `fn` for functions that return an AgentState."""
        return self.fn(
            func, name=name, map_result=lambda x: x.response, log_result=True
        )

    def context(
        self, name: str, /, content: Loggable | NotGiven = NOT_GIVEN, **kwargs
    ) -> Context:
        """Context manager that logs the start and end of an event."""
        return self.Context(self, name, content=content, **kwargs)

    class Context:
        def __init__(
            self,
            logger: ToolLogger,
            name: str,
            content: Loggable | NotGiven = NOT_GIVEN,
            **metadata: Loggable,
        ):
            self.logger = logger
            self.name = name
            self.enter_content = content
            self.enter_metadata = metadata
            self.content: Loggable | NotGiven = NOT_GIVEN
            self.metadata: dict[str, Loggable] = {}

        def __enter__(self) -> Self:
            self.logger.log(
                self.name,
                type="start",
                content=self.enter_content,
                metadata=self.enter_metadata,
            )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.logger.log(
                self.name,
                type="end",
                content=self.content,
                metadata=self.metadata,
            )

        def update(self, **kwargs):
            self.metadata.update(kwargs)

    def log_to(self, file: Path) -> LogFileContext:
        return self.LogFileContext(self, file)

    class LogFileContext:
        def __init__(self, logger: ToolLogger, file: Path):
            self.logger = logger
            self.file = file

        def __enter__(self) -> Self:
            self.logger.log_files.append(self.file)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                # Log the exception
                self.logger.log(
                    "exception",
                    content=tb.format_exception(exc_type, exc_value, traceback),
                )
            self.logger.log_files.remove(self.file)


def log_and_reraise[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception:
            tlog("error", traceback.format_exc())
            raise

    return wrapper


tlog = ToolLogger()
