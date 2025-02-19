from asyncio import Lock
from typing import Any, Callable, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request


class StatusResponse(BaseModel):
    status: Literal["ok"] = "ok"


class API(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _setup_api(self)


def _setup_api(app: FastAPI):
    lock = Lock()

    # Ensure that only one request is processed at a time
    @app.middleware("http")
    async def single_request_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        async with lock:
            response = await call_next(request)
        return response

    # Other endpoints
    @app.get("/alive")
    async def alive() -> StatusResponse:
        return StatusResponse()
