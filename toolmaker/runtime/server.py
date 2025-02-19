"""This is the client that runs inside the Docker container."""

import asyncio
import json
import shlex

from fastapi import Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.requests import Request

from toolmaker.actions import (
    ACTIONS,
    FunctionCallError,
    FunctionCallErrorObservation,
    Observation,
    observation_type_for_action,
)
from toolmaker.runtime.api import API
from toolmaker.runtime.code import FunctionCall, FunctionCallResult
from toolmaker.utils.bash import run_bash_command
from toolmaker.utils.env import get_env_dict_in_container
from toolmaker.utils.paths import (
    LOCAL_TOOLMAKER_FUNCTION_DIR,
    LOCAL_TOOLMAKER_RUNNER_PATH,
    LOCAL_WORKSPACE_DIR,
)

app = API()


# Register actions as endpoints
for name, action_type in ACTIONS.items():
    return_type: type[Observation] = observation_type_for_action(action_type)

    @app.post(f"/execute/{name}")
    async def execute_action(action: action_type) -> return_type:  # type: ignore
        result = action()  # type: ignore
        if asyncio.iscoroutine(result):
            return await result
        else:
            return result


@app.post("/run")
async def run_function(function: FunctionCall) -> FunctionCallResult:
    """Run a python function."""

    # Setup the function directory
    LOCAL_TOOLMAKER_FUNCTION_DIR.mkdir(parents=True, exist_ok=True)
    function_path = LOCAL_TOOLMAKER_FUNCTION_DIR.joinpath("function.py")
    info_path = LOCAL_TOOLMAKER_FUNCTION_DIR.joinpath("info.json")
    output_path = LOCAL_TOOLMAKER_FUNCTION_DIR.joinpath("output.json")
    with info_path.open("w") as f:
        json.dump(
            {
                "path": str(function_path.absolute()),
                **function.model_dump(),
                "output_path": str(output_path.absolute()),
            },
            f,
        )
    output_path.unlink(missing_ok=True)

    # Run the function
    logger.info(f"Running function {function.name}")
    cmd = await run_bash_command(
        shlex.join(
            [
                "python",
                str(LOCAL_TOOLMAKER_RUNNER_PATH.absolute()),
                str(info_path.absolute()),
            ]
        ),
        cwd=LOCAL_WORKSPACE_DIR,
        env=get_env_dict_in_container(),
    )

    # Return the result
    if cmd.return_code == 0:
        return FunctionCallResult(
            status="success",
            result=json.load(output_path.open("r"))["result"],
            stdout=cmd.output,
        )
    else:
        return FunctionCallResult(
            status="error",
            result=f"Process failed with return code {cmd.return_code}",
            stdout=cmd.output,
        )


# Handle exceptions
@app.exception_handler(FunctionCallError)
def function_call_error_handler(request: Request, exc: FunctionCallError) -> Response:
    logger.opt(exception=exc).error("Function call error, returning error observation")
    return JSONResponse(
        status_code=500,
        content=FunctionCallErrorObservation(error=exc.message).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
