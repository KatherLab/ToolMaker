# This image will be built as "toolmaker-runtime:latest"

ARG ARCH=cpu
ARG BASE=ghcr.io/katherlab/toolarena:${ARCH}
FROM ${BASE}

WORKDIR /toolmaker

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY scripts/toolmaker_function_runner.py scripts/subprocess_utils.py ./
COPY toolmaker ./toolmaker

WORKDIR /workspace

VOLUME /mount/input
VOLUME /mount/output

CMD /toolmaker/.venv/bin/uvicorn --app-dir /toolmaker --host ${HOST} --port ${PORT} toolmaker.runtime.server:app
