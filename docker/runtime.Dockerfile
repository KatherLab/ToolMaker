# This image will be built as "toolmaker-runtime:latest"

FROM python:3.12

ENV HOST=0.0.0.0
ENV PORT=8000

RUN mkdir -p /toolmaker
COPY pyproject.toml /toolmaker/pyproject.toml
COPY uv.lock /toolmaker/uv.lock
COPY toolmaker /toolmaker/toolmaker
COPY scripts/toolmaker_function_runner.py /toolmaker/toolmaker_function_runner.py
COPY scripts/subprocess_utils.py /toolmaker/subprocess_utils.py

RUN python -m pip install uv && \
    cd /toolmaker && \
    uv venv && \
    uv sync

RUN mkdir -p /workspace
WORKDIR /workspace

VOLUME /mount/input
VOLUME /mount/output

CMD /toolmaker/.venv/bin/uvicorn --app-dir /toolmaker --host ${HOST} --port ${PORT} toolmaker.runtime.server:app
