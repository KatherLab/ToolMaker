ARG ARCH=cpu
FROM ghcr.io/katherlab/toolmaker:${ARCH}

# Use bash (so we can use the "." command)
SHELL ["/bin/bash", "-c"]

RUN mkdir -p /toolmaker_runtime
COPY install.sh /toolmaker_runtime/install.sh

# TODO: use instead of COPY in the RUN command once BuildKit is supported on your host: --mount=type=bind,source=.env,target=/toolmaker_runtime/.env \
COPY .env /toolmaker_runtime/.env

RUN echo ">>>START INSTALL<<<" && \
    cd /workspace && \
    chmod +x /toolmaker_runtime/install.sh && \
    set -o allexport && \
    . /toolmaker_runtime/.env && \
    set +o allexport && \
    /toolmaker_runtime/install.sh && \
    echo ">>>END INSTALL<<<"
