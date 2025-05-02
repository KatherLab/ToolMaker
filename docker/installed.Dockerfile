ARG ARCH=cpu
FROM ghcr.io/katherlab/toolmaker:${ARCH}

RUN mkdir -p /toolmaker_runtime
COPY install.sh /toolmaker_runtime/install.sh
COPY .env /toolmaker_runtime/.env

RUN chmod +x /toolmaker_runtime/install.sh && \
    set -o allexport && \
    . /toolmaker_runtime/.env && \
    set +o allexport && \
    /toolmaker_runtime/install.sh
