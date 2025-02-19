FROM toolmaker-runtime:latest

RUN mkdir -p /toolmaker_runtime
COPY install.sh /toolmaker_runtime/install.sh
COPY .env /toolmaker_runtime/.env

RUN chmod +x /toolmaker_runtime/install.sh && \
    export $(grep -v '^#' /toolmaker_runtime/.env | xargs) && \
    /toolmaker_runtime/install.sh
