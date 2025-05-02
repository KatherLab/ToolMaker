#!/bin/bash

docker build -t ghcr.io/katherlab/toolmaker:cpu -f docker/runtime.Dockerfile --build-arg ARCH=cpu .
docker push ghcr.io/katherlab/toolmaker:cpu

docker build -t ghcr.io/katherlab/toolmaker:cuda -f docker/runtime.Dockerfile --build-arg ARCH=cuda .
docker push ghcr.io/katherlab/toolmaker:cuda

