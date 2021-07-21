#!/bin/bash
docker build \
    --pull \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=hoge --build-arg PASSWORD=fuga \
    -t naivete5656/semidetection:latest ./docker
