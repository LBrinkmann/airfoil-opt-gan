#!/usr/bin/env bash
DOCKER_ACC=levinb
DOCKER_REPO=mctest
IMG_TAG=tf3

docker build -t $DOCKER_ACC/$DOCKER_REPO:$IMG_TAG -f "./Dockerfile.tensorflow" "."
docker push $DOCKER_ACC/$DOCKER_REPO:$IMG_TAG
