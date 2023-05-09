#!/usr/bin/env bash

ARGS=("$@")

USER_NAME="user"
REPO_NAME="Deep_Learning"
CONTAINER_NAME="nycu-deep-learning-2023-spring"
REPOSITORY="sunfuchou/nycu-deep-learning-2023-spring"
TAG="cuda11.7-ubuntu22.04-update"
IMG="${REPOSITORY}:${TAG}"

CONTAINER_ID=$(docker ps -aqf "ancestor=${IMG}")
if [ $CONTAINER_ID ]; then
  echo "Attach to docker container $CONTAINER_ID"
  xhost +
  docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it ${CONTAINER_ID} bash
  xhost -
  return
fi

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
  echo "[$XAUTH] was not properly created. Exiting..."
  exit 1
fi

DOCKER_OPTS=

# Get the current version of docker-ce
# Strip leading stuff before the version number so it can be compared
DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
if dpkg --compare-versions 19.03 gt "$DOCKER_VER"; then
  echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
  if ! dpkg --list | grep nvidia-docker2; then
    echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
    exit 1
  fi
  DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
else
  echo "nvidia container toolkit"
  DOCKER_OPTS="$DOCKER_OPTS --gpus all"
fi

docker run \
    -it \
    --rm \
    -e DISPLAY \
    -e XAUTHORITY=$XAUTH \
    -v "$XAUTH:$XAUTH" \
    -v "/home/${USER}/${REPO_NAME}:/home/${USER_NAME}/${REPO_NAME}" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    --user "${USER_NAME}:root" \
    --workdir "/home/${USER_NAME}/${REPO_NAME}" \
    --name "${CONTAINER_NAME}" \
    --network host \
    --privileged \
    --security-opt seccomp=unconfined \
    $DOCKER_OPTS \
    "${IMG}" \
    bash

# -e "TERM=xterm-256color" \
