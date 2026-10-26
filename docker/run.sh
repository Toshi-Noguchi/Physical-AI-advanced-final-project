#!/usr/bin/env bash
# Run a Python script with Isaac Sim's python inside the official container.
#
#   docker/run.sh sim/run_env.py --headless          # offscreen
#   GUI=1 docker/run.sh sim/run_env.py               # X11 window on the host display
#
# The container runs as the host user so files written under the repo stay ours; Kit's caches,
# logs and config go to $ISAAC_SIM_HOME (persisted, so shader compilation is paid only once).
set -euo pipefail

IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.2.0.1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ISAAC_SIM_HOME=${ISAAC_SIM_HOME:-$HOME/.cache/pi-final-isaac}
mkdir -p "$ISAAC_SIM_HOME"/{home,kit_cache,kit_data,kit_logs,hub}
# Give our uid a name inside the container (the asset Hub and OptiX cache refuse to run without one).
if [[ ! -s "$ISAAC_SIM_HOME/passwd" ]]; then
  docker run --rm --entrypoint cat "$IMAGE" /etc/passwd > "$ISAAC_SIM_HOME/passwd"
  echo "$(id -un):x:$(id -u):$(id -g)::/home/isaac:/bin/bash" >> "$ISAAC_SIM_HOME/passwd"
fi

args=(
  --rm --runtime=nvidia --gpus all --network host
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  # /isaac-sim is 0750 isaac-sim:isaac-sim (gid 1234): join that group to read the install.
  --user "$(id -u):$(id -g)" --group-add 1234 -e HOME=/home/isaac -v "$ISAAC_SIM_HOME/home:/home/isaac"
  -v "$ISAAC_SIM_HOME/passwd:/etc/passwd:ro" -e "USER=$(id -un)"
  # Kit runs in portable mode and writes caches (shader DB, derived data) next to itself.
  -v "$ISAAC_SIM_HOME/kit_cache:/isaac-sim/kit/cache" -v "$ISAAC_SIM_HOME/kit_data:/isaac-sim/kit/data"
  -v "$ISAAC_SIM_HOME/kit_logs:/isaac-sim/kit/logs"
  -e HUB__CACHE__PATH=/home/isaac/.hub_cache -v "$ISAAC_SIM_HOME/hub:/home/isaac/.hub_cache"
  -v "$REPO:/workspace" -w /workspace
  --entrypoint /isaac-sim/python.sh
)
[[ -t 0 && -t 1 ]] && args+=(-it)
# Optional CJK font for the "不良" stamp variant.
[[ -d /usr/share/fonts/opentype/noto ]] && args+=(-v /usr/share/fonts/opentype/noto:/usr/share/fonts/opentype/noto:ro)
if [[ "${GUI:-0}" == 1 ]]; then
  args+=(-e "DISPLAY=${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix)
  xauth_file=${XAUTHORITY:-$HOME/.Xauthority}
  [[ -f "$xauth_file" ]] && args+=(-v "$xauth_file:/home/isaac/.Xauthority:ro" -e XAUTHORITY=/home/isaac/.Xauthority)
fi

exec docker run "${args[@]}" "$IMAGE" "$@"
