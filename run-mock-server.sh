#!/usr/bin/env bash
#
# Run the tracking server against a simulated table -- no cameras, no rig, no projector.
#
# The counterpart to run-server.ps1 for frontend development away from the tangible table. It runs
# the real server.py; only the camera detection thread is replaced (see mock_server.py). The
# websocket speaks the same protocol on the same port, so TOSCA-2 needs no changes to talk to it.
#
# An interactive prompt lets you move, turn, add and remove the physical blocks while the frontend
# watches. Type 'help' at that prompt for the command list.
#
#   ./run-mock-server.sh                  blocks stay put until you move them from the prompt
#   ./run-mock-server.sh --motion drift   blocks wander and spin on their own
#   ./run-mock-server.sh --reset          throw away mock_state/ first
#
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MOTION="jitter"
PORT="8053"
EXTRA_ARGS=()

usage() {
    cat <<'USAGE'
Run the tracking server against a simulated table -- no cameras, no rig, no projector.

It runs the real server.py; only the camera detection thread is replaced. The websocket speaks
the same protocol on the same port, so TOSCA-2 needs no changes to talk to it. An interactive
prompt lets you move the blocks while the frontend watches; type 'help' there for its commands.

  ./run-mock-server.sh                  blocks stay put until you move them from the prompt
  ./run-mock-server.sh --motion drift   blocks wander and spin on their own
  ./run-mock-server.sh --reset          throw away mock_state/ first

options
  --motion still|jitter|drift   still: frozen and exactly repeatable.
                                jitter: detection noise only (default).
                                drift: blocks move on their own.
  --port <n>                    websocket port (default 8053, the same as the rig)
  --reset                       start from a fresh copy of the real catalogs
  --no-sandbox                  write registrations to the REAL catalogs. Rarely what you want.
  --verbose                     echo every snapshot the server sends. Very noisy.
  --no-cli                      run headless, for scripts and CI
  -h, --help                    this text
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --motion)
            MOTION="${2:?--motion needs still, jitter or drift}"
            shift 2
            ;;
        --port)
            PORT="${2:?--port needs a number}"
            shift 2
            ;;
        --reset | --no-sandbox | --verbose | --no-cli)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MOTION" in
    still | jitter | drift) ;;
    *)
        echo "--motion must be still, jitter or drift (got '$MOTION')" >&2
        exit 2
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# macOS has its own pinned set -- the RealSense wheels the default file pulls in do not build
# there. The mock never opens a camera, but it does import server.py, which imports the capture
# stack, so the dependencies still have to resolve.
REQUIREMENTS="requirements.txt"
if [[ "$(uname -s)" == "Darwin" && -f "$PROJECT_DIR/requirements-mac.txt" ]]; then
    REQUIREMENTS="requirements-mac.txt"
fi

cd "$PROJECT_DIR"
# `${EXTRA_ARGS[@]+...}` rather than a bare `${EXTRA_ARGS[@]}`: under `set -u`, expanding an empty
# array is an error on the bash 3.2 that macOS still ships, which is exactly the platform
# requirements-mac.txt exists for.
exec uv run --python 3.13 --with-requirements "$REQUIREMENTS" -- \
    python mock_server.py --motion "$MOTION" --port "$PORT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
