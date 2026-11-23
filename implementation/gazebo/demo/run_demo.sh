#!/bin/sh
# One-command launcher.  Usage:  ./run_demo.sh [plan-file] [extra demo.py options]
#   ./run_demo.sh                                  # border map, GUI window, real time
#   ./run_demo.sh plans/paper-typical.json --terrain terrain/thar24
#   ./run_demo.sh plans/border-typical.json --headless --record demo.mp4
cd "$(dirname "$0")" || exit 1
command -v gz >/dev/null 2>&1 || { echo "Gazebo (gz) not found. On Ubuntu 24.04:  see README.md 'Install'"; exit 1; }
PLAN=plans/border-typical.json
if [ -n "$1" ] && [ "${1#-}" = "$1" ]; then PLAN="$1"; shift; fi
exec python3 demo.py "$PLAN" "$@"
