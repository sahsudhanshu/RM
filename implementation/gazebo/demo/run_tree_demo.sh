#!/bin/sh
# Fallen-tree scenario: a tree falls across a road, the original route (orange) is shown, then the replanned route (green).
#   ./run_tree_demo.sh                  # ring-road map (two blocked roads), Gazebo window, camera cuts to the event
#   ./run_tree_demo.sh border           # border map (one blocked road)
#   ./run_tree_demo.sh loop --terrain terrain/thar24
#   ./run_tree_demo.sh loop --headless --record tree.mp4 --max-sim 140
cd "$(dirname "$0")" || exit 1
command -v gz >/dev/null 2>&1 || { echo "Gazebo (gz) not found: see README.md 'Install'"; exit 1; }
MAP=loop
if [ -n "$1" ] && [ "${1#-}" = "$1" ]; then MAP="$1"; shift; fi
exec python3 demo.py "plans/tree-$MAP-event.json" --compare "plans/tree-$MAP-open.json" --show-event "$@"
