#!/bin/sh
# Diversion scenario: a UAV leaves its last checkpoint and flies to the UGV; a tree blocks the road so the UGV cannot reach the
# rendezvous; the team replans (the UAV diverts to a new rendezvous, the UGV takes another road) and the mission completes.
#   ./run_divert_demo.sh                 # border map (the open-loop plan fails here), Gazebo window, fleet camera
#   ./run_divert_demo.sh --terrain terrain/thar24
#   ./run_divert_demo.sh --headless --record divert.mp4 --max-sim 70      # video with battery readouts
# The terminal prints the story with plan times and UAV battery levels (fuel 287.7 kJ, hover power as in training).
cd "$(dirname "$0")" || exit 1
command -v gz >/dev/null 2>&1 || { echo "Gazebo (gz) not found: see README.md 'Install'"; exit 1; }
exec python3 demo.py plans/divert-border-event.json --compare plans/divert-border-open.json --show-event --cam fleet "$@"
