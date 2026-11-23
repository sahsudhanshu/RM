#!/bin/sh
# Multi-vehicle scenario: several UAVs and UGVs, a tree falls across the ring road, the team replans.
#   ./run_team_demo.sh                 # 4 UAVs + 2 UGVs on the ring-road map, Gazebo window, fleet camera
#   ./run_team_demo.sh 2u2g            # 2 UAVs + 2 UGVs (lighter)
#   ./run_team_demo.sh 4u2g --terrain terrain/thar24
#   ./run_team_demo.sh 4u2g --headless --record team.mp4 --max-sim 170
# Blue and green trucks are the two UGVs; the UAVs have different colours; the orange line is the route the open-loop plan would take, the green line the replanned one.
cd "$(dirname "$0")" || exit 1
command -v gz >/dev/null 2>&1 || { echo "Gazebo (gz) not found: see README.md 'Install'"; exit 1; }
TEAM=4u2g
if [ -n "$1" ] && [ "${1#-}" = "$1" ]; then TEAM="$1"; shift; fi
exec python3 demo.py "plans/team-$TEAM-event.json" --compare "plans/team-$TEAM-open.json" --show-event --cam fleet "$@"
