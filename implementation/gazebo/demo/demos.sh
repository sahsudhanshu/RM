#!/bin/sh
# Menu for the Gazebo demos.  Usage:
#   ./demos.sh              # asks which one
#   ./demos.sh simple       # 1  one mission: 2 UAVs + 1 UGV                      (= ./run_demo.sh)
#   ./demos.sh tree         # 2  a tree falls across the road, the mission replans (= ./run_tree_demo.sh)
#   ./demos.sh team         # 3  4 UAVs + 2 UGVs, a tree blocks the ring road     (= ./run_team_demo.sh)
#   ./demos.sh divert       # 4  a UAV flies from a checkpoint to the UGV, the road is blocked, the team replans (= ./run_divert_demo.sh)
# Extra options are passed on, e.g.  ./demos.sh team --headless --record team.mp4
cd "$(dirname "$0")" || exit 1
pick="$1"
if [ -z "$pick" ]; then
  echo "Gazebo demos:"
  echo "  1) simple  - one mission with 2 UAVs and 1 UGV"
  echo "  2) tree    - a tree falls across the road and the mission replans"
  echo "  3) team    - 4 UAVs and 2 UGVs, a tree blocks the ring road and the team replans"
  echo "  4) divert  - a UAV flies from a checkpoint to the UGV, the road is blocked, the team replans"
  printf "Choose 1, 2, 3 or 4: "; read pick
else shift; fi
case "$pick" in
  1|simple) exec ./run_demo.sh "$@" ;;
  2|tree)   exec ./run_tree_demo.sh "$@" ;;
  3|team|multi) exec ./run_team_demo.sh "$@" ;;
  4|divert) exec ./run_divert_demo.sh "$@" ;;
  *) echo "unknown choice: $pick"; exit 1 ;;
esac
