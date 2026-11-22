#!/bin/sh
# Fetch Gazebo run summaries and videos from the machine that ran replay.py (flat ground: ~/RM_gazebo/runs, terrain: ~/RM_gazebo_t/runs_terrain).
HOST=${1:-sudhanshu@10.215.19.219}; KEY=${2:-$HOME/.ssh/friend_gazebo}; D="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$D/results" "$D/videos" "$D/results_terrain" "$D/videos_terrain"
scp -q -i "$KEY" -o BatchMode=yes "$HOST:~/RM_gazebo/runs/*.json" "$D/results/"
scp -q -i "$KEY" -o BatchMode=yes "$HOST:~/RM_gazebo/runs/*.mp4" "$D/videos/"
scp -q -i "$KEY" -o BatchMode=yes "$HOST:~/RM_gazebo_t/runs_terrain/*.json" "$D/results_terrain/" 2>/dev/null
scp -q -i "$KEY" -o BatchMode=yes "$HOST:~/RM_gazebo_t/runs_terrain/*.mp4" "$D/videos_terrain/" 2>/dev/null
exit 0
