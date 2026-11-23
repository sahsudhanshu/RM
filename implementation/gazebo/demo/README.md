# Runnable 3D Gazebo demo (no experiments)

A ready-to-run 3D replay of a planned UAV–UGV mission in **Gazebo Harmonic**. Close-up 3D models, real-time playback, optional
real-terrain world. It replays a plan exported from our simulator; it does not run the learned policy and it is not an experiment.

## What you see
* **UGV**: wheeled truck with contact physics (DiffDrive), cab, mast and a yellow **landing pad** on its roof; it drives along the planned road edges.
* **UAV**: quadrotor with spinning rotors. It flies the planned legs, lands on the UGV roof, **charges while riding on it**, then takes off again.
* **Map**: roads, green task towers (turn grey when visited), yellow rendezvous pads, orange depot with a flag.
* **Road event** (blocked / slow): red or amber strip with cones appears on the affected roads at the planned time.
* Scale: space 1:50, time x60, so a ~200 min mission plays in ~3.5 min. Relative speeds are unchanged.

## Requirements
Ubuntu 22.04/24.04 with Gazebo Harmonic and a desktop session (the demo opens the Gazebo window). On Windows use **WSL2 + Ubuntu 24.04** (WSLg shows the window); native Windows is not supported by Gazebo Harmonic.

### Install (Ubuntu 24.04)
```bash
sudo apt-get update && sudo apt-get install -y curl lsb-release gnupg ffmpeg python3-pil python3-numpy
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt-get update && sudo apt-get install -y gz-harmonic
```
(`ffmpeg` is only needed for `--record`.) The driver uses the system Python 3 with the `gz.transport13` bindings that come with `gz-harmonic`.

## Run
```bash
unzip gazebo_demo.zip && cd demo
./run_demo.sh                                              # border map, Gazebo window, real time
./run_demo.sh plans/paper-typical.json                     # the base paper's map
./run_demo.sh plans/border-best-replanning.json            # a case where replanning pays off most
./run_demo.sh plans/border-typical.json --terrain terrain/thar24     # real-terrain world (open elevation data)
./run_demo.sh plans/border-typical.json --follow ugv0      # camera follows the UGV
./run_demo.sh plans/border-typical.json --headless --record demo.mp4 --max-sim 60   # no window: write a video
```
Options: `--speed N` time compression (default 60), `--scale N` space 1:N (default 50), `--rtf 0.5` slow motion, `--max-sim S` stop after S demo seconds, `--cam overview` for the recording camera.
The Gazebo window opens paused-looking until the first command; the mission starts automatically. When it finishes the world stays open; close the window to exit.

In the Gazebo window: left-drag rotates, middle-drag pans, scroll zooms; or use `--follow ugv0` / `--follow uav0`.

## Fallen-tree scenario: a tree blocks the road, then the mission replans
```bash
./run_tree_demo.sh                 # ring-road map: two blocked roads, Gazebo window
./run_tree_demo.sh border          # border map: one blocked road
./run_tree_demo.sh loop --terrain terrain/thar24
./run_tree_demo.sh border --headless --record tree.mp4 --max-sim 120    # no window: write a video
```
What happens (about 80 to 110 s into the demo, depending on the map): a tree stands beside the road; the **orange line shows the route the open-loop plan was going to use**; the tree **falls across the road** (animated) and striped barriers appear; the planner updates its road map and the **green line shows the replanned route**; the truck takes the detour and the UAVs reorder their sorties. In the Gazebo window the camera cuts to the event and back (`--show-event`). The plans are `plans/tree-*-event.json` (replanned) and `plans/tree-*-open.json` (the open-loop plan of the same case, used for the orange route). Options: `--event-style strip` for the old red strip instead of a tree, `--dry-run` to validate the world without Gazebo.
Preview videos: `../demo_preview_tree_border.mp4`, `../demo_preview_tree_loop.mp4`, `../demo_preview_tree_gazebo_window.mp4` (the Gazebo window captured in a virtual display).

## Plans included (`plans/`)
`border-typical`, `border-best-replanning`, `paper-typical`, `loop-typical` (event-triggered runs with a road event). More: `../export_plans.py` converts any case of `viewer/missions.json`.

## Honest limits
* The UGV motion is physics (wheels, friction, acceleration limits); UAV flight is velocity-controlled (kinematic), and landing/charging is a scripted hop onto the roof pad.
* Space is scaled 1:50 and time x60; the terrain relief is exaggerated x1.4 in `--terrain` mode.
* Tested on Ubuntu 24.04 with an RTX 3050: headless recordings and the real Gazebo window in a virtual display (including the fallen-tree scenario); not checked on a physical screen.
* In the fallen-tree scenario the tree fall is a scripted animation, and the replanned route is the one our planner produced for that case; the UGV then follows it with wheel physics.
