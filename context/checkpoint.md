# Checkpoint (2026-11-23, final)

## State
- Everything is committed and pushed to GitHub `sahsudhanshu/RM` (main). The only later additions are this note, `presentation_script.md` and the context updates (commit them if not yet pushed: `git status`).
- Deck: 29 slides (slides 15, 16, 17 = training setup, environment parameters, training curves; slide 28 = future plan) (`presentation_final/Presentation.pdf`, `Presentation_overleaf.zip`); the full 44-slide version is kept as `Presentation_full_44slides.pdf`.
- Script for the three speakers, demos and Q&A: `context/presentation_script.md`.
- Gazebo: 22 experiment replays done (`implementation/gazebo/gazebo_vs_plan.csv`); runnable 3D demo packaged (`implementation/gazebo/gazebo_demo.zip`, tested on the friend's Ubuntu laptop in headless mode, in the Gazebo window inside a virtual display and from a clean unzip).
- Full state, key numbers, limits and rebuild commands: `context/context.md` section 10.

## Before presenting
1. Open `implementation/viewer/index_offline.html` in a browser and press Play once.
2. Copy `implementation/gazebo/demo_preview_*.mp4` and the two replay videos named in `presentation_script.md` to the presenting laptop (videos are not in git).
3. Read `presentation_script.md` sections 0, 1 and your own part.

## Added last
- Training slides (15, 16) and the Gazebo fallen-tree scenario (`implementation/gazebo/demo/run_tree_demo.sh`, tested headless on both maps and in the Gazebo window in a virtual display); script renumbered for 28 slides.

- Recorded 3D viewer videos: `implementation/viewer/videos/` (three scenarios, made with `implementation/viewer/record_viewer.py`).

- Four Gazebo demos with a menu (`implementation/gazebo/demo/demos.sh`): simple, tree, team (4 UAVs + 2 UGVs), divert; previews `implementation/gazebo/demo_preview_*.mp4`.

## Open (nothing blocks the presentation)
Run the learned policy and replanner inside Gazebo; JIT mode in the 3D viewer; closing the 4U-2G gap by training; making candidates A and B work. The friend's laptop IP changes with the Wi-Fi (`hostname -I`); SSH key `~/.ssh/friend_gazebo`; remove access by deleting the last line of its `~/.ssh/authorized_keys`.
