# Showing the simulation to the professor

## Files (all in `implementation/viewer/`)
| File | Use |
|---|---|
| `index_offline.html` | **Main demo.** Double-click to open in any browser (Chrome/Edge/Firefox). Works with no internet (three.js is inlined; fonts fall back to system fonts). |
| `index.html` | Same page, loads three.js from the internet. This is the one published as the Claude artifact (private link; share it from the page's Share menu if the professor needs the link). |
| `../results/U15G5_2U1G/media/mission_slow_high_early.mp4` (+ `.gif`) | 2D video of one scenario, side by side. Backup if a browser misbehaves. |

## Recorded videos (backup, no browser needed)
`videos/3d_viewer_ringroad_best_case.mp4`, `videos/3d_viewer_border_worst_case.mp4`, `videos/3d_viewer_ringroad_cut_road.mp4`: the 3D viewer played through three scenarios (open-loop left, event-triggered replanning right, live stats and timeline). Re-record with `python3 record_viewer.py` (headless Chrome; time is set frame by frame through the scrubber, so the video is smooth); options `--case N --speed M --fps F --out file.mp4`.

## 90-second script
1. *(Scenario list, ring-road map "slow edges, high, early")* "Two UAVs and one UGV must visit all task points. The UGV is the charging station and is slow, so UAVs hover and wait for it. Both panels are the same instance and the same road event; only the controller differs."
2. *(Press Play, speed 12)* "At t = 47 min a stretch of road becomes slow (amber). Left: open-loop, the plan is never revisited. Right: our event-triggered replanning notices the event, checks whether each planned rendezvous is still reachable on time, and replans only then."
3. *(Point at the energy bars and chart)* "The bars show UAV energy so far against the budget. Left crosses the budget line, right stays under it. Waiting UAVs burn hover energy, which is what the road event inflates."
4. *(Switch to "Border surveillance")* "Same idea on an army border-surveillance map: patrol road along the border (dashed red), supply road behind it, connector roads for detours, forward outposts. Surveillance points sit in the shaded strip."
5. *(Switch to "Base paper's map")* "This is the base paper's own road. It is a tree, so we only slow roads here; a blocked road would disconnect it."
6. *(Open the scenario list: each setting has **best case / typical case / worst case**)* **Say this honestly:** "Best cases are where replanning helps most. Typical cases often change nothing (the event does not touch that mission). In the worst cases replanning makes it worse: +11 min on the ring-road map, +48 min on the border map, +9 min on the paper's map. Averaged over 100 instances on the ring-road map it saves about 7 % energy and 5 % time."
7. *(Pick "cut road (stress test)")* "If a bridge is cut so the map is disconnected, open-loop fails; replanning rescues some missions but still fails in most (e.g. 29 of 32 on the ring-road map). On the paper's tree both fail."


## Controls
Drag = orbit, scroll = zoom, right-drag (or shift-drag) = pan, **Top** = map view, **Space** = play/pause, click the chart to jump in time, speed 6-60 min/s.
UAV height is exaggerated for readability. UAV battery % and status (flying / hovering / charging) are shown per controller.

## Regenerate
```bash
.venv/bin/python scripts/export_viewer_data.py     # runs the missions, writes viewer/missions.json
.venv/bin/python scripts/build_viewer.py           # writes index.html and index_offline.html
```
