# Gazebo replay of the planned missions

**Runnable 3D demo (no experiments):** see `demo/README.md` and the package `gazebo_demo.zip` (`./run_demo.sh` opens a Gazebo window with close-up 3D vehicles, real-time playback, optional real terrain). Preview videos: `demo_preview_flat.mp4`, `demo_preview_terrain.mp4`, `demo_preview_gazebo_window.mp4`. The sections below describe the headless experiment replays.

Physics-based check of the plans produced by our simulator (`../viewer/missions.json`), run in **Gazebo Harmonic (gz-sim 8)**
on a separate GPU machine over SSH. It replays the plans, it does not run the learned policy inside Gazebo.

| piece | how it is simulated |
|---|---|
| UGV | wheeled DiffDrive vehicle on real contact physics (400 kg, accel 1.5 m/s², turn rate 1.2 rad/s); pure-pursuit along the planned road edges at the planned speed (slow roads = lower speed) |
| UAV | velocity-controlled body (kinematic, gravity off) flying the planned legs at the planned speed; vertical take-off/landing is not modelled |
| Roads, tasks, depot | visual overlays; road events (blocked / slow edges) are spawned at the planned event time |
| Coupling | a UAV is served only after the UGV has **actually arrived** and is free, so physical delays propagate into mission time and hover energy |
| Energy | UAV flight energy of the planned legs (same power model P(v)) + hover power × the waiting time measured in Gazebo |
| Terrain (optional) | real 24 x 24 km elevation (open SRTM-derived tiles, Thar desert, 144 m relief, up to 15 deg slopes) as a heightmap with the road map draped on it |

Google Earth imagery/terrain cannot be downloaded programmatically (licence); the open data behind the terrain tiles is the same kind of SRTM elevation.

## Files
* `export_plans.py` – `viewer/missions.json` → metres/seconds plans in `plans/` (open-loop and event-triggered run of every case).
* `make_world.py` – SDF world generator (flat ground or `--terrain`).
* `make_terrain.py <lat> <lon> <dir>` – builds `heightmap.png`, hill-shaded texture and the elevation lookup from open tiles.
* `replay.py` – headless replay in lock-step (paused world, 0.2 s per controller tick, ~11x real time), writes a JSON summary and an MP4.
* `collect.sh`, `analyze.py` – fetch results; `gazebo_vs_plan.csv` – planned vs Gazebo for every run; `videos/` – the recordings.

## Run (on the Gazebo machine; Ubuntu 24.04, `gz-harmonic`, system python with `gz.transport13`)
```bash
python3 export_plans.py                      # on the project machine, then copy plans/ and the scripts over
python3 replay.py "plans/<case>__event.json" --out runs                     # flat ground
python3 replay.py "plans/<case>__event.json" --terrain terrain/thar24 --out runs_terrain
```
Parallel runs need distinct `--partition` names (default: one per process).

## Results (22 replays: 16 flat ground, 6 real terrain; `gazebo_vs_plan.csv`)
Cases: border, paper and ring-road maps (typical, best and worst cases, open-loop and event-triggered); terrain: the three border cases.

| | n | mean abs. time difference | max | UAV energy difference (mean) | UGV arrival delay (mean / worst drive) |
|---|---|---|---|---|---|
| flat ground | 16 | 0.12 % | 0.75 min | +0.17 % | 11.0 s / 45 s |
| real terrain | 6 | 0.19 % | 0.77 min | +0.28 % | 15.9 s / 46 s |

The plans are physically achievable and the planned time and energy reproduce to within a fraction of a percent. This checks the plans; the learned policy is not run inside Gazebo, UAV flight is kinematic, and the UGV follows the planned timetable. Terrain renders show a small cosmetic speckle artefact; the elevation data has no spikes.

Gazebo's transport binds to the machine's network address; if the machine changes network mid-run the replays hang. Run with `GZ_IP=127.0.0.1` (all local) to avoid that.
