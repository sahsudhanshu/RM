# Checkpoint (2026-11-23 ~00:50 IST) — deadline 5:00 am the same day

## State
- GitHub `sahsudhanshu/RM` main = `152b92d` (deck 44 slides, code, results, notes). Local repo may be ahead by this checkpoint commit.
- Deck: `presentation_final/Presentation.pdf` (+ `Presentation_overleaf.zip`). Rebuild: `cd presentation_final && TEXMFHOME=$PWD/.texlocal/usr/share/texlive/texmf-dist python3 build.py` (needs `.texlocal`, not in git; figures: `python3 make_figs.py`).
- Done: P1 + JIT across 4 maps x 3 seeds, JIT re-prediction, adaptive sampling (neutral), 10240-sample decoding, fallback, 4U-2G more training (no gain), Gazebo test bed (flat + real terrain), 12 Gazebo replays collected (`implementation/gazebo/gazebo_vs_plan.csv`).

## In progress (on the friend's Ubuntu laptop, user `sudhanshu`, IP 10.215.19.219, key `~/.ssh/friend_gazebo`)
- 10 unfinished Gazebo replays relaunched with `GZ_IP=127.0.0.1` (the old ones hung after the Wi-Fi changed): flat `~/RM_gazebo/runs` (border typical/worst x open/event, loop typical open, loop best x2), terrain `~/RM_gazebo_t/runs_terrain` (border best open, worst x2). ETA about 01:30 IST. Needs the laptop awake and on the same network. Logs: `*.run3.log`.

## To finish (about 10 min after the runs end)
1. `cd implementation/gazebo && ./collect.sh && python3 analyze.py` (collect.sh already has the new IP).
2. Update the Gazebo slide sentence in `presentation_final/body.tex` (12 -> final n, numbers from analyze.py; remove "4 more border runs were not collected") and `implementation/gazebo/README.md` results table.
3. `cd presentation_final && python3 make_figs.py && TEXMFHOME=... python3 build.py`, re-zip Overleaf, commit, `git push` (retry on "hung up"; `http.postBuffer` is already set).
4. Copy for the demo: `implementation/viewer/index_offline.html` and `implementation/gazebo/videos*/` (videos are git-ignored).

## Not done / stated limits
Policy not run inside Gazebo; no JIT mode in the 3D viewer; 4U-2G gap not closed by training (search closes it); candidates A/B no better than control; terrain from open SRTM-derived data, not Google Earth.
