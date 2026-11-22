# Checkpoint (2026-11-23 ~00:50 IST) — deadline 5:00 am the same day

## State
- GitHub `sahsudhanshu/RM` main = `152b92d` (deck 44 slides, code, results, notes). Local repo may be ahead by this checkpoint commit.
- Deck: `presentation_final/Presentation.pdf` (+ `Presentation_overleaf.zip`). Rebuild: `cd presentation_final && TEXMFHOME=$PWD/.texlocal/usr/share/texlive/texmf-dist python3 build.py` (needs `.texlocal`, not in git; figures: `python3 make_figs.py`).
- Done: P1 + JIT across 4 maps x 3 seeds, JIT re-prediction, adaptive sampling (neutral), 10240-sample decoding, fallback, 4U-2G more training (no gain), Gazebo test bed (flat + real terrain), 12 Gazebo replays collected (`implementation/gazebo/gazebo_vs_plan.csv`).

## Gazebo runs STOPPED by the user (2026-11-23 ~00:55 IST)
- The 10 unfinished replays were killed on the friend's laptop (nothing is running; the laptop itself is untouched). 12 runs are collected and in the deck/CSV (9 flat, 3 terrain). Missing: flat border typical/worst x open/event, loop typical open, loop best x2, terrain border best open, worst x2.
- To redo them (about 45 min + 10 min wrap-up; only if started before ~3:00 am): the friend's Ubuntu laptop (user `sudhanshu`, key `~/.ssh/friend_gazebo`, IP changes with the Wi-Fi: ask `hostname -I`) must be awake and on the same network; run `ssh ... 'sh ~/relaunch.sh'` (already on the laptop; uses `GZ_IP=127.0.0.1`), then `implementation/gazebo/collect.sh` (edit the IP in it) and `analyze.py`.
- Otherwise freeze the deck as it is (12 runs, slide says 4 more were not collected; with this stop it is now 10 not collected: update that sentence to "10 more runs were not completed in time" in `presentation_final/body.tex` and `implementation/gazebo/README.md`).

## To finish (about 10 min after the runs end)
1. `cd implementation/gazebo && ./collect.sh && python3 analyze.py` (collect.sh already has the new IP).
2. Update the Gazebo slide sentence in `presentation_final/body.tex` (12 -> final n, numbers from analyze.py; remove "4 more border runs were not collected") and `implementation/gazebo/README.md` results table.
3. `cd presentation_final && python3 make_figs.py && TEXMFHOME=... python3 build.py`, re-zip Overleaf, commit, `git push` (retry on "hung up"; `http.postBuffer` is already set).
4. Copy for the demo: `implementation/viewer/index_offline.html` and `implementation/gazebo/videos*/` (videos are git-ignored).

## Not done / stated limits
Policy not run inside Gazebo; no JIT mode in the 3D viewer; 4U-2G gap not closed by training (search closes it); candidates A/B no better than control; terrain from open SRTM-derived data, not Google Earth.
