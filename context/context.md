# Project context — Energy-Constrained UAV–UGV Replanning under Dynamic Road Conditions

*Course:* Research Methodology (RM), IIIT Allahabad (IIT2024036 · IIT2024081 · IIT2024146).
*Team:* Mayank Anand, Sudhanshu (Sah), Sujal.
*Last updated:* 2026-11-23 (see §10 "Final state" at the bottom; §9 is the older status).

---

## 1. What is in this folder (`context/`)

| File | What it is |
|---|---|
| `p1.pdf` | **Base paper.** Mondal et al., *How to Coordinate UAVs and UGVs for Efficient Mission Planning? Optimizing Energy-Constrained Cooperative Routing with a DRL Framework*, arXiv:2504.21111 (Apr 2025). 19 pp. |
| `RM_IIT2024036_IIT2024081_IIT2024146 (1).pdf` | **Our submitted deck** (21 slides): research problem, literature review, proposed method and methodology. This is the specification we implement. |
| `main.tex` | Beamer source of our deck (`main (4).pdf` is an earlier 22-page compile of it). |
| `main (4).pdf` | Earlier compile of `main.tex` (22 pages). Superseded by the `RM_…(1).pdf` deck. |
| `gazebo-ugv-desert.png` | Screenshot of the Gazebo desert + 4-wheel UGV with a landing pad (used on slide 18, "Current work"). |
| `base_papers/` | **Downloaded 2026-11-21:** the base paper PDF, the authors' public code (`authors_code_multi-uav-ugv-drl/`, third-party, read-only, never executed), open-access related papers by the same group and the methods papers it builds on. `base_papers/README.md` lists sources and what could not be downloaded (paywalled). |
| `friend_presentation/` | A **different** group's work on the **same base paper** (see §4). Only used as a style/pipeline reference. |
| `presentation.zip` | Zip of `friend_presentation/` (identical content). |
| `context.md` | This file: project context and final state. |
| `chat.md` | Running log of the whole work (decisions, results, problems), newest at the bottom. |
| `checkpoint.md` | Short resume note: current state, how to rebuild and what is left. |
| `presentation_script.md` | **End-sem presentation script**: 3 parts for 3 speakers, every slide taught, the 3 demos, Q&A and cut list. |

The implementation lives **outside** this folder, in `../implementation/`.

---

## 2. Base paper (p1.pdf) — what we build on

**Problem.** A heterogeneous team of *M UAVs* and *N UGVs* must visit every task point in minimum time.
* Task points: **ground points** `M_g` (on the road network; reachable by UGV and UAV; also rendezvous/recharge points) and **aerial points** `M_a` (UAV only).
* UAV: speed `v_a = 10 m/s`, battery `F_a = 287.7 kJ`, power `P(v) = 0.0461 v³ − 0.5834 v² − 1.8761 v + 229.6 W` → ≈ 25 min endurance at 10 m/s (P(10)=198.6 W, P(0)=229.6 W).
* UGV: speed `v_g = 4.5 m/s`, road network only, acts as a mobile charger; **one UAV at a time**.
* All agents start at one depot; mission ends when all tasks are visited and UAVs finish their final recharge. Constant recharge time `T_R`. Failure penalty `P = 800`.
* Map 20×20 km; aerial points uniform within 7 km of road points; fixed road network; ground points random road nodes.
* Sizes: U15G5, U45G15 (train); U60G20, U75G25 (generalisation). Teams 1U-1G, 2U-1G, 2U-2G, 4U-2G (+5U-3G, 6U-4G).

**MDP.** State `(p_t, f_t, q_t)` = active-agent position, fuel, task visited flags. Action = visit task or recharge at ground point (infeasible masked). Reward = travel time (+`T_R` per recharge); return = max over agents of cumulative time + penalty on failure.

**Framework.** Encoder–decoder Transformer (encoder: 3 MHA layers, d=128, 8 heads, BN; decoder with context vector, glimpse MHA, single-head compat. with tanh clip C=10, masking). **Sortie-wise agent switching**: UAV with smallest time dimension flies a whole sortie (task visits … ends at a recharge), then next UAV; when all UAV sorties are done, UGVs act (service order via UGV decoder); **greedy UAV→UGV assignment by proximity** of landing locations; then UAVs again.
Training: REINFORCE with greedy-rollout baseline (updated by paired t-test), Adam 1e-4, decay 0.995/epoch, 5.12 M instances, batch 256, RTX 4090. Decoding: greedy or sampling (N=1024 / 10240).
Baselines: bilevel "UGV-first, UAV-second" (MSC + TSP + E-VRPTW) solved with OR-Tools CP-SAT + GLS / Tabu / SA.

**Table I (U15G5, mean mission time, min)** — used for trend checks:

| Team | DRL greedy | DRL(1024) | DRL(10240) | GLS | TS | SA |
|---|---|---|---|---|---|---|
| 1U-1G | 238.2 | 220.4 | 218.6 | 281.7 | 276.6 | 283.4 |
| 2U-1G | 229.7 | 203.1 | 199.7 | 232.8 | 230.8 | 242.8 |
| 2U-2G | 180.5 | 157.1 | 154.4 | 188.3 | 185.6 | 194.1 |
| 4U-2G | 188.9 | 165.6 | 159.9 | 219.7 | 219.0 | 220.3 |

Trends: more UGVs help a lot; more UAVs help little (waiting for single UGV); sampling beats greedy by ≈ 9–12 %; DRL beats metaheuristics; 4U-2G ≈ 2U-2G.
**Stated limitations** (our opening): fixed road network (no road changes), greedy UAV→UGV assignment, no energy/budget notion beyond the per-sortie battery.

**Things the paper does NOT state** (we treat as assumptions, see §6): value of `T_R`; exact road-network geometry; depot choice; how long UAVs may wait; whether waiting drains battery; UGV service order details.

---

## 3. Our project (the deck, RM_IIT2024036_IIT2024081_IIT2024146 (1).pdf)

**Central question.** Can a learned UAV–UGV routing policy control **total UAV energy** and remain **feasible when UGV roads become blocked or slow** during the mission?

**Research questions / hypotheses**
* RQ1 time–energy trade-off → **H1** energy can fall with only a small time increase.
* RQ2 keep energy within a given budget → **H2** a constrained policy beats a fixed penalty.
* RQ3 effect of blockages/delays on time, energy, failures → **H3** more disruption ⇒ more energy and more failures.
* RQ4 event-triggered vs periodic replanning → **H4** similar reliability with fewer replans.

**Problem statement.** Minimise mission time `T` subject to `E ≤ B`.
`T` = time the last agent finishes; `E = Σ (P_fly·t_fly + P_wait·t_wait)` over all UAVs (waiting energy only when the UAV waits **in the air**); `B = β·B0`, `β ∈ {0.70, 0.80, 0.90, 1.00}`, `B0` = energy of the base policy on the same static-road instance.
Always-hold rules: every task visited; each sortie within battery `F_a`; recharge meeting on a reachable road node; one UGV charges one UAV at a time.
`F_a` (per-sortie battery, inherited) vs `B` (mission-level cumulative budget, new).

**Road model.** Normal edge: original time. Slow edge: time × κ>1. Blocked edge: removed from the current UGV graph. Events: type (blocked / slow), level (low/med/high), time (early/middle/late). Policy learns of an event only when it occurs.

**Architecture (slide 13).** Instance → *unchanged* Transformer encoder → route decoder (candidate variant: + remaining-budget input) → *frozen* greedy UAV→UGV assignment (current road distance) → simulator rollout (T, E, failures) → dynamic road layer (events on `G_t`) → check recharge time/reachability → updated action masks on current roads → replan if needed.

**Two candidate energy-aware methods**
* **A – Lagrangian-constrained learning:** decoder unchanged; score `T + λ·max(0, E−B)`; dual update `λ ← [λ + η(Ē−B)]₊`.
* **B – Budget-conditioned routing:** decoder also receives `b_t = (B − E_used,t)/B`.

**Event-triggered replanning.** Observe event → update road graph → recompute affected UGV path/ETA → check each planned recharge point still reachable on time → if feasible continue, else keep completed route and replan the remaining tasks. Compared with *open-loop* (no replan) and *periodic* (every recharge, as in the base case study). Training regimes: static roads only vs random road events.

**Policies**

| | Objective | Tests |
|---|---|---|
| P0 Base | min T | original policy |
| P1 Route filter | fastest route with E ≤ B | post-hoc filtering enough? |
| P2 Fixed penalty | min T + αE | fixed reward shaping |
| P3 Lagrangian | min T s.t. E ≤ B | candidate A |
| P4 Budget-conditioned | budget-aware routing | candidate B |

**Ablations.** remove budget input; fixed penalty vs Lagrangian; none/periodic/event-triggered replanning; train with/without random road events; Euclidean vs road distance for assignment.

**Experimental design.** Axes: budget {0.7,0.8,0.9,1.0}·B0; disruption {none,low,med,high}; timing {early,mid,late}; type {block, slow}; teams {1U-1G,2U-1G,2U-2G,4U-2G}; policies P0–P4. Stage A static budgets (RQ1–2); Stage B blocked/slow roads (RQ3–4); Stage C larger problem sizes. Protocol: same test cases and events for all policies; 100 test instances; 3 training seeds; mean, std, 95 % CI.

**Validation (Milestone 1).** Reproduce base Table-I trends (U15G5 2U-1G greedy) within tolerance; hand-checkable case 1U-1G, 3 tasks, block the single edge to the next rendezvous → verify reroute, ETA, mask; unit test `E = Σ P·t` per sortie, check waiting model (hover vs landed).
**Risks.** time/energy move together; budget may be impossible (estimate min energy first); energy penalty unstable (small update rate); road graph may disconnect (first test blockages with an alternative path); scope creep (assignment rule fixed); base code incomplete (verify base first).
**Milestones.** M1 base env + hand-checked cases; M2 energy accounting, P0–P3 static (λ converges); M3 dynamic roads + event-triggered replanning; M4 ablations, seeds, scaling, report. Decision gate: if road events have little effect, report that and focus on the time–energy trade-off.
**Current work (slide 18).** Gazebo desert + 4-wheel UGV with landing pad; next: send waypoints / road events to the UGV, record arrival time + UAV waiting energy.

Literature used in the deck: Maini et al. 2019 (TAES), Fan et al. 2023 (TVT), Mondal et al. 2025 (arXiv, base), Shi et al. 2022 (CDC), Lin et al. 2022 (RA-L), Mondal et al. 2025 (ICRA, Ra-DRL). Gap: nobody jointly has learned multi-UAV-UGV policy + mission energy budget + roads that change during execution.

---

## 4. The friend's project (friend_presentation/) — DO NOT CONFUSE WITH OURS

Same base paper, **different problem**: *Learning Virtual-Slot Rendezvous Scheduling for Cooperative UAV–UGV Missions* (15 slides).
* They replace the base paper's **greedy UAV→UGV assignment** with a learned **queue-aware virtual-slot assignment** (cross-attention over UGVs × queue positions, Gumbel-Sinkhorn → Hungarian). Metric: landing wait, collisions, load balance, completion.
* Their roads are **static**; there is **no energy budget**, **no road events**, **no replanning**.
* Their code (`code/implementation/src/uav_ugv/…`, ~5.7 k lines, PyTorch + networkx): `base_map.py` (fixed 3-branch road tree around (6,8)), `mission_simulator.py` (2D sortie/fuel/rendezvous simulator with nearest-feasible teacher), `simulator.py` (charging round), `virtual_slots.py`, `sinkhorn.py`, `model.py`, `training/train_pair.py` (teacher imitation → REINFORCE, tiny CPU/GPU budgets), `evaluation/*`, and `results/plot_matrix.py`.
* Their *presentation style* (what we mimic): a scenario figure of generated instance, paired-dot plots of baseline vs proposed per cell, ranked-gain plot, ablation bar plot, `build.py` that regenerates figures from preserved CSVs and writes an `artifact_manifest.json` with hashes.

**What we take from them:** the *shape* of a CPU-feasible pipeline (shared mission simulator → teacher warm-start → REINFORCE → paired evaluation on frozen seeds → figures from CSV). **What we do not take:** virtual slots, Sinkhorn, their road tree (it has no cycles — useless for blockage rerouting), their configs or numbers.

---

## 5. Differences that matter (ours vs friend's)

| | Ours | Friend's |
|---|---|---|
| Assignment | frozen greedy (road distance) | learned virtual slots |
| New constraint | mission energy budget `E ≤ βB0` | none |
| Roads | change during mission (blocked/slow) | static |
| Replanning | open-loop / periodic / event-triggered | none |
| Candidate methods | Lagrangian, budget-conditioned | queue-aware assignment |
| Road graph | needs **cycles** so a blocked edge has a detour | tree |

---

## 6. Modelling assumptions we must state (not given in the paper)

1. `T_R = 15 min` - **verified** in the authors' code (900 s).
2. Road geometry: the base paper's own map is now implemented exactly (`paper`, see 8b); the dynamic-road experiments use maps **with loops** (`loop`, `border`, `geo`, ...).
3. Per-sortie battery feasibility `F_a` is checked on **flight energy** (inherited from the base paper); **waiting energy `P(0)=229.6 W·t_wait` is charged to mission energy `E`** only (deck: "waiting energy is added only when the UAV waits in the air"). A `landed` wait mode (0 W) exists for the hover-vs-landed unit test.
4. UGV service order within one UGV's queue: earliest touchdown first (frozen, like the assignment).
5. Replanning decisions are taken at recharge-round boundaries.
6. Training is **CPU-only and scaled down** (paper: 5.12 M instances on an RTX 4090). Warm start from a nearest-feasible teacher, then REINFORCE with greedy baseline.

---

## 7. Environment

Linux, Python 3.13, 12 cores, 7 GB RAM, **no GPU**. `numpy`, `scipy`, `matplotlib`, `pillow` available system-wide; PyTorch (CPU) + pytest installed in `implementation/.venv`. `pdflatex` and `ffmpeg` available.

---

## 8. Implementation (../implementation/) — summary

Package `edr` (deliberately not named like the friend's `uav_ugv`). Pure-numpy simulator + PyTorch policy. Full description, run commands,
results and deviations: `../implementation/README.md`; all tables: `../implementation/results/REPORT.md`.

* `env.py` sortie-wise mission simulator (fuel masks, greedy assignment, UGV queue, hover-waiting energy, mid-round UAV diversion);
  `roads.py` road map with blocked/slow edges + time-dependent UGV driving; `events.py`; `plan.py` open-loop / periodic / event-triggered controllers;
  `model.py` Transformer route policy (+ budget input); `rl.py` REINFORCE (greedy baseline; group baseline), P0–P4 objectives;
  `evaluate.py` paired evaluation incl. P1 route filter; `viz.py` scenario plots + side-by-side animation.
* 29 tests (hand-checked 1U-1G case: detour ETA 30 min, `E = ΣP·t`, hover vs landed, road mask, controller equivalence without events, regressions).
* Demonstration simulation: `results/U15G5_2U1G/media/mission_slow_high_early.mp4` (+ .gif, .png).

### Key findings (U15G5, 100 test instances, 1 training seed)
* M1 trend check passes (ordering of teams matches base Table I; P0 2U-1G 194.7 min vs paper greedy 229.7, sampled 199.7).
* H3 ✓ events cost the open-loop base policy +3…18 % T and +4.5…27 % E; early events worst.
* H4 ✓ event-triggered ≈ periodic (ΔT +0.1 ± 0.1 min) with ~5× fewer replans (0.47 vs 2.45).
* H1/H2 ✗/inconclusive: time and energy are aligned (hover waiting dominates E); P3 vs P2 indistinguishable, λ hit its cap (8) — never feasible;
  P4 ≈ control within seed noise. P1 best-of-64 route filtering is far stronger than any learned variant (−19 % T, −25 % E vs greedy P0).
* Disconnecting-bridge stress test: failures 85 % → 68 % (2U-1G), 75 % → 52 % (2U-2G); diversion limited by empty batteries.

### Not done / known limits
Single training seed (deck: 3); training is ~minutes on CPU (paper: 5.12 M instances on a GPU); no GLS/TS/SA metaheuristic baselines;
no Stage C (larger sizes); no Gazebo integration; replanning only at round boundaries (+ UAV diversion mid-round).

---

## 8b. Verified against the authors' code (base_papers/authors_code_multi-uav-ugv-drl)

* **Road graph**: three chains (CSV rows 0-6 west, 7-18 north-east, 19-29 south-east) that repeat the junction coordinate (6, 8), so networkx merges it into one node. 28 distinct points, a tree. Our `paper` map is exactly this graph (km units; the code converts km to miles). The CSV itself (`ugv_road.csv`) is not in the repo; the point list is in `scenario_generator_v2.py`.
* **Mandatory ground points** (3,8), (10,12), (10,5) always included; depot = random ground point; UAV tasks within 7 km of ground points. Same as ours.
* **Recharge time** `T_R = 900 s = 15 min` (speeds in ft/s: UAV 33 ≈ 10 m/s, UGV 15 ≈ 4.5 m/s; fuel 287 700 J; power 198 W at cruise). **Assumption 1 of §6 is therefore verified.**
* **Waiting**: UGV clock advances by drive time; UAV waits `max(0, UGV time − landing time)` then 900 s recharge; fuel drains only in flight. Same as our simulator.

## 8c. Road maps (implementation/src/edr/roads.py)

| map | what | nodes / edges / bridges |
|---|---|---|
| `paper` | the base paper's road (exact reconstruction) - use for the fair comparison with Table I | 28 / 27 / 27 (tree: only slow events are meaningful; a block disconnects it) |
| `loop` | our ring + chord + spurs (first map; all earlier results) | 41 / 42 / 12 |
| `grid` | city blocks with some links removed | 36 / 51 / 1 |
| `radial` | ring-radial town | 37 / 66 / 0 |
| `geo` | irregular planar district (Delaunay, pruned) | 44 / 63 / 16 |
| `border` | **army border-surveillance region**: patrol road along a slanted border, inner supply road, 4 connector roads (loops), 3 forward-outpost spurs (mandatory points); aerial surveillance points sampled in a strip along the border | 47 / 49 / 11 |

Map study (rule-based policy, 2U-1G, 60 instances; `scripts/map_study.py`, `results/map_study.csv`): events hurt an open-loop mission most on `border` (+21.7 % T, +29 % E) then `loop` (+12.7 / +18.9), `geo` (+11.5 / +14.5); replanning recovers most on `geo` (-33 min, -9.8 % E); `radial` is least informative (many detours: +4.3 % T). Maps differ in scale (static mission 160-346 min), so compare policies within a map.

## 8d. Kaggle

Kaggle CLI 2.2.4 installed in `implementation/.venv`; account `vichanshur` is already authenticated. Nothing has been uploaded yet. `implementation/kaggle/` holds the dataset (code zip) and one kernel per map; `scripts/kaggle_run.py` trains and evaluates map x seed x variants (resumable). A `--quick` smoke test of the full pipeline passed.

## 9. Status (2026-11-22) — superseded by §10

* [x] Context gathered; initial files in `context/`; base papers + authors' code downloaded to `context/base_papers/`.
* [x] Implementation, 35 tests, experiments (2U-1G/2U-2G local; **4 maps x 3 seeds on Kaggle**), figures, report, 2D video, **3D browser viewer (18 scenarios incl. worst and failed cases)**.
* [x] M1: base policy within ~10 % of the paper's greedy DRL on the paper's own map for 3 of 4 teams (4U-2G +16 %).
* [~] M2: lambda stabilises at feasible budgets (beta 1.0, 0.95), saturates at 0.9: partly met.
* [x] Stage C: zero-shot U45G15 done (1 seed, 50 instances); U75G25 not done.
* Main findings (details in `implementation/README.md`): replanning beats open loop on every map and seed; event-triggered = periodic with fewer replans; candidates A/B not better than control; sampling + budget filter (P1) strongest.
* [x] Closed since: 4 team sizes x 3 seeds, Stage C (45 and 75 tasks), GLS/Tabu/SA baselines, sampled decoding (1024), lambda convergence (cap 50), search distillation (negative).
* [x] Mid-sem idea: not achieved at route level (H1); achieved via just-in-time departure (-53..-74 % energy, +0 min; see README).
* [x] Loop-detection fallback (removes the bad seed's failures; seed stays ~2x slower), JIT across team sizes (-9..-70 % energy at +0 min), 35 tests.
* [ ] Still open: Gazebo coupling (skipped by user), 4U-2G base gap (9-14 %), candidates A/B negative.

* [x] 10240-sample decoding done: 197.5/157.2/135.3/135.2 min (paper 218.6/199.7/154.4/159.9).


---

## 10. Final state (end-semester, 2026-11-23)

### Deliverables
| What | Where |
|---|---|
| **Final deck, 29 slides** | `presentation_final/Presentation.pdf`; Overleaf upload: `presentation_final/Presentation_overleaf.zip`; source `body.tex` (+ `preamble.tex`, `extras.tex`, `make_figs.py`, `build.py`) |
| Full 44-slide deck (backup, has the removed slides) | `presentation_final/Presentation_full_44slides.pdf`, `body_full_44slides.tex` |
| **Presentation script** (3 speakers, per-slide teaching, demos, Q&A) | `context/presentation_script.md` |
| 3D browser viewer (**32 scenarios incl. 4 UAVs + 2 UGVs and 2 UAVs + 2 UGVs**, open-loop vs replanning) and its **recorded videos**; 2D animations of the teams in `implementation/results/U15G5_4U2G/media/` and `U15G5_2U2G/media/` | `implementation/viewer/index_offline.html` (works offline); videos `implementation/viewer/videos/*.mp4` (made by `record_viewer.py`); notes `implementation/viewer/DEMO.md` |
| Gazebo experiment replays (22 runs) and results | `implementation/gazebo/` (`gazebo_vs_plan.csv`, `results*/`, videos in `videos/` and `videos_terrain/`, git-ignored) |
| **Four runnable 3D Gazebo demos** (menu `demos.sh`): simple (`run_demo.sh`), **fallen tree** (`run_tree_demo.sh`), **team with 4 UAVs + 2 UGVs** (`run_team_demo.sh`), **diversion** (`run_divert_demo.sh`: UAV flies from a checkpoint to the UGV, road blocked by a tree, team replans; UAV battery bars) | `implementation/gazebo/demo/`, package `implementation/gazebo/gazebo_demo.zip`, previews `implementation/gazebo/demo_preview_*.mp4` |
| Code, tests (35), results, README | `implementation/` (`README.md` documents every experiment) |

Repository: GitHub `sahsudhanshu/RM` (branch `main`). Not in the repo (size or reproducible): `.venv`, `presentation_final/.texlocal` (local TeX, 230 MB), `*.pt` checkpoints, raw per-instance Kaggle tables, Gazebo videos/run folders.

### Story of the deck (29 slides)
Problem (2-5) -> literature and gap (6-8) -> before mid-sem (9-10) -> limits and what we did (11) -> method (12-18: reproduce base, event-triggered replanning, learning policies, **training setup and configuration (15)**, **environment parameters used in training (16)**, **training curves (17)**, experimental design) -> results (19-24) -> Gazebo (25) -> hypotheses (26) -> conclusion and limits (27) -> future plan with a success test per hypothesis (28) -> references (29).

### Key results (all with the numbers used in the deck)
* Road events hurt: up to +22 % time and +29 % energy (border map, open-loop rule policy, 18 events).
* Event-triggered replanning beats open loop in 12/12 (map, seed) combinations (15 to 41 min, 260 to 1000 kJ) and matches periodic replanning (within 0.3 min) with 1 to 3 fewer replans.
* Why the energy-aware methods have little to gain: time and energy correlate at 0.96 over sampled routes (hover waiting is about half of E); candidates A and B equal their control (negative result).
* Lever that works: just-in-time departure, -53 to -74 % UAV energy at +0 min (assumes free waiting on the ground); by team size with events -18 / -54 / -40 / -69 % (1U-1G, 2U-1G, 2U-2G, 4U-2G); re-predicting the UGV arrival at take-off cuts the remaining hover energy 241 -> 80 kJ.
* Sampling (64 samples, keep fastest within budget) + JIT, 4 maps x 3 seeds, 2U-1G: time 228 -> 182 min (-20 %), energy 4407 -> 1524 kJ (-65 %), over-budget 78 % -> 0 % (static); with events 270 -> 205 min, 5531 -> 1841 kJ, 91 % -> 2 %.
* Base reproduction: within about 10 % of the paper's greedy DRL for 3 of 4 teams; 4U-2G 9 to 14 % slower, more training does not help (211 -> 210 min); best of 10,240 samples gives 197.5 / 157.2 / 135.3 / 135.2 min vs the paper's 218.6 / 199.7 / 154.4 / 159.9 (reconstructed map: no claim to beat the paper).
* Gazebo: 22 replays (16 flat, 6 real terrain): time within 0.14 % of the plans (max 0.8 min), UAV energy within 0.5 %, UGV arrives 11 to 16 s late on average.

### Negative results and limits (always state them)
Candidates A and B no better than the control; search distillation and adaptive sampling no gain; 4U-2G gap not closed by training (search closes it); one bad seed (loop map, seed 2) loops, a fallback removes its failures but it stays about 2x slower; Gazebo replays plans only (the learned policy is not run inside it, UAV flight is kinematic); JIT assumes free ground waiting and predictable UGV arrival; the 3D viewer has no JIT mode; Google Earth data cannot be downloaded programmatically, the terrain uses open SRTM-derived data.

### How to rebuild and run
```bash
cd presentation_final && python3 make_figs.py && TEXMFHOME=$PWD/.texlocal/usr/share/texlive/texmf-dist python3 build.py   # needs .texlocal
open implementation/viewer/index_offline.html            # 3D viewer (any browser)
cd implementation/gazebo && unzip gazebo_demo.zip && cd demo && ./run_demo.sh     # 3D Gazebo demo (Ubuntu + gz-harmonic, desktop)
```
The Gazebo machine used so far is the friend's Ubuntu 24.04 laptop (RTX 3050) over SSH; always run replays with `GZ_IP=127.0.0.1` so a Wi-Fi change cannot hang them.

### Training details (also on slides 15 and 16)
Transformer encoder (3 layers, 8 heads, d = 128, LayerNorm) + glimpse-attention decoder (clip 10), 0.73 M parameters. Fresh random instance every step (15 aerial + 5 ground tasks), 64 validation and 100 held-out test instances. 1) Imitation of the nearest-feasible rule: 800 steps (2 x 400), batch 64, Adam 1e-3 then 3e-4, clip 1.0. 2) REINFORCE (P0): 1,200 steps on mission time, batch 64, Adam 1e-4 with decay 0.995 per 20-step epoch, greedy-rollout baseline. 3) Variants: 600 steps from the same P0, lr 3e-4, leave-one-out baseline over 8 samples; P2 T + 0.3 T_ref E/B0; P3 Lagrangian (dual step 0.1, cap 8); P4 hinge weight 3 + budget input. 3 seeds per map, Kaggle CPU 4 cores, about 41 to 53 min per P0 run and about 3 h per map; last checkpoint used. Curves (`presentation_final/assets/fig_training.png`, from `results_kaggle/loop/logs` and `train_*.json`): imitation cross-entropy 1.5 -> 0.26 for seeds 0 and 1, seed 2 stays at 1.6 (the weak seed); REINFORCE validation time about 300 -> 190-220 min (noisy, 64 instances); no variant separates from the control.

### Quick explainer: "sampling + budget filter" and "JIT" (slide 22)
* **Sampling + budget filter (P1):** at decision time sample N = 64 routes from the trained policy, simulate each, keep the fastest route with E <= B (else the lowest-energy route). No extra training. About -20% mission time.
* **JIT (just-in-time departure):** a UAV that expects to wait for the UGV delays take-off and waits on the ground (free) instead of hovering; it arrives as the UGV arrives, so mission time is unchanged and hover energy disappears (about -65% UAV energy). Assumes free ground waiting and a predictable UGV arrival.
* **Combined (4 maps x 3 seeds, 2U-1G, static roads):** base 228 min / 4.4 MJ / 78% over budget -> both 182 min / 1.5 MJ / 0% over budget.

### Environment parameters (slide 16; `implementation/src/edr/params.py`)
Workspace 20 x 20 km (aerial tasks within 7 km of the depot); UAV 10 m/s; UGV 4.5 m/s on roads only (divided by kappa on slow roads, blocked roads removed); UAV fuel F_a = 287.7 kJ (about 24 min, 14.5 km); power P(v) = 0.0461 v^3 - 0.5834 v^2 - 1.8761 v + 229.6 W (198.6 W cruise, 229.6 W hover); recharge T_R = 15 min (900 s, confirmed from the authors' code); failure penalty 800 min; trained on U15G5, zero-shot U45G15 and U75G25; teams 1U-1G, 2U-1G, 2U-2G, 4U-2G. Ours: 18 road events, budget B = beta B0, rendezvous mask with a 30 min wait limit, replan if a UAV waits 5 min longer. In the environment hovering counts in the mission energy E but not in the per-sortie battery F_a.

### Gazebo demo 4 (diversion) and the team demo
Border-map "cut road" case: UAV 1 visits checkpoints, flies back to the UGV (arrives with 31% battery), a tree blocks the UGV's road, the UAV hovers, the team replans, the UAV diverts to a new rendezvous (lands with 3%), the open-loop plan of the same case fails. Team demo: 4 UAVs + 2 UGVs on the ring road, tree blocks the road, team replans (open-loop 272 min vs replanned 166 min); trucks drive in separate lanes and do not collide with each other (without this the two trucks jammed at the depot).

### Multiple vehicles in the viewers (2026-11-23)
The 3D viewer and the 2D animation now draw several UGVs and UAVs: per-truck routes and colours, four UAV colours, a battery/state row per UAV, an adaptive legend, and team groups in the scenario list (`scripts/export_viewer_multi.py` appends the 4U-2G and 2U-2G ring-road cases to `viewer/missions.json`). Best 4U-2G case: blocked road, open-loop 272 min over budget vs replanning 166 min within budget.
