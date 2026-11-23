# Chat log

Running log of the working conversation (requests, decisions, outcomes). Newest at the bottom.

---

## 2026-11-20 — Session 1

**User request**
> Take all the context from the folder. Check `friend_presentation/` — a friend on the same base paper but a different problem; use a similar way to create a simulation to show. Implement `RM_IIT2024036_IIT2024081_IIT2024146 (1).pdf`; `p1.pdf` is the base — check properly.
> 1. Make a `context` folder, transfer all the initial files into it, update `context.md` and `chat.md` there.
> 2. Implement properly; don't get confused with my friend's implementation.
> Get the context first, properly.

**What I did first (context gathering)**
* Read all 19 pages of `p1.pdf` (base paper), all 21 slides of our deck, `main.tex`, identified `main (4).pdf` as an earlier compile of `main.tex`, `presentation.zip` as a copy of `friend_presentation/`.
* Read the friend's code (domain, mission simulator, road network, base map, route transformer, trainer, visualisation) and deck outline.
* Conclusion: friend's problem = learned virtual-slot UAV→UGV assignment on static roads; ours = mission energy budget + blocked/slow roads + event-triggered replanning with the base assignment frozen. Their road graph is a tree → cannot reroute → we need our own graph with loops.
* Environment check: no GPU, no torch, no networkx; numpy/scipy/matplotlib present. → venv + CPU torch.

**Decisions**
* Moved all initial files into `context/`; implementation goes to `implementation/` (outside `context/`), package name `edr` (Energy + Dynamic Roads) to keep it clearly apart from the friend's `uav_ugv`.
* Modelling assumptions recorded in `context.md` §6.
* Training is scaled-down CPU training with teacher warm start; results will be reported as reduced-scale, not as a reproduction of the paper's absolute numbers.

**Implementation session (same day)**

* Built `implementation/` (see `context.md` §8). Order: params/roads/scenario → events/env → tests (hand-checked case) → heuristic + controllers →
  Transformer policy + RL → evaluation → experiments → figures/animation → README/REPORT.
* Problems found and fixed on the way (worth remembering):
  * Road-aware mask livelock (UAVs shuttled between two landing sites when every rendezvous was "late") → prefer rendezvous from which work remains.
  * Imitation of the teacher failed with the paper's decoder + BatchNorm at this scale → LayerNorm + two dynamic geometry features.
  * `NeuralPolicy` cached encoder output by `id(scenario)` → stale embeddings after GC (found because P0 gave 228 vs 195 min on the same seeds) → identity check + regression test.
  * Greedy-baseline REINFORCE fine-tuning (200 steps) barely moved P2/P3/P4 → switched fine-tuning to a group baseline (8 samples/instance); P0+ control added for fairness.
  * Stress test failures looked absurdly high → events were placed so UAVs had already committed; added mid-round UAV diversion for replanning controllers.
* Honest outcomes: H3 and H4 supported; H1/H2 not supported at this scale; P1 route filter strongest; 1 training seed only. See README "Deviations".
* The user said "Try again" after the animation render; I re-inspected the frames, fixed the legend/label overlap, re-rendered (mp4 + gif) and finished the experiments.

**Maps + Kaggle session (2026-11-21)**

* User asked for better/multiple maps, the base paper's own map for a fair comparison, a border-surveillance map, all base papers downloaded, and Kaggle for heavy training.
* Downloaded into `context/base_papers/` (paper, authors' repo, 8 open-access related/method papers; paywalled ones listed in its README).
* Read the authors' code (read-only): confirmed road graph structure, `T_R` = 900 s, waiting semantics, units -> `paper` map implemented exactly; `border` map added (plus grid/radial/geo).
* Map study: `border` most event-sensitive, `geo` largest replanning gain, `radial` least informative. Not cherry-picking: all maps are reported; `paper` is for the Table I comparison, the others for dynamic-road claims.
* Kaggle CLI installed; already authenticated as `vichanshur`; packaged dataset + 4 kernels; smoke test passed locally. Nothing uploaded without the user's go-ahead.
* Local P0 training on the paper map for 1U-1G/2U-1G/2U-2G/4U-2G was still running at the time of writing.
* Subagent report (authors' repo facts) treated as data and checked against the code myself.

**Kaggle upload (2026-11-21, user approved "Upload and run")**
* Created private dataset `vichanshur/edr-code` (code zip, 68 KB) and pushed kernels `vichanshur/edr-border` and `vichanshur/edr-geo` (both RUNNING). `edr-paper` and `edr-loop` are packaged but not pushed yet (2 concurrent sessions).
* Check: `kaggle kernels status vichanshur/edr-border`; download when COMPLETE: `kaggle kernels output vichanshur/edr-border -p out/border` -> `results_bundle.zip`.

* All four kernels pushed and RUNNING (edr-border, edr-geo, edr-paper, edr-loop) - Kaggle accepted 4 concurrent.
* User reminder: the work is **base paper + the user's idea, not the friend's**. Verified: no friend code/concepts in `implementation/` (grep for sinkhorn / virtual slot / hungarian / uav_ugv / friend: clean). Clarified that the base-paper part is our own re-implementation guided by the paper and the authors' code, not the authors' code itself.

**Energy-trade-off session (2026-11-21, "continue")**
* New `scripts/tradeoff_study.py`: with hover waiting corr(T,E)=0.96 and the min-energy route is only 0.7 % slower / saves 0.9 % -> no trade-off. With landed waiting corr=0.39, min-E route +23 % time for -7.9 % energy.
* Added `--wait-mode landed`; retrained p0c/p2/p3_b90/p4/p4_nb under it (P0 reused). Stage A: only P1 (search) traces a Pareto curve; learned variants ~ control.
* Diagnosed that P4's budget input is ignored (weight 10x smaller; identical output for all beta). Added `p4x` (budget + pace feature, 10x lr): weights grow but behaviour is still beta-independent. Conclusion recorded in implementation/README.md.
* Kaggle kernels (border/geo/paper/loop) still running; local paper-map P0 training still running.

**Landed Kaggle pass (2026-11-21)**
* Added `--wait-mode landed` to kaggle_run/package scripts (variants incl. p4x), versioned dataset vichanshur/edr-code, pushed `edr-border-landed`. `edr-loop-landed` was rejected: free tier allows 5 concurrent CPU sessions (4 hover kernels + border-landed). geo/paper/loop landed kernels are packaged in `implementation/kaggle/kernels/` for later.
* Hover kernels started ~13:11 IST; expected ~6-8 h in total (estimate; the CLI shows no live log, the kernel web page does). Free Kaggle: 12 h CPU session limit; no accelerator used on purpose (workload is CPU-bound Python simulation).

**Scope reset + second Kaggle account (2026-11-21)**
* User: do required deck items first, extras later; deadline tomorrow EOD. Required still open: M1 (base policy 11-19 % off Table I on the paper map), M2 (lambda converges), Stage C (larger sizes), 3 seeds (Kaggle).
* Started locally: `train.py p0more` (continue P0 on paper map, 4 teams, 1500 steps) and P3 with feasible budgets (p3_b100, p3_b95) for the lambda-convergence demo. Added `--small` / zero-shot model lookup to experiments.py for Stage C.
* Built `implementation/kaggle_second_account/` (dataset zip, 8 kernels, README with website-only and CLI-second-account steps). Kaggle CLI supports one login at a time; switch via KAGGLE_CONFIG_DIR.

**Demo for the professor (2026-11-21)**
* User: the simulation must be shown to the professor. Viewer extended to 9 scenarios on 3 maps (ring-road, **border surveillance** with border line/strip/outposts, **base paper's map**), auto-framed camera, honesty note (scenarios are best-of-40), offline build `viewer/index_offline.html`, demo script `viewer/DEMO.md`. Artifact republished (v2, same private link).

* User asked for bad examples too, not only best ones. Viewer now has 18 scenarios: per setting a **best / typical / worst** case (picked from 40 test instances by the gain of replanning) plus a **cut-road stress** case per map with failed missions shown (stranded UAV, "MISSION FAILED" badge). Worst cases: replanning +103 min (paper map), +48 min (border), +11 min (ring-road). Typical cases often unchanged. Stress: open-loop fails 32/32, replanning 29/32 (ring-road).

**Required items progress (2026-11-21 ~15:30)**
* M1: continued P0 on the paper map (+1500 steps): 228 / 210 / 185 / 219 min vs paper greedy 238 / 230 / 181 / 189 (within ~10 % for 3 of 4 teams; 4U-2G +16 %).
* M2: lambda did NOT converge even at beta=1.0 (sampled-batch violation stays > 0: sampled routes cost more energy than the greedy B0 reference). Fix tried: lambda driven by the greedy validation violation (variants p3g_b100/b95/b90, 800 steps) - running.
* Stage C: zero-shot U45G15 running (A then small B); U75G25 next.
* Kaggle kernels still running (hover x4 + border-landed).

**Kaggle results collected (2026-11-22)**
* All 5 kernels COMPLETE (each ~2.9 h, all 30 jobs rc=0; faster than the 6-8 h estimate). Downloaded to `implementation/results_kaggle/<map>/` (kept separate from local results so the loop kernel does not overwrite local seed-0 models). Aggregated with `scripts/aggregate_seeds.py <tag> <root>` -> `REPORT_SEEDS_<tag>.md`.
* Findings (3 seeds x 4 maps): replanning beats open loop in 12/12 (map,seed) combos; event-triggered = periodic (|dT| <= 0.3 min) with 1-3 fewer replans; P3/P4 not distinguishable from control; P1 strongest; seed 2 gave a weak base on loop/geo. Landed waiting (border): P1 saves ~13 % energy at no time cost.
* M2: lambda driven by greedy validation violation stabilises at beta 1.0/0.95, saturates at 0.9 -> partly met. M1: P0 within ~10 % of paper greedy for 3 of 4 teams on the paper map. Stage C U45G15 done (U75G25 not).

**Erratum (2026-11-22):** earlier entries/messages quote a "+103 min" worst case on the paper's map. That came from viewer data exported at 15:09, before the improved paper-map base policy (finished 15:22). Re-exported with the current model: the paper-map worst case is **+9 min** (193 -> 202); ring-road (+11) and border (+48) worst cases are unchanged. Greedy decoding is deterministic across 1/2/4/8 threads (checked). DEMO.md corrected.

**New presentation (2026-11-22)**
* User asked for a new end-sem deck: story flow problem -> intuition -> literature -> pre-mid-sem work (2-3 summary slides) -> its limitations -> post-mid-sem work -> method/experiments/results, IEEE citations, flow diagrams, text-light, same white format as the pre-mid-sem deck.
* Built `presentation_final/` (33 slides): `Presentation.tex` (preamble copied verbatim from `context/main.tex`), `Presentation.pdf`, `Presentation_overleaf.zip` (tex + 12 assets), `make_figs.py` (all result figures computed from real experiment files), `build.py` (orders the bibliography by first citation, compiles).
* Local compile needed Beamer/TikZ, which the system TeX lacked: unpacked the Ubuntu texlive-latex-recommended/pictures/extra/fonts-recommended debs into `presentation_final/.texlocal` (no sudo, system untouched); tectonic was tried and removed (package server unreachable).
* To verify before presenting: reference [11] (Gazebo, Koenig & Howard 2004) and the page ranges were written from memory/base-paper reference lists, not fetched.

**Closing the six open items (2026-11-22, user: "implement this", then "train on Kaggle, not local; use accelerator if needed")**
* Item 2 (other teams) -> Kaggle kernels `edr-loop-1u1g`, `edr-loop-2u2g`, `edr-loop-4u2g` (3 seeds each). Item 3 (4U-2G gap) -> `edr-paper-4u2g-long` (3000 REINFORCE steps) + local sampled-decoding evaluation (`scripts/sampling_eval.py`; ours best-of-1024 on paper map 1U-1G = 199.5 min vs paper DRL(1024) 220.4).
* Items 4/5 (lambda at 0.9; candidates A/B) -> new kernel `edr-loop-bon`: search distillation (`expert_iteration` in rl.py: sample N routes, imitate the best under the budget-aware cost; variants bon_t / bon_nb / bon_p4x) and high-cap lambda (p3h_b90/b100, 1500 steps), starting from the 3 trained base policies uploaded as dataset `vichanshur/edr-p0-loop`.
* Item 6 (GLS/TS/SA) -> `src/edr/bilevel.py` (min set cover + stop-order TSP + OR-Tools capacity-constrained sortie routing with GLS/TS/SA, executed in our simulator), `scripts/bilevel_eval.py`; evaluation running locally.
* Item 1 (Stage C) -> running locally (evaluation only) with the Kaggle-trained loop models: U45G15 and U75G25, 3 seeds, 50 instances -> `results_stageC/`.
* Accelerator: not used on purpose. Profiling: time is the serial Python simulator, not the network; GPU kernels allow only 2 concurrent sessions. Training was moved off the laptop (stopped local bon runs).
* Kaggle: 5 CPU kernels running (free tier limit 5 concurrent).

**All five Kaggle kernels finished (2026-11-22 ~18:20)** (every job rc=0; 2.1-4.7 h each). Results: `implementation/results_kaggle/{loop-1u1g,loop-2u2g,loop-4u2g,paper-4u2g-long,loop-bon}`; Stage C `results_stageC/`; baselines `results/bilevel_eval_paper.csv`; sampling `results/sampling_eval_1024.csv`. Summary in the README section "Closing the loose ends". Negative results kept: distillation did not help; 4U-2G gap not closed; one seed fails on larger instances. Presentation now 37 slides (+4); Overleaf zip refreshed. Leftover download folders in `implementation/out/` were NOT deleted (the safety check blocked the removal; harmless, user can delete).

**The mid-sem idea ("more time saves a lot of energy") - answered (2026-11-22 evening).** User asked whether it was achieved. Honest answer: not at the route level (H1 unsupported: +0.7 % time / -0.9 % energy under hover; +23 % / -7.9 % with free waiting; energy/km is lowest near 15 m/s so slowing down does not help). Found and tested a scheduling lever: just-in-time departure (`env.jit`, `scripts/jit_study.py`, `results/jit_study.csv`): -53..-74 % energy at +0.00 min on 4 maps x 3 seeds; budget 0.9 violations 75-100 % -> ~0 %. Caveats: free ground waiting, UGV is the bottleneck, delays must be predictable (events leave 11-36 % hovering). Presentation now 39 slides (JIT x2, results reordered, hypotheses H1 now partial, conclusion updated).

**Final deck for the end-sem evaluation (2026-11-22).** Presentation.pdf is now 40 slides: title "End-Semester Evaluation", results in a logical order (incl. just-in-time departure, other team sizes, Stage C, baselines/sampling, two fixes that did not work, simulation, hypotheses), new "Planned versus delivered" slide (M1 partly, M2 partly, M3/M4 done, Gazebo not done), limitations updated. Corrected a claim: the weak seed is a bad base policy (loops in 5/30 instances even at 15 tasks), not a size-generalisation failure. Reference [11] verified via Crossref (Koenig & Howard, IROS 2004, vol 3, pp 2149-2154, DOI 10.1109/IROS.2004.1389727). Gazebo skipped by the user.

**Leftovers closed (2026-11-22 night).** Implemented the loop-detection fallback (`stalled_rounds`, `NeuralPolicy(fallback=True)`; 35 tests pass): identical on good seeds, bad seed 0 failures (892 -> 302 min mean at 15 tasks) but still ~2x slower; seed 1 at 75 tasks keeps 1 failure. JIT extended to 4 team sizes and to the sampling policy (`results/jit_teams.csv`): -18/-54/-40/-69 % energy at +0 min with events; P1+JIT -46 % energy. 10,240-sample kernel `edr-sampling10240` failed twice because Kaggle auto-extracts the single-zip dataset (entry.py looked for edr_code.zip); fixed (copy the extracted dir) and re-pushed as version 3, running. Deck now 41 slides (new JIT-by-team-size slide). Gazebo still skipped.

**10,240-sample decoding finished (2026-11-22).** Kaggle kernel COMPLETE (50 instances, paper map): ours best-of-10240 = 197.5 / 157.2 / 135.3 / 135.2 min (1U-1G/2U-1G/2U-2G/4U-2G) vs paper 218.6 / 199.7 / 154.4 / 159.9; gain over 1024 samples only 1-4%. Saved `implementation/results/sampling_eval_10240.csv`; README and deck (Result 9) updated. Git: repo initialised and pushed to sahsudhanshu/RM main (commit ce5bc22; .venv, .texlocal, checkpoints and raw stage CSVs ignored).

**Night session (2026-11-22, deadline 5 am).** (1) Gazebo test bed on the friend's Ubuntu 24.04 laptop (RTX 3050) over SSH (key login, LAN): `implementation/gazebo/` (plan exporter, SDF world generator, headless lock-step replay with wheeled UGV physics, MP4 recording, real-terrain heightmap from open SRTM-derived tiles because Google Earth data cannot be downloaded programmatically). Flat-ground replays match the plans to within ~0.2 % in time. (2) P1 + JIT across 4 maps x 3 seeds: -20 % time, -65 % energy, over-budget 78 % -> 0 % (static). (3) JIT re-prediction at takeoff: hover energy 241 -> 80 kJ. (4) Adaptive sampling: neutral (matches fixed-64); samples-quality curve recorded. (5) Kaggle kernel `edr-4u2g-more` launched (3000 more REINFORCE steps for the 4U-2G base policy, 3 seeds); included only if it finishes before the deadline. Viewer JIT mode was not implemented (would need re-timed legs).

**Late night 2026-11-23.** Kaggle `edr-4u2g-more` finished: extra 3000 steps do not close the 4U-2G gap (mean greedy 210.8 -> 209.7 min vs paper 188.9). Friend laptop moved off the network (new subnet 10.215.19.x, host not reachable); Gazebo results used: 12 runs (9 flat, 3 terrain), mean time error 0.13 % / 0.21 %. 4 border reruns not collected.

**Gazebo finished (2026-11-23 ~04:50).** All 22 replays done (16 flat, 6 terrain): mean |dT| 0.12 % / 0.19 %, dE +0.17 % / +0.28 %, UGV arrival delay 11 / 16 s. Earlier hang cause: Gazebo transport bound to the old Wi-Fi address after the network changed; fixed by GZ_IP=127.0.0.1. Deck 44 slides, text fixes (Gazebo, 35 tests).

**Deck cut to 25 slides (2026-11-23).** Removed the roadmap, energy timeline, base-paper flow, 2 of 3 mid-sem summaries, plan-followed, simulator, candidates-vs-control, JIT-by-map/team sizes/larger instances, two failed fixes, simulation-we-can-show, planned-vs-delivered; merged the two Gazebo slides and conclusion+limitations; removed the 'Thank you' line; references on one slide. Full 44-slide version kept as `presentation_final/Presentation_full_44slides.pdf` and `body_full_44slides.tex`.

**Runnable 3D Gazebo demo (2026-11-23).** User asked for a runnable 3D Gazebo implementation, no experiments. Built `implementation/gazebo/demo/` (scaled 1:50 / x60 world, detailed UGV with roof landing pad and quadrotor UAVs with spinning rotors that ride the UGV while charging, task markers, road-event cones, optional real terrain, real-time driver, camera follow) and the package `gazebo/gazebo_demo.zip`. Verified on the friend's Ubuntu laptop: headless recordings, the real Gazebo window in a virtual display (screen-captured), and a clean unzip run. Not checked on a physical screen. Preview videos in `implementation/gazebo/demo_preview_*.mp4`.

**Presentation script (2026-11-23).** User asked to divide the presentation and demo into 3 parts with a script and a proper explanation of every slide, and to update `context/`. Wrote `context/presentation_script.md` (3 speakers: slides 1-8 / 9-16 / 17-25; Demo 1 after slide 4 = 3D viewer 'the problem', Demo 2 after slide 13 = viewer replanning best/worst/cut-road, Demo 3 after slide 22 = Gazebo videos; per-slide on-screen/teach/say/numbers, glossary, 15-question Q&A, time plan, 20-min cut list, delivery tips). Updated `context.md` (new section 10, file table) and `checkpoint.md`.

**Future-plan slide (2026-11-23).** Added slide 25 'Future Plan: Improving the Method and Proving the Hypotheses' before the references (deck now 26 slides): per open point (H1 partly, H2 rejected, 4U-2G gap, policy not in Gazebo, cut bridge) what to implement and a success test. Script, context and checkpoint updated for the new numbering.
