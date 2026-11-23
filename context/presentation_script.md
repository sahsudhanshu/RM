# End-Semester Presentation: Script, Teaching Notes and Demo Plan

Deck: `presentation_final/Presentation.pdf` (26 slides). Full 44-slide version for backup: `presentation_final/Presentation_full_44slides.pdf`.
Three presenters (the three authors on the title slide). Assign names as you like: **Speaker 1, Speaker 2, Speaker 3**.

| Part | Speaker | Slides | Theme | Demo | Time |
|---|---|---|---|---|---|
| 1 | Speaker 1 | 1 to 8 | The problem, the intuition, the literature and our questions | **Demo 1** (3D viewer: "the problem") after slide 4 | about 10 min |
| 2 | Speaker 2 | 9 to 16 | Before mid-sem, its limits, and what we built | **Demo 2** (3D viewer: "replanning at work, honestly") after slide 13 | about 11 min |
| 3 | Speaker 3 | 17 to 26 | The results, Gazebo, conclusion, future plan | **Demo 3** (Gazebo videos) after slide 22 | about 13 min |

Total about 34 min including demos. **If you only have 20 min**, use the cut list in section 6.

---

## 0. Before you start (checklist)

1. Open `implementation/viewer/index_offline.html` in Chrome or Firefox **before** the session (double-click; works offline). Press Space once to check that it plays. Keep it in a second tab.
2. Copy the Gazebo videos to the presenting laptop and test they play:
   `implementation/gazebo/demo_preview_flat.mp4`, `demo_preview_terrain.mp4`, `demo_preview_gazebo_window.mp4`
   and the experiment replays `implementation/gazebo/videos/border-block-medium-middle-best_case__event.mp4`, `videos_terrain/border-block-medium-middle-typical_case__event.mp4`.
3. Backup if the browser misbehaves: `implementation/results/U15G5_2U1G/media/mission_slow_high_early.mp4` (2D video of the same scenario).
4. Have the PDF open full screen, and `Presentation_full_44slides.pdf` ready for questions about removed slides.
5. Practise the three hand-overs (end of each part) so the transition takes 5 seconds.

### Viewer controls (for the demos)
Scenario list (top), **3D / Top** view, **Play/Pause** (Space), **Restart**, **Speed** slider (6 to 60 min per second), click the chart to jump in time. Left panel = open-loop (never replans), right panel = event-triggered replanning. Both panels run the **same instance and the same road event**; only the controller differs. Amber road = slow, red = blocked. UAV height is exaggerated for readability.

---

## 1. The whole story in five sentences (memorise this)

1. A team of fast UAVs with small batteries and one slow UGV charging truck must visit all task points as fast as possible, and the UGV is the bottleneck.
2. The base paper solves this with a learned policy, but assumes fixed roads and has no energy budget.
3. We add an energy budget and **roads that change during the mission** (blocked or slow), and we replan only when an event actually breaks the plan.
4. Road events cost up to +29% energy; **event-triggered replanning** recovers it with few replans; the energy-aware learning methods we proposed did not beat their control, but we found out why and found a lever that works: **just-in-time departure** (the UAV waits on the ground instead of hovering), which cuts UAV energy by 53 to 74% at no cost in time, and together with **sampling plus a budget filter** gives about 20% less time and 65% less energy with the budget met.
5. We checked the plans in **Gazebo** physics (22 replays, time within 0.14%), and we are honest about what we did not do (the learned policy is not run inside Gazebo; the learned energy-aware methods did not help).

## 2. Glossary (teach these words early)

| Term | Meaning |
|---|---|
| UAV / UGV | Flying drone / ground vehicle (the charging truck) |
| Task point | A location a UAV must visit (aerial task) |
| Ground point / rendezvous | A road location where a UAV can land on the UGV to be recharged |
| Sortie | One flight of a UAV from take-off (full battery) to landing on the UGV |
| T | Mission time: when the **last** vehicle finishes |
| E | Mission energy: sum of UAV flight energy plus **hover** energy while waiting in the air |
| B, β, B0 | Energy budget B = β·B0, where B0 is the energy the base policy uses on the same instance without events; β = 0.9 means "10% less than the base" |
| T_R | UAV recharge time at the UGV, 15 min (900 s, confirmed from the authors' code) |
| Open-loop | Plan once, never replan |
| Periodic | Replan after every recharge |
| Event-triggered | Replan only if the plan is broken by a road event |
| P0 | Base policy (imitation, then REINFORCE on mission time) |
| P1 | At decision time, sample N routes, keep the fastest one that fits the budget |
| A / B | Our two candidate methods: A = Lagrangian penalty, B = budget input to the decoder |
| JIT | Just-in-time departure: a UAV that expects to wait delays take-off and waits on the ground |
| Team names | "2U-1G" = 2 UAVs and 1 UGV, etc. |

Physical facts to quote: UAV battery 287.7 kJ (about 24 min of flight, about 14.5 km); UAV speed 10 m/s, power 198.6 W cruising and 229.6 W hovering; UGV speed 4.5 m/s; recharge 15 min.

---

# PART 1: Problem, intuition, literature, questions (Speaker 1, slides 1 to 8)

## Slide 1: Title (30 s)
**On screen:** "Energy-Constrained UAV-UGV Replanning under Dynamic Road Conditions", End-Semester Evaluation, authors.
**Say:** "Good morning. We are presenting our end-semester work on energy-constrained UAV-UGV replanning when the roads change during a mission. I will cover the problem and the literature, my teammate will explain what we built, and the third will present the results and the Gazebo check. We have three short demos: two in our 3D viewer and one in Gazebo."

## Slide 2: Problem: a team of UAVs and a UGV charger (1 min)
**On screen:** Fig. 1 of the base paper: depot, UAV 1 and 2, UGV 1 and 2, ground points, aerial points, rendezvous points; three boxes on the right.
**Teach:**
- UAVs are **fast but battery-limited**: about 24 minutes of flight. They cannot finish the mission on one charge.
- The UGV is a **mobile charging station**. It is slow, stays on roads, and charges **one UAV at a time**.
- So a UAV flies a **sortie**: take off, visit some task points, land on the UGV at a rendezvous point on the road, get recharged, repeat.
- The goal: **visit every task point as fast as possible**.
**Say:** "Imagine inspecting a large area. Drones are fast but can fly only about 24 minutes. A ground vehicle can carry chargers but is slow and must follow roads. In the picture, the red circles are points only a drone can visit, the blue dots are points on the road where a drone can land on the truck, and the black circles are the actual rendezvous. The goal is simple: visit every point as fast as possible. This setting is from the base paper by Mondal et al." (cite [1])
**Numbers:** 24 min flight; one UAV charged at a time.
**If asked "why not more chargers?":** the problem as defined in the base paper has one charging truck per team; more UGVs are covered by the 2U-2G and 4U-2G teams.

## Slide 3: Intuition: one sortie, step by step (1 min)
**On screen:** flowchart: Take off full battery, Visit task points (battery check), Battery low or all done?, Fly to rendezvous and land on UGV, Hover until UGV arrives, Recharge T_R = 15 min, Tasks left?, Mission done.
**Teach (read the chart clockwise):**
- The UAV keeps visiting points until the battery is low or everything is done.
- Then it must fly to a rendezvous the UGV can reach and **land**. If the UAV arrives before the UGV, it **hovers** in the air and burns energy: this "hover until UGV arrives" box is the heart of the whole project.
- Recharge takes 15 minutes. If tasks are left, a new sortie starts.
**Say:** "Here is one sortie. The drone visits points until its battery is low, then flies to a meeting point and lands on the truck. The truck is slow: it drives 4.5 metres per second while the drone flies 10. So the drone often arrives first and has to hover, burning energy, until the truck shows up. The truck is the bottleneck of the whole system."
**Numbers:** UGV 4.5 m/s, UAV 10 m/s, T_R = 15 min.

## Slide 4: Why it gets harder: roads change during the mission (1 min)
**On screen:** causal chain: road blocked or slowed, UGV detours and arrives late, UAV hovers longer, extra energy used, over budget or missed recharge. Below: a hand-checked case: direct road 2.7 km, 10 min; detour 8.1 km, 30 min.
**Teach:**
- In the base paper the roads are **fixed**. In reality a road can be blocked or slowed during the mission.
- A longer UGV drive makes the UAV hover longer. In our hand-checked example the UGV's trip goes from 10 to 30 minutes, so the UAV hovers **19 more minutes**, which costs **262 kJ** of extra energy (19 min times 13.8 kJ per minute of hovering). That is almost a full battery.
- So a road problem becomes an **energy** problem.
**Say:** "Now the part the base paper does not cover. Suppose the direct road to the meeting point is blocked. The truck detours: 8 kilometres and 30 minutes instead of 2.7 kilometres and 10. The drone that was waiting 5 minutes now hovers for 24. In our hand-checked example that wastes 262 kilojoules, about a whole battery. So we need a planner that notices road changes, and an energy budget so that the mission stays efficient."
**Transition to Demo 1:** "Let me show you what this looks like."

## DEMO 1 (about 1.5 min): "The problem" in the 3D viewer (Speaker 1)
1. In the viewer choose **Ring-road map, slow edges, high, early, best case**. Use **3D** view.
2. Press Play, set speed to 12.
3. **Say:** "Two drones and one truck. Both panels have the same road event: early in the mission a stretch of road becomes slow, shown in amber. **Look at the left panel only**: the plan is never revisited. The truck crawls through the slow road, the drones arrive and hover, and the energy bar climbs towards and past the budget line. On the right, the same event is handled by our replanning, which my teammate will explain."
4. Do **not** dwell on the right panel; say "we will come back to it in Part 2".
5. Honest note if asked: "This is a best case chosen to be visible; typical cases often show no difference because the event does not touch that mission."

## Slide 5: Problem statement (1 min)
**On screen:** Inputs (task points, fleet, road graph G_t, budget B), Planner (learned policy), Outputs (UAV sorties, UGV routes, rendezvous schedule); road events feed back; formulation box; constraint chips; energy formula.
**Teach:**
- **Minimise the mission time T** (when the last vehicle finishes), **subject to** energy E ≤ B = β·B0.
- B0 = the energy the base policy uses on the same instance with no events. β ∈ {0.7, 0.8, 0.9, 1.0}, so β = 0.9 asks the planner to use 10% less energy than the base.
- E = flight energy + hover energy: E = Σ(P_fly·t_fly + P_hover·t_wait), with the power model of the base paper (198.6 W at 10 m/s, 229.6 W hovering).
- Constraints: every task visited; every sortie within battery; the rendezvous must be on a reachable road node; one UAV charged per UGV at a time.
- The planner must **replan when the road graph G_t changes**.
**Say:** "Formally: minimise mission time, subject to an energy budget. The budget is set relative to what the base policy would use, so 0.9 means ten percent less. Energy is the flight energy plus hover energy while waiting. Inputs are the task points, the fleet, the road graph and the budget. The road graph can change during the mission, which is why the planner must be able to replan."
**If asked "why a budget at all?":** because without it the planner is free to waste energy hovering; the budget makes the energy cost explicit.

## Slide 6: Literature: how the field got here (1 min 15 s)
**On screen:** three lanes: Optimisation (Maini 2019, Mondal 2023), Learning-based routing (Fan 2023, Mondal 2025 = base paper), Energy and uncertainty (Shi 2022, Lin 2022, Mondal 2025 risk-aware).
**Teach (left to right = later work):**
- **Optimisation lane:** Maini et al. use MILP and heuristics for 1 UAV + 1 UGV; Mondal et al. 2023 do bilevel planning ("UGV route first, then UAV") for large coverage.
- **Learning lane:** Fan et al. use deep RL for one UAV with many charging stations; the **base paper [1]** (Mondal et al. 2025) uses a Transformer-based DRL for multiple UAVs and a UGV, and beats the bilevel heuristics.
- **Uncertainty lane:** chance-constrained rendezvous, robust surveillance with charging stations, risk-aware attention RL with uncertain fuel.
**Say:** "Early work solved this with optimisation: mixed-integer programs and bilevel heuristics, which plan the truck first and the drones second. Later work uses deep reinforcement learning. Our base paper is the transformer-based learned policy for several drones and a truck, which outperforms the bilevel heuristics. A third line looks at uncertainty, mostly in fuel and obstacles."
**Numbers:** none needed.

## Slide 7: Research gap (45 s)
**On screen:** comparison table of five works over four columns: learned policy, multi UAV-UGV, mission energy budget, roads change during mission.
**Teach:** read the last two columns. The base paper handles learned multi-UAV-UGV planning but has **no energy budget** and **fixed roads**. Others cover fuel risk or obstacles but are not learned multi-UAV-UGV planners. **This work** is the only row with all four ticks.
**Say:** "Putting these together: no reviewed work combines a learned multi-drone-and-truck policy, a mission energy budget, and roads that change during the mission. That is our gap."
**Caution:** say "among the works we reviewed", not "nobody has done this".

## Slide 8: From the gap to our questions (1 min), then HAND OVER
**On screen:** gap box leads to RQ1 to RQ4, each with a hypothesis H1 to H4.
**Teach:**
- **RQ1 Is there a time-energy trade-off?** H1: energy falls while time barely rises ("a little more time saves a lot of energy", the idea from the mid-sem).
- **RQ2 Can we stay within a budget?** H2: a constrained method beats a fixed penalty.
- **RQ3 What do blockages do?** H3: more disruption means more energy.
- **RQ4 Event-triggered vs periodic replanning?** H4: same reliability, fewer replans.
- We will come back to these four hypotheses in the scorecard slide (23); two are confirmed, one is rejected and one is partly true.
**Say:** "From the gap we ask four questions with four hypotheses. First, is there a time-energy trade-off, that is, can we spend a little more time to save a lot of energy? Second, can a constrained method keep us within the budget better than a fixed penalty? Third, how much do blockages cost? Fourth, can we replan only when needed, instead of regularly? I will hand over to my teammate, who will show what we did before mid-sem and what we built since."

---

# PART 2: What we built (Speaker 2, slides 9 to 16)

## Slide 9: Before mid-sem: what we did (45 s)
**On screen:** flow: studied the base paper, found its limits, formalised min T s.t. E ≤ B, road/energy/budget models, candidate methods A and B, replanning design; below: experimental plan (policies P0 to P4, stages A, B, C) and the Gazebo testbed started (desert terrain, 4-wheel UGV).
**Teach:** before mid-sem was a **proposal and design phase with no learned results**. We defined the models, proposed the two candidates, and designed the replanning and experiments.
**Say:** "Before mid-sem we did the design: we studied the base paper, found its limits, formalised the problem, defined the road, energy and budget models, proposed two candidate methods, and designed event-triggered replanning. We had no learned results yet; the energy-saving idea was only illustrated."

## Slide 10: Before mid-sem: the proposed framework (1 min 15 s)
**On screen:** pipeline: instance (x_i, y_i, b_i), Encoder (unchanged), Route decoder (+ budget input in B), Greedy assignment (frozen), Simulator (T, E), Road layer (events on G_t), Check reach and time, Updated action masks, replan; two amber boxes for A and B; legend: kept from base / new / candidate.
**Teach:**
- **Kept from the base paper (grey):** the Transformer encoder, the greedy UAV-to-UGV assignment (frozen), and the simulator that returns T and E.
- **New (teal):** a **road layer** that updates the road graph when events happen, then **checks reachability and time** and updates the **action masks** so the policy cannot choose a rendezvous that is unreachable or too late; this loops back as **replan**.
- **Candidate A (Lagrangian):** train with cost T + λ·max(0, E − B): penalise energy only above the budget, with λ learned.
- **Candidate B (budget-conditioned):** give the decoder the remaining budget b_t = (B − E_t)/B as an input so it can adapt to the budget.
**Say:** "Our framework keeps the base paper's encoder, its greedy assignment and the simulator. We add a road layer that updates the road graph when an event happens, and masks that remove choices which are no longer reachable or on time. We proposed two ways to make the policy energy-aware: A adds a Lagrangian penalty for exceeding the budget; B feeds the remaining budget to the decoder."
**Remember:** these are the two methods that later turn out to be no better than the control (slide 18, 23).

## Slide 11: Challenges and limits before mid-sem and what we did about them (1 min)
**On screen:** six rows, before mid-sem (red) leading to after mid-sem (teal).
**Teach each row (one sentence):**
1. No implementation, only a plan, so a **simulator and learned policy with 35 tests**.
2. Base method not verified, so **we compared with the paper on its own map**.
3. Unknown details (recharge time T_R and the road map), so **confirmed from the authors' code**: T_R = 900 s, 28 road points in 3 chains.
4. Only tree roads (no detours), so **maps with loops** (ring road, border, district) where a blockage has a detour.
5. Energy gain only illustrated, so **measured on 4 maps × 3 seeds × 100 instances**.
6. Physics not validated, so **a 3D replay and a Gazebo replay of the plans**.
**Say:** "These were the limits of mid-sem and what we did about each. We built a tested simulator, verified it against the base paper, confirmed the unknown details from the authors' code, added maps with loops so blockages have detours, ran real measurements, and added a 3D viewer and a Gazebo replay."

## Slide 12: Step 1: Reproduce the base on its own map (1 min)
**On screen:** bar chart of mission time per team (1U-1G, 2U-1G, 2U-2G, 4U-2G): paper greedy DRL, paper 10,240 samples, ours greedy; checklist: authors' code, 28 road points / 3 chains, T_R = 900 s, our simulator and policy.
**Teach:**
- To trust any later result we first **reproduce the base paper** on its own road map (rebuilt from the authors' code).
- Our trained greedy policy is **within about 10%** of the paper's greedy DRL for three of the four teams (1U-1G 228 vs 238 min, 2U-1G 210 vs 230, 2U-2G 185 vs 181).
- **4U-2G is 9 to 14% slower** than the paper, even after 3000 more training steps, over 3 seeds. We say this plainly; sampling closes it later (slide 21).
**Say:** "Step one is to check we can reproduce the base paper. On its own map, our greedy policy is within about ten percent for three of four team sizes. For four drones and two trucks we are still nine to fourteen percent slower, and more training did not fix it. We report this honestly."
**If asked why the gap:** the 4U-2G policy has more coordination to learn and we trained on free CPUs for a few hours; the paper's training is much larger.

## Slide 13: Step 2: Event-triggered replanning (1 min 15 s)
**On screen:** left flowchart: road event observed, update road graph, re-time the remaining plan, plan still valid? (yes: keep; no: replan remaining tasks, keeping the completed route); right table: three controllers; note about UAV diversion.
**Teach:**
- When a road event is observed, update the road graph and **re-time the remaining plan** on the new roads.
- The plan is **invalid** if (i) a rendezvous is unreachable, (ii) the energy would exceed the budget, or (iii) a UAV would wait more than 5 minutes longer than planned.
- Only then **replan the remaining tasks**, keeping the completed route.
- Three controllers: **open-loop** never replans; **periodic** replans after every recharge; **event-triggered** replans only when the plan breaks.
- A UAV whose rendezvous becomes unreachable can **divert** using its remaining battery.
**Say:** "Here is the replanning logic. When a road event is observed we update the road graph and re-time the rest of the plan. If every rendezvous is still reachable, the budget is met and no UAV waits more than five minutes longer than planned, we keep the plan. Otherwise we replan only the remaining tasks and keep what is already done. We compare three controllers: never replan, replan every recharge, and replan only when needed."

## DEMO 2 (about 2 min): "Replanning at work, honestly" in the 3D viewer (Speaker 2)
1. **Best case again:** Ring-road, slow edges, high, early, best case. Play at speed 12. **Say:** "Same instance as before. Left never replans. Right notices the slow road, checks every planned rendezvous, and replans: the energy bar stays under the budget line and the mission finishes about 87 minutes earlier."
2. **Honest worst case:** choose **Border surveillance, block edges, medium, middle, worst case**. **Say:** "Replanning is not always better. Here replanning made this instance about 48 minutes slower than open-loop. Averaged over the test instances it still wins clearly, and we report cases like this one instead of hiding them."
3. **Cut road:** choose **cut road (stress test)** on the border or ring-road map. **Say:** "If a bridge is cut and the map is disconnected, open-loop fails and the mission is never completed. Replanning rescues some missions, but not all, because some UAVs have already committed their last battery."
4. Switch to **Top** view briefly to show the border strip and roads if time allows.

## Slide 14: Step 3: Learning the policies (1 min 15 s)
**On screen:** pipeline: clone the rule policy (imitation), REINFORCE on time = P0, fan-out to P0+ control, P2 fixed penalty, P3 (A) Lagrangian, P4 (B) budget input; dashed to P1 (at inference): sample N routes, keep the fastest with E ≤ B; caption on Transformer, attention model and REINFORCE.
**Teach:**
- **P0 (the base):** first **imitate a simple rule policy** (nearest feasible), then **REINFORCE** to minimise mission time.
- Every variant **starts from the same P0**, so differences come from the method, not from luck.
- **P0+ (control):** P0 trained for the same extra steps with no energy term; this is the fair comparison for A and B.
- **P2:** fixed penalty T + α·E. **P3 = A:** Lagrangian. **P4 = B:** budget input.
- **P1:** needs no training: at decision time **sample N routes from P0, keep the fastest that fits the budget**.
**Say:** "We train a base policy by imitating a rule and then using REINFORCE on mission time. Every other variant starts from this same policy, so differences come from the method. The control is the same policy trained for the same extra steps without any energy term. P1 is different: it needs no training; at decision time it samples many routes and keeps the fastest one that fits the budget."
**Why the control matters:** a method counts as a success only if it beats P0+, not just P0.

## Slide 15: Step 4: Experimental design (45 s)
**On screen:** six maps (base paper's tree, ring road, city grid, radial town, irregular district, border surveillance); chips: 4 maps and 4 team sizes, 3 training seeds, 100 test instances, 18 road events, Kaggle CPU about 3 h per map; the 18 events = {blocked, slow} × {low, medium, high} × {early, middle, late}.
**Teach:**
- Squares are ground points, circles are aerial tasks; the shaded strip and dashed line in the border map is the surveillance zone.
- **18 road events** = type (blocked or slow) × severity (low, medium, high) × timing (early, middle, late).
- **3 seeds**: each is a separate training run; confidence intervals are over seeds. All heavy training ran on **Kaggle CPUs** (about 3 hours per map), evaluation locally.
- Four maps are used for the main results: ring road, border, district (geo) and the paper's map.
**Say:** "We evaluate on several maps, including the base paper's map and a border-surveillance map. Every setting uses 100 test instances and 18 kinds of road events: blocked or slow, three severities, three timings. We train three seeds per map to see how stable the results are. Training ran on free Kaggle CPUs."

## Slide 16: Result 1: Road events really hurt (45 s), then HAND OVER
**On screen:** bars per map: increase of mission time and UAV energy versus static roads for an open-loop mission (rule policy, 18 events).
**Teach (read the bars):**
- Radial: +4% time, +6% energy. Paper map: +6%, +8%. Grid: +7%, +9%. District: +12%, +15%. Ring road: +13%, +19%. **Border: +22% time, +29% energy**.
- **Energy grows faster than time** (hovering).
- Maps with many detours (radial) suffer least; the border map, where roads are few and long, suffers most. This answers **H3: more disruption means more energy**.
**Say:** "First result: do road events really matter? Yes. On every map an open-loop mission takes longer and uses more energy, and energy grows faster than time. On the border map the loss is 22 percent in time and 29 percent in energy; on the radial map, full of detours, it is only 4 and 6. This confirms our third hypothesis. Next, my teammate will show how much replanning recovers."

---

# PART 3: Results, Gazebo, conclusion, future plan (Speaker 3, slides 17 to 26)

## Slide 17: Result 2: Replanning recovers time and energy, with few replans (1 min)
**On screen:** left: change in mission time (min) per map for event-triggered vs open-loop, one dot per training seed, line = median; right: change in UAV energy (kJ).
**Teach:**
- Every dot below zero means replanning saved time or energy compared with open-loop. All **12 map-seed combinations** are below zero.
- Saving: **15 to 41 minutes** and **260 to 1000 kJ** on average.
- Event-triggered replanning **matches periodic** replanning in mission time (within 0.3 minutes) but needs **1 to 3 fewer replans per mission** (for example 0.5 versus 2.8 on the ring road). This answers **H4**.
- Honesty: some individual instances are worse (we showed one in Demo 2); the claim is on averages.
**Say:** "Replanning helps. In all twelve map and seed combinations it beats open-loop, saving fifteen to forty-one minutes and hundreds of kilojoules. And replanning only when needed gives the same mission time as replanning after every recharge, with one to three fewer replans."

## Slide 18: Result 3: Why the energy methods have so little to gain (1 min 30 s)
**On screen:** left bars: correlation between time and energy over sampled routes (hover as modelled 0.96, hover at 0.5× 0.97, hover at 0.25× 0.90, landed i.e. free waiting 0.39); right bars: extra time and energy saved by an energy-aware route versus the fastest; chain: hover waiting is about half of E, faster routes wait less, fastest is about the lowest-energy route, no trade-off to learn (H1).
**Teach (this is the key insight for the "energy budget" idea):**
- The mid-sem idea: "spend a little more time and save a lot of energy." For that, a **lower-energy route must exist that is only slightly slower**.
- We sampled many routes from the policy for the same instance and measured the correlation of time and energy: **0.96**. They move **together**, because **hover waiting is about half of E**, and a **faster** route also **waits less**, so the fastest route is almost the lowest-energy route.
- A fully energy-aware choice costs +0.7% time to save 0.9% energy: nothing to gain. Only if waiting were **free** (landed) does a trade-off appear (correlation 0.39: +23% time for 7.9% energy).
- Conclusion: at the route level **there is no trade-off to learn**, so H1 is not supported for routes, and this is why candidates A and B (and the fixed penalty) cannot beat their control.
**Say:** "Here is why our energy-aware methods had little to gain. We asked: is there a slightly slower route that uses much less energy? The answer is no. Time and energy are correlated at 0.96. Hovering is about half of the energy, and a faster route hovers less. So the fastest route is nearly the lowest-energy route. There is no trade-off for a policy to learn at the route level. This is a negative result, and it is an important one: it tells us where the energy actually goes."
**If asked "so the mid-sem idea failed?":** "At the level of choosing routes, yes. The energy is saved by changing **when** UAVs take off, which is the next slide."

## Slide 19: Result 4: A lever that realises the idea (1 min 30 s)
**On screen:** three timelines (Before, After, UGV) with markers "UGV arrives: same moment in both"; flow: UAV expects to arrive before the UGV, delay take-off by the expected wait, arrives as the UGV arrives (no hovering), same mission time, far less energy; assumption note.
**Teach (read the timelines):**
- **Before:** the UAV flies, arrives early and **hovers** (orange) until the UGV arrives, then charges.
- **After (just-in-time departure):** the UAV **waits on the ground** (grey, free), then flies and **arrives exactly as the UGV arrives**; no hovering.
- The UGV arrives at the same moment in both cases, so **mission time is unchanged**; only the hover energy disappears.
- **How the UAV knows how long to wait:** it predicts the UGV's arrival from the road state it knows (including events it has observed), and it re-checks at take-off.
- **Assumption (state it!):** a UAV can wait on the ground at its charging point at no energy cost, and the UGV is the bottleneck. With free ground waiting this saves **53 to 74% of UAV energy at +0 minutes** (4 maps × 3 seeds), and brings budget violations at β = 0.9 from 75 to 100% down to about 0%.
- By team size with road events: 1U-1G −18%, 2U-1G −54%, 2U-2G −40%, 4U-2G −69%. The more UAVs share a UGV, the more they wait, so the more there is to save.
**Say:** "The energy is in the waiting, so we attack the waiting. In just-in-time departure, a drone that expects to wait delays its take-off and waits on the ground, where it costs nothing, instead of hovering in the air. It arrives as the truck arrives. The truck's arrival time is unchanged, so mission time is unchanged, but the hover energy disappears. This saves fifty-three to seventy-four percent of the drone energy at no cost in time. The assumption is that a drone can wait on the ground for free, and that its wait can be predicted from the road state it knows."
**If asked "is this realistic?":** it needs the UAV to idle without battery loss and decent UGV-arrival prediction; road events mid-flight cannot be predicted, so 11 to 36% of the waiting remains after events (re-predicting at take-off removes two thirds of that: hover energy 241 to 80 kJ).

## Slide 20: Result 5: Sampling and just-in-time together (1 min 15 s)
**On screen:** three panels (mission time, UAV energy, share over the 0.9 budget), each with static roads and road events, four bars: base policy, + just-in-time, sampling + budget filter, both.
**Teach (read per panel, static roads):**
- **Time:** base 228 min; JIT 228 (unchanged); sampling + budget filter (P1, 64 samples) **183**; both 182. So **sampling makes it about 20% faster**.
- **Energy:** base 4.4 MJ; JIT **1.6 MJ**; P1 3.2 MJ; both **1.5 MJ**. So **JIT cuts about 65%**, sampling alone about 26%.
- **Over budget (β = 0.9):** base 78%; JIT 0%; P1 16%; both **0%**.
- With road events: time 270 to 205 min, energy 5.5 to 1.8 MJ, violations 91% to 2%.
- Message: the two levers are **complementary**: sampling picks a faster route, JIT removes the waiting, and the combination meets the budget.
- Setting: 2U-1G, 4 maps × 3 seeds (12 combinations), 24 instances per combination.
**Say:** "Putting the two levers together. Sampling many routes and keeping the fastest one within the budget cuts mission time by about twenty percent. Just-in-time departure cuts drone energy by about sixty-five percent. Together the gains add up: the mission is about twenty percent faster, uses sixty-five percent less energy, and meets the budget, with zero percent violations on static roads and two percent with events."
**If asked "why does sampling help and not training?":** we tried training a policy to imitate the best sampled routes (search distillation), but it did not help in our setting, so we keep sampling as a decision-time search.

## Slide 21: Result 6: Baselines and sampled decoding on the paper's map (1 min)
**On screen:** left: GLS baseline, paper vs ours, per team; right: greedy vs sampled: paper greedy, ours greedy, paper 10,240 samples, ours best of 10,240.
**Teach:**
- **Left:** our re-implementation of the bilevel heuristic "UGV first, UAV second" with OR-Tools (GLS, Tabu, SA) lands **in the paper's range** (close for 1U-1G, ±10 to 20% otherwise). It shows our simulator is comparable to the paper's.
- **Right:** greedy decoding is within about 10% except 4U-2G. With **10,240 samples** (the paper's own setting) we reach **197.5, 157.2, 135.3 and 135.2 minutes** versus the paper's **218.6, 199.7, 154.4 and 159.9**. So **search closes the 4U-2G gap** and goes below the paper's sampled numbers.
- Honesty: our map and instances are **reconstructions**, so this is **not a claim to beat the paper**.
**Say:** "For comparison with the base paper: our re-implementation of the bilevel heuristic lands in the paper's range. With the same sampled decoding the paper uses, our policy reaches lower times than the paper reports, which also closes the gap we saw for four drones. But our map and instances are reconstructed, so we do not claim to beat the paper."

## Slide 22: Gazebo test bed: do the plans hold up in physics? (1 min)
**On screen:** flow: planned mission, world with roads/tasks/events, wheeled UGV + flying UAV + lock-step replay, compare time/energy/UGV timing; two stills: flat ground (border map) and real terrain; summary box.
**Teach:**
- We take a plan from our simulator, build a **Gazebo** world (roads, tasks, events), and replay it with a **wheeled UGV with contact physics** and **UAVs flying the planned legs**. A UAV is served only after the UGV has **actually arrived**, so physical delays propagate.
- **22 replays** (16 flat ground, 6 real terrain from open elevation data, 144 m relief): mission time within **0.14%** of the plan (at most 0.8 minutes), UAV energy within 0.5%, and the UGV reaches rendezvous **11 to 16 s late on average**.
- **Honesty:** this checks the **plans** are physically achievable. The **learned policy is not run inside Gazebo**, and UAV flight is velocity-controlled (kinematic). Google Earth data cannot be downloaded programmatically, so the terrain uses open SRTM-derived elevation data.
**Say:** "Do the plans hold up in physics? We replayed twenty-two of them in Gazebo, on flat ground and on real terrain, with a wheeled truck that has real contact physics. Mission time matches the plan within 0.14 percent and energy within half a percent; the truck arrives eleven to sixteen seconds late on average. This is a consistency check of the plans. The learned policy itself is not run inside Gazebo."
**Transition to Demo 3:** "Let me show you."

## DEMO 3 (about 2.5 min): Gazebo (Speaker 3)
1. **Gazebo replay video (flat ground):** play `implementation/gazebo/videos/border-block-medium-middle-best_case__event.mp4` (28 s, 400× speed). **Say:** "This is the plan replayed in Gazebo. The title shows the plan time. The road event appears as a red strip."
2. **Real terrain:** play `videos_terrain/border-block-medium-middle-typical_case__event.mp4`. **Say:** "The same kind of mission on real desert terrain from elevation data: the truck follows the road over hills, with the same timing within a fraction of a percent."
3. **Close-up 3D demo:** play `demo_preview_gazebo_window.mp4` (or `demo_preview_flat.mp4`). **Say:** "And a close-up of the runnable 3D demo: the truck with its landing pad, a quadrotor landing on the roof, charging while riding along, then taking off again. We provide this as a one-command Gazebo demo (`./run_demo.sh`)."
4. If the live Gazebo window is available on the machine, run it; otherwise use the recorded videos (safer).

## Slide 23: Did the hypotheses hold? (1 min)
**On screen:** table H1 to H4 with partial, rejected, confirmed, confirmed.
**Teach, one row at a time:**
- **H1 (energy falls, time barely rises): partly.** Rejected for choosing routes (time and energy move together); **confirmed for departure timing**: −53 to −74% energy at +0 minutes.
- **H2 (constrained beats fixed penalty): rejected.** Candidates A and B equal the control on all maps.
- **H3 (more disruption means more energy): confirmed.** Up to +29% energy, early events are worst.
- **H4 (same reliability, fewer replans): confirmed.** Same mission time, 1 to 3 fewer replans.
- Footnote: failures appear only when a bridge is cut and the map is disconnected.
**Say:** "Scorecard. Hypothesis one is partly true: not for choosing routes, but yes for departure timing. Hypothesis two is rejected: our two energy-aware methods did not beat a control trained the same way. Hypotheses three and four are confirmed. Rejecting a hypothesis taught us where the energy really goes, which led to just-in-time departure."

## Slide 24: Conclusion and limitations (1 min 30 s)
**On screen:** left "What we showed", right "Limitations and next steps".
**Say (left):** "We showed four things: road events cost up to twenty-nine percent more energy; event-triggered replanning recovers it with few replans; just-in-time departure cuts drone energy by fifty-three to seventy-four percent at no cost in time; and with sampling and a budget filter we get about twenty percent less time and sixty-five percent less energy with the budget met. Gazebo reproduces the plans to within 0.14 percent in time."
**Say (right, honest):** "The limitations: our learned energy-aware methods A and B did not beat the control; the four-drone policy is nine to fourteen percent slower than the paper in greedy mode, and more training did not help, search does; we replan only at recharge boundaries and there is one bad training seed that a fallback fixes in terms of failures but not speed; and Gazebo replays plans only, so running the policy and the replanner inside Gazebo is the next step."
**Next:** "To finish, here is what we will do next."

## Slide 25: Future plan: improving the method and proving the hypotheses (1 min)
**On screen:** a table with three columns: open point, what we will implement, proved if.
**Teach (one row at a time; each row ends in a test that says when the hypothesis counts as proved):**
1. **H1 is only partly true**, because the 53 to 74% saving relies on UAVs waiting on the ground for free. **Plan:** add a standby energy on the ground and UGV-arrival prediction errors to the simulator, and learn the departure delay instead of using the rule. **Proved if:** the saving stays above 40% at +0 minutes even when ground standby costs 10 to 20% of the hover power.
2. **H2 was rejected** (A and B equal the control) because there is no trade-off at the route level. **Plan:** make the policy energy-aware where the energy is: learn **when to depart** and **which UGV serves which UAV**, not only the route. **Proved if:** it beats the control by at least 5% energy at equal time over 3 seeds, with a paired confidence interval that excludes zero.
3. **4U-2G is 9 to 14% slower than the paper.** **Plan:** longer training, a larger model, a curriculum from small to large teams, and search distillation with a tuned stopping rule. **Proved if:** greedy time is within 5% of the paper.
4. **The policy is not run inside Gazebo.** **Plan:** a closed loop in Gazebo: measure physical delays, replan on them, and add slopes, wind and a UAV battery model. **Proved if:** replanning still saves time under Gazebo disturbances.
5. **Cut bridge: 52 to 68% failures.** **Plan:** replan inside a round and keep a fuel reserve so stranded UAVs can divert. **Proved if:** failures are halved in the stress test.
- Bottom line: H3 and H4 are confirmed; the plan is to prove H1 and H2 under realistic assumptions and to close the loop in Gazebo.
**Say:** "Looking ahead, each open point has a concrete plan and a test that says when it counts as proved. For the first hypothesis we will make the assumptions realistic, with standby energy on the ground and prediction errors, and learn the departure delay. For the second, we will move the learning to where the energy is: departure timing and the assignment of drones to trucks. We will also work on the four-drone gap, run the policy and replanner inside Gazebo, and handle cut bridges with mid-round replanning and a fuel reserve. Thank you; we are happy to take questions."

## Slide 26: References
Keep on screen during questions (or return to slide 23 or 25 if asked about hypotheses or future work). Numbering follows first citation (IEEE). [1] is the base paper (Mondal et al., arXiv:2504.21111); [10] is the authors' code; [9] is Gazebo (Koenig and Howard).

---

# 4. Question and answer preparation

| Question | Answer |
|---|---|
| What is new compared with the base paper? | The energy budget, **dynamic road events** with event-triggered replanning, a road layer that masks unreachable choices, and an analysis of why energy-aware learning does not help, plus just-in-time departure. The base paper has fixed roads and no energy budget. |
| Why did your energy-aware methods (A, B) not work? | Over sampled routes, time and energy are correlated at 0.96 because hover waiting is about half of the energy and faster routes wait less. There is no trade-off for the policy to learn, so A and B equal their control. We report this as a negative result. |
| Is just-in-time departure realistic? | It assumes a UAV can wait on the ground at no energy cost and that the wait can be predicted from road state. Events after take-off cannot be predicted, so some waiting remains (11 to 36% after events); re-predicting at take-off removes two thirds of that. It is a scheduling lever, not a new policy. |
| Isn't JIT just "don't hover"? | Yes in effect, and that is the finding: the energy is in the waiting, and the fastest route does not reduce waiting. The saving is large because hover is about half of E. |
| What is B0? | The energy the base policy uses on the same instance with static roads. B = β·B0; β = 0.9 means a ten percent lower budget. |
| Why not beat the base paper? | Greedy decoding is within about 10% of it for three of four teams and 9 to 14% slower for 4U-2G. With the paper's sampled decoding (10,240 samples) we reach lower times, but our map is a reconstruction from the authors' code, so we make no claim to beat it. |
| How do you know the simulator is right? | 35 automated tests including a hand-checkable 1U-1G case; recharge time and the road map confirmed from the authors' code; Gazebo replays match within 0.14% in time. |
| What does Gazebo prove and not prove? | Proves the **plans** are physically achievable (UGV with wheel physics, real terrain). It does not run the learned policy; UAV flight is kinematic; the UGV follows the planned timetable. |
| Why only 3 seeds? | Training ran on free Kaggle CPUs (about 3 h per map). We report spread over seeds and one seed (loop map, seed 2) is a bad base policy that loops; a fallback to the rule policy removes its failures but it stays about 2× slower. |
| Why event-triggered rather than periodic? | Same mission time (within 0.3 min) with 1 to 3 fewer replans per mission, i.e. less computation and more stable plans. |
| What if a bridge is cut? | The map is disconnected: open-loop fails; replanning rescues some missions (UAVs can divert on remaining battery), but not all, because some UAVs have already committed their last battery. A fuel reserve policy is future work. |
| Does it scale to larger instances? | Trained on 15 aerial and 5 ground tasks; zero-shot on 45 and 75 tasks, two of three seeds still work and replanning still helps (see backup slides in the full deck / README). |
| Why a Transformer policy? | It is the base paper's model; we keep its encoder unchanged and change the decoder input/masks only. |
| What is "landed waiting" in the energy-correlation slide? | A variant where waiting costs no energy (the UAV waits on the ground). It lowers the correlation between time and energy to 0.39, which is why JIT, which creates exactly that, works. |
| Did you use Google Earth for the terrain? | No: Google Earth data cannot be downloaded programmatically. We used open SRTM-derived elevation data for a Thar-desert area. |
| What would you do next? | Run the policy and replanner inside Gazebo, replan mid-round, add a fuel reserve for stranded UAVs, longer training or a larger model for 4U-2G, and test just-in-time departure with realistic UAV standby energy. |

---

# 5. Slide-by-slide time plan

| Slide | Time | Slide | Time | Slide | Time |
|---|---|---|---|---|---|
| 1 Title | 0:30 | 10 Framework | 1:15 | 19 Lever (JIT) | 1:30 |
| 2 Problem | 1:00 | 11 Limits to actions | 1:00 | 20 Sampling + JIT | 1:15 |
| 3 Sortie | 1:00 | 12 Reproduce base | 1:00 | 21 Baselines | 1:00 |
| 4 Roads change | 1:00 | 13 Replanning | 1:15 | 22 Gazebo | 1:00 |
| **Demo 1** | 1:30 | **Demo 2** | 2:00 | **Demo 3** | 2:30 |
| 5 Problem statement | 1:00 | 14 Policies | 1:15 | 23 Hypotheses | 1:00 |
| 6 Literature | 1:15 | 15 Experiments | 0:45 | 24 Conclusion | 1:30 |
| 7 Gap | 0:45 | 16 Result 1 | 0:45 | 25 Future plan | 1:00 |
| 8 Questions | 1:00 | 17 Result 2 | 1:00 | 26 References | (Q&A) |
| | | 18 Result 3 | 1:30 | | |
| **Part 1 total** | about 10 min | **Part 2 total** | about 11 min | **Part 3 total** | about 13 min |

# 6. Cut list for a shorter talk (about 20 min)

Skip or merge, in this order: slide 6 (literature; say two sentences while showing slide 7), slide 11 (limits to actions, merge into slide 9), slide 15 (experimental design), slide 21 (baselines), Demo 2 steps 2 and 4 (keep best case and cut road), the 3D demo in Demo 3 (keep the two Gazebo videos), and slide 16 (merge into slide 17). Slide 25 can be shortened to two sentences. Never cut: slides 4, 13, 18, 19, 20, 23, 24.

# 7. Delivery tips

* Use the **story in five sentences** (section 1) as your spine; every slide serves one of those sentences.
* Say the **negative results out loud** (A and B did not help; 4U-2G gap; Gazebo does not run the policy). They show scientific honesty and professors reward it.
* State the **JIT assumption** every time you quote the 53 to 74% number.
* During demos talk about **one thing at a time** (left panel = problem, right panel = replanning) and do not apologise for the speed of the video; it is time-compressed.
* If a demo fails, say "here is the recorded version" and use the backup videos; do not debug live.
