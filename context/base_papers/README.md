# Base papers and related material (downloaded 2026-11-21)

## Base paper
| File | Source |
|---|---|
| `arxiv_2504.21111v1.pdf` | Mondal et al., *How to Coordinate UAVs and UGVs for Efficient Mission Planning?…*, arXiv:2504.21111v1 (same paper as `../p1.pdf`) |
| `authors_code_multi-uav-ugv-drl/` | The authors' public code, `git clone --depth 1 https://github.com/imsafwan/multi-uav-ugv-cooperative-routing-drl` (**third-party code, read-only reference; not executed**). Its road-edge file `ugv_road.csv` is referenced but is **not in the repository**; the 28 road points are in `scenario_generator_v2.py` (`all_ugv_points`). |

## Related papers by the same group (arXiv, open access) — `related_open_access/`
| File | arXiv | Relation to our deck |
|---|---|---|
| `mondal2023_bilevel_fuel_constrained_uav_ugv_MRS.pdf` | 2303.02315 | base paper ref [31] (bilevel planning, MRS 2023) |
| `mondal2023_optiroute_heuristic_assisted_drl.pdf` | 2309.09942 | earlier DRL UAV-UGV routing (OptiRoute) |
| `mondal2023_cooperative_multi_agent_planning_fuel_constrained.pdf` | 2309.03397 | multi-agent fuel-constrained UAV-UGV planning |
| `mondal2025_persistent_surveillance_energy_aware_uav_ugv.pdf` | 2502.02666 | energy-aware persistent surveillance (disaster management) |
| `shi2022_risk_aware_uav_ugv_rendezvous_cmdp_CDC.pdf` | 2204.04767 | deck reference [4] (Shi et al., CDC 2022) |

## Methods the base paper builds on — `related_open_access/`
`kool2019_attention_learn_to_solve_routing_problems.pdf` (1803.08475), `bello2016_neural_combinatorial_optimization_rl.pdf` (1611.09940), `vinyals2015_pointer_networks.pdf` (1506.03134).

## Cited by our deck but NOT downloaded (not openly available / not found on arXiv)
* Maini et al., TAES 2019 (IEEE, paywalled)
* Fan et al., IEEE TVT 2023 (IEEE, paywalled)
* Lin & Yazicioğlu & Aksaray, IEEE RA-L 2022 (IEEE; a preprint may exist elsewhere)
* Mondal et al., ICRA 2025, *Risk-aware energy-constrained UAV-UGV cooperative routing using attention-guided reinforcement learning* (no arXiv entry found by title search)
