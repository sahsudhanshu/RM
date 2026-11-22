# Multi-seed results — U15G5_2U1G_border_landed

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 154.9 ± 14.6 | 936 ± 67 | 100 ± 0 | [152, 162, 151] |
| P0+ (control) | 155.2 ± 6.1 | 954 ± 34 | 86 ± 31 | [155, 158, 153] |
| P1 route filter | 154.3 ± 6.2 | 829 ± 52 | 39 ± 7 | [153, 157, 153] |
| P2 fixed penalty | 152.9 ± 5.4 | 934 ± 33 | 80 ± 21 | [155, 151, 153] |
| P4 budget-conditioned | 156.2 ± 14.2 | 924 ± 39 | 81 ± 19 | [156, 162, 151] |
| P4 no budget input | 157.5 ± 10.4 | 921 ± 12 | 79 ± 22 | [155, 162, 155] |
| P4x budget + pace | 153.8 ± 7.1 | 915 ± 13 | 76 ± 7 | [155, 156, 151] |
| P3 Lagrangian | 156.0 ± 5.0 | 917 ± 25 | 77 ± 10 | [154, 158, 157] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 154.9 ± 14.6 | 936 ± 67 | 0 ± 0 | [152, 162, 151] |
| P0+ (control) | 155.2 ± 6.1 | 954 ± 34 | 47 ± 40 | [155, 158, 153] |
| P1 route filter | 136.5 ± 3.8 | 888 ± 51 | 0 ± 0 | [136, 138, 136] |
| P2 fixed penalty | 152.9 ± 5.4 | 934 ± 33 | 44 ± 4 | [155, 151, 153] |
| P4 budget-conditioned | 156.2 ± 14.2 | 924 ± 39 | 43 ± 22 | [156, 162, 151] |
| P4 no budget input | 157.5 ± 10.4 | 921 ± 12 | 42 ± 25 | [155, 162, 155] |
| P4x budget + pace | 153.8 ± 7.1 | 915 ± 13 | 40 ± 21 | [155, 156, 151] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -0.8 ± 3.0 | -125 ± 47 | [-117, -147, -111] |
| P2 fixed penalty | -2.3 ± 9.3 | -21 ± 44 | [-4, -40, -18] |
| P3 Lagrangian | 0.8 ± 7.3 | -38 ± 37 | [-23, -53, -37] |
| P4 budget-conditioned | 1.0 ± 8.3 | -31 ± 73 | [+3, -47, -49] |
| P4 no budget input | 2.3 ± 6.4 | -34 ± 34 | [-18, -44, -38] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 192.3 ± 21.3 | 936 ± 67 | 100 ± 0 | 0.00 ± 0.00 | [189, 202, 186] |
| P0 periodic + road layer | 176.8 ± 4.3 | 961 ± 29 | 100 ± 2 | 1.14 ± 0.07 | [176, 179, 176] |
| P0 event + road layer | 176.9 ± 4.3 | 961 ± 29 | 100 ± 2 | 0.28 ± 0.02 | [176, 179, 176] |
| P0+ (control) open-loop | 185.1 ± 9.3 | 954 ± 34 | 86 ± 31 | 0.00 ± 0.00 | [188, 186, 181] |
| P0+ (control) event + road layer | 172.4 ± 11.1 | 978 ± 42 | 88 ± 24 | 0.29 ± 0.02 | [177, 171, 169] |
| P0+ event, road-distance assignment | 172.4 ± 11.1 | 978 ± 42 | 88 ± 24 | 0.29 ± 0.02 | [177, 171, 169] |
| P4 open-loop | 185.4 ± 16.1 | 924 ± 39 | 81 ± 19 | 0.00 ± 0.00 | [186, 191, 179] |
| P4 periodic + road layer | 169.8 ± 11.7 | 955 ± 40 | 86 ± 17 | 1.14 ± 0.13 | [174, 171, 165] |
| P4 event + road layer | 169.8 ± 11.7 | 955 ± 40 | 86 ± 17 | 0.38 ± 0.08 | [174, 171, 165] |
| P3 (beta=0.9) event + road layer | 170.7 ± 3.0 | 951 ± 3 | 84 ± 11 | 0.28 ± 0.02 | [170, 170, 172] |
| P4 event, road-distance assignment | 169.8 ± 11.7 | 955 ± 40 | 86 ± 17 | 0.38 ± 0.08 | [174, 171, 165] |
| P0 event, road-distance assignment | 176.9 ± 4.3 | 961 ± 29 | 100 ± 2 | 0.28 ± 0.02 | [176, 179, 176] |
| P1 route filter, event | 175.2 ± 5.8 | 850 ± 53 | 56 ± 4 | 0.32 ± 0.01 | [175, 177, 173] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.1 ± 0.2 | -0 ± 0 | [+0.1, +0.0, +0.0] |
| P4: event − periodic | 0.0 ± 0.1 | 0 ± 0 | [+0.0, -0.0, +0.0] |
| P0: event + layer − open loop | -15.4 ± 17.1 | 25 ± 39 | [-13.1, -23.1, -9.9] |
| P0+ control: event + layer − open loop | -12.7 ± 4.9 | 24 ± 10 | [-11.0, -14.9, -12.2] |
| P4: event + layer − open loop | -15.6 ± 10.3 | 31 ± 16 | [-12.6, -20.3, -13.8] |
| P4 − P0+ control (both event + layer) | -2.6 ± 5.4 | -24 ± 81 | [-3.6, -0.1, -4.1] |
| P3(0.9) − P0+ control (both event + layer) | -1.7 ± 13.5 | -28 ± 43 | [-7.5, -1.0, +3.4] |
| P1 route filter − P0+ control | 2.8 ± 10.5 | -128 ± 49 | [-1.9, +6.3, +4.0] |

Replans event − periodic (P0): -0.86 ± 0.06 per mission.

