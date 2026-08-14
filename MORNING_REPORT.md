# TreeWM overnight design-space search — morning report

Screening = 1 seed (**preliminary**); Phase 2 = 3 seeds (**more robust**). RandomTreeWM (`A0_random`) is the control in every comparison.


## Phase 1 — mechanism screening (1 seed, pointmaze-medium-stitch)


### Axis A — recursive training

| recipe | success | goal_dist | val_loss | EBF | leaf_depth |
|---|---|---|---|---|---|
| `A0_random` _(baseline)_ | 0.520 | 4.319 | 1.417 | 1.764 | 3.430 |
| `A0_bfs` | 0.440 | 4.232 | 1.399 | 1.766 | 2.466 |

**Best on this axis:** `A0_bfs` (success 0.440, -0.080 vs baseline; 1 seed -> **preliminary**).

### Axis B — branch proposal

| recipe | success | goal_dist | val_loss | EBF | leaf_depth |
|---|---|---|---|---|---|
| `A0_random` _(baseline)_ | 0.520 | 4.319 | 1.417 | 1.764 | 3.430 |
| `B1_k8` | 0.540 | 3.478 | 1.421 | 2.750 | 2.285 |
| `B3_k8_short` | 0.460 | 5.558 | 1.277 | 2.701 | 2.365 |

**Best on this axis:** `B1_k8` (success 0.540, +0.020 vs baseline; 1 seed -> **preliminary**).

### Axis C — temporal horizon

| recipe | success | goal_dist | val_loss | EBF | leaf_depth |
|---|---|---|---|---|---|
| `A0_random` _(baseline)_ | 0.520 | 4.319 | 1.417 | 1.764 | 3.430 |
| `C1_short` | 0.560 | 4.498 | 1.258 | 1.770 | 2.911 |

**Best on this axis:** `C1_short` (success 0.560, +0.040 vs baseline; 1 seed -> **preliminary**).

### Axis G — capacity / training

| recipe | success | goal_dist | val_loss | EBF | leaf_depth |
|---|---|---|---|---|---|
| `A0_random` _(baseline)_ | 0.520 | 4.319 | 1.417 | 1.764 | 3.430 |
| `G3_z256` | 0.560 | 3.694 | 1.398 | 1.756 | 3.520 |

**Best on this axis:** `G3_z256` (success 0.560, +0.040 vs baseline; 1 seed -> **preliminary**).

### Axis H — loss design

| recipe | success | goal_dist | val_loss | EBF | leaf_depth |
|---|---|---|---|---|---|
| `A0_random` _(baseline)_ | 0.520 | 4.319 | 1.417 | 1.764 | 3.430 |

## Phase 2 — promoted recipes (3 seeds)


### Phase 2 — success vs node budget, pointmaze-medium-stitch (3 seeds)

| recipe | 16 | 32 | 64 | 128 | 256 | AUC |
|---|---|---|---|---|---|---|
| `C1_short` | 0.460 | 0.520 | 0.567 | 0.560 | 0.600 | **0.541** |
| `P_k8_short_ms` | 0.513 | 0.493 | 0.487 | 0.447 | 0.487 | **0.485** |
| `A0_random` | 0.427 | 0.467 | 0.467 | 0.487 | 0.520 | **0.473** |
| `G3_z256` | 0.413 | 0.467 | 0.467 | 0.507 | 0.447 | **0.460** |
| `B1_k8` | 0.440 | 0.427 | 0.493 | 0.440 | 0.387 | **0.437** |
| `B3_k8_short` | 0.393 | 0.420 | 0.380 | 0.433 | 0.467 | **0.419** |
| `A0_bfs` | 0.413 | 0.220 | 0.307 | 0.180 | 0.280 | **0.280** |

### Phase 2 — coverage vs node budget (distinct regions)


**pointmaze-medium-stitch**

| recipe | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `A0_bfs` | 6.4 | 9.2 | 14.6 | 21.3 | 29.1 |
| `A0_random` | 8.2 | 13.6 | 23.0 | 36.1 | 55.4 |
| `B1_k8` | 8.2 | 9.4 | 15.6 | 26.9 | 41.9 |
| `B3_k8_short` | 7.8 | 8.7 | 13.9 | 21.8 | 32.2 |
| `C1_short` | 6.8 | 11.3 | 19.5 | 29.7 | 42.3 |
| `G3_z256` | 7.1 | 11.6 | 20.4 | 32.6 | 50.4 |
| `P_k8_short_ms` | 8.9 | 10.1 | 16.7 | 26.3 | 39.3 |

## Tracks D / E / F — no retraining


### Axis D — tree structure / allocation

| setting | 128 | 32 | 64 | goal_dist | leaf_depth |
|---|---|---|---|---|---|
| `bfs` | 0.312 | 0.333 | 0.292 | 6.129 | 2.619 |
| `broad_to_focused` | 0.354 | 0.396 | 0.417 | 5.372 | 4.131 |
| `depth_balanced` | 0.396 | 0.458 | 0.417 | 4.497 | 3.544 |
| `diverse_goal` | 0.229 | 0.375 | 0.333 | 6.037 | 4.107 |
| `goal` | 0.354 | 0.333 | 0.354 | 6.284 | 4.268 |
| `random` | 0.458 | 0.438 | 0.458 | 4.230 | 3.484 |
| `root_quota` | 0.375 | 0.458 | 0.500 | 4.178 | 3.492 |

### Axis E — execution cadence

| setting | env_steps | exec_len | goal_dist | replans | success |
|---|---|---|---|---|---|
| `clipped16` | 360.792 | 14.500 | 4.070 | 25.458 | 0.479 |
| `fixed1` | 411.896 | 1.000 | 7.665 | 411.896 | 0.292 |
| `fixed4` | 401.958 | 4.000 | 6.738 | 100.646 | 0.333 |
| `fixed8` | 387.583 | 7.989 | 5.287 | 49.000 | 0.354 |
| `full64` | 406.417 | 24.839 | 5.002 | 17.875 | 0.333 |

### Axis F — planner interface

| setting | goal_dist | leaf_depth | success |
|---|---|---|---|
| `ancestor_0.5` | 4.294 | 3.452 | 0.438 |
| `endpoint_0.0` | 4.361 | 3.385 | 0.396 |
| `path_aware_0.005` | 4.632 | 2.540 | 0.396 |
| `path_aware_0.02` | 6.896 | 1.069 | 0.417 |

**Tree shape by policy (budget 64):**

| policy | mean_depth | unique_root_subtrees_explored | top2_root_subtree_fraction |
|---|---|---|---|
| `random` | 2.925 | 4.000 | 0.698 |
| `bfs` | 2.578 | 4.000 | 0.667 |
| `depth_balanced` | 2.976 | 4.000 | 0.719 |
| `root_quota` | 2.917 | 4.000 | 0.670 |
| `goal` | 3.148 | 4.000 | 0.841 |
| `diverse_goal` | 3.098 | 4.000 | 0.832 |
| `broad_to_focused` | 3.109 | 4.000 | 0.750 |

### Axis: generalisation — AntMaze

_AntMaze results missing (pipeline did not reach this stage)._
