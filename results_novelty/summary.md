
### Primary (success vs matched node budget) — pointmaze-medium

arm                   n   steps        success      goal_dist   nodes/replan      chunk_len
-------------------------------------------------------------------------------------------
fixedtreewm           3   19950    0.213±0.050   10.864±1.218   64.000±0.000   14.125±1.419
randomtreewm          3   19950    0.233±0.062    9.689±0.583   64.000±0.000   13.803±0.972
noveltyq              3   19950    0.187±0.111   10.849±2.087   64.000±0.000   14.055±1.596
learnedq              3   19950    0.227±0.084   11.033±2.120   64.000±0.000   14.373±1.513
noveltyz              3   19950    0.180±0.071   10.345±1.119   64.000±0.000   13.676±1.334
learnedz              3   19950    0.173±0.050   12.017±1.206   64.000±0.000   14.123±1.493

### Diagnostics — pointmaze-medium

arm                   n   steps            EBF    rare_recall    supp_recall       coverage       cov/node       gain_rho     branch~div            q-z         q-rand
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
fixedtreewm           3   19950    2.284±0.014    0.924±0.004    0.974±0.001    9.891±1.182    0.618±0.074    0.811±0.003    0.425±0.021   -0.218±0.028   -0.217±0.029
randomtreewm          3   19950    2.277±0.023    0.924±0.001    0.974±0.001    9.513±0.750    0.595±0.047    0.800±0.010    0.435±0.026   -0.243±0.010   -0.242±0.012
noveltyq              3   19950    2.284±0.014    0.924±0.004    0.974±0.001    9.900±1.179    0.619±0.074    0.811±0.003    0.425±0.021   -0.218±0.028   -0.217±0.029
learnedq              3   19950    2.284±0.014    0.924±0.004    0.974±0.001    9.896±1.181    0.619±0.074    0.811±0.003    0.425±0.021   -0.218±0.028   -0.217±0.029
noveltyz              3   19950    2.283±0.014    0.924±0.002    0.974±0.001    9.085±0.087    0.568±0.005    0.830±0.002    0.417±0.027   -0.228±0.023   -0.226±0.020
learnedz              3   19950    2.283±0.014    0.924±0.002    0.974±0.001    9.084±0.088    0.568±0.006    0.830±0.002    0.417±0.027   -0.228±0.023   -0.226±0.020

### Novelty-target diagnostics — pointmaze-medium

arm                   n   steps       spearman        pearson       coverage   cov/expanded      redundant     mean_depth      depth_std    front_decay
-------------------------------------------------------------------------------------------------------------------------------------------------------
fixedtreewm           3   19950    0.811±0.003    0.811±0.004    9.891±1.182    1.978±0.236    0.812±0.039    1.625±0.000    0.599±0.000    0.199±0.003
randomtreewm          3   19950    0.800±0.010    0.808±0.013    9.513±0.750    1.903±0.150    0.809±0.031    1.625±0.000    0.599±0.000    0.200±0.009
noveltyq              3   19950    0.811±0.003    0.811±0.004    9.900±1.179    1.980±0.236    0.812±0.039    1.625±0.000    0.599±0.000    0.199±0.003
learnedq              3   19950    0.811±0.003    0.811±0.004    9.896±1.181    1.979±0.236    0.812±0.040    1.625±0.000    0.599±0.000    0.199±0.003
noveltyz              3   19950    0.830±0.002    0.854±0.002    9.085±0.087    1.817±0.017    0.789±0.032    1.625±0.000    0.599±0.000    0.883±0.021
learnedz              3   19950    0.830±0.002    0.854±0.002    9.084±0.088    1.817±0.018    0.790±0.032    1.625±0.000    0.599±0.000    0.883±0.021

### Ladder verdict — pointmaze-medium

  training parity: all arms at ~19950 steps

  multimodal futures matter                     flatkwm    nan vs singlewm              nan   -> no data
  recursive prediction matters              fixedtreewm  0.213 vs flatkwm               nan   -> no data
  learned allocation matters                     treewm    nan vs fixedtreewm         0.213   -> no data
  *learned* beats merely adaptive                treewm    nan vs heuristictreewm       nan   -> no data

### Novelty-target verdict — pointmaze-medium

  training parity: all arms at ~19950 steps

  Q1. Does learned novelty close the gap to its direct heuristic?
  q-novelty      direct=0.187 learned=0.227 (rel +21%, spearman 0.81) -> learned EXCEEDS direct
  z-novelty      direct=0.180 learned=0.173 (rel -4%, spearman 0.83) -> CLOSED -- bad gain target was the problem; learned allocation viable

  Q2. Is q-novelty actually better than z-novelty?
  direct: q=0.187 vs z=0.180 (rel +4%) -> q ~= z -- do NOT attribute the gain to controllability-aware q

  Q3. Controls
  randomtreewm         0.233
  fixedtreewm          0.213

### Primary (success vs matched node budget) — pointmaze-medium-stitch

arm                   n   steps        success      goal_dist   nodes/replan      chunk_len
-------------------------------------------------------------------------------------------
fixedtreewm           3   19950    0.193±0.111   10.790±1.922   64.000±0.000   13.844±1.348
randomtreewm          3   19950    0.267±0.009    7.689±0.707   64.000±0.000   14.279±0.076
noveltyq              3   19950    0.153±0.047   10.843±1.295   64.000±0.000   13.106±0.805
learnedq              3   19950    0.087±0.047   12.101±1.773   64.000±0.000   14.411±1.143
noveltyz              3   19950    0.173±0.066   11.166±0.885   64.000±0.000   13.339±1.019
learnedz              3   19950    0.207±0.050   10.008±1.259   64.000±0.000   14.480±0.574

### Diagnostics — pointmaze-medium-stitch

arm                   n   steps            EBF    rare_recall    supp_recall       coverage       cov/node       gain_rho     branch~div            q-z         q-rand
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
fixedtreewm           3   19950    1.757±0.013    0.980±0.002    0.995±0.001    8.022±1.051    0.501±0.066    0.796±0.031    0.827±0.016   -0.076±0.035   -0.077±0.034
randomtreewm          3   19950    1.758±0.006    0.980±0.002    0.995±0.001    7.838±0.397    0.490±0.025    0.780±0.031    0.809±0.012   -0.078±0.025   -0.074±0.026
noveltyq              3   19950    1.757±0.013    0.980±0.002    0.995±0.001    8.008±1.040    0.501±0.065    0.796±0.031    0.827±0.016   -0.076±0.035   -0.077±0.034
learnedq              3   19950    1.757±0.013    0.980±0.002    0.995±0.001    8.013±1.042    0.501±0.065    0.796±0.031    0.827±0.016   -0.076±0.035   -0.077±0.034
noveltyz              3   19950    1.762±0.004    0.980±0.003    0.995±0.001    7.941±0.493    0.496±0.031    0.906±0.016    0.831±0.014   -0.070±0.038   -0.069±0.036
learnedz              3   19950    1.762±0.004    0.980±0.003    0.995±0.001    7.941±0.493    0.496±0.031    0.906±0.016    0.831±0.014   -0.070±0.038   -0.069±0.036

### Novelty-target diagnostics — pointmaze-medium-stitch

arm                   n   steps       spearman        pearson       coverage   cov/expanded      redundant     mean_depth      depth_std    front_decay
-------------------------------------------------------------------------------------------------------------------------------------------------------
fixedtreewm           3   19950    0.796±0.031    0.782±0.053    8.022±1.051    1.604±0.210    0.826±0.027    1.625±0.000    0.599±0.000    0.148±0.008
randomtreewm          3   19950    0.780±0.031    0.737±0.052    7.838±0.397    1.568±0.079    0.858±0.019    1.625±0.000    0.599±0.000    0.147±0.005
noveltyq              3   19950    0.796±0.031    0.782±0.053    8.008±1.040    1.602±0.208    0.826±0.026    1.625±0.000    0.599±0.000    0.148±0.008
learnedq              3   19950    0.796±0.031    0.782±0.053    8.013±1.042    1.603±0.208    0.826±0.027    1.625±0.000    0.599±0.000    0.148±0.008
noveltyz              3   19950    0.906±0.016    0.890±0.013    7.941±0.493    1.588±0.099    0.855±0.011    1.625±0.000    0.599±0.000    1.029±0.027
learnedz              3   19950    0.906±0.016    0.890±0.013    7.941±0.493    1.588±0.099    0.855±0.010    1.625±0.000    0.599±0.000    1.029±0.027

### Ladder verdict — pointmaze-medium-stitch

  training parity: all arms at ~19950 steps

  multimodal futures matter                     flatkwm    nan vs singlewm              nan   -> no data
  recursive prediction matters              fixedtreewm  0.193 vs flatkwm               nan   -> no data
  learned allocation matters                     treewm    nan vs fixedtreewm         0.193   -> no data
  *learned* beats merely adaptive                treewm    nan vs heuristictreewm       nan   -> no data

### Novelty-target verdict — pointmaze-medium-stitch

  training parity: all arms at ~19950 steps

  Q1. Does learned novelty close the gap to its direct heuristic?
  q-novelty      direct=0.153 learned=0.087 (rel -43%, spearman 0.80) -> STILL FAR BELOW despite high target correlation -> inspect best-first batching / tree-context feedback, not the representation
  z-novelty      direct=0.173 learned=0.207 (rel +19%, spearman 0.91) -> learned EXCEEDS direct

  Q2. Is q-novelty actually better than z-novelty?
  direct: q=0.153 vs z=0.173 (rel -12%) -> q ~= z -- do NOT attribute the gain to controllability-aware q

  Q3. Controls
  randomtreewm         0.267
  fixedtreewm          0.193