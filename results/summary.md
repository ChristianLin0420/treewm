
### Primary (success vs matched node budget) — pointmaze-medium

arm                   n   steps        success      goal_dist   nodes/replan      chunk_len
-------------------------------------------------------------------------------------------
singlewm              2   19950    0.000±0.000   15.156±0.777   64.000±0.000   11.738±0.179
flatkwm               2   18125    0.080±0.000   12.194±0.346   64.000±0.000   15.974±0.010
fixedtreewm           2   19950    0.140±0.060   13.214±0.853   64.000±0.000   15.030±0.077
randomtreewm          2   19950    0.270±0.030    8.884±0.035   64.000±0.000   14.449±0.237
uncertaintytreewm     2   17825    0.150±0.010   10.522±0.094   64.000±0.000   13.901±0.165
heuristictreewm       2   12800    0.100±0.000   13.703±0.000   64.000±0.000   11.268±0.000
treewm                2   11750    0.220±0.000    7.444±0.000   64.000±0.000   12.912±0.000

### Diagnostics — pointmaze-medium

arm                   n   steps            EBF    rare_recall    supp_recall       coverage       cov/node       gain_rho     branch~div            q-z         q-rand
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
singlewm              2   19950    1.000±0.000    0.119±0.001    0.338±0.001   11.999±1.663    0.750±0.104    0.343±0.056    0.000±0.000   -0.300±0.013   -0.302±0.015
flatkwm               2   18125   72.145±2.477    1.000±0.000    1.000±0.000    9.254±1.454    0.578±0.091    0.365±0.096    0.007±0.069   -0.140±0.009   -0.144±0.001
fixedtreewm           2   19950    2.313±0.017    0.925±0.004    0.974±0.001    9.483±0.589    0.593±0.037    0.493±0.012    0.416±0.025   -0.231±0.002   -0.228±0.003
randomtreewm          2   19950    2.310±0.001    0.925±0.002    0.974±0.001    9.526±0.710    0.595±0.044    0.485±0.016    0.425±0.033   -0.244±0.004   -0.236±0.004
uncertaintytreewm     2   17825    2.337±0.008    0.921±0.000    0.973±0.000   10.676±1.936    0.667±0.121    0.480±0.030    0.447±0.013   -0.230±0.018   -0.225±0.019
heuristictreewm       2   12800    2.269±0.048    0.924±0.001    0.973±0.000   10.924±1.592    0.683±0.099    0.459±0.014    0.432±0.002   -0.193±0.009   -0.192±0.008
treewm                2   11750    2.275±0.003    0.922±0.000    0.973±0.000   10.620±1.234    0.664±0.077    0.452±0.034    0.397±0.038   -0.196±0.026   -0.192±0.032

### Ladder verdict — pointmaze-medium

  training parity: arms still training (max=19950): flatkwm@18125, heuristictreewm@12800, treewm@11750, uncertaintytreewm@17825

  *** VERDICT SUPPRESSED: arms are at different training steps, so any
  *** ordering below would confound allocation with training budget.

  multimodal futures matter                     flatkwm  0.080 vs singlewm            0.000   -> (unreliable) supported
  recursive prediction matters              fixedtreewm  0.140 vs flatkwm             0.080   -> (unreliable) supported
  learned allocation matters                     treewm  0.220 vs fixedtreewm         0.140   -> (unreliable) supported
  *learned* beats merely adaptive                treewm  0.220 vs heuristictreewm     0.100   -> (unreliable) supported