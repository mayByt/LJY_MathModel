# 问题二修复后计算与验收结果

> 本报告由 `问题二/problem2.py` 从结果 CSV/JSON 自动生成；本轮按用户要求未生成 PDF 图表。

## 运行环境与边界

- 随机种子：`2026`
- bootstrap 次数：`500`
- 绘图：已跳过，仅完成建模、验收和结果报告。
- GPU策略：硬件和依赖支持且实测更快时优先；本轮 scipy 拟合路径使用 CPU，并在 `run_log.json` 记录回退原因。

- 问题三接口状态：`{'CENTRAL_MODEL_READY': True, 'FIXED_P_READY': True, 'JOINT_P_READY': False, 'P12_FREEZE_READY': True}`。

## 数据审计

B1 共 `1176` 行、`8` 个模型规模；B7 为 `45` 个 `(N,D)` 单元，每单元 `10`-`10` 个 Q。
B6/B7 共享键 `360` 条，最大 Loss 差 `0.000e+00`。B8 已隔离，仅作压力测试。
B2/B3 规则选择 B2 作族外验证；B3 的 `8` 条插值轨迹文件仅审计，不作为新增独立证据。
B11 审计 `18` 行、B12 审计 `1386` 行。

B1–B12 使用账本：
| 附件   | 角色                                         |
|:-------|:---------------------------------------------|
| B1     | classic_fit_cv_bootstrap                     |
| B2     | selected_family_out_validation               |
| B3     | audited_only_B2_selected_under_B2_or_B3_rule |
| B4     | external_family_validation                   |
| B5     | published_validation                         |
| B6     | strict_subset_integrity_check_only           |
| B7     | quality_fit_blocked_cv                       |
| B8     | direction_and_floor_stress_test_only         |
| B9     | extended_N_D_support_boundary                |
| B10    | estimated_large_model_scale_check            |
| B11    | family_source_provenance_audit               |
| B12    | checkpoint_index_integrity_audit             |

## 经典标度律

获选经典模型：`M0`，规则：`one_standard_error`。
|      E |       A |       B |    alpha |     beta |
|-------:|--------:|--------:|---------:|---------:|
| 1.6898 | 0.35398 | 1.24031 | 0.339977 | 0.279878 |

经典模型留一规模均值：
| model   |     RMSE |      MAE |    nRMSE |   Spearman_D |
|:--------|---------:|---------:|---------:|-------------:|
| M0      | 0.000147 | 0.000108 | 0.000374 |     1.000000 |
| MC      | 0.084910 | 0.053853 | 0.216303 |     1.000000 |
| MND     | 0.000148 | 0.000108 | 0.000376 |     1.000000 |

早期轨迹敏感性：删除 `D<2` B tokens 的 `32` 行后重拟合，alpha/beta 最大相对变化 `0.052965%`；该结果不参与主模型选择。
| parameter   |   full_value |   trimmed_value |   relative_change |   abs_relative_change |
|:------------|-------------:|----------------:|------------------:|----------------------:|
| E           |   1.68979756 |      1.69002953 |        0.00013727 |            0.00013727 |
| A           |   0.35398032 |      0.35398652 |        0.00001750 |            0.00001750 |
| B           |   1.24030558 |      1.24025964 |       -0.00003704 |            0.00003704 |
| alpha       |   0.33997658 |      0.33997043 |       -0.00001811 |            0.00001811 |
| beta        |   0.27987813 |      0.28002637 |        0.00052965 |            0.00052965 |

外部与外推验证（B10 为估算对照，不是独立观测）：
| source   |    n | evidence_type   |     RMSE |      MAE |   centered_MAE |   Spearman |
|:---------|-----:|:----------------|---------:|---------:|---------------:|-----------:|
| B2       | 1029 | semi_synthetic  | 1.243085 | 1.178442 |       1.178442 |   0.788102 |
| B4       |   57 | external_family | 0.292667 | 0.227263 |       0.112277 |   0.982982 |
| B5       |   44 | published       | 0.197595 | 0.170722 |       0.102567 |   0.958808 |
| B10      |  128 | estimated       | 0.001094 | 0.000614 |       0.000614 |   0.999994 |

## 质量模型

锚定状态：`absolute_anchor_compatible`；Q=1 锚点 nRMSE=`0.113203`。
获选质量模型：`GQ2`；家族等权 one-SE 原始候选：`GQ2`。
|     c_Q |     nu_Q |
|--------:|---------:|
| 0.36204 | 0.990288 |

四类切分的无泄漏绝对 Loss 验证（每类选模权重均为 0.25）：
| model   | split_type   |   family_RMSE |   family_nRMSE |   folds |
|:--------|:-------------|--------------:|---------------:|--------:|
| GQ1     | leave_D      |      0.127452 |       0.264300 |       5 |
| GQ1     | leave_N      |      0.113737 |       0.235859 |       9 |
| GQ1     | leave_Q      |      0.119541 |       0.247895 |      10 |
| GQ1     | leave_cell   |      0.108506 |       0.225010 |      45 |
| GQ2     | leave_D      |      0.073338 |       0.152083 |       5 |
| GQ2     | leave_N      |      0.074643 |       0.154789 |       9 |
| GQ2     | leave_Q      |      0.071086 |       0.147414 |      10 |
| GQ2     | leave_cell   |      0.068770 |       0.142610 |      45 |
| GQ3     | leave_D      |      0.076619 |       0.158886 |       5 |
| GQ3     | leave_N      |      0.075974 |       0.157549 |       9 |
| GQ3     | leave_Q      |      0.073105 |       0.151599 |      10 |
| GQ3     | leave_cell   |      0.071830 |       0.148954 |      45 |
| GQ4     | leave_D      |      0.084075 |       0.174349 |       5 |
| GQ4     | leave_N      |      0.081051 |       0.168077 |       9 |
| GQ4     | leave_Q      |      0.091701 |       0.190163 |      10 |
| GQ4     | leave_cell   |      0.078497 |       0.162781 |      45 |

## 问题一桥接与配比迁移

主锚点使用 `p_ref`，`Q_ref=0.5389559588`；`p_star` 仅作敏感性，`Q_star=0.5453680376`。
问题一 WARN 已传播：`MAP02, O01, Q02`。
桥接接口版本：`p1_to_p2_v2`；问题二直接读取附件 A：`False`。

配比迁移指标：
|    slope |   intercept |   spearman_R_loss |     rmse | scale   | loss_domain   |   n | role      |   domain_median_slope |   domain_median_spearman_R_loss |   sign_probability_positive | delta_p_profile_role   |
|---------:|------------:|------------------:|---------:|:--------|:--------------|----:|:----------|----------------------:|--------------------------------:|----------------------------:|:-----------------------|
| 0.268871 |    5.566847 |          0.812811 | 0.139874 | 1M      | __composite__ | 256 | composite |              0.308757 |                        0.302226 |                    1.000000 | paired_estimate        |
| 0.238602 |    4.152130 |          0.819749 | 0.123886 | 60M     | __composite__ | 256 | composite |              0.256855 |                        0.300365 |                    1.000000 | paired_estimate        |
| 0.083987 |    2.646454 |          0.734158 | 0.046505 | 1B      | __composite__ |  64 | composite |              0.046092 |                        0.132372 |                    0.924000 | paired_estimate        |

Scheffé 真实重拟合 bootstrap 有效率 `1.000`，原始系数最大抽样标准差 `2092.24`；原始项存在共线与尺度差异，解释以归一化响应 `R(p)` 而非单个系数为准。
`R(p_star)` 点值 `-2.289763`，bootstrap 中位数 `-1.697580`，95%区间 [`-3.568433`, `0.113031`]。

领域交互 bootstrap 分类：`{'complementary': 75, 'uncertain': 61}`。负的标准化二阶交互表示联合增加相对线性叠加更能降低 Loss（互补），正值表示竞争同一配比预算（替代）；区间跨零只记为不确定。
| domain_i        | domain_j          |   standardized_interaction |      ci_low |    ci_high |   probability_negative | relation      | interpretation              |
|:----------------|:------------------|---------------------------:|------------:|-----------:|-----------------------:|:--------------|:----------------------------|
| nih_exporter    | ubuntu_irc        |                 -68.540954 | -134.645900 | -13.287530 |               0.994000 | complementary | negative_joint_loss_synergy |
| ubuntu_irc      | hackernews        |                 -46.624483 |  -72.711034 |  -7.548095 |               0.992000 | complementary | negative_joint_loss_synergy |
| europarl        | hackernews        |                 -44.537607 |  -74.255322 |  -2.054280 |               0.978000 | complementary | negative_joint_loss_synergy |
| dm_mathematics  | hackernews        |                 -34.100525 |  -57.277397 |  -5.316547 |               0.994000 | complementary | negative_joint_loss_synergy |
| pubmed_central  | hackernews        |                 -30.510342 |  -43.213410 | -11.958681 |               1.000000 | complementary | negative_joint_loss_synergy |
| hackernews      | uspto_backgrounds |                 -30.042008 |  -44.375073 | -10.193024 |               1.000000 | complementary | negative_joint_loss_synergy |
| arxiv           | hackernews        |                 -29.572594 |  -42.784856 |  -8.657971 |               1.000000 | complementary | negative_joint_loss_synergy |
| dm_mathematics  | ubuntu_irc        |                 -27.843123 |  -42.086709 | -15.676619 |               1.000000 | complementary | negative_joint_loss_synergy |
| gutenberg_pg_19 | hackernews        |                 -27.533189 |  -43.477690 |  -5.786611 |               0.998000 | complementary | negative_joint_loss_synergy |
| wikipedia_en    | hackernews        |                 -26.879595 |  -41.557987 |  -5.519382 |               1.000000 | complementary | negative_joint_loss_synergy |
| hackernews      | pubmed_abstracts  |                 -26.350662 |  -39.667228 |  -6.008447 |               0.998000 | complementary | negative_joint_loss_synergy |
| gutenberg_pg_19 | ubuntu_irc        |                 -25.850681 |  -37.226465 | -14.887084 |               1.000000 | complementary | negative_joint_loss_synergy |

## 广义模型与等价量

最终接口角色：`L0_plus_quality_hs_plus_scale_dependent_Rp`；质量映射情景 `[0.5, 1.0, 1.5]`；配比强度 `scenario_not_identified_across_A_and_B_units`。
联合 bootstrap 有效率 `1.000`；配比强度使用 B7 Loss IQR 显式换算的预注册 profile，不视为附件直接识别值。
主身份映射且固定 `p_ref` 时：`L=1.689798+0.353980n^(-0.339977)+1.240306d^(-0.279878)+0.362040(1-Q)^(0.990288)`，其中 `n=N/1e9,d=D/1e9`。
标准弹性列 `epsilon_N/D/Q` 等于 `-x/L*dL/dx`；`epsilon_*_reducible` 另表示 `-x/(L-E)*dL/dx`，当 `L<=E` 时不定义。
非零配比物理门槛：`BLOCK_JOINT_P`；最低预测 `1.047781`，不可约损失 `E=1.689798`，低于 E 的网格行数 `52`。
| p_scenario         |   mixture_strength_profile |   rows |   min_prediction |   below_E_rows |
|:-------------------|---------------------------:|-------:|-----------------:|---------------:|
| p_ref              |                   0.250000 |     81 |         2.074832 |              0 |
| p_ref              |                   0.500000 |     81 |         2.074832 |              0 |
| p_ref              |                   1.000000 |     81 |         2.074832 |              0 |
| p_star_sensitivity |                   0.250000 |     81 |         1.818069 |              0 |
| p_star_sensitivity |                   0.500000 |     81 |         1.561306 |              4 |
| p_star_sensitivity |                   1.000000 |     81 |         1.047781 |             48 |
|      s_Q |      Q_A |   delta_Q |   finite_N_multiplier_minus_1 |   finite_D_multiplier_minus_1 | root_status_N   | root_status_D   |
|---------:|---------:|----------:|------------------------------:|------------------------------:|:----------------|:----------------|
| 0.500000 | 0.400000 |  0.050000 |                      0.078884 |                      0.113179 | ok_observed     | ok_observed     |
| 0.500000 | 0.400000 |  0.100000 |                      0.166380 |                      0.243352 | ok_observed     | ok_observed     |
| 0.500000 | 0.538956 |  0.050000 |                      0.079000 |                      0.113349 | ok_observed     | ok_observed     |
| 0.500000 | 0.538956 |  0.100000 |                      0.166645 |                      0.243754 | ok_observed     | ok_observed     |
| 0.500000 | 0.800000 |  0.050000 |                      0.079278 |                      0.113755 | ok_observed     | ok_observed     |
| 0.500000 | 0.800000 |  0.100000 |                      0.167285 |                      0.244724 | ok_observed     | ok_observed     |
| 1.000000 | 0.400000 |  0.050000 |                      0.166149 |                      0.243003 | ok_observed     | ok_observed     |
| 1.000000 | 0.400000 |  0.100000 |                      0.371683 |                      0.567339 | ok_observed     | ok_observed     |
| 1.000000 | 0.538956 |  0.050000 |                      0.166645 |                      0.243754 | ok_observed     | ok_observed     |
| 1.000000 | 0.538956 |  0.100000 |                      0.372985 |                      0.569474 | ok_observed     | ok_observed     |
| 1.000000 | 0.800000 |  0.050000 |                      0.168296 |                      0.246257 | ok_observed     | ok_observed     |
| 1.000000 | 0.800000 |  0.100000 |                      0.377592 |                      0.577040 | ok_observed     | ok_observed     |
| 1.500000 | 0.400000 |  0.050000 |                      0.262961 |                      0.392575 | ok_observed     | ok_observed     |
| 1.500000 | 0.400000 |  0.100000 |                      0.628050 |                      1.006994 | ok_observed     | ok_observed     |
| 1.500000 | 0.538956 |  0.050000 |                      0.264159 |                      0.394461 | ok_observed     | ok_observed     |
| 1.500000 | 0.538956 |  0.100000 |                      0.631713 |                      1.013550 | ok_observed     | ok_observed     |
| 1.500000 | 0.800000 |  0.050000 |                      0.249473 |                      0.371391 | ok_observed     | ok_observed     |
| 1.500000 | 0.800000 |  0.100000 |                      0.249473 |                      0.371391 | ok_observed     | ok_observed     |

## 验收

总体状态：**WARN**；PASS=37，WARN=3，FAIL=0。

WARN 降级说明：
- `M02` 1B独立验证：1B仅保留方向敏感性，不作精确迁移校准
- `M03` 绝对强度识别：保持强度情景，不报告伪精确点
- `M05` 非零配比不可约下界：存在违反时禁止联合优化p，固定p_ref主接口不受影响

## 修复前—修复后对照

- 修复前验收：`unknown` {}；修复后：`WARN` {'PASS': 37, 'WARN': 3, 'FAIL': 0}。两轮规则集合不同，PASS 数量不可直接作为优劣分数；以新增 RF01–RF11 及最终封版门槛是否通过为准。
- 经典模型：`unknown` → `M0`。
- 质量模型：`unknown` → `GQ2`；关键变化是验证口径、抽样和可调用接口，不强制模型名称改变。

## 结果文件

核心文件位于 `问题二/results_p12_final_v2/`：`scaling_data_audit.json`、`classic_scaling_parameters.json`、`classic_warmup_sensitivity.csv`、`classic_external_validation.csv`、`quality_model_parameters.json`、`problem1_bridge.json`、`domain_substitution_complementarity.csv`、`problem1_mixture_response_bootstrap.npz`、`mixture_transfer_bootstrap.csv.gz`、`mixture_physical_gate.json`、`elasticity_grid.csv`、`generalized_scaling_bootstrap.csv.gz`、`units_contract.json`、`problem2_acceptance_report.json`。

## 可复现运行方式

```bash
python 问题二/problem2.py all --config <config>
```
