# 问题二最终封版修复规划

> 本文件是问题二最后一次修复的执行规范。修复完成后冻结问题一、问题二接口，后续仅消费冻结产物进入问题三和问题四。旧版 `results_p12_final_v1/` 与对应报告保留，新结果写入 `results_p12_final_v2/`。

## 1. 修复目标

1. 严格落实题面“问题二通过问题一输出引入附件 A 信息，不重复读取原始文件”。问题一新增 `problem2_bridge_input.npz`，问题二只读取该冻结接口及问题一已有 CSV/JSON/NPZ 结果。
2. 同时输出标准总 Loss 弹性与可约 Loss 弹性：
   - 标准弹性 `epsilon_x_total=-x/L*dL/dx`；
   - 可约弹性 `epsilon_x_reducible=-x/(L-E)*dL/dx`。
   论文主文采用标准弹性，可约弹性作为机制补充；质量—参数替代率不受共同分母选择影响。
3. 对所有非零配比强度情景显式检查 `L>=E`。只要存在违反，不允许联合优化 `p`，但保留情景结果用于说明加性迁移接口的边界。
4. 结果报告补入 B2/B4/B5/B10 外部验证、B11/B12 辅助审计、B3 未使用的替代关系，以及问题一 Scheffé 交互导出的领域替代/互补表。
5. 补做 B1 早期检查点敏感性：删除 `D<2` B tokens 的 32 点后重拟合 M0，与全量参数比较，不以删点结果反向调参。

## 2. 冻结接口

### 2.1 问题一输出

`problem2_bridge_input.npz` 至少包含：

- 17 个训练域与 13 个验证 Loss 域的固定顺序；
- A4/A5 的闭合配比、Loss、训练 index；
- A6–A11 三个检验尺度的配比、Loss 与 index；
- 二阶 Scheffé 点估计系数、训练行 bootstrap 系数和对应 `p_ref` 抽样；
- A5 域内 IQR、`p_ref`、逐域上界、支持尺度和支持半径；
- Ridge alpha、抽样索引 hash 与接口版本。

该文件属于问题一派生产物。问题二不得再读取 `A_data_value/regmix_tables/`。

### 2.2 问题二主接口

完整结构保留为

```math
L(N,D,Q_A,p)=L_0(N,D)+\Delta_Q(h_{s_Q}(Q_A))
+\lambda_0(N/N_0)^{\delta_p}R(p).
```

主接口固定 `p=p_ref` 且 `lambda_0=0`；非零 `lambda_0` 为不可识别强度情景。只有迁移、绝对强度、支持域和 `L>=E` 四类门槛全部通过时，才允许 `JOINT_P_READY=true`。

## 3. 新版本输出

- 问题一桥接版结果：`问题一/results_p12_final_v2/`
- 问题一桥接版报告：`问题一/reports_p12_final_v2/RESULTS_REPORT.md`
- 问题二最终结果：`问题二/results_p12_final_v2/`
- 问题二最终报告：`问题二/reports_p12_final_v2/RESULTS_REPORT.md`
- 最终汇总：`reports/P2_FINAL_REPAIR_RESULTS.md`

旧目录不删除、不覆盖。

## 4. 验收标准

1. 问题二源码中不存在对 `A_data_value` 或 `regmix_tables` 的运行时读取。
2. 问题一桥接文件被 manifest 登记，问题二校验其 hash 后消费。
3. `elasticity_grid.csv` 同时含 `epsilon_*_total` 与 `epsilon_*_reducible`，原 `epsilon_*` 明确等于标准总 Loss 弹性。
4. 非零配比情景输出最小 Loss、低于 E 的行数和门槛状态；存在违反时 `JOINT_P_READY=false`。
5. 32 点 warm-up 敏感性有独立结果文件；若主要指数相对变化不超过 1%，视为主结论稳健。
6. B1–B12 均有明确账本角色；B3 因题面允许 B2/B3 二选一，可审计但不作为独立证据重复使用。
7. 全部测试通过，最终验收无 FAIL，manifest 写后 hash 校验 PASS。
