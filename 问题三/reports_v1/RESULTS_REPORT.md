# 问题三建模与验收结果报告

> 版本：problem3_v1。按用户要求，本轮不生成图表；本报告完全由数值结果文件自动生成。

## 1. 结论与接口状态

- 总体验收：**PASS**；PASS=30，WARN=0，FAIL=0。
- 接口状态：`{'PROBLEM3_CENTRAL_READY': True, 'PROBLEM3_UNCERTAINTY_READY': True, 'PROBLEM3_READY': True}`。
- 主模型严格承接问题二冻结版本 M0+GQ2：固定 `p=p_ref`、`lambda0=0`，配比不参与联合优化。
- C7 只用于核验上下文长度情景；其他架构字段即使存在异常也不进入 Loss 或成本公式。
- 主回答使用 operational_extended 支持域，同时单独运行 strict_joint 以标记观测支持与外推风险。

## 2. 模型与约束

$$L=E+A(N/10^9)^{-\alpha}+B(D/10^9)^{-\beta}+c_Q(1-Q_{eff})^{\nu_Q},$$

其中 $Q_{eff}=Q_0+s_Q(Q_A-Q_0)$，主情景 $s_Q=1$。成本约束为

$$6ND+D[g(Q_A)-g(Q_0)]_++\eta NDL_{ctx}\le C,$$

且主值 $\eta=2\times10^{-4}$。计算中 $N,D$ 使用绝对单位，结果表同时输出十亿单位。

## 3. 4096 上下文的三档预算结果

|质量成本|预算 FLOPs|N(B)|D(B tokens)|Q_A|预测 Loss|训练份额|质量份额|注意力份额|证据标签|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|exponential|1e+19|0.259187|4.82101|0.671056|3.16898|0.749724|0.147913|0.102362|one_component_extrapolated|
|exponential|1e+22|5.44886|244.275|0.999999|2.15491|0.798614|0.0923488|0.109037|joint_observed|
|exponential|1e+24|39.5804|3653.81|0.999999|1.91601|0.867715|0.0138133|0.118472|bridge_gap_extrapolated|
|logarithmic|1e+19|0.337313|2.95276|0.999999|3.11805|0.597603|0.320805|0.0815927|one_component_extrapolated|
|logarithmic|1e+22|5.04774|281.627|0.999999|2.14976|0.852947|0.0305976|0.116456|joint_observed|
|logarithmic|1e+24|39.1302|3732.41|0.999999|1.91566|0.876301|0.00405511|0.119644|bridge_gap_extrapolated|
|power|1e+19|0.215205|6.8142|0.538956|3.17962|0.879869|0|0.120131|one_component_extrapolated|
|power|1e+22|5.5584|235.394|0.999999|2.15634|0.785049|0.107766|0.107185|joint_observed|
|power|1e+24|39.7119|3631.33|0.999999|1.91611|0.865241|0.0166246|0.118134|bridge_gap_extrapolated|

上述每个方案均另存 17 个领域的 Token 数，满足 $D_i=p_{ref,i}D$；这就是固定配比在问题三中的实际用途。

## 4. 不确定性

- 三档锚点使用 500 个有效联合参数抽样计算 95% 区间。
- 连续路径按预注册主成分分层选取 200 个抽样，用于估计结构转移出现概率。
- 只有出现概率不低于 70% 的同类事件才标为稳定；其余只称中心参数下候选转移。

## 5. 结构性转移

|成本函数|上下文|类型|事件|预算中点|log10区间宽|出现概率|稳定|
|---|---:|---|---|---:|---:|---:|---|
|exponential|2048|secondary|Q_increment_slope_break|9.085e+19|0.0833333|1|是|
|exponential|2048|primary|quality_saturation|1.263e+20|0.00520833|1|是|
|exponential|2048|secondary|D_B_slope_break|2.371e+20|0.0833333|1|是|
|exponential|4096|secondary|Q_increment_slope_break|9.085e+19|0.0833333|1|是|
|exponential|4096|primary|quality_saturation|1.234e+20|0.00520833|1|是|
|exponential|4096|secondary|D_B_slope_break|2.371e+20|0.0833333|1|是|
|exponential|8192|secondary|Q_increment_slope_break|9.085e+19|0.0833333|1|是|
|exponential|8192|primary|quality_saturation|1.176e+20|0.00520833|1|是|
|exponential|8192|secondary|D_B_slope_break|2.371e+20|0.0833333|0.99|是|
|exponential|32768|secondary|Q_increment_slope_break|7.499e+19|0.0833333|1|是|
|exponential|32768|primary|quality_saturation|9.362e+19|0.00520833|1|是|
|exponential|32768|secondary|D_B_slope_break|1.957e+20|0.0833333|0.855|是|
|exponential|131072|primary|quality_saturation|6.007e+19|0.00520833|1|是|
|exponential|131072|secondary|Q_increment_slope_break|6.190e+19|0.0833333|1|是|
|power|2048|primary|quality_investment_start|1.407e+19|0.00520833|1|是|
|power|2048|secondary|Q_increment_slope_break|4.217e+19|0.0833333|1|是|
|power|2048|primary|quality_saturation|6.854e+19|0.00520833|1|是|
|power|2048|secondary|N_B_slope_break|1.101e+20|0.0833333|1|是|
|power|2048|secondary|D_B_slope_break|1.101e+20|0.0833333|1|是|
|power|4096|primary|quality_investment_start|1.342e+19|0.00520833|1|是|
|power|4096|secondary|Q_increment_slope_break|4.217e+19|0.0833333|1|是|
|power|4096|primary|quality_saturation|6.691e+19|0.00520833|1|是|
|power|4096|secondary|N_B_slope_break|1.101e+20|0.0833333|1|是|
|power|4096|secondary|D_B_slope_break|1.101e+20|0.0833333|1|是|
|power|8192|primary|quality_investment_start|1.248e+19|0.00520833|1|是|
|power|8192|secondary|Q_increment_slope_break|5.109e+19|0.0833333|1|是|
|power|8192|primary|quality_saturation|6.378e+19|0.00520833|1|是|
|power|8192|secondary|N_B_slope_break|1.101e+20|0.0833333|1|是|
|power|8192|secondary|D_B_slope_break|1.101e+20|0.0833333|1|是|
|power|32768|secondary|Q_increment_slope_break|3.481e+19|0.0833333|1|是|
|power|32768|primary|quality_saturation|5.264e+19|0.00520833|1|是|
|power|32768|secondary|N_B_slope_break|9.085e+19|0.0833333|1|是|
|power|32768|secondary|D_B_slope_break|9.085e+19|0.0833333|1|是|
|power|131072|secondary|Q_increment_slope_break|3.481e+19|0.0833333|1|是|
|power|131072|primary|quality_saturation|3.502e+19|0.00520833|1|是|
|power|131072|secondary|N_B_slope_break|7.499e+19|0.0833333|0.795|是|
|power|131072|secondary|D_B_slope_break|7.499e+19|0.0833333|1|是|

注意力临界上下文长度为 30000。它比较的是不同上下文场景的成本结构，不是预算路径上的结构转移，因此未混入转移事件表。

## 6. 敏感性与支持边界

共运行 234 条敏感性结果，覆盖质量映射斜率、质量成本幅度和形状、注意力系数、经验质量上界以及 strict_joint 支持域。
中心 45 个主场景中，外推标记数量为 30/45。外推 Loss 是模型推断，不作为新的实测证据。

## 7. 数据审计

- 问题二冻结输入 hash 不匹配数：0。
- C7 形状：[45, 7]；上下文核验：PASS；架构字段 WARN 行：6。
- C7 非上下文字段进入问题三公式：[]。

## 8. 验收与使用边界

全部硬门禁通过。问题四只能读取 `problem3_bridge.json` 中 `PROBLEM3_READY=True` 的条目，并保留 evidence/support 标签。

局限：配比绝对强度仍未被问题二识别，所以本题没有把 $p$ 当连续决策变量；质量成本函数为题设情景，并非从训练日志估计；超出问题二联合观测域的结果必须按外推解释。

## 9. 复现

```bash
python 问题三/problem3.py --config 问题三/config/problem3_v1.json --stage all
```
