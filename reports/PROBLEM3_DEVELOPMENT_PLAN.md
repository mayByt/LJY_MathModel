# 问题三开发与验收规划

> 本文是 ANALYSIS_MODELING_REPORT.md 第 7 节的工程化补充，承接问题一与问题二最终冻结版本。本文只规定问题三的数据处理、异常处理、模型实现、求解、结果文件和验收标准，不开始编写或运行问题三模型，也不覆盖问题一、问题二的任何结果。问题三本轮只交付建模、验收和结果报告，暂不生成图表。

## 1. 目标、边界与冻结结论

### 1.1 问题三必须回答的内容

1. 在预算 $C$ 下，求参数量 $N$、训练 Token 数 $D$ 和质量水平 $Q$ 的最优配置。
2. 领域配比 $p$ 由前两问确定，并把最优总 Token 数拆为 $D_i=p_iD$；说明为什么本问不再联合优化 $p$。
3. 对题设三种质量成本函数分别求解，比较成本函数选择对最优配置和最优 Loss 的影响。
4. 依据 C7 设置上下文长度 $L_{ctx}$ 外生情景，给出 $L_{ctx}^{crit}=6/\eta$ 并完成可行取值敏感性。
5. 在 $10^{19}$、$10^{22}$、$10^{24}$ FLOPs 三档预算及连续预算路径上识别结构性转移。
6. 传播问题一、问题二参数与质量映射不确定性，输出区间、外推等级和稳定转移概率。
7. 生成可供问题四读取的冻结接口、manifest、验收报告和结果报告。

### 1.2 已冻结的上游状态

问题三只能读取问题一、二的最终 V2 产物。当前问题二接口状态为：

| 接口 | 状态 | 问题三动作 |
| --- | --- | --- |
| CENTRAL_MODEL_READY | True | 允许读取中心广义标度律 |
| FIXED_P_READY | True | 允许固定 $p=p_{ref}$ 优化 |
| JOINT_P_READY | False | 禁止把 17 维 $p$ 放入问题三寻优 |
| P12_FREEZE_READY | True | 允许开始问题三 |

JOINT_P_READY=False 不是问题三失败，而是已经识别出的能力边界。原因是 1B 配比迁移指标略低于预注册阈值、A/B 绝对配比强度未识别、非零配比情景在部分网格违反 $L\ge E$。问题三不得绕过该门禁。

### 1.3 明确不做的事情

- 不读取附件 A、B 原始数据重新拟合问题一或问题二。
- 不修改问题一、问题二参数、阈值、WARN 或结果文件。
- 不把 $p_{star}$ 当成唯一最优配比，也不联合优化 17 个配比分量。
- 不把 C7 的行频数解释为现实模型市场中的上下文长度概率。
- 不把 max_position_embeddings 解释为训练时真实平均序列长度。
- 不用 C7 的层数、头数、宽度或训练数据量拟合问题三；这些字段存在已发现的错误。
- 不从三种题设质量成本函数中按拟合优度挑选“真实函数”；附件没有质量处理成本观测，三者是并列情景。
- 不把估算或边界支持区间内的结果写成观测内结论。
- 不生成 PDF 或其他图表；只生成 CSV、JSON、日志和 RESULTS_REPORT.md。

## 2. 输入数据与接口契约

### 2.1 允许读取的上游文件

问题三从 问题二/results_p12_final_v2 读取：

| 文件 | 用途 | 入口检查 |
| --- | --- | --- |
| generalized_scaling_parameters.json | 中心模型、参数、支持域、接口状态 | 模型必须为 M0+GQ2，FIXED_P_READY=True |
| generalized_scaling_bootstrap.csv.gz | 500 组联合参数抽样 | success 比例至少 95%，参数有限且物理符号正确 |
| problem1_bridge.json | $p_{ref}$、$Q_{ref}$、领域顺序、WARN 传播 | 17 域齐全，$p_{ref}$ 非负且和为 1 |
| units_contract.json | 十亿单位与绝对单位换算 | $N_{abs}=10^9n$，$D_{abs}=10^9d$ |
| problem2_acceptance_report.json | 上游 PASS/WARN/FAIL 与接口门禁 | FAIL=0，P12_FREEZE_READY=True |
| problem2_manifest.json | 输入、输出和代码指纹 | 被消费文件 hash 全部匹配 |
| quality_parameter_equivalence.csv | 结果解释与回归对照 | 只作诊断，不参与问题三参数拟合 |
| elasticity_grid.csv | 边际效用对照 | 只作诊断，不替代问题三 KKT 计算 |

问题三直接读取附件 C 的唯一文件是：

- real_attachments/C_efficiency_evolution/model_architecture_metadata.csv，即 C7。

### 2.2 问题二中心公式回归基准

主场景固定 $p=p_{ref}$、$s_Q=1$、$\lambda_0=0$：

$$
L(N,D,Q)=
1.6897975628980855
+0.3539803206859227\,n^{-0.33997658188127156}
+1.2403055835550174\,d^{-0.27987812854272615}
+0.3620399283468315(1-Q)^{0.9902880682591864},
$$

其中 $n=N_{abs}/10^9$、$d=D_{abs}/10^9$，$Q_{ref}=0.5389559587975633$。

程序必须在一组固定锚点上同时调用问题二预测函数和问题三重建函数，最大绝对差不超过 $10^{-10}$。若不满足，问题三停止。

### 2.3 问题一、二 WARN 的强制传播

| 来源 | 已知 WARN | 问题三强制动作 |
| --- | --- | --- |
| 问题一 MAP02 | 17 域推断映射排序不稳定 | 使用 $Q_{ref}$ 抽样，不解释 inferred 域精确排序 |
| 问题一 O01 | $p_{star}$ 在边界且非唯一 | 主配比固定 $p_{ref}$，不把 $p_{star}$ 当最优真值 |
| 问题一 Q02 | 极端样本名单方案敏感 | 不使用样本 Top/Bottom 排名 |
| 问题二 M02 | 1B 配比迁移 Spearman 0.734158 | 不做精确跨规模配比强度推断 |
| 问题二 M03 | A/B 绝对配比强度未识别 | 固定 $\lambda_0=0$ |
| 问题二 M05 | 非零配比情景有 52 行 $L<E$ | 禁止联合优化 $p$ |

### 2.4 bootstrap 重组规则

generalized_scaling_bootstrap.csv.gz 中每行同时保存了参数抽样和当时轮换的 $s_Q$、配比情景。问题三只读取以下抽样参数：

- classic_E、classic_A、classic_B、classic_alpha、classic_beta；
- quality_c_Q、quality_nu_Q；
- Q_ref_draw。

问题三不得直接沿用每行原有的 s_Q、mixture_strength_scenario 或 p_source。每个抽样都重新施加问题三冻结场景：

$$
p=p_{ref},\qquad \lambda_0=0,\qquad s_Q\in\{0.5,1,1.5\}.
$$

这样可以避免把问题二用于覆盖不同情景的轮换字段误当成问题三随机参数。

## 3. C7 数据审计与异常处理

### 3.1 当前文件的可复现事实

C7 当前为 45 行、7 列，无缺失、无重复模型名。max_position_embeddings 的离散分布为：

| $L_{ctx}$ | 行数 |
| ---: | ---: |
| 2048 | 18 |
| 4096 | 8 |
| 8192 | 7 |
| 32768 | 11 |
| 131072 | 1 |

这五个值只用于构造五个外生情景，不按 18:8:7:11:1 加权。

### 3.2 已确认的数据错误风险

“无缺失”不代表 C7 完全正确。简单一致性检查 $d_{model}/n_{heads}$ 是否为整数，至少发现 6 行不满足，包括 Pythia-12B、Falcon-40B、Cerebras-GPT-111M、590M、1.3B 和 DeepSeek-LLM-7B。官方配置抽查进一步确认：

- Pythia-12B 的 hidden_size 应为 5120，而 C7 写为 4096；
- Falcon-40B 的注意力头数应为 128，而 C7 写为 232；
- Cerebras-GPT-111M、590M、1.3B 的头数分别应为 12、12、16，而 C7 写为 10、18、24；
- DeepSeek-LLM-7B 的头数应为 32，而 C7 写为 30。

另有能通过整除检查但仍可能与官方配置不一致的记录。因此开发阶段采用“原始表不改写、修正表另存、问题三只信任已核验的上下文列”的规则。

### 3.3 C7 清洗规则

1. 原始 C7 只读，保存 SHA-256。
2. 输出 c7_context_audit.csv，保留原始 7 列，并增加：
   - schema_valid；
   - duplicate_model；
   - positive_integer_context；
   - architecture_consistency_flag；
   - official_context_status；
   - official_source；
   - p3_use_flag；
   - exclusion_reason。
3. n_layers、n_heads、d_model、vocab_size、training_data_TB 均不进入问题三公式。
4. max_position_embeddings 必须为正整数；每个离散情景至少有一个经官方配置或模型卡核验的代表模型。
5. 单行其他架构字段错误不自动删除其上下文值；上下文值独立核验。
6. 若某上下文值没有任何可核验代表，则从主情景中移除并记 WARN；不得自行补造新值。
7. C7 的最大窗口只是架构配置上限。报告统一称“外生上下文窗口情景”，不称“实际训练序列长度”。
8. Mistral 等滑动窗口架构的真实硬件成本可能低于题设代理；主模型仍严格使用题设 $C_{attn}=\eta NDL_{ctx}$，差异只写入局限性。

## 4. 变量、单位和场景注册

### 4.1 决策变量

| 变量 | 含义 | 内部单位 |
| --- | --- | --- |
| $N$ | 参数个数 | 绝对参数数 |
| $D$ | 训练 Token 数 | 绝对 Token 数 |
| $Q_A$ | 问题一质量尺度上的投入后质量 | 无量纲，$[Q_0,1]$ |

### 4.2 固定量和外生情景

| 符号 | 含义 | 取值 |
| --- | --- | --- |
| $p$ | 17 域配比 | 固定为 $p_{ref}$ |
| $Q_0$ | 质量投资起点 | 中心值 $Q_{ref}$，bootstrap 用 $Q_{ref}^{(b)}$ |
| $L_{ctx}$ | 上下文窗口 | C7 核验后的五个离散值 |
| $C$ | 总 FLOPs 预算 | 三档锚点加连续网格 |
| $g$ | 质量成本函数 | 指数、幂函数、对数渐进三种 |
| $s_Q$ | A 到 B 的质量尺度斜率 | 主值 1，敏感性 0.5、1.5 |
| $\eta$ | 注意力成本系数 | 主值 $2\times10^{-4}$ |

### 4.3 质量映射

问题三的成本函数使用题面和问题一同尺度的 $Q_A$。Loss 中使用问题二映射后的有效质量：

$$
Q_{eff}=Q_0+s_Q(Q_A-Q_0).
$$

主实现不依赖静默 clip。数值上把质量上界设置为

$$
Q_{sat}=\min\left(1,\;Q_0+\frac{1-\varepsilon-Q_0}{s_Q}\right),
$$

因为超过该点只增加成本、不再降低问题二预测 Loss。$\varepsilon=10^{-6}$。另做 $Q_A\le0.880183$ 的问题一经验范围敏感性。

### 4.4 配比对应的领域 Token

固定配比并不等于忽略领域资源分配。每个最优方案输出：

$$
D_i=p_{ref,i}D,\qquad i=1,\ldots,17.
$$

必须检查 $D_i\ge0$ 且 $\sum_iD_i=D$。配比无额外算力成本，因此它不出现在成本约束中。

## 5. 目标函数与成本约束

### 5.1 冻结广义标度律

对每个中心或 bootstrap 参数组：

$$
L=E+A\left(\frac{N}{10^9}\right)^{-\alpha}
+B\left(\frac{D}{10^9}\right)^{-\beta}
+c_Q(1-Q_{eff})^{\nu_Q}.
$$

主问题固定 $p=p_{ref}$ 和 $\lambda_0=0$，因此不再包含未识别的配比绝对强度项。

### 5.2 三类成本

$$
C_{train}=6ND,
$$

$$
C_Q=D\,[g(Q_A)-g(Q_0)]_+,
$$

$$
C_{attn}=\eta NDL_{ctx}.
$$

题设三种 $g$ 为：

$$
g_{exp}(Q)=10^7e^{6Q},
\qquad
g_{pow}(Q)=5\times10^9Q^4,
\qquad
g_{log}(Q)=2\times10^9\ln(1+10Q).
$$

总约束：

$$
C_{train}+C_Q+C_{attn}\le C.
$$

所有成本使用绝对 $N,D$。禁止把十亿参数、十亿 Token 直接代入成本公式。

### 5.3 支持域和外推等级

主程序同时运行两种支持模式：

1. strict_joint：问题二经典项与质量项共同观测覆盖，
   $N\in[0.070542,11.965825]$ B，
   $D\in[10,299.893]$ B。
2. operational_extended：为回答题设三档预算，使用问题二已声明的边界包络，
   $N\in[0.070542,10000]$ B，
   $D\in[0.1,36000]$ B。

operational_extended 是主求解路径，但每个结果必须标记：

- joint_observed；
- one_component_extrapolated；
- bridge_gap_extrapolated；
- boundary_supported_extrapolated；
- out_of_contract。

strict_joint 不可行不是求解失败，应输出 infeasible_under_strict_support。任何超出 operational_extended 的解均为 FAIL，不允许继续扩大边界。

### 5.4 预算和上下文场景

必须包含题设三档预算：

$$
C\in\{10^{19},10^{22},10^{24}\}\ {\rm FLOPs}.
$$

为寻找转移点，在 $10^{19}$ 到 $10^{24}$ 之间生成 61 个对数均匀预算点，并显式并入三个锚点。每个质量成本函数和每个上下文值分别求一条连续路径。

## 6. 解析降维与数值求解

### 6.1 消去 $D$

总成本可重写为：

$$
C_{tot}
=D\left[(6+\eta L_{ctx})N+\Delta g(Q_A)\right],
\qquad
\Delta g(Q_A)=[g(Q_A)-g(Q_0)]_+.
$$

对固定 $(N,Q_A)$，Loss 对 $D$ 单调下降，因此：

$$
D^{max}(N,Q_A)
=\frac{C}{(6+\eta L_{ctx})N+\Delta g(Q_A)}.
$$

考虑上界后取

$$
D^*(N,Q_A)
=\min\{D_{max},D^{max}(N,Q_A)\},
$$

并要求 $D^*\ge D_{min}$。问题由三维约束优化降为二维有界优化。

### 6.2 中心求解流程

每个 $(g,L_{ctx},C,s_Q,support\_mode)$ 场景：

1. 在 $(\log N,Q_A)$ 上建立粗网格，剔除 $D^*<D_{min}$ 的点。
2. 选取目标最小的前 20 个不同网格点作局部初值。
3. 使用 L-BFGS-B 或 trust-constr 做二维局部优化。
4. 沿预算递增路径使用上一预算最优解作 continuation 初值。
5. 取可行且目标最小的候选。
6. 恢复 $D^*$，复算三类成本、约束、Loss、梯度、KKT 和领域 Token。
7. 对三档预算的全部 $3\times5\times3=45$ 个中心场景，再运行完整三维 SLSQP/trust-constr 作独立复核。

低维问题通常 CPU 更快。仅当批量 bootstrap 的 GPU 向量化实测更快且数值结果一致时使用 GPU；否则记录 CPU 回退原因。

### 6.3 经典闭式解回归

在 $Q_A=Q_0$ 且无边界激活时，令

$$
\bar C=\frac{C}{(6+\eta L_{ctx})10^{18}},
$$

则经典部分有：

$$
n^*=
\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\bar C^{\beta/(\alpha+\beta)},
\qquad
d^*=\frac{\bar C}{n^*}.
$$

该闭式解只用作初值和回归测试，不替代含质量投资的正式优化。

## 7. KKT 解释与结构性转移

### 7.1 边际 Loss 降低/算力

在内点且预算活跃时，KKT 要求：

$$
\frac{-L_N}{C_N}
=\frac{-L_D}{C_D}
=\frac{-L_Q}{C_Q'},
$$

其中：

$$
C_N=(6+\eta L_{ctx})D,
$$

$$
C_D=(6+\eta L_{ctx})N+\Delta g(Q_A),
$$

$$
C_Q'=Dg'(Q_A).
$$

这三个比值分别表示把一单位额外 FLOPs 投给参数、Token 和质量时能降低多少 Loss。

### 7.2 主要结构转移定义

固定成本函数和上下文情景，记最优活跃集合为：

$$
\mathcal A(C)=
\{Q=Q_0,\ Q=Q_{sat},\
N=N_{min/max},\
D=D_{min/max},\
C_{tot}=C\}.
$$

若相邻预算区间两侧满足

$$
\mathcal A(C^-)\ne\mathcal A(C^+),
$$

则定义为主要结构性转移。重点事件包括：

- 质量投入开启：$Q$ 离开 $Q_0$；
- 质量饱和：$Q$ 到达 $Q_{sat}$；
- 参数或 Token 离开/到达支持边界；
- strict_joint 从不可行转为可行。

### 7.3 次要路径断点

对 $x\in\{N,D,Q-Q_0\}$ 计算局部配置弹性：

$$
\theta_x(C)=\frac{d\log x^*}{d\log C}.
$$

若活跃集合未变化，但分段线性拟合相对单直线的 BIC 改善至少 10，且断点前后斜率差至少 0.15，则记为次要结构候选。阈值 0.10、0.20 作敏感性。

转移位置先由 61 点路径夹逼，再对预算做二分细化，直到 $\log_{10}C$ 区间宽度不超过 0.01。

### 7.4 注意力临界值

$$
\frac{C_{attn}}{C_{train}}=\frac{\eta L_{ctx}}6,
\qquad
L_{ctx}^{crit}=\frac6\eta=30000.
$$

32768 和 131072 位于临界值以上。该临界值描述上下文场景的成本结构，不是预算变化导致的转移，因为固定 $L_{ctx}$ 时该比值与 $C,N,D,Q$ 均无关。

## 8. 不确定性和敏感性

### 8.1 上游参数传播

- 三档锚点：使用全部 500 个问题二联合参数抽样。
- 连续转移路径：从 500 个抽样中预注册分层选择 200 个，覆盖 $E,A,B,\alpha,\beta,c_Q,\nu_Q,Q_0$ 的联合分布；保存 draw_id，不按结果挑选。
- 每个抽样重新计算 $Q_0$、质量映射、最优解、成本和转移事件。

### 8.2 必做敏感性

| 维度 | 主值 | 敏感性 |
| --- | --- | --- |
| 质量映射 | $s_Q=1$ | 0.5、1.5 |
| 质量成本 | 三种函数并列 | 各参数 $\pm20\%$ |
| 注意力系数 | $2\times10^{-4}$ | $\pm20\%$ |
| Q 上界 | 题设 $Q\le1$ | 问题一经验上界 0.880183 |
| 支持域 | operational_extended | strict_joint |
| 配比 | 固定 $p_{ref}$ | 仅传播 $p_{ref}$ 与 $Q_{ref}$ 抽样，不联合优化 |
| 转移斜率阈值 | 0.15 | 0.10、0.20 |

只有在至少 70% 的有效转移路径抽样中出现同类型事件，才称“稳定结构性转移”；否则称“中心参数下的候选转移”。

## 9. 代码结构与执行阶段

### 9.1 建议目录

~~~text
问题三/
├── config/
│   └── problem3_v1.json
├── src/problem3/
│   ├── data.py
│   ├── contracts.py
│   ├── cost.py
│   ├── objective.py
│   ├── optimize.py
│   ├── transitions.py
│   ├── uncertainty.py
│   ├── acceptance.py
│   └── reporting.py
├── tests/
├── problem3.py
├── results_v1/
└── reports_v1/
~~~

正式结果写入新目录，不覆盖问题一、问题二及以后已验收的问题三版本。

### 9.2 命令阶段

| 阶段 | 任务 | 主要输出 |
| --- | --- | --- |
| audit | 上游 hash、接口状态、C7 审计 | problem3_data_audit.json、c7_context_audit.csv |
| scenarios | 注册预算、上下文、成本、支持域和映射情景 | scenario_registry.csv |
| solve | 中心二维优化和三维复核 | optimal_allocations.csv、constraint_checks.csv |
| transitions | 连续预算追踪和断点细化 | transition_points.csv、kkt_diagnostics.csv |
| uncertainty | 500 锚点抽样、200 路径抽样和敏感性 | bootstrap_optimal_allocations.csv.gz、transition_bootstrap.csv.gz |
| accept | 全部硬门禁与 WARN | problem3_acceptance_report.csv/json |
| report | 自动读取结果生成报告 | reports_v1/RESULTS_REPORT.md |
| all | 按上述顺序执行 | manifest、run_log 和全部产物 |

## 10. 结果文件与问题四接口

### 10.1 问题三正式输出

| 文件 | 内容 |
| --- | --- |
| problem3_data_audit.json | 上游状态、hash、C7 schema、错误和可用情景 |
| c7_context_audit.csv | C7 原始字段、核验状态和问题三使用标志 |
| scenario_registry.csv | 全部预注册场景及角色 |
| optimal_allocations.csv | 中心最优 $N,D,Q,p$、Loss、成本、支持等级 |
| domain_token_allocations.csv | 每个最优方案的 17 域 $D_i$ |
| constraint_checks.csv | 预算、边界、非负、单位和求解状态 |
| kkt_diagnostics.csv | 边际收益/FLOP、乘子、残差、活跃集合 |
| transition_points.csv | 中心转移类型、区间和细化结果 |
| bootstrap_optimal_allocations.csv.gz | 三档预算的 500 抽样结果 |
| transition_bootstrap.csv.gz | 200 条不确定路径的转移事件 |
| sensitivity_summary.csv | $s_Q$、成本参数、$\eta$、Q 上界和支持域敏感性 |
| problem3_acceptance_report.csv/json | PASS/WARN/FAIL 及接口状态 |
| problem3_manifest.json | 配置、代码、输入输出 hash 和运行环境 |
| run_log.json | 阶段耗时、CPU/GPU、WARN、异常和恢复动作 |

### 10.2 给问题四的桥接

problem3_bridge.json 只收录通过验收的三档预算方案，字段至少包括：

- budget_FLOPs、context_length、quality_cost_type；
- N_abs、D_abs、Q_A、Q_eff、p_ref、domain_D；
- predicted_loss 及参数区间；
- 三类成本和份额；
- support_status、extrapolation_flag；
- transition_regime；
- 上游 manifest hash；
- PROBLEM3_READY。

问题四只允许读取 PROBLEM3_READY=True 的条目。问题三的外推 Loss 必须带 evidence 标签进入问题四，不能与观测内结果等权。

## 11. 自动测试与验收标准

### 11.1 数据与接口门禁

| ID | 检查 | PASS 标准 | 失败动作 |
| --- | --- | --- | --- |
| D01 | 上游冻结状态 | P12_FREEZE_READY=True、FIXED_P_READY=True、FAIL=0 | 停止 |
| D02 | 上游 hash | 所有消费文件与问题二 manifest 一致 | 停止 |
| D03 | C7 schema | 45×7、固定列、模型名无重复 | 停止并报告版本变化 |
| D04 | C7 上下文 | 正整数，五类当前回归值可复现 | 未核验类别移除并 WARN |
| D05 | C7 错误隔离 | 非上下文字段不进入问题三公式 | 停止 |
| D06 | 配比 | 17 维非负、和误差不超过 $10^{-12}$ | 停止 |

### 11.2 公式与单位

| ID | 检查 | PASS 标准 |
| --- | --- | --- |
| M01 | Loss 重建 | 与问题二预测函数锚点最大差 $\le10^{-10}$ |
| M02 | 质量零增量 | $C_Q(Q_0)=0$，绝对误差 $\le10^{-6}$ FLOPs |
| M03 | 成本单调性 | 三种 $g$ 在合法区间有限且严格递增 |
| M04 | 绝对单位 | $C_{train}=6(n10^9)(d10^9)$ 回代一致 |
| M05 | 上下文临界 | $L_{ctx}^{crit}=30000$，五个成本比回代一致 |
| M06 | 质量映射 | $Q_A=Q_0$ 时 $Q_{eff}=Q_0$，且不超 $[0,1]$ |

### 11.3 优化与物理门禁

| ID | 检查 | PASS 标准 | 备注 |
| --- | --- | --- | --- |
| O01 | 约束可行 | 最大归一化违反量 $\le10^{-8}$ | 不只看 solver success |
| O02 | 成本非负 | 三类成本均 $\ge0$ | 任何负值 FAIL |
| O03 | 预算使用 | 非全变量上界时相对松弛 $\le10^{-6}$ | 全上界允许剩余预算 |
| O04 | 二维/三维一致 | 45 个中心场景最优 Loss 相对差 $\le10^{-6}$ | 配置差异另报 |
| O05 | 网格复核 | 加密网格与最终最优 Loss 相对差 $\le10^{-4}$ | 防局部最优 |
| O06 | 可重复性 | 同种子中心结果差 $\le10^{-10}$ | |
| O07 | 预算单调值函数 | 同情景最优 Loss 随预算不增加 | 单个 $N,D,Q$ 不强制单调 |
| O08 | 上下文有序性 | 同预算、同成本函数下更长上下文不产生更低最优 Loss | 容差 $10^{-8}$ |
| O09 | 基线支配 | 正式最优 Loss 不劣于可行 $Q=Q_0$ 基线 | |
| O10 | 领域 Token | $\sum_iD_i=D$，误差 $\le10^{-12}D$ | |
| O11 | 联合配比门禁 | 任何结果均为 $p=p_{ref},\lambda_0=0$ | 否则 FAIL |

### 11.4 不确定性与转移

| ID | 检查 | PASS 标准 |
| --- | --- | --- |
| U01 | 三档 bootstrap | 500 抽样中有效率至少 95% |
| U02 | 转移路径抽样 | 200 抽样中有效率至少 95% |
| U03 | 区间顺序 | 下界 $\le$ 中位数 $\le$ 上界 |
| T01 | 主要转移 | 活跃集合确有变化，二分区间宽度 $\le0.01$ log10 FLOPs |
| T02 | 次要断点 | BIC 改善 $\ge10$ 且斜率差 $\ge0.15$ |
| T03 | 稳定标签 | 同类事件出现概率 $\ge70\%$ 才称稳定 |
| T04 | 场景/预算区分 | 注意力临界值不误报为预算转移 |

### 11.5 总体验收

问题三最终状态分为：

- PROBLEM3_CENTRAL_READY：中心 45 个锚点场景和连续路径通过；
- PROBLEM3_UNCERTAINTY_READY：bootstrap 与敏感性通过；
- PROBLEM3_READY：数据、中心、转移、不确定性、manifest 全部无 FAIL。

最终验收至少满足：

1. 所有硬门禁无 FAIL；
2. 三档预算、三种质量成本、五个上下文情景都有明确结果或可解释的 strict_support 不可行状态；
3. operational_extended 的每个解都有支持等级，不隐藏外推；
4. $p$ 固定策略与 JOINT_P_READY=False 完全一致；
5. C7 架构错误已隔离，问题三只使用核验后的上下文情景；
6. 所有结果可由 manifest、配置和命令复现；
7. RESULTS_REPORT.md 自动从结果文件生成；
8. 本轮不生成图表。

## 12. 失败降级规则

- 上游 hash 或冻结状态失败：停止，不复制点参数继续。
- C7 非上下文字段错误：记录 WARN 并隔离，不阻断已核验上下文情景。
- 某上下文类别无法核验：移除该类别并 WARN；不得人工猜值。
- strict_joint 不可行：保留为数据支持结论，使用 operational_extended 回答题目并显式标注外推。
- 二维求解与三维复核不一致：加密网格并扩大多起点；仍不一致则该场景 FAIL。
- bootstrap 有效率不足 95%：中心结果可保留，但 PROBLEM3_UNCERTAINTY_READY=False。
- 转移概率低于 70%：只报告候选，不下稳定结构转移结论。
- GPU 与 CPU 不一致或 GPU 不更快：固定使用 CPU，并在 run_log.json 说明。

## 13. 开发顺序

1. P3-00：建立配置、输入 manifest 与只读上游桥接。
2. P3-01：完成 C7 上下文审计和错误隔离。
3. P3-02：实现 Loss、三类成本、单位和解析 $D$。
4. P3-03：完成中心场景二维求解、三维复核和 KKT。
5. P3-04：完成 61 点连续预算路径和转移细化。
6. P3-05：完成 500/200 抽样传播和全部敏感性。
7. P3-06：运行测试、验收、manifest 写后 hash 校验。
8. P3-07：生成 RESULTS_REPORT.md 和 problem3_bridge.json。

未完成 P3-00 至 P3-02 的全部硬测试前，不开始批量优化。

## 14. 方法依据与引用边界

- Hoffmann et al. (NeurIPS 2022) 支撑经典计算最优训练和 $6ND$ 近似；本题参数采用问题二冻结估计。
- Vaswani et al. (NeurIPS 2017) 说明全注意力对序列长度的二次复杂度；本题实际使用题设给定的线性化附加代理，不能混为同一精确硬件公式。
- Dao et al. (NeurIPS 2022) 说明 IO-aware 实现会改变真实长上下文速度和内存成本；因此 $\eta NDL_{ctx}$ 只解释为题设政策代理。
- Boyd and Vandenberghe 的 KKT 条件用于解释内点边际收益相等和活跃约束变化；本题目标未预先证明全局凸，因此 KKT 只作必要条件与诊断，全球性由网格和独立求解复核。
- C7 错误核验优先使用模型发布方官方 config 或模型卡。第三方复制仓库只能作为辅助，不覆盖官方来源。

