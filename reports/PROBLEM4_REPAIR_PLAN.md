# 问题四最终修复与冻结规划

> 文档状态：已执行并通过 V2 验收；本文保留为最终修复规范，不作为执行日志。  
> 制定日期：2026-09-24。  
> 本文只规划问题四的最后一次修复；当前阶段不修改代码、不运行新模型、不覆盖 `问题四/results_v1/` 与 `问题四/reports_v1/`。修复执行后应写入新的 `results_v2/`、`reports_v2/` 和 `config/problem4_v2.json`。问题一、问题二、问题三保持冻结，问题四只能读取问题三现有冻结接口。

> 执行结果：`问题四/results_v2/problem4_acceptance_report.json` 为 PASS=29、WARN=1、FAIL=0，五个 READY 接口全部为 True；唯一 WARN 是 MATH hard 未达到逐任务官方分数 95% 精确复现门槛，已按本规划降级，正式 C8 证据采用通过验收的 BBH。最终报告见 `问题四/reports_v2/RESULTS_REPORT.md`。

## 1. 修复结论与边界

问题四的总体方向仍然保留：

1. 用 C1/C2 定义现代开源模型的六维综合能力；
2. 用 C4 的训练算力、数据量、开放权重和证据字段构造确认性预训练面板；
3. 用 0.90 条件分位动力学前沿分离“规模扩张相关贡献”和“控制规模后的非规模残余技术贡献”；
4. 用 C6 建立 Loss–Benchmark 单调桥接，并把问题三的 45 个冻结情景映射为能力分数；
5. 在算力增长延续、减半和降至四分之一的情景下预测 12/24 个月能力前沿。

本轮不更换核心研究问题，不把观察性分解改写成严格因果识别，也不为了提高拟合指标引入高自由度黑箱。需要修复的是实现与验收闭环：当前 V1 的未来预测存在确定的数值一致性错误，C8 的 BBH 聚合不符合官方归一化口径，C3 尚未形成实质分析，算力增长和预测区间没有完整传播不确定性，部分开发规划承诺也没有落实。

修复完成后分别冻结以下接口：

- `P4_DATA_READY`：附件、去重、匹配、开源口径、C3 和 C8 使用合格；
- `P4_BRIDGE_READY`：Average 主桥接通过，单任务桥接各自标 READY/WEAK；
- `P4_DECOMPOSITION_READY`：历史动力学前沿、回测和贡献分解通过；
- `P4_FORECAST_READY`：算力趋势、预测起点、三分位预测、区间和外推标记通过；
- `PROBLEM4_READY`：上述四项均为 True 且没有硬 FAIL。

## 2. 题意复核

题目要求问题四同时回答三件事：

1. 分别量化规模扩张与非规模技术进步的贡献占比；
2. 在算力增长放缓情景下预测未来 12 或 24 个月开源大语言模型能力前沿，并给出不确定性；
3. 将前三问的交叉熵 Loss 通过 C5/C6 或可核验文献映射到多维 Benchmark，并讨论映射误差。

附件的硬性使用要求是：

- C1 或 C2 与 C3 必须使用；
- C4 中算力、数据量、开源权重等字段必须使用；
- C8 至少完成一项逐任务聚合，不能只读汇总表；
- pretrained 与 chat/finetuned 必须区分；
- 开源筛选、许可证和时间轴口径必须说明。

因此，“文件被读取”不等于“数据已经使用”。C3 必须产生独立的方向性或敏感性结果；C4 的数据量字段必须至少进入 M-ND 机制诊断；C8 必须能从子任务记录重建至少一个排行榜指标。

## 3. 证据来源及可信度分级

### 3.1 A 级：同行评议论文与官方方法

| 资料 | 支撑内容 | 本轮采用方式 |
| --- | --- | --- |
| [Chernozhukov、Fernández-Val 与 Galichon，Econometrica 2010](https://doi.org/10.3982/ECTA7880) | 分位曲线交叉可用单调重排修正，bootstrap 应作用于完整分位过程 | 每个 bootstrap 抽样同时拟合 0.85/0.90/0.95，并在抽样内部重排；不能只重排中心值 |
| [Ho 等，NeurIPS 2024](https://openreview.net/pdf?id=5qPmQtfvhy) | 扩展标度律可用于分离计算规模和算法进步，但短期与未来解释存在限制 | 保留规模项与时间残余项的分解，同时坚持观察性、非因果措辞 |
| [Gadre 等，ICLR 2025](https://openreview.net/pdf?id=iZeQBqJamf) | 聚合下游能力与预训练 Loss 可以建立规律，但依赖模型族和训练分布 | 保留单调 Loss–Benchmark 桥接，来源分组验证，不把单任务桥接写成普适定律 |
| [Schaeffer 等，ICML 2025](https://proceedings.mlr.press/v267/schaeffer25b.html) | 多选题得分的离散转换会削弱与规模的可预测关系 | 单任务桥接弱时降级，未来预测以综合能力为主，并扩大外推解释边界 |
| [Künsch，Annals of Statistics 1989](https://doi.org/10.1214/aos/1176347265) | 相关时间序列需要块 bootstrap | 时间维使用真正的连续月份块，不再把 `to_period("2M")` 当成两月块 |
| [Hagemann，JASA 2017](https://doi.org/10.1080/01621459.2016.1148610) | 分位回归可采用簇稳健 bootstrap 传播异质簇相关 | 模型家族按簇重采样/乘子加权，不能把同一家族模型当独立样本 |
| [Hyndman 与 Athanasopoulos：滚动预测起点验证](https://otexts.com/fpp3/tscv.html) | 时序预测应按时间滚动验证，多步预测要按对应步长评价 | 完成 3 月、6 月回测；历史允许时增加 9 月，不使用随机 K 折 |

### 3.2 B 级：官方数据与评测说明

| 资料 | 已核实事实 | 对修复的直接影响 |
| --- | --- | --- |
| [Hugging Face Open LLM Leaderboard 得分归一化说明](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/normalization) | BBH/MUSR 先按每个子任务的随机猜测下界归一化，再对归一化子任务等权平均；MATH 使用 exact match，随机下界视为 0 | 当前按样本数直接平均 BBH 是错误口径，必须改为官方子任务归一化流程 |
| [Open LLM Leaderboard 指标代码](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/blob/40bc934ecf7be415c33092927c71051362f86d9f/src/display/utils.py) | BBH 使用 `acc_norm,none`，MATH hard 使用 `exact_match,none` | 固定指标键，并在 manifest 中保存所参考的仓库 revision |
| [Epoch AI 训练算力估计说明](https://epoch.ai/data/ai-models-documentation/estimation) | 算力来源包括直接报告、运算计数和硬件估计；硬件估计包含利用率假设；由 Benchmark 反推的算力不应再用于 Benchmark–算力关系 | 主面板排除 benchmark-based compute，利用 Confidence 与上下界传播算力测量误差 |

### 3.3 C 级：预印本、公开题解与网络帖子

- [Owen，How predictable is language model benchmark performance?](https://arxiv.org/abs/2401.04757) 显示聚合 Benchmark 比单任务更可预测，并给出随外推距离增大的误差证据；只用于解释“为什么主预测选择综合能力”，不作为附件数据的替代。
- 已检索 2026 华为杯 F 题公开资料、竞赛官网、题目转载和一般性解题博客。检索日尚未找到组委会发布的官方解答或经过同行评议的本题解法；网络上可见内容主要是题面转载、选题建议或未验证思路。它们只用于检查是否遗漏题目要求，不用于确定公式、阈值或替换附件数据。
- 非官方资料如果与题面、附件或官方评测代码冲突，以题面、附件和版本化官方代码为准。

## 4. V1 基线与已定位问题

V1 保留为修复前对照，不删除、不改写。基线事实如下：

- C1/C2 原始 4,576 行，版本级去重后 4,546 行；
- C1/C2–C4 接受匹配 66 行，确认性预训练前沿 41 行、7 个观测月份、13 个家族；
- C3 有 4,573 条同源排行榜记录和 26 条历史记录；
- C8 有 1,863 个目录，1,860 个至少存在一个可解析 JSON，4 个损坏 JSON，3 个目录无可解析文件；
- 问题三 45 个情景全部进入桥接输出；
- V1 验收为 PASS=13、WARN=1、FAIL=0，但验收没有覆盖以下错误。

| 编号 | 问题 | 性质 | V1 观测 | 预计影响 |
| --- | --- | --- | --- | --- |
| R4-01 | 有界 link 不是严格互逆 | 确定代码错误 | 使用 `logit((S+0.5)/101)`，逆变换却用 `100 sigmoid(y)` | 所有桥接/前沿分数最多产生约 0.5 分系统偏差 |
| R4-02 | 分位重排后只改 score、不改 logit | 确定代码错误 | 18 行预测中 12 行不满足 score–logit 恒等式，最大差 5.6686 分 | 当前预测表内部不自洽 |
| R4-03 | 三个分位共用同一套区间 | 确定实现错误 | 每个情景/时长内 0.85、0.90、0.95 的四个区间端点完全相同 | 0.85/0.95 不能称为带区间的敏感性预测 |
| R4-04 | 预测算力起点取全样本全局 0.90 分位 | 模型口径错误 | 使用 24.2007；最后能力月份为 23.9694，C4 2025-03 月度前沿为 24.3788 | 12/24 月中心预测可发生实质改变 |
| R4-05 | 算力增速未按证据层/模型族处理且未进入预测 bootstrap | 规划未落实 | 预测抽样固定同一条年化增速 | 区间遗漏未来算力趋势误差，长期区间可能偏窄 |
| R4-06 | `to_period("2M")` 没有构成相邻两月块 | 确定实现错误 | 月份仍是一月一个标签 | 时间相关性没有按规划保留 |
| R4-07 | 非规模状态斜率固定为局部/全局各 0.5 | 规划未落实 | 收缩强度没有通过回测选择 | 未来技术残余趋势依赖人工常数 |
| R4-08 | C3 只读取和计数，没有结果输出 | 题意覆盖不足 | 26 条历史记录没有方向性检验文件 | 不能充分证明“使用了 C3” |
| R4-09 | BBH 按样本量平均，遗漏官方随机基线归一化 | 确定口径错误 | BBH 官方精确复现数为 0 | C8 BBH 聚合结论无效；不影响 C1 官方 Average 主表 |
| R4-10 | 桥接验证只完整输出 MAE | 规划未落实 | 缺少 RMSE、Spearman、区间覆盖及折外预测明细 | 桥接误差讨论不完整 |
| R4-11 | 不确定性文件未按估计对象拆分来源 | 规划未落实 | 只有若干参数分位数，没有分量宽度 | 无法回答映射误差、算力趋势误差各占多少 |
| R4-12 | strict unrestricted 未同时核验许可证 | 口径不足 | 主要依赖 C4 accessibility 字符串 | 严格开源敏感性不够严格 |
| R4-13 | 验收只查边界，未查 link、分位、区间来源和 C3 实质使用 | 验收漏洞 | V1 得到 `PROBLEM4_READY=True` | 必须重写门禁，V1 READY 不再作为最终状态 |

## 5. 修复后的数据合同

### 5.1 上游冻结

问题四 V2 只读取：

- `问题三/results_v1/problem3_bridge.json`；
- `problem3_acceptance_report.json`；
- `problem3_manifest.json`；
- `bootstrap_optimal_allocations.csv.gz`；
- `bootstrap_intervals.csv`。

必须保持 45 个问题三情景、预算、上下文、质量成本函数、Loss、区间和 evidence 标签不变。问题四不得重新拟合问题一至三，不得把 45 条情景加入 C1/C2/C3/C4 或 C6 的训练数据。

### 5.2 统一的能力 link

桥接、动力学和预测统一使用

$$
g_\epsilon(S)=\operatorname{logit}\left(\frac{S+\epsilon}{100+2\epsilon}\right),
\qquad
g_\epsilon^{-1}(y)=(100+2\epsilon)\sigma(y)-\epsilon,
$$

主值为 $\epsilon=0.5$，敏感性为 0.1 和 1.0。反变换不得写成 $100\sigma(y)$，也不得在结果末端用 `clip(0,100)` 掩盖错误。数值误差容许量为 $10^{-10}$；浮点误差导致的极小越界须首先检查公式，不能静默裁剪。

### 5.3 开源与类型合同

- 主口径：`open_weight_verified`，C2 或接受的 C4 匹配明确 `Open model weights?=Yes`；
- 严格口径：同时满足开放权重、C4 `Open weights (unrestricted)`、排行榜许可证属于预注册的研究/复现允许清单；未知、`other`、限制性和非商业许可证不得进入严格层；
- 宽口径：`leaderboard_candidate` 只作覆盖率敏感性；
- 确认性贡献分解仍只使用 pretrained/continuously pretrained；
- posttrained 当前只有约 6 条合格算力记录，预注册为 `INSUFFICIENT_SUPPORT` 时只给描述性结果，不强拟合后训练贡献；
- merge 不分摊完整预训练算力，只作样本计数和能力前沿对照。

## 6. C8 聚合修复

### 6.1 BBH 官方归一化

当前“子任务分数按样本数加权”被废止。对每个 BBH 子任务 $j$，从与附件版本相匹配的官方任务定义中取得随机猜测下界 $l_j$，计算

$$
z_{ij}=
\begin{cases}
0, & s_{ij}<l_j,\\
100\dfrac{s_{ij}-l_j}{1-l_j}, & s_{ij}\ge l_j.
\end{cases}
$$

模型 $i$ 的 BBH 汇总为完整子任务的等权平均：

$$
S_i^{BBH}=\frac{1}{|J_i|}\sum_{j\in J_i}z_{ij}.
$$

每个子任务必须保存：任务键、`acc_norm,none`、随机下界、归一化得分、effective/original 样本数、是否完整。样本数只用于完整性和审计，不再作为 BBH 官方聚合权重。随机下界映射必须来自版本化的官方配置或归一化资产，保存 URL/revision/hash；不能由当前排行榜分数反推。

### 6.2 MATH hard

MATH hard 继续提取 7 个类别的 `exact_match,none`，按有效样本数重建整体 exact-match。官方说明中生成式 MATH 的随机下界视为 0。保留当前 737 条 exact 复现作为 V2 回归基准，但重新执行后以新结果为准。

### 6.3 C8 READY 规则

- `C8_MATH_READY=True`：7 类完整记录可回代，匹配到排行榜的记录中存在精确复现且差异分布有日志；
- `C8_BBH_READY=True`：官方子任务下界表版本固定、完整子任务归一化可回代，唯一匹配且完整记录的绝大多数差异不超过 $10^{-6}$；
- 题目只要求至少一项逐任务聚合，因此只要 MATH 或 BBH 至少一项 READY，C8 不阻断问题四；另一项若不通过必须 WARN 并明确不作为论文主复现结果；
- 不得为了提高 exact 数量覆盖 C1/C2 官方分数。

## 7. C3 实质使用

C3 的 4,573 条排行榜记录与 C1/C2 同源，只用于 schema、年份和分数一致性，不重复加入现代面板。26 条 `Historical (papers/reports)` 因年份、任务覆盖和评测口径异质，只做长期方向敏感性。

### 7.1 历史方向模型

在 Average 非缺失、参数量为正的历史记录上拟合稳健模型：

$$
g_{0.5}(S_i^{avg})
=a+b\log_{10}N_i+c(Year_i-2019)+e_i.
$$

同时计算年度 0.90 分位 Average 的 Theil–Sen 趋势。模型族由规范化名称提取，使用家族 bootstrap 与 leave-one-family-out 敏感性，输出 $c$ 的中心、80%/95%区间、正号概率和逐家族删除结果。

### 7.2 使用边界

- 若 $P(c>0)\ge0.8$ 且逐家族删除符号稳定，记为“长期方向支持非规模残余非负”；
- 否则记为“历史口径下方向不确定”；
- 无论结果如何，C3 不进入现代 M-C 主拟合，不把 C3 系数强制设为现代状态趋势先验，也不改变贡献中心值；
- 论文只把该结果作为方向性敏感性，不能把 26 条异口径记录包装成独立现代验证集。

新增 `c3_historical_sensitivity.csv` 和 `c3_historical_summary.json`。只有生成并解释这两项结果，才算 C3 被实质使用。

## 8. Loss–Benchmark 桥接修复

### 8.1 保留部分

继续使用 C6 的 75 条记录作为唯一桥接主表；C5 仅验证其 43 条是否为 C6 子集。保持高可比权重 1、中可比主权重 0.25，以及 0、0.10、0.50 敏感性。候选仍为常数、单调线性、低自由度单调样条和保序回归，所有非恒定候选强制 Loss 增大时能力不增加。

### 8.2 验证补全

每个目标输出逐折预测，按来源族留一验证，记录：

- 加权 MAE；
- 加权 RMSE；
- Spearman；
- 单调违反数；
- 80%/95%区间覆盖率与平均宽度；
- 相对常数基线的误差差和标准误；
- 一个标准误差规则下的最终选择。

样本过少导致 Spearman 或覆盖率不可定义时记为 `insufficient_fold_support`，不得填 0。Average 主桥接未优于常数基线时 `P4_BRIDGE_READY=False`；单任务未优于常数基线只将该任务标为 WEAK。

### 8.3 映射不确定性分解

对问题三每个情景至少形成三组抽样：

1. `upstream_only`：抽问题三 Loss，固定中心桥接；
2. `bridge_only`：固定问题三中心 Loss，重抽来源族并重拟合桥接；
3. `joint`：问题三 Loss 与桥接同时抽样。

分别报告 80%/95%区间宽度。该分解只属于“问题三 Loss → Benchmark”估计对象，不与历史前沿或未来能力预测的区间混在一起。

## 9. C4 算力趋势与预测起点修复

### 9.1 算力趋势样本

主样本要求：

- `Domain` 包含 Language；
- `Open model weights?=Yes`；
- 发布日期不晚于能力面板截止日；
- 训练算力为正；
- 算力估计方法不是由 Benchmark 表现反推；
- 同一模型族、同一月份的多个版本先作版本审计，构造家族月前沿，避免同一家族密集发布重复计权。

主中心允许 `reported`、`operation_counting`、`hardware_based`。`third_party_or_benchmark` 与 `derived_6ND` 只作敏感性，其中 benchmark-based compute 禁止进入 Benchmark–compute 主关系，避免循环论证。

### 9.2 证据权重与测量误差

优先使用 C4 的 `Training compute lower bound`、`Training compute upper bound` 和 `Confidence`。在 $\log_{10}C$ 尺度上用区间半宽构造相对精度权重；没有上下界时依据 Epoch 官方 Confidence 含义使用预注册的 90%乘数范围，而不是按结果调权。

每个 bootstrap 抽样同时：

1. 重采样模型家族；
2. 在允许的算力上下界内抽取该模型的 $\log_{10}C$；
3. 重新构造家族月前沿与月度 0.90 分位；
4. 重新估计近期增速和截止日算力水平。

抽样分布主用 log 尺度三角分布，以点估计为众数、上下界为边界；log-uniform 作为敏感性。若原始上下界缺失，必须在结果中标记所用替代范围。

### 9.3 增速与端点

在近 24 个月的证据加权月度前沿上拟合稳健局部线性趋势

$$
x_m=a_C+r_C m+u_m,
\qquad x_m=Q_{0.90}^{w}(\log_{10}C\mid m).
$$

中心斜率使用 Theil–Sen 或等价低自由度稳健线性估计；18 个月和 30 个月为窗口敏感性。未来预测的算力起点固定为该趋势在排行榜截止月 $m_0$ 的拟合值

$$
x_{C,0}=\widehat a_C+\widehat r_Cm_0,
$$

不再使用全部 41 个能力前沿样本的全局 0.90 分位。另输出“截止月原始月度前沿”和“最后三个月中位前沿”作为端点敏感性。

每个增长 bootstrap draw 自带 $(x_{C,0}^{(b)},r_C^{(b)})$，二者共同进入未来预测。中心情景仍为 $r_C$、$0.5r_C$、$0.25r_C$；若中心增速非正或区间跨 0，额外输出 flat 情景。抽样中的负斜率保留为估计不确定性，不通过事后截断删除。

## 10. 动力学前沿与贡献分解修复

### 10.1 候选模型角色

- `M-C`：训练算力 + 月度非规模状态，仍为贡献分解主候选；
- `M-time`：只有时间状态的基线；
- `M-scale`：只有训练算力的基线；
- `M-N`：只用参数量的广覆盖诊断，必须输出但不能替代 M-C；
- `M-ND`：参数量和训练数据量同时进入的机制诊断；当前约 26 条完整记录、条件数约 388，预期只作共线性敏感性，不作为主模型；
- `M-post`：只有合格 posttrained 样本达到预注册的最小覆盖才拟合，否则输出 `INSUFFICIENT_SUPPORT`。当前约 6 条合格记录，预计不放行正式后训练贡献。

M-C 的中心公式仍为

$$
Q_\tau(Y_i\mid C_i,t_i)
=\beta_0+\beta_C\,\widetilde{\log_{10}C_i}+a_{m(i)},
\qquad \tau\in\{0.85,0.90,0.95\}.
$$

不同时把 $C,N,D$ 塞进主模型，避免 $C\approx6ND$ 的机械共线性。

### 10.2 时间状态外推

未来状态斜率写为

$$
v_{future}(\rho)=\rho v_{local}+(1-\rho)v_{global},
\qquad \rho\in\{0,0.25,0.5,0.75,1\}.
$$

$\rho$ 与状态平滑参数 $\lambda$ 必须只通过滚动起点回测选择；不再固定 $\rho=0.5$。按简约原则，如果多个组合在最优值一个标准误差内，选择更平滑、局部权重更小的组合。

### 10.3 回测

按月份滚动：每折只使用截止月及以前的数据，分别预测后 3 月和 6 月；若存在至少 3 个有效折再增加 9 月。每折必须重新拟合变换、状态和模型参数。输出：

- pinball loss、MAE、RMSE；
- 与 M-time、M-scale 的成对误差差和标准误；
- 分位覆盖、区间宽度和分位交叉率；
- 每个预测步长的有效折数。

12/24 月没有同口径真实值时，不把 3/6 月回测伪装成长期验证，只用来筛除明显失效模型。

### 10.4 贡献分解

继续用两因素 Shapley 路径平均分解规模与非规模残余。修复后必须使用精确的 $g^{-1}_\epsilon$ 重新计算得分尺度四个反事实和贡献，因此 V2 中心值允许相对 V1 小幅变化。

主报告仍只在总变化和两项均为正时给普通百分比；含负项时报告带符号贡献、绝对份额、正号概率和区间。正式术语固定为：

- “规模扩张相关贡献”；
- “控制已观测规模后的非规模残余技术贡献”。

不得写成“训练算力的严格因果效应”和“纯算法进步”。

### 10.5 必做敏感性

在不改变主模型选择的前提下输出：

1. strict unrestricted 口径；
2. Publication Date 时间轴；
3. 训练窗内拟合 ECDF 的六维秩正态平衡能力分；
4. M-N 与 M-ND；
5. 算力证据只保留 reported/operation-counting；
6. 月块长度 1、2、3；
7. $\epsilon=0.1,0.5,1.0$。

若某敏感性样本不足，只能输出 `INSUFFICIENT_SUPPORT` 和计数，不得填造结果。

## 11. 三分位联合预测与 bootstrap

### 11.1 正确的抽样单位

现代能力面板只有约 7 个有效月份，因此区间解释为“小样本下的经验稳定性区间”，不夸大为精确渐近置信区间。主 bootstrap 至少获得 1,000 个有效 draw：

- 模型家族作为横截面簇；
- 月份使用真正连续的 moving blocks；主块长 2 个月，1 和 3 个月作敏感性；
- 每个被抽中的连续块给对应月份统一时间权重；
- 家族权重和时间块权重共同作用于该行；
- 算力趋势面板独立执行家族与月份重采样，并抽取算力测量误差。

禁止再次使用 `submission_date.dt.to_period("2M")` 代替连续两月块。

### 11.2 每个 draw 的预测步骤

对每个有效 draw $b$：

1. 重新拟合 $\tau=0.85,0.90,0.95$ 三条动力学前沿；
2. 抽取并重估 $(x_{C,0}^{(b)},r_C^{(b)})$；
3. 在三种算力情景和 12/24 月时点预测三条原始分位；
4. 对同一情景、同一时点的三条预测做单调重排；
5. 使用精确逆 link 同步保存重排后的 score 与 logit；
6. 保存原始预测、重排预测、是否交叉、模型参数、增长参数和 draw provenance。

最终中心预测也执行同一重排。区间必须由各自 $\tau$ 的重排后抽样计算，不能把 0.90 区间复制给 0.85/0.95。

### 11.3 预测结果字段

`frontier_forecast.csv` 每行至少包含：

- `tau`、`scenario_factor`、`horizon_months`、`forecast_month`；
- `forecast_score_raw`、`forecast_link_raw`；
- `forecast_score`、`forecast_link`；
- `quantile_rearranged`；
- `compute_endpoint_log10`、`compute_growth_annual_log10`；
- `p10/p90/p025/p975`；
- `interval_tau`、`bootstrap_draw_count`、`extrapolation_flag`；
- `forecast_support` 与 `uncertainty_scope`。

12 月固定标 `extrapolation`，24 月固定标 `long_horizon_extrapolation`。不输出脱离区间的单一“确定预测”。

## 12. 按估计对象拆分不确定性

不再声称每个结果都包含同一套“五类不确定性”。不同估计对象的来源应分开：

| 估计对象 | 必须传播的来源 |
| --- | --- |
| 历史规模/非规模贡献 | 家族抽样、月份块、前沿参数、link 口径 |
| 未来 12/24 月前沿 | 历史前沿参数、未来状态斜率、算力端点、算力增速、算力测量误差、分位重排 |
| 问题三 Loss → Benchmark | 问题三 Loss 抽样、桥接来源族抽样、桥接模型参数、支持域外推 |

新增 `uncertainty_budget.csv`，长表字段至少包括：`estimand`、`scenario_id`、`component`、`p10`、`p90`、`p025`、`p975`、`width80`、`width95`、`draws`、`interpretation`。分量宽度用于解释，不把各宽度简单相加冒充联合区间。

## 13. 自动测试与验收标准

### 13.1 单元测试

| ID | 检查 | PASS 标准 |
| --- | --- | --- |
| UT01 | link 往返 | $\max|g^{-1}(g(S))-S|\le10^{-10}$，覆盖 0、5、50、99、100 |
| UT02 | 分位重排 | 任意交叉三元组修复后满足 $q_{.85}\le q_{.90}\le q_{.95}$ |
| UT03 | score–link 同步 | 重排后每行 $|S-g^{-1}(y)|\le10^{-10}$ |
| UT04 | 区间来源 | 三个 $\tau$ 均从带自身 tau 的 draw 计算，禁止 merge 时丢失 tau |
| UT05 | 连续月份块 | 块内月份连续、块长正确；不能退化成逐月唯一标签 |
| UT06 | C8 BBH 归一化 | 手工小例与官方下界公式一致，子任务等权而非样本量加权 |
| UT07 | MATH 聚合 | 7 类样本加权手工例可回代 |
| UT08 | 未来泄漏 | 截止日之后的 C4 行进入任一拟合/增速估计数为 0 |
| UT09 | Shapley 闭合 | link 与得分尺度误差均 $\le10^{-10}$ |
| UT10 | 上游隔离 | 问题三 45 条预测进入前沿或桥接训练数为 0 |

### 13.2 数据与题意门禁

| ID | 检查 | PASS 标准 | 失败动作 |
| --- | --- | --- | --- |
| D01 | 上游接口 | P3 READY、45 情景及所有 hash 一致 | FAIL，停止 |
| D02 | C1/C2 | 4,576 行、六项完整、Average 重算误差 $\le10^{-10}$ | FAIL，停止 |
| D03 | C3 | 4,573 同源行不重复；26 历史行生成方向敏感性结果 | FAIL，停止 |
| D04 | C4 字段 | compute、data、open weights、method、confidence、上下界均有审计和使用记录 | FAIL，停止 |
| D05 | 开源严格层 | 未知/限制许可证进入 strict unrestricted 数为 0 | FAIL strict sensitivity |
| D06 | 类型分层 | pretrained/posttrained/merge/excluded 无未分类行 | FAIL，停止 |
| D07 | C8 | MATH 或 BBH 至少一项 READY；另一项若失败有完整日志 | 无 READY 则 FAIL |
| D08 | C5/C6 | C5 为 C6 子集，主拟合只使用 C6 一次 | FAIL，停止 |

### 13.3 桥接门禁

| ID | 检查 | PASS 标准 |
| --- | --- | --- |
| B01 | 单调性 | 稠密 Loss 网格最大上升量 $\le10^{-10}$ |
| B02 | 折外预测 | 每个可用来源族都有折外记录，无训练/验证混用 |
| B03 | 指标 | MAE、RMSE、Spearman、覆盖、宽度和基线差均输出；不可定义项有原因 |
| B04 | 选模 | 一个标准误差规则可回代；Average 不劣于常数基线 |
| B05 | 支持域 | 45 情景均有 high/expanded/out-of-range 与最终证据层 |
| B06 | 区间 | 80%/95%端点有序且在 link 自然边界内 |
| B07 | 误差分解 | upstream-only、bridge-only、joint 三类区间齐全 |

### 13.4 动力学与贡献门禁

| ID | 检查 | PASS 标准 |
| --- | --- | --- |
| M01 | M-C 可识别 | 设计矩阵满秩，月份/家族覆盖达标，bootstrap 有效率 $\ge95\%$ |
| M02 | 规模方向 | $\beta_C\ge0$ 或区间含 0 但不显著反向 |
| M03 | 回测 | 3/6 月有效折完成，M-C 不劣于最佳简单基线超过一个标准误差 |
| M04 | 趋势收缩 | $\rho$ 和 $\lambda$ 均由训练期滚动回测选择 |
| M05 | M-ND | 完整样本数、条件数/VIF、系数及 bootstrap 符号齐全；共线则明确降级 |
| M06 | 贡献闭合 | 两尺度闭合误差 $\le10^{-10}$ |
| M07 | 因果措辞 | 自动报告只使用“相关贡献/残余”，不出现无条件因果断言 |

### 13.5 算力与预测门禁

| ID | 检查 | PASS 标准 | 失败动作 |
| --- | --- | --- | --- |
| F01 | 预测端点 | 来自截止月的拟合算力前沿，不能等于全样本无时间条件分位的硬编码结果 | FAIL |
| F02 | 增长抽样 | 每个有效 draw 都重估端点和增速，增长 draw 非退化 | FAIL |
| F03 | 三分位 | 所有情景/时长满足 0.85≤0.90≤0.95 | FAIL |
| F04 | link 一致 | 所有中心和 draw 的 score–link 误差 $\le10^{-10}$ | FAIL |
| F05 | tau 区间 | 每个 tau 的区间由本 tau 重排后 draw 计算并保留 provenance | FAIL |
| F06 | 情景 | 1、0.5、0.25 完整；必要时含 flat | FAIL |
| F07 | 外推标记 | 12 月 extrapolation，24 月 long_horizon_extrapolation | FAIL |
| F08 | 结果边界 | 通过 link 自然落在 0–100；事后裁剪次数为 0 | FAIL |

### 13.6 READY 逻辑

V2 总体状态不能再使用“只要 FAIL=0 且分解通过就全部 READY”的简化逻辑。四个子接口独立判断：

```text
P4_DATA_READY          = D 类硬门禁通过
P4_BRIDGE_READY        = DATA + Average 桥接 B 类门禁通过
P4_DECOMPOSITION_READY = DATA + M 类门禁通过
P4_FORECAST_READY      = DECOMPOSITION + F 类门禁通过
PROBLEM4_READY         = DATA & BRIDGE & DECOMPOSITION & FORECAST
```

WARN 不自动阻断 READY，但必须有明确降级动作；例如 BBH 未复现但 MATH 已通过时，只允许以 MATH 作为论文的 C8 正式聚合案例。

## 14. V2 输出和版本冻结

所有结果写入 `问题四/results_v2/`，报告写入 `问题四/reports_v2/`。至少新增或更新：

| 文件 | 内容 |
| --- | --- |
| `repair_baseline_v1.json` | V1 关键数值、文件 hash 和已知问题，保证对照可追溯 |
| `c8_subtask_normalization.csv.gz` | BBH/MATH 子任务原始值、下界、归一化、样本数和完整性 |
| `task_level_aggregates.csv` | 修复后的模型级 BBH/MATH 重算与官方差异 |
| `c3_historical_sensitivity.csv` | C3 历史方向、bootstrap 和逐家族删除结果 |
| `c3_historical_summary.json` | C3 结论及能否支持方向性判断 |
| `bridge_cv_predictions.csv.gz` | 桥接逐来源族折外预测 |
| `bridge_cv_results.csv` | MAE/RMSE/Spearman/覆盖/宽度与选模结果 |
| `loss_benchmark_predictions.csv` | 45 情景的 V2 能力映射与证据层 |
| `loss_benchmark_uncertainty.csv` | upstream-only、bridge-only、joint 宽度分解 |
| `compute_growth_monthly.csv` | 证据加权、家族去重的月度算力前沿 |
| `compute_growth_bootstrap.csv.gz` | 每个 draw 的端点、增速、窗口、证据和有效性 |
| `frontier_model_comparison.csv` | M-C、M-time、M-scale、M-N、M-ND 及敏感性 |
| `frontier_bootstrap_draws.csv.gz` | 家族×连续月份块的历史前沿与贡献抽样 |
| `frontier_forecast_draws.csv.gz` | 带 tau 的原始/重排预测和增长参数 |
| `frontier_forecast.csv` | 三分位、三情景、12/24 月中心与区间 |
| `frontier_sensitivity.csv` | unrestricted、发布日期、平衡分、证据层、块长、epsilon |
| `uncertainty_budget.csv` | 按估计对象和来源拆分的区间宽度 |
| `repair_comparison_v1_v2.csv` | V1/V2 数据量、贡献、预测、区间、READY 对照 |
| `problem4_acceptance_report.csv/json` | 完整门禁和独立 READY 状态 |
| `problem4_manifest.json` | 输入、代码、配置、参考版本与全部输出 hash |
| `run_log.json` | 环境、种子、耗时、CPU/GPU、异常与降级 |
| `reports_v2/RESULTS_REPORT.md` | 只从 V2 CSV/JSON 自动生成的中文报告 |

manifest 必须保存代码文件指纹，不只保存输出指纹；写完全部输出后重新计算 hash，再写最终 manifest。任何 V1 文件不得被修改。

## 15. 开发顺序

1. 建立 `problem4_v2.json`、`results_v2/`、`reports_v2/`，写入 V1 baseline hash；
2. 抽取统一 link 工具并完成严格往返测试；
3. 按官方归一化修复 C8，先用少量 JSON 手工回代，再全量执行；
4. 增加 C3 历史方向性分析，确认其不进入现代主拟合；
5. 修复 strict unrestricted 许可证合同和 C4 证据/上下界处理；
6. 补全桥接折外预测、指标和两来源不确定性分解；
7. 修复连续月份块 bootstrap；
8. 完成 M-N/M-ND 和必做口径敏感性；
9. 将状态斜率收缩权重纳入滚动回测；
10. 重建算力月度前沿、截止月端点和增长 bootstrap；
11. 实现三分位联合预测、draw 内重排和 tau 专属区间；
12. 重算贡献、未来预测与问题三能力映射；
13. 执行单元测试、集成测试、验收和双次复现；
14. 生成 V1/V2 对照、manifest 和新结果报告；
15. 审核通过后，才把问题四 V2 写入总大纲的冻结状态并进入论文最终撰写。

如 GPU 对三个分位×1,000 次 bootstrap 确实更快，可优先使用 GPU；必须先做 CPU/GPU 小样本一致性测试，中心输出差异不超过预注册数值容差。GPU 不可用或更慢时使用 CPU，不改变模型。

## 16. 预期变化与不预设结果

- C8 的 MATH 结果应大体保持；BBH 将因采用官方随机基线归一化而明显改变，预计与官方分数大幅接近，但不预先保证 exact 数量；
- 历史贡献的中心值可能因精确逆 link、证据权重和块 bootstrap 小幅改变，区间可能变宽；不得为了保持 V1 的“规模 99.7%”而锁定参数；
- C3 可能支持、反对或无法判断非规模长期方向，任何结果都只作敏感性；
- 未来 12/24 月预测可能发生实质变化，因为算力端点、增长率、技术趋势和三分位区间都将重新估计；
- 问题一至三的模型、参数和冻结结果不改变；问题三只会通过既有 Loss 抽样被重新映射到问题四能力尺度；
- 如果修复后 `P4_FORECAST_READY=False`，允许保留历史贡献与桥接结果，但论文不得给出未经放行的未来点预测。这比用错误区间勉强回答更符合题意和学术规范。

## 17. 最终验收标准

问题四只有同时满足以下条件才可冻结并进入论文定稿：

1. V1 文件零改动，V2 有独立配置、结果、报告和 manifest；
2. 统一 link 严格互逆，所有结果 score/link 自洽；
3. C3 有实质敏感性结果但不与现代面板重复计数；
4. C4 的 compute、data、open weights、method、confidence/上下界有使用记录；
5. C8 至少 MATH 或 BBH 一项按官方口径通过逐任务聚合；
6. Average 桥接通过来源族折外验证，映射误差和问题三上游误差分别报告；
7. M-C 的时间回测、识别性、规模方向和 Shapley 闭合通过；
8. 未来算力起点来自截止月趋势水平，增长率在每个 bootstrap draw 重估；
9. 0.85/0.90/0.95 在每个中心和 draw 内无交叉，且各自区间来源正确；
10. 12/24 月结果均带外推、区间和适用边界；
11. `P4_DATA_READY`、`P4_BRIDGE_READY`、`P4_DECOMPOSITION_READY`、`P4_FORECAST_READY` 和 `PROBLEM4_READY` 全部为 True；
12. 同一配置连续运行两次的中心参数、贡献和预测在 $10^{-10}$ 数值容差内一致，所有输出 hash 写后核验通过。

达到以上标准后，问题四与问题一至三一起冻结；后续论文和图表只能消费 V2 冻结结果，不得在写作阶段重新计算或手工修饰数值。
