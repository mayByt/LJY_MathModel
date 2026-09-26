# 问题一开发与验收文档

> 本文是 [`ANALYSIS_MODELING_REPORT.md`](./ANALYSIS_MODELING_REPORT.md) 第 5 节的工程补充。主报告回答“为什么这样建模”，本文固定“数据怎样用、代码怎样拆、怎样验收、失败后怎样降级”。本文不替代后续 `RESULTS_REPORT.md`，所有性能阈值均为开发前验收标准，不是已经得到的实验结论。

## 1. 目标、边界与最终接口

### 1.1 问题一必须完成的四项交付

1. 将 A1–A3 的 22 个异构质量字段统一为 `[0,1]` 且越大越好的标量，得到样本质量 `Q_i`、五个语义块分数、冲突指数 `K_i/QCI_i` 和冲突类型。
2. 得到 7 个质量域的稳健质量 `q_d`，并利用 A16/A18 将其映射为 17 个 RegMix 配方域质量及不确定区间。
3. 用 A4/A5 拟合 17 域配比到 13 域 Loss 的二阶 Scheffé-Ridge 响应面，用 A6–A11 做真实留出检验，用 A12–A15 做外推敏感性分析。
4. 在训练支持域内求综合 Loss 最优配比 `p*`，输出主效应、方向边际效应、稳定交互和不确定区间，为问题二提供 `Q`、`q`、`p*`、`J(p)` 与 `H(p)`。

### 1.2 明确不做的事情

- 不把 A1 与 A2/A3 当成独立样本；A1 的 arXiv/GitHub 分别全部包含在 A2/A3 中。
- 不把 A12–A15 的估算 Loss 混入主模型训练或主验证。
- 不把质量域分数 `q_d` 直接塞入 Scheffé 一阶项后宣称识别了独立质量因果效应；`q(p)=p^Tq` 与配比一阶项共线。
- 不使用 A6–A11 调参，不根据测试集表现改变变换、正则网格或交互筛选阈值。
- 不用树模型的特征重要性代替 Scheffé 系数和方向导数的机制解释。

### 1.3 下游接口

问题二读取以下冻结结果：

- `quality_domain.csv`：7 个质量域及 17 个配方域的 `q_d`、区间、映射类型与证据等级；
- `optimal_mixture.csv`：`p*`、约束余量、支持域距离、bootstrap 区间；
- `mixture_metrics.csv`：A6–A11 的误差、秩相关、Top-k 和 Regret；
- `mixture_coefficients.csv.gz`：13 个验证域的主效应和交互系数；
- `problem1_manifest.json`：配置、输入指纹、版本、种子、训练/测试切分和验收状态。

## 2. 数据契约与使用规划

### 2.1 输入文件角色

| 组 | 文件 | 规模 | 允许用途 | 禁止用途 |
| --- | --- | ---: | --- | --- |
| A1 | `slimpajama_quality_signal_sample.jsonl.xz` | 51,230×27 | 22 指标变换、正文验证、冲突案例、5 个非扩展域的全量质量 | 与 A2/A3 重复累计样本量 |
| A2 | `arxiv_part-...jsonl.xz` | 17,523×24 | arXiv 全域质量、A1 子集与余集代表性比较 | 正文语义案例分析 |
| A3 | `github_part-...jsonl.xz` | 203,752×24 | GitHub 全域质量、A1 子集与余集代表性比较 | 正文语义案例分析 |
| A4/A5 | `train_mixture_1m.csv` / `train_pile_loss_1m.csv` | 512 组 | 全部配比模型拟合、内部交叉验证 | 外部验证 |
| A6/A7 | `test_mixture_1m.csv` / `test_pile_loss_1m.csv` | 256 组 | 同规模留出检验 | 调参 |
| A8/A9 | `test_mixture_60m.csv` / `test_pile_loss_60m.csv` | 256 组 | 跨规模排序、符号和再校准检验 | 调参；把与 A6/A7 相同的配比重复计为另外 256 个独立配比 |
| A10/A11 | `test_mixture_1B.csv` / `test_pile_loss_1B.csv` | 64 组 | 独立配比、独立规模检验 | 与 A6/A8 按 index 对齐 |
| A12–A15 | `est_*_10b/70b.csv` | 各 63 组 | 外推敏感性、区间与排序稳定性 | 主拟合、主验收 |
| A16 | `domain_mapping_guide.csv` | 17×4 | direct/near_direct/inferred 先验 | 把 inferred 当成已知真值 |
| A17 | `regmix_domain_summary.csv` | 17×5 | A18 样本量诊断、域不平衡记录 | 直接替代文本特征 |
| A18 | `regmix_domain_sample.jsonl.xz` | 138,034×3 | 17 域文本分布特征、软映射 | 直接计算附件没有提供的 22 个质量指标 |

### 2.2 去重后的质量参考宇宙

质量标度拟合只使用一次每条记录：

```text
A1 中非 arXiv/GitHub 的 5 域：51,230 - 1,419 - 10,000 = 39,811
A2 arXiv 全量：17,523
A3 GitHub 全量：203,752
合计：261,086
```

唯一键优先使用 `(规范化域名, id)`。若同键内容字段不一致，保留扩展集 A2/A3 的质量字段，A1 只保留正文和来源元数据，并在 `duplicate_conflicts.csv` 中记录。该冲突计数应为 0；非 0 时中止质量建模，先核查 ID 语义。

为防止 GitHub 占 78% 左右样本而主导标度，每一指标的参考分布定义为七域等权经验分布：

```math
F_{ref,j}(x)=\frac{1}{7}\sum_{d=1}^{7}F_{j,d}(x).
```

实现时可保存各域排序数组并求平均 CDF；bootstrap 重拟合时按域重采样。不得把全部 261,086 条简单拼接后直接求 pooled ECDF 作为主口径，pooled ECDF 仅作敏感性对照。

### 2.3 22 字段的数据字典

| 语义块 | 原字段 | 原始结构 | 主标量/方向 | 信号来源 |
| --- | --- | --- | --- | --- |
| 教育与领域价值 | `fineweb_edu` | 长度 1 列表 | 唯一元素，正向 `F_ref` | Model |
| 教育与领域价值 | `qurater` | 长度 4 列表 | 四分量分别分位化后等权平均 | Model |
| 教育与领域价值 | `dsir_books` | 标量 | 正向 `F_ref` | DSIR |
| 教育与领域价值 | `dsir_wiki` | 标量 | 正向 `F_ref` | DSIR |
| 教育与领域价值 | `dsir_math` | 标量 | 正向 `F_ref` | DSIR |
| 表达与整洁 | `fluency_en` | `[not_fluent, fluent]` logits | 稳定 softmax 的 `P(fluent)` | Model |
| 表达与整洁 | `modernbert_readability` | 6 级 logits | softmax 等级期望除以 5 | Model |
| 表达与整洁 | `modernbert_cleanliness` | 6 级 logits | softmax 等级期望除以 5 | Model |
| 表达与整洁 | `rps_lines_ending_with_terminal_punctution_mark` | 标量 | 正向；域内方向作敏感性 | Rule |
| 推理与信息 | `modernbert_reasoning` | 6 级 logits | softmax 等级期望除以 5 | Model |
| 推理与信息 | `modernbert_professionalism` | 6 级 logits | softmax 等级期望除以 5 | Model |
| 推理与信息 | `rps_doc_unigram_entropy` | 标量 | 正向至锚点上限，极端截尾 | Rule |
| 推理与信息 | `rps_doc_frac_unique_words` | 标量 | 正向至锚点上限，极端截尾 | Rule |
| 噪声与重复 | `ad_en` | `[has_ad, no_ad]` logits | 稳定 softmax 的 `P(no_ad)` | Model |
| 噪声与重复 | `rps_doc_frac_no_alph_words` | 标量 | 负向 `1-F_ref` | Rule |
| 噪声与重复 | `rps_doc_frac_chars_top_2gram` | 标量 | 负向 `1-F_ref` | Rule |
| 噪声与重复 | `rps_doc_frac_chars_top_3gram` | 标量 | 负向 `1-F_ref` | Rule |
| 噪声与重复 | `rps_lines_uppercase_letter_fraction` | 标量 | 负向 `1-F_ref` | Rule |
| 噪声与重复 | `rps_lines_numerical_chars_fraction` | 标量 | 通常负向；数学/GitHub 域适中型 | Rule |
| 结构充分性 | `rps_doc_word_count` | 标量 | 对数尺度适中/饱和型 | Rule |
| 结构充分性 | `rps_doc_num_sentences` | 标量 | 对数尺度适中/饱和型 | Rule |
| 结构充分性 | `rps_doc_mean_word_length` | 标量 | 对数尺度适中型 | Rule |

注意字段名中的 `punctution` 是附件原始拼写，代码必须原样读取，不得自动改成 `punctuation` 后造成静默缺列。

### 2.4 非有限值与缺失策略

已知基准：A1 有 18 条记录出现 PRRC 整向量 NaN，共 108 个非有限元素；A3 有 1 条 professionalism 整向量 NaN，共 6 个非有限元素；A2 无非有限值。

- logits 任一分量非有限时，整个字段记缺失，不对残余类别重新 softmax，避免改变类别空间。
- 样本某字段缺失时，只在所属语义块内对剩余字段权重重新归一。
- 若整个语义块缺失，对剩余块重新归一并设置 `missing_block_flag=1`；这类样本单独统计，不进入典型冲突文本展示。
- 域中位数插补只用于敏感性方案，不进入主结果。
- 所有处理前后缺失计数、记录 ID、字段名写入 `data_audit.json`，不得静默丢行。

## 3. 模型实现规格

### 3.1 阶段 Q0：数据审计与冻结

执行顺序：

1. 读取 `source_manifest.json` 和全部 A 文件，保存文件大小、SHA-256、表头和行数。
2. 校验 A4–A15 配比表与 Loss 表的 `index` 一一对应；排序必须显式按 `index`，禁止依赖原文件行序。
3. 校验 A6 与 A8 配比矩阵逐元素一致；A10 与二者不按 index 对齐。
4. 记录闭合前每行配比和、最小值、最大偏差与零比例；只对浮点误差做非负截断和重新闭合。
5. 冻结 `config/problem1.yaml` 和输入清单；后续测试数据不得反馈修改配置。

### 3.2 阶段 Q1：标量化、同向化与锚点拟合

稳定 softmax 使用 `a-max(a)`，六级 logits 的期望为：

```math
z=\frac{1}{5}\sum_{\ell=0}^{5}\ell\frac{e^{a_\ell-\max(a)}}{\sum_r e^{a_r-\max(a)}}.
```

主变换参数仅从 2.2 的质量参考宇宙拟合。单调字段先按七域等权参考分布的 0.5%/99.5% 分位 winsorize，再做 `F_ref` 或 `1-F_ref`。结构型字段在高可信集合 `P(no_ad)>0.8`、`P(fluent)>0.8`、`cleanliness>=0.8` 中估计对数中位数 `m_j` 和 `MAD s_j`：

```math
T_{mid}(x)=\exp\{-\frac{[\log(x+\epsilon)-m_j]^2}{2\max(s_j,s_{min})^2}\}.
```

`s_min` 防止 MAD 为零，默认取该字段对数值域 IQR 的 `0.05` 倍。所有锚点、截尾边界和方向必须写入 `quality_transform_spec.csv`。

### 3.3 阶段 Q2：稳定 CRITIC、五块评分和冲突

在每个语义块内计算 Spearman 相关和 CRITIC 信息量。为避免样本量不平衡，标准差和相关矩阵使用各域等量分层样本；默认每域最多 10,000 条，种子固定为 2026。稳定性修正来自 500 次域内 bootstrap 的域中位数变异。

五块权重固定为 `0.2`，不是待调超参数。样本先计算块分数 `B_ib`，再计算：

```math
K_i=\max_b B_{ib}-\min_b B_{ib},
\qquad
QCI_i=\sum_jw_j|z_{ij}-m_i|.
```

显著冲突必须同时满足：

1. `K_i` 超过本域第 90 百分位；
2. 至少一块 `>=0.75`；
3. 至少一块 `<=0.25`。

非冲突样本用五块等权均值；冲突样本用 Huber M 估计，调节常数 1.345、IRLS 最多 50 次、收敛阈值 `1e-8`。所有样本均保留 Q、五块分数、K、QCI、强冲突标志、最大差异块对和缺失标志。

### 3.4 阶段 Q3：域质量与 A1 代表性

域质量主估计使用样本 Q 的 Huber 均值，10% 截尾均值为对照。每域 500 次域内 bootstrap，报告点估计、标准误、2.5%/97.5% 分位。

arXiv/GitHub 的代表性比较严格使用：

```text
A1 子集 vs A2/A3 中排除相同 ID 后的余集
```

分别报告均值差、中位数差、Wasserstein 距离、KS 统计量及 bootstrap 区间。该比较回答“抽样是否有偏”，不是两个独立总体的常规显著性检验。A2/A3 全量结果用于最终域质量；A1 只承担正文解释。

为做非循环的方向核验，只在 A1 正文上额外提取六个**未进入 Q** 的代理：Unicode 控制/替换字符率、HTML/脚本标签密度、仅含 URL 的行比例、完全重复行比例、异常超长 token 比例和跨文档高频模板行比例。六项均预期在高 Q 组更低。它们不是人工质量真值，只用于发现方向写反或某一规则信号支配总分，不能据此训练或调权。

### 3.5 阶段 Q4：17 域质量映射

A16 中 3 个 direct 域和 3 个 near_direct 域先按确定映射赋值，并保留映射类型。其余 11 个 inferred 域使用 A1 与 A18 的文本分布相似度。

流式文本特征至少包括：`log_chars`、`log_lines`、字母/数字/大写/标点/空白比例、URL 比例、代码符号比例、词汇丰富度、重复 2-gram/3-gram 代理。每个域保存中位数、IQR 和预设分位点，不保存完整 A18 文本副本。

标准化 Wasserstein 距离与对角 Mahalanobis 距离各占 0.5，温度 `tau` 只通过 leave-one-known-domain-out 选择。软权重必须非负且和为 1；inferred 质量向七域全局均值收缩，收缩强度随 A18 样本量减少和最近邻距离增大而增强。

对仅 16、67、79 条等极小样本域，必须保留 `low_support_flag=1`，其区间不得窄于对应最近邻直接域区间。若映射留一误差不达标，inferred 域不输出强排序，只输出区间和探索性点估计。

### 3.6 阶段 M1：配比数据准备

- 17 个配比分量按固定列顺序读取并闭合为单纯形；闭合后每行和误差必须 `<=1e-12`。
- A4 的二阶特征维度必须为 `17+C(17,2)=153`，不加截距。
- 13 个 Loss 域各自建模；训练目标不先跨域平均，避免难度尺度掩盖域差异。
- A4/A5 内部使用固定、可复现的 5 折划分。若存在生成批次信息则按批次分组；否则对 17 维配比做 KMeans 分层后分折，使极端配比不集中到单折。
- A6–A11 在全部模型和参数冻结后只运行一次正式验收。

### 3.7 阶段 M2：候选模型、选模与解释

每个验证域比较：

1. 线性 Scheffé-Ridge；
2. 二阶 Scheffé-Ridge（解释主模型）；
3. 二阶 Scheffé-Elastic Net（交互稀疏性对照）；
4. ExtraTrees / Random Forest；
5. 环境可用时的 XGBoost。

Ridge `alpha` 网格默认 `10^{-6}` 至 `10^4` 共 41 个对数点；Elastic Net 的 `l1_ratio={0.05,0.1,0.25,0.5,0.75,0.9}`，正则强度由内层 CV 选择。树模型只允许在 A4/A5 的内部 CV 中调参。

模型角色只依据 A4/A5 内部 CV 预先冻结：若二阶 Scheffé-Ridge 的综合秩相关/Regret 相对最佳黑箱不劣于 5%，则同时作为预测与解释主模型；若内部 CV 已明显落后，则预先采用“双轨结果”——黑箱负责预测、Scheffé 只负责局部方向与交互解释。A6/A7 只检验该预设角色是否得到外部支持，不再用于调参或改选模型；若测试结论相反，只如实报告泛化失败和性能差距。

交互项只有同时满足以下条件才进入论文机制图：bootstrap 非零符号概率 `>=0.80`、至少 3 个验证域方向一致或对综合目标显著、且不由单个高杠杆配比驱动。否则保留在结果表但标记为不稳定。

### 3.8 阶段 M3：跨规模验证和仿射校准

- A6/A7：检验 1M 同规模的绝对误差与排序，是主测试集。
- A8/A9：配比与 A6 相同，但规模为 60M；可以在每个验证域交叉拟合 `L_60=a_k+b_k Lhat_1m` 的两参数校准，绝对误差由 out-of-fold 预测评价；排序、Top-k 和方向符号必须在校准前评价。
- A10/A11：64 个新配比、1B 模型；同样只允许 5 折交叉拟合整体仿射层比较绝对误差，不允许利用 64 条数据重新拟合 153 个混料系数。样本较少时同时报告不校准的秩指标。
- A12–A15：只报告排序、Top-k、Regret 和预测区间覆盖，不进入主 PASS/FAIL。

### 3.9 阶段 M4：综合目标与约束优化

13 域 Loss 先用 A5 训练分布的中位数/IQR 标准化，再等权构造 `J(p)`。优化约束包括：

- `p_i>=0`、`sum p_i=1`；
- 每域不超过训练配比第 99 百分位 `u_{0.99,i}`；
- 到训练支持域的距离不超过训练样本第 99 百分位阈值；
- 至少 30 个 Dirichlet 初值；最终另用 10,000 个可行随机点检查局部最优质量。

`p*` 由主模型产生，黑箱最优配比只作预测对照。每个 bootstrap 重拟合主模型并重新优化，输出配比区间、进入零边界的概率和目标值区间。若多起点结果不一致或 `p*` 位于支持边界，论文只给出“近优可行配比集合”，不宣称唯一最优点。

## 4. 代码与配置拆分

建议目录如下；开发阶段可以先由一个入口脚本调度，核心函数必须可单测。

```text
code/
  problem1.py                 # CLI 入口与阶段编排
  problem1_data.py            # 流式读取、审计、去重、配比对齐
  problem1_quality.py         # softmax、变换、CRITIC、冲突、域聚合
  problem1_mapping.py         # A18 文本特征、17 域映射
  problem1_mixture.py         # Scheffé 特征、CV、跨规模评估、优化
  problem1_acceptance.py      # 硬门槛与性能门槛汇总
  utils.py                    # 种子、hash、bootstrap、结果写出
config/
  problem1.yaml
results/problem1/
figures/
tests/
  test_problem1_quality.py
  test_problem1_mixture.py
  test_problem1_regression.py
```

配置文件至少冻结：主种子、分块大小、截尾分位、五块字段映射、方向、锚点条件、bootstrap 次数、冲突阈值、CV 折数、正则网格、支持域阈值、优化初值数、Top-k 比例和全部输入路径。

推荐入口：

```bash
python code/problem1.py audit --config config/problem1.yaml
python code/problem1.py quality --config config/problem1.yaml
python code/problem1.py mapping --config config/problem1.yaml
python code/problem1.py mixture --config config/problem1.yaml
python code/problem1.py validate --config config/problem1.yaml
python code/problem1.py all --config config/problem1.yaml
```

默认结果格式使用 CSV/CSV.gz 和 JSON，避免把 Parquet 作为硬依赖；若 `pyarrow` 可用可额外写 Parquet，但论文读取的唯一真源仍是 CSV/JSON。

## 5. 输出文件契约

| 文件 | 最小字段/内容 |
| --- | --- |
| `data_audit.json` | 输入 hash、行列数、字段、缺失/非有限、重复、ID 交集、闭合误差 |
| `quality_transform_spec.csv` | field、block、source、direction、winsor bounds、anchor、MAD、version |
| `quality_weights.csv` | block、field、CRITIC、stability、conditional weight、bootstrap CI |
| `quality_sample.csv.gz` | id、domain、22 个效用、5 块分数、Q、K、QCI、conflict_type、flags |
| `quality_domain.csv` | domain、n、q、CI、trimmed_mean、conflict_rate、mapping_type、evidence |
| `sample_remainder_comparison.csv` | arxiv/github 的 subset、remainder、差值、Wasserstein、KS、CI |
| `domain_mapping.csv` | mixture_domain、anchor_domain、weight、distance、tau、q、CI、flags |
| `mixture_metrics.csv` | model、split、scale、loss_domain、RMSE、MAE、R²、Spearman、Kendall、Top-k、Regret |
| `mixture_coefficients.csv.gz` | loss_domain、term_type、domain_i、domain_j、coef、CI、sign_prob |
| `interaction_stability.csv` | interaction、sign_prob、domains_supported、leverage_flag、publishable |
| `optimal_mixture.csv` | domain、p_star、CI、zero_probability、upper_bound、constraint_margin |
| `acceptance_report.json` | 每条规则的 PASS/WARN/FAIL、观测值、阈值、证据文件 |
| `problem1_manifest.json` | git/代码版本、配置 hash、输入 hash、种子、运行时间、输出 hash |

## 6. 测试设计

### 6.1 单元测试

- softmax 对极大/极小 logits 不溢出，概率和误差 `<1e-12`；常数 logits 得到均匀分布。
- 二分类顺序固定：`ad_en` 取索引 1 的 `no_ad`，`fluency_en` 取索引 1 的 `fluent`。
- 六级期望：单点极大 logit 应接近对应等级除以 5；整向量 NaN 返回缺失而不是 0。
- `F_ref` 单调且输出在 `[0,1]`；负向变换方向相反；适中型在锚点附近最大。
- CRITIC 块内权重非负、和为 1；缺失字段重归一后仍为 1。
- Huber IRLS 在无异常值时接近均值，在加入极端块分数后移动幅度小于普通均值。
- 17 域线性 Scheffé 特征为 17 维，二阶特征严格为 153 维且无截距列。
- 方向导数与有限差分在 `delta=1e-6` 时相对误差 `<1e-5`。
- 任意合法配比闭合后非负且和误差 `<=1e-12`。
- 文本映射权重非负且行和误差 `<=1e-12`。

### 6.2 集成测试

- 在每个源文件前 1,000 行上完成 `audit -> quality -> mapping` 小流程，字段与正式结果一致。
- 用合成已知二阶混料函数生成 512/256 条数据，Scheffé-Ridge 能恢复预测面，测试 R² `>=0.99`。
- 人为打乱 Loss 表 index，连接函数必须仍按 index 对齐；重复或缺失 index 必须报错退出。
- 人为把 A6/A7 加入训练列表，数据泄漏检测必须 FAIL。
- 同一配置连续运行两次，非时间戳数值输出在 `1e-10` 内一致。

### 6.3 数据回归测试

以下是当前附件的固定基准，任何变化都必须先人工确认是不是换了数据版本：

- A1/A2/A3 行数分别为 51,230、17,523、203,752；字段数分别为 27、24、24。
- A1 arXiv 1,419 条全部出现在 A2；A1 GitHub 10,000 条全部出现在 A3。
- A2/A3 排除 A1 后分别剩 16,104、193,752 条；去重质量参考宇宙 261,086 条。
- A4/A5 为 512 行，A6/A7 与 A8/A9 各 256 行，A10/A11 为 64 行，A12–A15 各 63 行。
- 配比列 17 个、Loss 列 13 个；A4 零分量 3,928/8,704，即约 45.13%。
- A18 为 138,034 条、17 域。

## 7. 验收标准与失败降级

### 7.1 状态定义

- **PASS**：满足规则，可进入下一阶段并用于论文主结论。
- **WARN**：结果可继续使用，但必须在论文中写明限制，并采用规定的降级口径。
- **FAIL**：阻断相关下游阶段，不能通过删样本、改测试集或放宽阈值掩盖。

### 7.2 数据与数值硬门槛

| 编号 | 验收项 | PASS 标准 | 失败处理 |
| --- | --- | --- | --- |
| D01 | 文件与行列数 | 与 6.3 基准一致，或有经确认的新 manifest | FAIL，停止 |
| D02 | 字段和数组长度 | 22 字段齐全；二分类 2、PRRC 6、QuRating 4、FineWeb 1 | FAIL，停止 |
| D03 | index 对齐 | A4–A15 每对表一一对应，无重复/缺失 | FAIL，停止 |
| D04 | ID 重叠 | 1,419/10,000 全包含关系被复现 | FAIL，停止并核查版本 |
| D05 | 去重冲突 | 同 `(domain,id)` 的质量字段冲突数为 0 | FAIL，人工判定保留规则 |
| D06 | 非有限处理 | 已知计数被复现，处理后不静默丢行 | FAIL，停止 |
| D07 | 质量范围 | 所有效用、块分数、Q 在 `[0,1]`，有限 | FAIL，停止 |
| D08 | 权重 | 块内和及五块总和误差 `<=1e-12` | FAIL，停止 |
| D09 | 单纯形 | 闭合后非负、行和误差 `<=1e-12` | FAIL，停止 |
| D10 | 数据泄漏 | A6–A15 未参与预处理调参、候选角色选择或正则选择；校准层仅 out-of-fold 评价 | FAIL，重跑 |
| D11 | bootstrap | 500 次中成功率 `>=99%` | WARN；低于 95% 为 FAIL |
| D12 | 优化可行性 | 所有约束余量 `>=-1e-10`，支持距离不过界 | FAIL，停止 |
| D13 | 可复现性 | 两次运行关键数值差 `<1e-10` | WARN；差异影响排名则 FAIL |

### 7.3 质量分可信度目标

这些门槛判断“能否把五块压成一个 Q”，达不到时优先保留多维报告。

| 编号 | 指标 | 目标 | 未达标降级 |
| --- | --- | --- | --- |
| Q01 | 主方案与至少 3 个敏感性方案的域排名 | Kendall `tau>=0.80` | 不给精确全序，只给分层/区间 |
| Q02 | 样本前后 10% 重合率 | 各域中位数 `>=0.70` | 降低单一 Q 的结论强度，分块展示 |
| Q03 | 未入 Q 的正文代理核验 | 高低 Q 组至少 4/6 个代理方向符合预期，至少 3 个 95% CI 不跨 0 | 复核方向和锚点；仍失败则 Q 仅作相对指标 |
| Q04 | 权重集中度 | 任一字段总权重 `<0.25`，任一来源贡献 `<0.70` | WARN，报告等权结果为主对照 |
| Q05 | A1 代表性 | 不设“必须无差异”阈值，但比较表必须完整 | 有差异时 arXiv/GitHub 只用 A2/A3 全量质量 |
| Q06 | Huber 收敛 | 样本收敛率 `>=99.99%` | 未收敛样本用加权中位数并 WARN；超过 0.1% 为 FAIL |

### 7.4 17 域映射目标

| 编号 | 指标 | 目标 | 未达标降级 |
| --- | --- | --- | --- |
| MAP01 | 已知域留一 MAE | `<=0.10` | inferred `q_d` 标记探索性，不用于强排序 |
| MAP02 | 已知域留一 Spearman | `>=0.70` | 只用 A16 direct/near_direct，inferred 回归全局均值 |
| MAP03 | 权重合法性 | 非负、和误差 `<=1e-12` | FAIL |
| MAP04 | 小样本不确定性 | low-support inferred 域 CI 不窄于最近邻直接域 | 扩大区间并 WARN |
| MAP05 | 最近邻可解释性 | 每个 inferred 域输出前三锚点及距离 | 缺失则不允许写入论文映射解释 |

### 7.5 配比模型性能目标

以下为预注册目标。应同时报告 13 域分布，不能只报告平均值。

| 编号 | 测试集 | 指标与目标 | 未达标降级 |
| --- | --- | --- | --- |
| M01 | A6/A7 1M | 13 域 Spearman 中位数 `>=0.80`、Kendall 中位数 `>=0.60` | 二阶模型若仍优于线性则保留排序用途；否则只报告描述性响应面 |
| M02 | A6/A7 1M | 13 域 R² 中位数 `>=0.50`，标准化 RMSE 优于线性基线 | 绝对 Loss 不用于优化，只优化秩/经验候选 |
| M03 | A8/A9 60M | 校准前 Spearman 中位数 `>=0.80`；Top 10% 重合率中位数 `>=0.50` | 仅声称局部规模迁移，不用于 60M 绝对预测 |
| M04 | A10/A11 1B | Spearman 中位数 `>=0.75`，且优于线性基线 | `p*` 限定在小模型结论，不外推 1B |
| M05 | A6–A11 | 归一化 Regret 中位数 `<=0.25 IQR` | 输出近优候选集合，不给唯一 `p*` |
| M06 | bootstrap | 进入论文的交互符号概率 `>=0.80` | 删除该机制解释，不删除原始结果 |
| M07 | A4/A5 内部 CV 的主模型竞争力 | Scheffé-Ridge 综合表现距最佳黑箱 `<=5%` | 在读取测试集前冻结为黑箱预测 + Scheffé 解释双轨 |
| M08 | 质量-边际效应关联 | 报告 Spearman/Kendall 与 CI，不设必须显著 | 不显著即如实报告“未发现稳定关联” |

### 7.6 优化验收

| 编号 | 验收项 | PASS 标准 | 未达标降级 |
| --- | --- | --- | --- |
| O01 | 多起点一致性 | 最佳 5 个可行解目标相对极差 `<=1e-6` | 增加初值；仍失败则报告近优集合 |
| O02 | 随机可行点对照 | `p*` 不劣于 10,000 个随机可行点最佳值超过数值容差 | FAIL，检查优化器/梯度 |
| O03 | 支持域 | 距离不超过 `r_0.99`，各分量不超过训练 `u_0.99` | FAIL，禁止外推最优解 |
| O04 | bootstrap 稳定性 | 至少 80% 重采样解落在原支持域内 | WARN；否则只给宽松近优区间 |
| O05 | 目标改善 | 相对训练集最佳观测配比有改善时必须给 CI | CI 跨 0 时不得声称“显著优于已观测最佳” |

## 8. 敏感性矩阵

正式运行至少完成下列预设对照，不得只挑有利结果：

| 模块 | 主口径 | 对照 |
| --- | --- | --- |
| logits | softmax 期望 | argmax |
| QuRating | 四分量先变换后平均 | 四分量展开为 25 维 |
| 参考分布 | 七域等权 ECDF | pooled ECDF、域内 ECDF |
| 截尾 | 0.5%/99.5% | 0/100%、1%/99% |
| 权重 | 五块等权 + 稳定 CRITIC | 字段等权、普通 CRITIC、熵权、Meta-rater 权重 |
| 冲突阈值 | 90% + 0.75/0.25 | 85%、95%；0.70/0.30 |
| 冲突消解 | Huber M | 普通均值、加权中位数、指数可靠度 |
| 域聚合 | Huber mean | 中位数、10% 截尾均值 |
| 映射距离 | Wasserstein + 对角 Mahalanobis | 单一距离、硬最近邻 |
| 组成数据 | 原单纯形 Scheffé | CLR/ILR 多伪计数 |
| 配比模型 | 二阶 Scheffé-Ridge | 线性、Elastic Net、RF/ExtraTrees/XGBoost |
| 综合目标 | 13 域标准化等权 | 原始 Loss 等权、留一验证域 |
| 支持约束 | 分量/距离 99% | 95%、97.5% |

敏感性结果统一用域排名 Kendall、样本 Top/Bottom 10% 重合、交互符号概率、`p*` 的 L1 距离和 Regret 变化评价。

## 9. 开发顺序与完成定义

| 里程碑 | 工作内容 | 完成定义 |
| --- | --- | --- |
| P1-00 | 数据审计、hash、schema、index、重叠和非有限值 | D01–D06 全 PASS，生成 `data_audit.json` |
| P1-01 | 22 字段标量化、七域等权参考变换 | 单元测试通过，生成变换规格和小样本输出 |
| P1-02 | CRITIC、五块、冲突、Q、域聚合 | D07–D11 通过，生成 7 域质量和代表性比较 |
| P1-03 | A18 文本特征和 17 域映射 | MAP01–MAP05 有结论，明确 direct/inferred 证据等级 |
| P1-04 | Scheffé 特征、内部 CV、候选模型 | A4/A5 内部结果冻结，未读取测试标签调参 |
| P1-05 | A6–A11 一次性验收、A12–A15 敏感性 | M01–M08 形成 PASS/WARN/FAIL 与降级决定 |
| P1-06 | 约束优化和 bootstrap | O01–O05 形成 `p*` 或近优集合 |
| P1-07 | 全链复现、结果清单和开发验收 | `acceptance_report.json`、`problem1_manifest.json` 完整；可交给 3coding-visual |

问题一“完成”不是所有性能指标必须漂亮，而是：数据硬门槛无失败；测试集只被正式使用一次；每个性能未达标项都执行了预先规定的降级；全部论文数字可以从 `results/problem1/` 追溯到配置、输入和代码版本。

## 10. 论文写作时的引用提醒

方法编号 `[P1-R1]` 至 `[P1-R14]` 的完整书目和引用边界见主报告 5.8；可直接用于 LaTeX 的条目见 [`problem1_references.bib`](../references/problem1_references.bib)。写作时按以下措辞区分来源与创新：

- “依据 Meta-rater/QuRating 定义质量维度”，不要写成“本文复现了 Meta-rater 权重学习”；本题没有训练数百个质量代理模型。
- “在 CRITIC 基础上加入 bootstrap 稳定惩罚”，不要把稳定项写成原始 CRITIC 公式。
- “采用 Huber M 估计稳健聚合，并自行定义块冲突门槛”，不要把 K/QCI 归因于 Huber。
- “采用 Scheffé 二阶混料响应面并用 Ridge 正则”，分别引用混料实验和 Ridge；支持域约束与方向边际效应是本文工程设计。
- “借鉴 RegMix 的小模型回归选配比思想”，不要写成完全复现 RegMix；本题的回归器、损失标准化和解释框架均不同。
