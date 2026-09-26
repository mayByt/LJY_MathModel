from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def write_results_report(report_dir: Path, center: pd.DataFrame, intervals: pd.DataFrame,
                         transitions: pd.DataFrame, sensitivity: pd.DataFrame,
                         acceptance: dict, audit: dict, cfg: dict) -> Path:
    op = center[center["support_mode"].eq("operational_extended")].copy()
    main = op[op["context_length"].eq(4096)].sort_values(["quality_cost_type", "budget_FLOPs"])
    lines = [
        "# 问题三建模与验收结果报告",
        "",
        "> 版本：problem3_v1。按用户要求，本轮不生成图表；本报告完全由数值结果文件自动生成。",
        "",
        "## 1. 结论与接口状态",
        "",
        f"- 总体验收：**{acceptance['overall']}**；PASS={acceptance['summary']['PASS']}，WARN={acceptance['summary']['WARN']}，FAIL={acceptance['summary']['FAIL']}。",
        f"- 接口状态：`{acceptance['interface_state']}`。",
        "- 主模型严格承接问题二冻结版本 M0+GQ2：固定 `p=p_ref`、`lambda0=0`，配比不参与联合优化。",
        "- C7 只用于核验上下文长度情景；其他架构字段即使存在异常也不进入 Loss 或成本公式。",
        "- 主回答使用 operational_extended 支持域，同时单独运行 strict_joint 以标记观测支持与外推风险。",
        "",
        "## 2. 模型与约束",
        "",
        r"$$L=E+A(N/10^9)^{-\alpha}+B(D/10^9)^{-\beta}+c_Q(1-Q_{eff})^{\nu_Q},$$",
        "",
        r"其中 $Q_{eff}=Q_0+s_Q(Q_A-Q_0)$，主情景 $s_Q=1$。成本约束为",
        "",
        r"$$6ND+D[g(Q_A)-g(Q_0)]_++\eta NDL_{ctx}\le C,$$",
        "",
        r"且主值 $\eta=2\times10^{-4}$。计算中 $N,D$ 使用绝对单位，结果表同时输出十亿单位。",
        "",
        "## 3. 4096 上下文的三档预算结果",
        "",
        "|质量成本|预算 FLOPs|N(B)|D(B tokens)|Q_A|预测 Loss|训练份额|质量份额|注意力份额|证据标签|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in main.itertuples():
        lines.append(f"|{row.quality_cost_type}|{row.budget_FLOPs:.0e}|{_fmt(row.N_B)}|{_fmt(row.D_B)}|{_fmt(row.Q_A)}|{_fmt(row.predicted_loss)}|{_fmt(row.share_train)}|{_fmt(row.share_quality)}|{_fmt(row.share_attention)}|{row.support_status}|")
    lines += [
        "",
        "上述每个方案均另存 17 个领域的 Token 数，满足 $D_i=p_{ref,i}D$；这就是固定配比在问题三中的实际用途。",
        "",
        "## 4. 不确定性",
        "",
        f"- 三档锚点使用 {int(intervals['valid_draws'].max()) if len(intervals) else 0} 个有效联合参数抽样计算 95% 区间。",
        f"- 连续路径按预注册主成分分层选取 {cfg['bootstrap_path_draws']} 个抽样，用于估计结构转移出现概率。",
        "- 只有出现概率不低于 70% 的同类事件才标为稳定；其余只称中心参数下候选转移。",
        "",
        "## 5. 结构性转移",
        "",
        "|成本函数|上下文|类型|事件|预算中点|log10区间宽|出现概率|稳定|",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    if transitions.empty:
        lines.append("|—|—|—|在扫描范围内未识别到满足门槛的转移|—|—|—|—|")
    else:
        for row in transitions.sort_values(["quality_cost_type", "context_length", "budget_mid"]).itertuples():
            lines.append(f"|{row.quality_cost_type}|{row.context_length}|{row.transition_class}|{row.event_type}|{row.budget_mid:.3e}|{_fmt(row.log10_interval_width)}|{_fmt(row.stability_probability)}|{'是' if row.stable else '否'}|")
    lines += [
        "",
        "注意力临界上下文长度为 30000。它比较的是不同上下文场景的成本结构，不是预算路径上的结构转移，因此未混入转移事件表。",
        "",
        "## 6. 敏感性与支持边界",
        "",
        f"共运行 {len(sensitivity)} 条敏感性结果，覆盖质量映射斜率、质量成本幅度和形状、注意力系数、经验质量上界以及 strict_joint 支持域。",
        f"中心 45 个主场景中，外推标记数量为 {int(op['extrapolation_flag'].sum())}/{len(op)}。外推 Loss 是模型推断，不作为新的实测证据。",
        "",
        "## 7. 数据审计",
        "",
        f"- 问题二冻结输入 hash 不匹配数：{len(audit['hash_mismatches'])}。",
        f"- C7 形状：{audit['c7']['shape']}；上下文核验：{audit['c7']['context_gate']}；架构字段 WARN 行：{audit['c7']['architecture_warning_rows']}。",
        f"- C7 非上下文字段进入问题三公式：{audit['c7']['non_context_fields_used_by_problem3']}。",
        "",
        "## 8. 验收与使用边界",
        "",
    ]
    failed = acceptance["hard_gate_failures"]
    if failed:
        lines.append(f"存在未通过硬门禁：{failed}。问题四不得读取 `PROBLEM3_READY=False` 的条目。")
    else:
        lines.append("全部硬门禁通过。问题四只能读取 `problem3_bridge.json` 中 `PROBLEM3_READY=True` 的条目，并保留 evidence/support 标签。")
    lines += [
        "",
        "局限：配比绝对强度仍未被问题二识别，所以本题没有把 $p$ 当连续决策变量；质量成本函数为题设情景，并非从训练日志估计；超出问题二联合观测域的结果必须按外推解释。",
        "",
        "## 9. 复现",
        "",
        "```bash",
        "python 问题三/problem3.py --config 问题三/config/problem3_v1.json --stage all",
        "```",
    ]
    path = report_dir / "RESULTS_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
