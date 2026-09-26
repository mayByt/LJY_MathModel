from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def write_results_report(path: Path, cfg: dict[str, Any], data: dict[str, Any], classic: dict[str, Any], quality: dict[str, Any], bridge: dict[str, Any], generalized: dict[str, Any], acceptance: dict[str, Any]) -> None:
    ccv = classic["cv"].groupby("model")[["RMSE", "MAE", "nRMSE", "Spearman_D"]].mean().reset_index()
    qcv = quality["family_scores"].copy()
    lines = []
    lines.append("# 问题二修复后计算与验收结果\n")
    lines.append("> 本报告由 `问题二/problem2.py` 从结果 CSV/JSON 自动生成；本轮按用户要求未生成 PDF 图表。\n")
    lines.append("## 运行环境与边界\n")
    lines.append(f"- 随机种子：`{cfg['seed']}`")
    lines.append(f"- bootstrap 次数：`{cfg['bootstrap_reps']}`")
    lines.append("- 绘图：已跳过，仅完成建模、验收和结果报告。")
    lines.append("- GPU策略：硬件和依赖支持且实测更快时优先；本轮 scipy 拟合路径使用 CPU，并在 `run_log.json` 记录回退原因。\n")
    lines.append(f"- 问题三接口状态：`{acceptance['interface_state']}`。\n")
    lines.append("## 数据审计\n")
    checks = data["audit"]["checks"]
    lines.append(f"B1 共 `{checks['b1_rows']}` 行、`{checks['b1_n_levels']}` 个模型规模；B7 为 `{checks['b7_nd_cells']}` 个 `(N,D)` 单元，每单元 `{checks['b7_q_per_cell_min']}`-`{checks['b7_q_per_cell_max']}` 个 Q。")
    lines.append(f"B6/B7 共享键 `{checks['b6_in_b7_matches']}` 条，最大 Loss 差 `{checks['b6_in_b7_max_loss_diff']:.3e}`。B8 已隔离，仅作压力测试。")
    lines.append(f"B2/B3 规则选择 B2 作族外验证；B3 的 `{checks['b2_or_b3_validation_choice']['B3_files']}` 条插值轨迹文件仅审计，不作为新增独立证据。")
    lines.append(f"B11 审计 `{checks['b11_source_audit']['rows']}` 行、B12 审计 `{checks['b12_checkpoint_audit']['rows']}` 行。")
    ledger = pd.DataFrame([{"附件": k, "角色": v} for k, v in data["audit"]["usage_ledger"].items()])
    lines.append("\nB1–B12 使用账本：")
    lines.append(ledger.to_markdown(index=False))
    lines.append("\n## 经典标度律\n")
    lines.append(f"获选经典模型：`{classic['summary']['selected_model']}`，规则：`{classic['summary']['selection_rule']}`。")
    lines.append(pd.DataFrame(classic["summary"]["parameters"], index=[0]).to_markdown(index=False))
    lines.append("\n经典模型留一规模均值：")
    lines.append(ccv.to_markdown(index=False, floatfmt=".6f"))
    warmup = classic["summary"]["warmup_sensitivity"]
    lines.append(f"\n早期轨迹敏感性：删除 `D<2` B tokens 的 `{warmup['removed_rows']}` 行后重拟合，alpha/beta 最大相对变化 `{warmup['alpha_beta_max_abs_relative_change']:.6%}`；该结果不参与主模型选择。")
    lines.append(classic["warmup"].to_markdown(index=False, floatfmt=".8f"))
    lines.append("\n外部与外推验证（B10 为估算对照，不是独立观测）：")
    lines.append(classic["external"].to_markdown(index=False, floatfmt=".6f"))
    lines.append("\n## 质量模型\n")
    lines.append(f"锚定状态：`{quality['summary']['fit_mode']}`；Q=1 锚点 nRMSE=`{quality['summary']['q1_anchor_nrmse']:.6f}`。")
    lines.append(f"获选质量模型：`{quality['summary']['selected_model']}`；家族等权 one-SE 原始候选：`{quality['summary']['raw_selected_model']}`。")
    lines.append(pd.DataFrame(quality["summary"]["parameters"], index=[0]).to_markdown(index=False))
    lines.append("\n四类切分的无泄漏绝对 Loss 验证（每类选模权重均为 0.25）：")
    lines.append(qcv.to_markdown(index=False, floatfmt=".6f"))
    lines.append("\n## 问题一桥接与配比迁移\n")
    b = bridge["bridge"]
    lines.append(f"主锚点使用 `p_ref`，`Q_ref={b['Q_ref']:.10f}`；`p_star` 仅作敏感性，`Q_star={b['Q_star']:.10f}`。")
    lines.append(f"问题一 WARN 已传播：`{', '.join(b['propagated_warn_ids'])}`。")
    lines.append(f"桥接接口版本：`{b['problem1_bridge_interface']}`；问题二直接读取附件 A：`{b['raw_attachment_A_read_by_problem2']}`。")
    lines.append("\n配比迁移指标：")
    transfer_view = bridge["transfer"][bridge["transfer"]["role"] == "composite"].copy()
    lines.append(transfer_view.to_markdown(index=False, floatfmt=".6f"))
    lines.append(f"\nScheffé 真实重拟合 bootstrap 有效率 `{b['scheffe_bootstrap_valid_rate']:.3f}`，原始系数最大抽样标准差 `{b['scheffe_bootstrap_max_std']:.6g}`；原始项存在共线与尺度差异，解释以归一化响应 `R(p)` 而非单个系数为准。")
    lines.append(f"`R(p_star)` 点值 `{b['R_star']:.6f}`，bootstrap 中位数 `{b['R_star_bootstrap']['median']:.6f}`，95%区间 [`{b['R_star_bootstrap']['ci_low']:.6f}`, `{b['R_star_bootstrap']['ci_high']:.6f}`]。")
    stable_interaction = bridge["interaction"][bridge["interaction"]["relation"] != "uncertain"].head(12)
    lines.append(f"\n领域交互 bootstrap 分类：`{b['interaction_relation_counts']}`。负的标准化二阶交互表示联合增加相对线性叠加更能降低 Loss（互补），正值表示竞争同一配比预算（替代）；区间跨零只记为不确定。")
    if len(stable_interaction):
        lines.append(stable_interaction.to_markdown(index=False, floatfmt=".6f"))
    lines.append("\n## 广义模型与等价量\n")
    gp = generalized["parameters"]
    lines.append(f"最终接口角色：`{gp['final_formula_role']}`；质量映射情景 `{gp['quality_mapping_slopes']}`；配比强度 `{gp['mixture_strength_role']}`。")
    lines.append(f"联合 bootstrap 有效率 `{gp['joint_bootstrap_valid_rate']:.3f}`；配比强度使用 B7 Loss IQR 显式换算的预注册 profile，不视为附件直接识别值。")
    cp, qp = gp["classic_parameters"], gp["quality_parameters"]
    lines.append(
        f"主身份映射且固定 `p_ref` 时：`L={cp['E']:.6f}+{cp['A']:.6f}n^(-{cp['alpha']:.6f})+"
        f"{cp['B']:.6f}d^(-{cp['beta']:.6f})+{qp['c_Q']:.6f}(1-Q)^({qp['nu_Q']:.6f})`，"
        "其中 `n=N/1e9,d=D/1e9`。"
    )
    lines.append("标准弹性列 `epsilon_N/D/Q` 等于 `-x/L*dL/dx`；`epsilon_*_reducible` 另表示 `-x/(L-E)*dL/dx`，当 `L<=E` 时不定义。")
    physical = generalized["physical_gate"]
    lines.append(f"非零配比物理门槛：`{physical['status']}`；最低预测 `{physical['min_nonzero_scenario_prediction']:.6f}`，不可约损失 `E={physical['E']:.6f}`，低于 E 的网格行数 `{physical['below_E_rows']}`。")
    lines.append(generalized["physical_by_scenario"].to_markdown(index=False, floatfmt=".6f"))
    eq = generalized["equivalence"][["s_Q", "Q_A", "delta_Q", "finite_N_multiplier_minus_1", "finite_D_multiplier_minus_1", "root_status_N", "root_status_D"]]
    lines.append(eq.to_markdown(index=False, floatfmt=".6f"))
    lines.append("\n## 验收\n")
    lines.append(f"总体状态：**{acceptance['overall']}**；PASS={acceptance['summary']['PASS']}，WARN={acceptance['summary']['WARN']}，FAIL={acceptance['summary']['FAIL']}。")
    warn = [r for r in acceptance["rules"] if r["status"] == "WARN"]
    if warn:
        lines.append("\nWARN 降级说明：")
        for r in warn:
            lines.append(f"- `{r['id']}` {r['name']}：{r.get('action') or '保留为风险说明'}")
    output_root = cfg.get("output_root", "results")
    baseline_path = path.parent.parent / output_root / "repair_baseline.json"
    if baseline_path.exists():
        old = json.loads(baseline_path.read_text(encoding="utf-8"))
        old_acc = old.get("summaries", {}).get("problem2_acceptance_report.json", {})
        old_c = old.get("summaries", {}).get("classic_scaling_parameters.json", {})
        old_q = old.get("summaries", {}).get("quality_model_parameters.json", {})
        lines.append("\n## 修复前—修复后对照\n")
        lines.append(f"- 修复前验收：`{old_acc.get('overall', 'unknown')}` {old_acc.get('summary', {})}；修复后：`{acceptance['overall']}` {acceptance['summary']}。两轮规则集合不同，PASS 数量不可直接作为优劣分数；以新增 RF01–RF11 及最终封版门槛是否通过为准。")
        lines.append(f"- 经典模型：`{old_c.get('selected_model', 'unknown')}` → `{classic['summary']['selected_model']}`。")
        lines.append(f"- 质量模型：`{old_q.get('selected_model', 'unknown')}` → `{quality['summary']['selected_model']}`；关键变化是验证口径、抽样和可调用接口，不强制模型名称改变。")
    lines.append("\n## 结果文件\n")
    lines.append(f"核心文件位于 `问题二/{output_root}/`：`scaling_data_audit.json`、`classic_scaling_parameters.json`、`classic_warmup_sensitivity.csv`、`classic_external_validation.csv`、`quality_model_parameters.json`、`problem1_bridge.json`、`domain_substitution_complementarity.csv`、`problem1_mixture_response_bootstrap.npz`、`mixture_transfer_bootstrap.csv.gz`、`mixture_physical_gate.json`、`elasticity_grid.csv`、`generalized_scaling_bootstrap.csv.gz`、`units_contract.json`、`problem2_acceptance_report.json`。")
    lines.append("\n## 可复现运行方式\n")
    lines.append("```bash\npython 问题二/problem2.py all --config <config>\n```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
