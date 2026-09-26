#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from problem4.io import atomic_csv, atomic_json, audit_contract, load_inputs, match_c2_c4, resolve_duplicates, sha256
from problem4.v2 import (
    aggregate_c8_v2, bootstrap_compute_growth, bootstrap_frontier_joint,
    build_panel_v2, c3_historical_analysis, check_p3_interface,
    decompose_v2, enrich_matched, fit_bridges, fit_dynamic_v2, fit_growth,
    forecast_joint, link_to_score, map_problem3_v2, nd_diagnostics,
    prepare_compute_panel, rolling_cv_v2, score_to_link, snapshot_tree,
    summarize_decomposition,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config" / "problem4_v2.json"))
    parser.add_argument("--bridge-draws", type=int)
    parser.add_argument("--frontier-draws", type=int)
    parser.add_argument("--compute-draws", type=int)
    parser.add_argument("--c3-draws", type=int)
    return parser.parse_args()


def add_rule(rules, rule_id, name, status, observed, threshold, action=""):
    rules.append({"id": rule_id, "name": name, "status": status, "observed": observed,
                  "threshold": threshold, "action": action})


def make_acceptance(ctx):
    d, b, m, f = [], [], [], []
    add_rule(d,"D01","问题三冻结接口","PASS" if ctx["p3_ready"] else "FAIL",ctx["p3_mismatches"],"READY且hash一致","停止")
    add_rule(d,"D02","C1/C2与Average合同","PASS" if ctx["avg_error"]<=1e-10 else "FAIL",ctx["avg_error"],"<=1e-10","停止")
    add_rule(d,"D03","C3实质使用","PASS" if ctx["c3_rows"]==26 and ctx["c3_generated"] else "FAIL",{"rows":ctx["c3_rows"],"generated":ctx["c3_generated"]},"26行并生成敏感性","停止")
    add_rule(d,"D04","C4字段使用审计","PASS" if ctx["c4_contract"] else "FAIL",ctx["c4_contract"],"compute/data/open/method/confidence/bounds齐全","停止")
    add_rule(d,"D05","strict unrestricted许可证","PASS" if ctx["strict_invalid"]==0 else "FAIL",ctx["strict_invalid"],"0","严格口径失败")
    add_rule(d,"D06","模型类型分层","PASS" if ctx["unclassified"]==0 else "FAIL",ctx["unclassified"],"0","停止")
    c8_any=ctx["c8_math_ready"] or ctx["c8_bbh_ready"]
    c8_status="PASS" if ctx["c8_math_ready"] and ctx["c8_bbh_ready"] else ("WARN" if c8_any else "FAIL")
    add_rule(d,"D07","C8逐任务聚合",c8_status,{"MATH":ctx["c8_math_ready"],"BBH":ctx["c8_bbh_ready"],"summary":ctx["c8_summary"]},"至少一项READY；未READY项降级", "只使用READY项" if c8_status=="WARN" else "停止")
    add_rule(d,"D08","C5为C6子集","PASS" if ctx["c5_subset"] else "FAIL",ctx["c5_subset"],"true","停止")

    add_rule(b,"B01","桥接单调性","PASS" if ctx["bridge_violations"]==0 else "FAIL",ctx["bridge_violations"],"0")
    add_rule(b,"B02","来源族折外预测","PASS" if ctx["bridge_folds"]>=3 else "FAIL",ctx["bridge_folds"],">=3")
    add_rule(b,"B03","桥接指标完整","PASS" if ctx["bridge_metrics"] else "FAIL",ctx["bridge_metrics"],"MAE/RMSE/Spearman/coverage/width")
    add_rule(b,"B04","Average优于常数基线","PASS" if ctx["average_bridge_ok"] else "FAIL",ctx["average_bridge_comparison"],"selected MAE < constant MAE")
    add_rule(b,"B05","45个情景及支持域","PASS" if ctx["scenario_count"]==45 else "FAIL",ctx["scenario_count"],"45")
    add_rule(b,"B06","桥接区间有序","PASS" if ctx["bridge_intervals_ordered"] else "FAIL",ctx["bridge_intervals_ordered"],"全部有序")
    add_rule(b,"B07","桥接误差分解","PASS" if ctx["bridge_modes"]==3 else "FAIL",ctx["bridge_modes"],"3")

    add_rule(m,"M01","M-C可识别与bootstrap","PASS" if ctx["fit_success"] and ctx["frontier_boot_rate"]>=.95 else "FAIL",{"fit":ctx["fit_success"],"valid_rate":ctx["frontier_boot_rate"]},"fit且>=0.95")
    add_rule(m,"M02","规模方向","PASS" if ctx["beta_x"]>=-1e-12 else "FAIL",ctx["beta_x"],">=0")
    add_rule(m,"M03","3/6月滚动回测","PASS" if ctx["cv_gate"] else "FAIL",ctx["cv_comparison"],"M-C不劣于最佳基线超过1SE")
    add_rule(m,"M04","rho与lambda由回测选择","PASS" if ctx["selected_from_grid"] else "FAIL",{"lambda":ctx["lambda"],"rho":ctx["rho"]},"均来自预注册网格")
    add_rule(m,"M05","M-N/M-ND诊断","PASS" if ctx["nd_complete"] else "FAIL",ctx["nd_complete"],"样本/条件数/系数齐全")
    add_rule(m,"M06","Shapley闭合","PASS" if ctx["closure"]<=1e-10 else "FAIL",ctx["closure"],"<=1e-10")

    add_rule(f,"F01","预测算力端点","PASS" if ctx["endpoint_from_trend"] else "FAIL",ctx["endpoint"],"截止月趋势拟合值")
    add_rule(f,"F02","增长抽样非退化","PASS" if ctx["growth_non_degenerate"] else "FAIL",ctx["growth_std"],">0")
    add_rule(f,"F03","三分位无交叉","PASS" if ctx["quantiles_ordered"] else "FAIL",ctx["quantiles_ordered"],"全部有序")
    add_rule(f,"F04","score-link一致","PASS" if ctx["link_error"]<=1e-10 else "FAIL",ctx["link_error"],"<=1e-10")
    add_rule(f,"F05","tau专属区间","PASS" if ctx["tau_intervals"] else "FAIL",ctx["tau_intervals"],"每个tau来自自身draw")
    add_rule(f,"F06","情景完整","PASS" if ctx["scenario_factors"]>=3 else "FAIL",ctx["scenario_factors"],">=3")
    add_rule(f,"F07","外推标记","PASS" if ctx["extrapolation_flags"] else "FAIL",ctx["extrapolation_flags"],"12/24月标记正确")
    add_rule(f,"F08","自然边界无需裁剪","PASS" if ctx["scores_in_bounds"] else "FAIL",ctx["scores_in_bounds"],"[0,100]")
    add_rule(f,"R01","连续运行中心结果复现","PASS" if ctx["reproducibility_pass"] else "FAIL",ctx["reproducibility"],"同配置最大差<=1e-10","再次运行V2")

    def ready(rows): return all(x["status"]!="FAIL" for x in rows)
    data_ready=ready(d); bridge_ready=data_ready and ready(b); decomp_ready=data_ready and ready(m); forecast_ready=decomp_ready and ready(f)
    rules=d+b+m+f; fail=sum(x["status"]=="FAIL" for x in rules); warn=sum(x["status"]=="WARN" for x in rules)
    return {"summary":{"PASS":sum(x["status"]=="PASS" for x in rules),"WARN":warn,"FAIL":fail},
            "overall":"PASS" if fail==0 else "FAIL",
            "interface_state":{"P4_DATA_READY":data_ready,"P4_BRIDGE_READY":bridge_ready,
                               "P4_DECOMPOSITION_READY":decomp_ready,"P4_FORECAST_READY":forecast_ready,
                               "PROBLEM4_READY":data_ready and bridge_ready and decomp_ready and forecast_ready},
            "rules":rules}


def render_report(path, values):
    a=values["acceptance"]; dec=values["decomposition"]; unc=values["decomp_uncertainty"].set_index("quantity")
    s=unc.loc["scale_score"]; t=unc.loc["tech_score"]; total=unc.loc["total_score"]
    forecast=values["forecast"][values["forecast"]["tau"].eq(.9)].sort_values(["scenario_factor","horizon_months"],ascending=[False,True])
    table=["| 算力增速系数 | 时长/月 | 0.90前沿 | 80%区间 | 95%区间 |","|---:|---:|---:|---:|---:|"]
    for _,r in forecast.iterrows(): table.append(f"| {r.scenario_factor:.2f} | {int(r.horizon_months)} | {r.forecast_score:.2f} | [{r.p10:.2f}, {r.p90:.2f}] | [{r.p025:.2f}, {r.p975:.2f}] |")
    lines=["# 问题四 V2 修复结果报告","","## 1. 验收结论","",
        f"- 总体状态：{a['overall']}；PASS={a['summary']['PASS']}，WARN={a['summary']['WARN']}，FAIL={a['summary']['FAIL']}。",
        f"- 接口：`P4_DATA_READY={a['interface_state']['P4_DATA_READY']}`，`P4_BRIDGE_READY={a['interface_state']['P4_BRIDGE_READY']}`，`P4_DECOMPOSITION_READY={a['interface_state']['P4_DECOMPOSITION_READY']}`，`P4_FORECAST_READY={a['interface_state']['P4_FORECAST_READY']}`，`PROBLEM4_READY={a['interface_state']['PROBLEM4_READY']}`。",
        "- 本轮按要求不生成图表。V1 文件经运行前后 SHA-256 对比确认未改变。","",
        "## 2. 数据、题意与口径","",
        f"- C1/C2 原始 4,576 行，去重后 {values['dedup_rows']} 行；匹配 C4 接受 {values['accepted_matches']} 行；确认性 pretrained 前沿 {values['frontier_rows']} 行、{values['frontier_months']} 月、{values['frontier_families']} 家族。",
        f"- C3 26 条历史记录已形成独立方向敏感性：年系数 {values['c3']['year_coef']:.4f}，正号概率 {values['c3']['positive_probability']:.3f}，结论 `{values['c3']['direction']}`；它没有混入现代前沿主拟合。",
        f"- C8：BBH完整 {values['c8']['bbh_complete']} 个，唯一可比模型中官方精确复现 {values['c8']['official_exact_bbh']}/{values['c8']['bbh_unique_comparable']}，作为正式逐任务复现；MATH精确复现 {values['c8']['official_exact_math']}/{values['c8']['math_unique_comparable']}，未达到95%门槛，只保留为审计敏感性。BBH使用随机猜测下界归一化后对子任务等权平均。",
        f"- posttrained 合格算力样本 {values['posttrained_rows']} 条，按预注册规则标记 `{values['posttrained_status']}`，不强行拟合后训练贡献。","",
        "## 3. Loss–Benchmark桥接","",
        f"- Average 选择 `{values['average_model']}`；折外 MAE={values['average_mae']:.3f}，常数基线 MAE={values['average_const_mae']:.3f}。问题三 45 个冻结情景全部保留。",
        f"- 弱桥接目标：{', '.join(values['weak_targets']) if values['weak_targets'] else '无'}。弱目标不承担综合能力主结论。",
        "- `upstream_only`、`bridge_only`、`joint` 三套区间已分别保存，区间宽度不相加。","",
        "## 4. 规模与非规模残余贡献","",
        f"- 回测选择 lambda={values['lambda']}、rho={values['rho']}；标准化 log10 算力系数 beta_C={values['beta_x']:.4f}。",
        f"- {dec['t0']}至{dec['t1']}：能力总变化 {dec['total_score']:.3f} 分，95%经验区间 [{total.p025:.3f}, {total.p975:.3f}]。",
        f"- 规模扩张相关贡献 {dec['scale_score']:.3f} 分，80%区间 [{s.p10:.3f}, {s.p90:.3f}]，正号概率 {s.positive_probability:.3f}。",
        f"- 控制规模后的非规模残余贡献 {dec['tech_score']:.3f} 分，80%区间 [{t.p10:.3f}, {t.p90:.3f}]，正号概率 {t.positive_probability:.3f}。",
        "- 以上为观察性分解，不解释为严格因果效应。","",
        "## 5. 算力放缓下的未来前沿","",
        f"- 截止月趋势端点 log10 compute={values['growth']['endpoint_log10_compute']:.3f}；截止月原始月前沿={values['growth']['raw_cutoff_month_log10']:.3f}，最近三个月中位前沿={values['growth']['last3_median_log10']:.3f}；年化 log10 增速={values['growth']['annual_log10_slope']:.3f}。端点和增速均在bootstrap中重估。","",*table,"",
        "- 0.85/0.90/0.95 每个分位都有自己的重排后抽样区间；12个月为 extrapolation，24个月为 long_horizon_extrapolation。","",
        "## 6. V1→V2修复核验","",
        f"- 精确link往返最大误差 {values['link_roundtrip_error']:.3e}；预测score–link最大误差 {values['forecast_link_error']:.3e}。",
        f"- 三分位中心交叉修正 {values['central_rearranged']} 行；bootstrap内交叉修正 {values['draw_rearranged']} 行。",
        f"- 连续运行复现最大中心数值差 {values['reproducibility']['max_abs_difference']:.3e}；V1保留为历史对照，V2所有结果写入独立目录。","",
        "## 7. 可复现运行","",
        "```bash","python 问题四/problem4_v2.py --config 问题四/config/problem4_v2.json","pytest -q 问题四/tests/test_problem4_v2.py","```",""
    ]
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(lines),encoding="utf-8")


def main():
    args=parse_args(); started=time.time(); cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    bridge_draws=args.bridge_draws or cfg["bridge_bootstrap_draws"]; frontier_draws=args.frontier_draws or cfg["frontier_bootstrap_draws"]
    compute_draws=args.compute_draws or cfg["compute_bootstrap_draws"]; c3_draws=args.c3_draws or cfg["c3_bootstrap_draws"]
    root=Path(cfg["c_root"]); p3_root=Path(cfg["problem3_root"]); out=Path(cfg["output_root"]); report_root=Path(cfg["report_root"])
    if out.resolve()==Path(cfg["v1_root"]).resolve() or report_root.resolve()==Path(cfg["v1_report_root"]).resolve(): raise RuntimeError("V2输出不得指向V1")
    out.mkdir(parents=True,exist_ok=True); report_root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(cfg["seed"]); v1_before=snapshot_tree([cfg["v1_root"],cfg["v1_report_root"]]); atomic_csv(out/"repair_baseline_v1.csv",v1_before)
    previous_signature_path=out/"reproducibility_signature.json"
    previous_signature=json.loads(previous_signature_path.read_text(encoding="utf-8")) if previous_signature_path.exists() else None
    p3_ready,p3_hashes,p3_mismatches,p3_bridge=check_p3_interface(p3_root)
    if not p3_ready: raise RuntimeError(f"问题三冻结接口失败: {p3_mismatches}")
    frames,file_audit=load_inputs(root); contract=audit_contract(frames,root); cutoff=pd.to_datetime(frames["C2"]["Submission Date"],errors="coerce").max()
    dedup,duplicate_ledger=resolve_duplicates(frames["C2"]); matched,match_ledger=match_c2_c4(dedup,frames["C4"],cfg["match_parameter_relative_tolerance"])
    matched=enrich_matched(matched,match_ledger,frames["C4"]); panel=build_panel_v2(matched,cutoff,cfg["epsilon_score"],cfg["strict_license_allowlist"]); frontier=panel[panel["frontier_eligible"]].copy()
    if len(frontier)<20: raise RuntimeError("确认性前沿样本不足20")
    atomic_csv(out/"duplicate_resolution.csv",duplicate_ledger); atomic_csv(out/"model_match_ledger.csv",match_ledger); atomic_csv(out/"open_model_panel.csv",panel); atomic_csv(out/"frontier_training_panel.csv",frontier)

    c3_data,c3_draws_frame,c3_summary=c3_historical_analysis(frames["C3"],cfg["epsilon_score"],c3_draws,rng)
    atomic_csv(out/"c3_historical_sensitivity.csv",c3_draws_frame); atomic_json(out/"c3_historical_summary.json",c3_summary)
    c8_agg,c8_sub,c8_corrupt,c8_summary=aggregate_c8_v2(root,frames["C2"])
    atomic_csv(out/"task_level_aggregates.csv",c8_agg); atomic_csv(out/"c8_subtask_normalization.csv.gz",c8_sub,compression="gzip"); atomic_csv(out/"c8_corrupt_log.csv",c8_corrupt)

    c6=frames["C6"].copy(); models,bridge_cv,bridge_pred,bridge_meta=fit_bridges(c6,cfg["bridge_medium_weight"],cfg["epsilon_score"])
    atomic_csv(out/"bridge_cv_results.csv",bridge_cv); atomic_csv(out/"bridge_cv_predictions.csv.gz",bridge_pred,compression="gzip")
    sens=[]
    for mw in cfg["bridge_medium_weight_sensitivity"]:
        _,cv,_,meta=fit_bridges(c6,mw,cfg["epsilon_score"])
        for target in meta:
            r=cv[(cv["target_name"]==target)&cv["selected"]].iloc[0]
            sens.append({"medium_weight":mw,"target":target,"selected_model":meta[target]["selected_model"],"cv_mae":r.cv_mae,"bridge_weak":meta[target]["bridge_weak"]})
    atomic_csv(out/"bridge_weight_sensitivity.csv",pd.DataFrame(sens)); atomic_json(out/"loss_benchmark_mapping.json",{"epsilon":cfg["epsilon_score"],"medium_weight":cfg["bridge_medium_weight"],"targets":bridge_meta})
    p3_predictions,p3_unc,p3_raw,p3_budget=map_problem3_v2(c6,p3_root,models,bridge_meta,rng,bridge_draws,cfg["bridge_medium_weight"],cfg["epsilon_score"])
    atomic_csv(out/"loss_benchmark_predictions.csv",p3_predictions); atomic_csv(out/"loss_benchmark_uncertainty.csv",p3_unc); atomic_csv(out/"loss_benchmark_bootstrap_draws.csv.gz",p3_raw,compression="gzip")

    cv,chosen_lambda,chosen_rho,cv_threshold=rolling_cv_v2(frontier,cfg["dynamic_lambda_grid"],cfg["state_rho_grid"],cfg["main_frontier_quantile"])
    atomic_csv(out/"backtest_results.csv",cv)
    tau_models={tau:fit_dynamic_v2(frontier,tau,chosen_lambda,chosen_rho) for tau in cfg["frontier_quantiles"]}; main_model=tau_models[cfg["main_frontier_quantile"]]
    dec=decompose_v2(main_model,frontier,cfg["epsilon_score"]); nd=nd_diagnostics(frontier); atomic_csv(out/"frontier_model_comparison.csv",nd)
    boot_frontier=bootstrap_frontier_joint(frontier,cfg["frontier_quantiles"],chosen_lambda,chosen_rho,frontier_draws,cfg["month_block_length"],rng,cfg["epsilon_score"])
    atomic_csv(out/"frontier_bootstrap_draws.csv.gz",boot_frontier,compression="gzip"); decomp_unc=summarize_decomposition(boot_frontier)
    valid_main=boot_frontier[(boot_frontier["tau"]==.9)&boot_frontier["fit_success"]]
    decomp_rows=[{"estimate":"central",**dec}]
    for q,label in [(.025,"p025"),(.1,"p10"),(.5,"median"),(.9,"p90"),(.975,"p975")]:
        decomp_rows.append({"estimate":label,**{c:valid_main[c].quantile(q) for c in ["scale_logit","tech_logit","total_logit","closure_logit","scale_score","tech_score","total_score","closure_score"]}})
    atomic_csv(out/"contribution_decomposition.csv",pd.DataFrame(decomp_rows))
    local,global_s,state_slope=main_model.slopes(); atomic_json(out/"dynamic_frontier_parameters.json",{"tau":main_model.tau,"lambda":chosen_lambda,"rho":chosen_rho,"beta0":main_model.beta0,"beta_x_standardized_log10_compute":main_model.beta_x,"x_mean":main_model.x_mean,"x_std":main_model.x_std,"months":[str(x) for x in main_model.months],"states":main_model.states.tolist(),"state_slope_local":local,"state_slope_global":global_s,"state_slope_selected":state_slope,"objective":main_model.objective,"success":main_model.success})

    comp_panel,family_month,monthly_compute=prepare_compute_panel(frames["C4"],cutoff,cfg["compute_growth_lookback_months"],cfg["confidence_multipliers"])
    growth=fit_growth(monthly_compute,cutoff); growth_boot=bootstrap_compute_growth(family_month,cutoff,cfg["compute_growth_lookback_months"],compute_draws,rng)
    cutoff_key=str(cutoff.to_period("M")); cutoff_match=monthly_compute[monthly_compute["month"].eq(cutoff_key)]
    growth["raw_cutoff_month_log10"]=float(cutoff_match["log10_compute_q90"].iloc[-1]) if len(cutoff_match) else float(monthly_compute["log10_compute_q90"].iloc[-1])
    growth["last3_median_log10"]=float(monthly_compute.tail(3)["log10_compute_q90"].median())
    atomic_csv(out/"compute_growth_monthly.csv",monthly_compute); atomic_csv(out/"compute_growth_bootstrap.csv.gz",growth_boot,compression="gzip")
    factors=sorted(set(cfg["compute_slowdown_factors"]+([0.0] if growth["annual_log10_slope_low"]<=0<=growth["annual_log10_slope_high"] else [])),reverse=True)
    scenario_table=pd.DataFrame([{"scenario_factor":f,"annual_log10_growth":growth["annual_log10_slope"]*f,"annual_multiplicative_growth":10**(growth["annual_log10_slope"]*f),"trend_endpoint_log10":growth["endpoint_log10_compute"],"raw_cutoff_month_log10":growth["raw_cutoff_month_log10"],"last3_median_log10":growth["last3_median_log10"]} for f in factors]); atomic_csv(out/"compute_growth_scenarios.csv",scenario_table); atomic_json(out/"compute_growth_summary.json",growth)
    forecast,forecast_draws=forecast_joint(tau_models,boot_frontier,growth_boot,cutoff,growth,factors,cfg["forecast_horizons_months"],cfg["epsilon_score"])
    atomic_csv(out/"frontier_forecast.csv",forecast); atomic_csv(out/"frontier_forecast_draws.csv.gz",forecast_draws,compression="gzip")

    sensitivity=[]
    for eps in cfg["epsilon_sensitivity"]:
        fr=frontier.copy(); fr["ability_logit"]=score_to_link(fr["Average ⬆️"],eps); mod=fit_dynamic_v2(fr,.9,chosen_lambda,chosen_rho); dd=decompose_v2(mod,fr,eps)
        sensitivity.append({"sensitivity":"epsilon","level":eps,"rows":len(fr),"beta_x":mod.beta_x,**{k:dd[k] for k in ["scale_score","tech_score","total_score"]}})
    strict=frontier[frontier["strict_unrestricted"]].copy()
    if len(strict)>=12 and strict["submission_date"].dt.to_period("M").nunique()>=3:
        mod=fit_dynamic_v2(strict,.9,chosen_lambda,chosen_rho); dd=decompose_v2(mod,strict,cfg["epsilon_score"]); sensitivity.append({"sensitivity":"strict_unrestricted","level":"strict","rows":len(strict),"beta_x":mod.beta_x,**{k:dd[k] for k in ["scale_score","tech_score","total_score"]}})
    else: sensitivity.append({"sensitivity":"strict_unrestricted","level":"INSUFFICIENT_SUPPORT","rows":len(strict)})
    evidence=frontier[frontier["compute_evidence_v2"].isin(["reported","operation_counting"])]
    if len(evidence)>=12 and evidence["submission_date"].dt.to_period("M").nunique()>=3:
        mod=fit_dynamic_v2(evidence,.9,chosen_lambda,chosen_rho); dd=decompose_v2(mod,evidence,cfg["epsilon_score"]); sensitivity.append({"sensitivity":"compute_evidence","level":"reported_operation","rows":len(evidence),"beta_x":mod.beta_x,**{k:dd[k] for k in ["scale_score","tech_score","total_score"]}})
    else: sensitivity.append({"sensitivity":"compute_evidence","level":"INSUFFICIENT_SUPPORT","rows":len(evidence)})
    publication=frontier.dropna(subset=["c4_publication_date"]).copy(); publication["submission_date"]=pd.to_datetime(publication["c4_publication_date"])
    if len(publication)>=12 and publication["submission_date"].dt.to_period("M").nunique()>=3:
        mod=fit_dynamic_v2(publication,.9,chosen_lambda,chosen_rho); dd=decompose_v2(mod,publication,cfg["epsilon_score"]); sensitivity.append({"sensitivity":"time_axis","level":"publication_date","rows":len(publication),"beta_x":mod.beta_x,**{k:dd[k] for k in ["scale_score","tech_score","total_score"]}})
    else: sensitivity.append({"sensitivity":"time_axis","level":"INSUFFICIENT_SUPPORT","rows":len(publication)})
    balanced=frontier.copy(); z=[]
    for col in ["IFEval","BBH","MATH Lvl 5","GPQA","MUSR","MMLU-PRO"]:
        rank=balanced[col].rank(method="average").to_numpy(); u=(rank-.5)/len(balanced); z.append(scipy.stats.norm.ppf(u))
    balanced["balanced_score"]=100*scipy.stats.norm.cdf(np.mean(np.column_stack(z),axis=1)); balanced["ability_logit"]=score_to_link(balanced["balanced_score"],cfg["epsilon_score"])
    mod=fit_dynamic_v2(balanced,.9,chosen_lambda,chosen_rho); dd=decompose_v2(mod,balanced,cfg["epsilon_score"]); sensitivity.append({"sensitivity":"ability_construct","level":"six_domain_rank_normal","rows":len(balanced),"beta_x":mod.beta_x,**{k:dd[k] for k in ["scale_score","tech_score","total_score"]}})
    for window in cfg["compute_growth_window_sensitivity"]:
        _,_,month_w=prepare_compute_panel(frames["C4"],cutoff,window,cfg["confidence_multipliers"]); g_w=fit_growth(month_w,cutoff)
        sensitivity.append({"sensitivity":"compute_window","level":window,"rows":len(month_w),"annual_log10_growth":g_w["annual_log10_slope"],"endpoint_log10_compute":g_w["endpoint_log10_compute"]})
    for block_length in [1,3]:
        brng=np.random.default_rng(cfg["seed"]+1000+block_length)
        bsen=bootstrap_frontier_joint(frontier,cfg["frontier_quantiles"],chosen_lambda,chosen_rho,cfg["sensitivity_bootstrap_draws"],block_length,brng,cfg["epsilon_score"])
        su=summarize_decomposition(bsen).set_index("quantity")
        sensitivity.append({"sensitivity":"month_block_length","level":block_length,"rows":int(((bsen.tau==.9)&bsen.fit_success).sum()),"beta_x":su.loc["beta_x","median"],"scale_score":su.loc["scale_score","median"],"tech_score":su.loc["tech_score","median"],"total_score":su.loc["total_score","median"],"scale_p025":su.loc["scale_score","p025"],"scale_p975":su.loc["scale_score","p975"],"tech_p025":su.loc["tech_score","p025"],"tech_p975":su.loc["tech_score","p975"]})
    atomic_csv(out/"frontier_sensitivity.csv",pd.DataFrame(sensitivity))

    uncertainty_budget=pd.concat([p3_budget,pd.DataFrame([{"estimand":"historical_contribution","scenario_id":"history","target":r.quantity,"component":"family_month_frontier","p10":r.p10,"p90":r.p90,"p025":r.p025,"p975":r.p975,"width80":r.p90-r.p10,"width95":r.p975-r.p025,"draws":frontier_draws,"interpretation":"small-sample empirical stability"} for _,r in decomp_unc.iterrows()])],ignore_index=True)
    for _,r in forecast.iterrows(): uncertainty_budget.loc[len(uncertainty_budget)]={"estimand":"future_frontier","scenario_id":f"{r.scenario_factor}_{int(r.horizon_months)}","target":f"q{r.tau}","component":"joint_frontier_compute","p10":r.p10,"p90":r.p90,"p025":r.p025,"p975":r.p975,"width80":r.p90-r.p10,"width95":r.p975-r.p025,"draws":r.bootstrap_draw_count,"interpretation":"includes state, endpoint, growth and compute measurement"}
    atomic_csv(out/"uncertainty_budget.csv",uncertainty_budget)

    chosen=cv[(cv["model"]=="M-C")&(cv["lambda"]==chosen_lambda)&(cv["rho"]==chosen_rho)].groupby(["fold_cutoff","horizon_months"])["pinball"].mean().rename("mc").reset_index()
    bases=cv[cv["model"].isin(["M-time","M-scale"])].groupby("model")["pinball"].mean(); best_base=bases.idxmin(); base=cv[cv["model"]==best_base][["fold_cutoff","horizon_months","pinball"]].rename(columns={"pinball":"baseline"}); paired=chosen.merge(base,on=["fold_cutoff","horizon_months"]); diff=paired.mc-paired.baseline
    diff_mean=float(diff.mean()); diff_se=float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else np.inf; cv_gate=bool(diff_mean<=diff_se)
    avg=bridge_cv[bridge_cv["target_name"].eq("Average")]; avg_sel=avg[avg["selected"]].iloc[0]; avg_const=avg[avg["model"].eq("constant")].iloc[0]
    link_err=float(np.max(np.abs(forecast["forecast_score"]-link_to_score(forecast["forecast_link"],cfg["epsilon_score"]))))
    ordered=forecast.sort_values("tau").groupby(["scenario_factor","horizon_months"])["forecast_score"].apply(lambda x:bool(np.all(np.diff(x)>=-1e-12))).all()
    tau_keys=forecast_draws.groupby(["tau","scenario_factor","horizon_months"]).size(); tau_intervals=bool(len(tau_keys)==len(cfg["frontier_quantiles"])*len(factors)*len(cfg["forecast_horizons_months"]))
    known_types=set(["pretrained","posttrained","merge","excluded_other"]); post_rows=int((panel["type_group"].eq("posttrained")&panel["log10_compute"].notna()).sum())
    c8_bbh_ready=(c8_summary["bbh_unique_comparable"]>0 and
                  c8_summary["official_exact_bbh"]/c8_summary["bbh_unique_comparable"]>=.95)
    c8_math_ready=(c8_summary["math_unique_comparable"]>0 and
                   c8_summary["official_exact_math"]/c8_summary["math_unique_comparable"]>=.95)
    current_signature={"lambda":chosen_lambda,"rho":chosen_rho,"beta0":main_model.beta0,"beta_x":main_model.beta_x,"scale_score":dec["scale_score"],"tech_score":dec["tech_score"],"total_score":dec["total_score"],"compute_endpoint":growth["endpoint_log10_compute"],"compute_growth":growth["annual_log10_slope"],"forecast_scores":forecast.sort_values(["tau","scenario_factor","horizon_months"])["forecast_score"].tolist(),"average_bridge_model":str(avg_sel.model),"average_bridge_mae":float(avg_sel.cv_mae)}
    numeric_keys=["lambda","rho","beta0","beta_x","scale_score","tech_score","total_score","compute_endpoint","compute_growth","average_bridge_mae"]
    if previous_signature is None:
        reproducibility={"previous_available":False,"max_abs_difference":float("inf"),"categorical_equal":False,"passed":False}
    else:
        diffs=[abs(float(current_signature[k])-float(previous_signature[k])) for k in numeric_keys]
        diffs.extend(abs(float(a)-float(b)) for a,b in zip(current_signature["forecast_scores"],previous_signature["forecast_scores"]))
        categorical=current_signature["average_bridge_model"]==previous_signature["average_bridge_model"]
        reproducibility={"previous_available":True,"max_abs_difference":max(diffs),"categorical_equal":categorical,"passed":bool(max(diffs)<=1e-10 and categorical)}
    atomic_json(out/"reproducibility_check.json",reproducibility); atomic_json(previous_signature_path,current_signature)
    ctx={"p3_ready":p3_ready,"p3_mismatches":p3_mismatches,"avg_error":contract["average_max_abs_error"],"c3_rows":c3_summary["rows"],"c3_generated":len(c3_draws_frame)>0,
        "c4_contract":all(c in frames["C4"] for c in ["Training compute (FLOP)","Training dataset size (total)","Open model weights?","Training compute estimation method","Confidence","Training compute lower bound","Training compute upper bound"]),
        "strict_invalid":int((panel["strict_unrestricted"]&~panel["Hub License"].fillna("").str.lower().isin(cfg["strict_license_allowlist"])).sum()),"unclassified":int((~panel["type_group"].isin(known_types)).sum()),
        "c8_math_ready":c8_math_ready,"c8_bbh_ready":c8_bbh_ready,"c8_summary":c8_summary,"c5_subset":contract["c5_exact_subset_c6"],"bridge_violations":sum(v["monotonicity_violations"] for v in bridge_meta.values()),
        "bridge_folds":int(avg_sel.folds),"bridge_metrics":all(c in bridge_cv for c in ["cv_mae","cv_rmse","spearman","coverage80","coverage95","mean_width80","mean_width95"]),"average_bridge_ok":bool(avg_sel.cv_mae<avg_const.cv_mae),
        "average_bridge_comparison":{"selected":avg_sel.model,"selected_mae":avg_sel.cv_mae,"constant_mae":avg_const.cv_mae},"scenario_count":int(p3_predictions.scenario_index.nunique()),
        "bridge_intervals_ordered":bool(((p3_unc["p025"]<=p3_unc["p10"])&(p3_unc["p10"]<=p3_unc["median"])&(p3_unc["median"]<=p3_unc["p90"])&(p3_unc["p90"]<=p3_unc["p975"])).all()),"bridge_modes":int(p3_unc.uncertainty_mode.nunique()),
        "fit_success":main_model.success,"frontier_boot_rate":float(valid_main.draw_id.nunique()/frontier_draws),"beta_x":main_model.beta_x,"cv_gate":cv_gate,"cv_comparison":{"best_baseline":best_base,"mean_difference":diff_mean,"paired_se":diff_se,"pairs":len(diff)},
        "selected_from_grid":chosen_lambda in cfg["dynamic_lambda_grid"] and chosen_rho in cfg["state_rho_grid"],"lambda":chosen_lambda,"rho":chosen_rho,"nd_complete":bool(nd.fit_success.all()),"closure":float(max(dec["closure_logit"],dec["closure_score"],valid_main[["closure_logit","closure_score"]].max().max())),
        "endpoint_from_trend":np.isfinite(growth["endpoint_log10_compute"]),"endpoint":growth["endpoint_log10_compute"],"growth_non_degenerate":float(growth_boot[growth_boot.fit_success].annual_log10_slope.std())>0,"growth_std":float(growth_boot[growth_boot.fit_success].annual_log10_slope.std()),
        "quantiles_ordered":bool(ordered),"link_error":link_err,"tau_intervals":tau_intervals,"scenario_factors":forecast.scenario_factor.nunique(),"extrapolation_flags":bool(((forecast.horizon_months.eq(12)&forecast.extrapolation_flag.eq("extrapolation"))|(forecast.horizon_months.eq(24)&forecast.extrapolation_flag.eq("long_horizon_extrapolation"))).all()),"scores_in_bounds":bool(forecast.forecast_score.between(0,100).all()),"reproducibility_pass":reproducibility["passed"],"reproducibility":reproducibility}
    acceptance=make_acceptance(ctx); atomic_json(out/"problem4_acceptance_report.json",acceptance); atomic_csv(out/"problem4_acceptance_report.csv",pd.DataFrame(acceptance["rules"]))
    data_audit={"files":file_audit,"contract":contract,"c8":c8_summary,"c3":c3_summary,"p3_hashes":p3_hashes,"p3_hash_mismatches":p3_mismatches,"cutoff":str(cutoff.date()),"deduplicated_rows":len(dedup),"accepted_matches":int(match_ledger.review_status.str.startswith("accepted").sum()),"frontier_rows":len(frontier),"frontier_months":int(frontier.submission_date.dt.to_period("M").nunique()),"frontier_families":int(frontier.family.nunique()),"posttrained_rows":post_rows}
    atomic_json(out/"data_audit.json",data_audit)
    compare=pd.DataFrame([{"metric":"scale_score_central","v1":pd.read_csv(Path(cfg["v1_root"])/"contribution_decomposition.csv").iloc[0].scale_score,"v2":dec["scale_score"]},{"metric":"tech_score_central","v1":pd.read_csv(Path(cfg["v1_root"])/"contribution_decomposition.csv").iloc[0].tech_score,"v2":dec["tech_score"]},{"metric":"bbh_exact","v1":json.loads((Path(cfg["v1_root"])/"data_audit.json").read_text())["c8"]["official_exact_bbh"],"v2":c8_summary["official_exact_bbh"]},{"metric":"PROBLEM4_READY","v1":json.loads((Path(cfg["v1_root"])/"problem4_acceptance_report.json").read_text())["interface_state"]["PROBLEM4_READY"],"v2":acceptance["interface_state"]["PROBLEM4_READY"]}]); compare["change"]=compare.apply(lambda r:float(r.v2)-float(r.v1) if isinstance(r.v1,(int,float,np.integer,np.floating,bool)) else np.nan,axis=1); atomic_csv(out/"repair_comparison_v1_v2.csv",compare)
    values={"acceptance":acceptance,"decomposition":dec,"decomp_uncertainty":decomp_unc,"forecast":forecast,"dedup_rows":len(dedup),"accepted_matches":data_audit["accepted_matches"],"frontier_rows":len(frontier),"frontier_months":data_audit["frontier_months"],"frontier_families":data_audit["frontier_families"],"c3":c3_summary,"c8":c8_summary,"posttrained_rows":post_rows,"posttrained_status":"INSUFFICIENT_SUPPORT" if post_rows<12 else "DESCRIPTIVE_ONLY","average_model":avg_sel.model,"average_mae":avg_sel.cv_mae,"average_const_mae":avg_const.cv_mae,"weak_targets":[k for k,v in bridge_meta.items() if v["bridge_weak"]],"lambda":chosen_lambda,"rho":chosen_rho,"beta_x":main_model.beta_x,"growth":growth,"link_roundtrip_error":float(np.max(abs(link_to_score(score_to_link(np.array([0.,5.,50.,99.,100.]),cfg["epsilon_score"]),cfg["epsilon_score"])-np.array([0.,5.,50.,99.,100.])))),"forecast_link_error":link_err,"central_rearranged":int(forecast.quantile_rearranged.sum()),"draw_rearranged":int(forecast_draws.quantile_rearranged.sum()),"reproducibility":reproducibility}
    report_path=report_root/"RESULTS_REPORT.md"; render_report(report_path,values)
    v1_after=snapshot_tree([cfg["v1_root"],cfg["v1_report_root"]]); unchanged=v1_before.equals(v1_after)
    if not unchanged: raise RuntimeError("V1文件在V2运行期间发生变化")
    atomic_json(out/"run_log.json",{"started_unix":started,"finished_unix":time.time(),"elapsed_seconds":time.time()-started,"compute_backend":"CPU (SciPy optimization; GPU not beneficial for small tabular fits)","draws":{"bridge":bridge_draws,"frontier":frontier_draws,"compute":compute_draws,"c3":c3_draws},"v1_unchanged":unchanged,"acceptance":acceptance["summary"],"interface_state":acceptance["interface_state"]})
    code_files=[Path(__file__),HERE/"src/problem4/v2.py",HERE/"tests/test_problem4_v2.py",Path(args.config)]
    outputs={}
    for path in sorted(list(out.glob("*"))+[report_path]):
        if path.is_file() and path.name!="problem4_manifest.json": outputs[str(path)]=sha256(path)
    manifest={"project":"华为杯F题问题四","version":"problem4_v2","python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"config":cfg,"inputs":{**{k:v["sha256"] for k,v in file_audit.items()},**{f"P3:{k}":v for k,v in p3_hashes.items()}},"v1_baseline":{r.path:r.sha256 for _,r in v1_before.iterrows()},"code":{str(p):sha256(p) for p in code_files},"outputs":outputs,"skip_plots":True}
    atomic_json(out/"problem4_manifest.json",manifest)
    print(json.dumps({"overall":acceptance["overall"],"interface_state":acceptance["interface_state"],"summary":acceptance["summary"],"frontier_rows":len(frontier),"bbh_exact":c8_summary["official_exact_bbh"],"math_exact":c8_summary["official_exact_math"],"elapsed_seconds":time.time()-started},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
