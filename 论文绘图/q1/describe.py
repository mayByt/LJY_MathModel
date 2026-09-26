"""Build explanations from each figure's recorded statistics, without redrawing."""
import json
from pathlib import Path
import config as C
from .data import DOMAIN,FIELDS

def quantitative(n,s):
 if n==2:return '去重记录共'+str(sum(x['n'] for x in s.values()))+'条；各域词数中位数：'+'、'.join(f"{DOMAIN[k]} {v['quantiles'][4]:,.0f}" for k,v in s.items())+'。'
 if n==5:return f"七域各 {s['n_per_domain']} 条，共 {s['n_balanced']} 条；包含 {len(s['fields'])} 项指标。相关矩阵以固定随机种子2026构建。"
 if n==6:
  r=max(s['rows'],key=lambda r:r['total_weight']);return f"最大单项综合权重为{FIELDS[r['field']]} {r['total_weight']:.2%}；五个块总权重均为20%。"
 if n==7:return '各域Q中位数：'+'、'.join(f"{DOMAIN[k]} {v['quartiles'][1]:.3f}" for k,v in s['domains'].items())+'。'
 if n==8:return '；'.join(f"{DOMAIN[k]}抽样/扩展非重合样本数 {v['n_a1']:,}/{v['n_remainder']:,}，Wasserstein距离 {v['wasserstein']:.6f}" for k,v in s.items())+'。'
 if n==9:
  hi=max(s['values'],key=lambda r:r['q']);lo=min(s['values'],key=lambda r:r['q']);return f"{DOMAIN[hi['domain']]}评分最高（{hi['q']:.6f}），{DOMAIN[lo['domain']]}最低（{lo['q']:.6f}）；其余估计及区间逐域保存在同名JSON。"
 if n==10:return f"共有 {s['n']:,} 条样本，其中 {s['n_conflict']:,} 条触发冲突（{s['conflict_rate']:.4%}）；图面显示其中 {s['displayed_conflict']} 条空心点。"
 if n==11:return f"总体冲突记录 {s['n_all_conflict']:,} 条，本图按规则选取 {len(s['selected'])} 条；全体有符号修正量中位数 {s['delta_quantiles']['0.5']:.6f}。"
 if n==13:return f"17个域中 {s['n_inferred']} 个使用推断回退；这些域的q均为 {next(x['q'] for x in s['rows'] if x['mapping_type']=='inferred'):.6f}，不得据此强排序。"
 if n==15:return f"组成矩阵为 {s['shape'][0]}×{s['shape'][1]}，精确零份额占 {s['zero_fraction']:.4%}。"
 if n==16:return f"非零领域数中位数为 {s['active_quantiles'][2]:g}，有效领域数中位数为 {s['effective_quantiles'][2]:.4f}；两者描述不同的集中度属性。"
 if n==17:
  q=next(r for r in s['models'] if r['model']=='quadratic_ridge');t=next(r for r in s['models'] if r['model']=='extra_trees');return f"二阶Ridge NRMSE={q['normalized_rmse']:.6f}；Extra Trees Spearman={t['median_spearman']:.6f}。"
 if n==18:return '；'.join(f"{name}最优α={v['best']['alpha']:.6g}，NRMSE={v['best']['normalized_rmse']:.6f}" for name,v in s.items())+'。'
 if n==19:return '；'.join(f"{k} n={s[k]['n']}、Spearman={s[k]['spearman']:.6f}" for k in ['test_1m','test_60m','test_1B'])+'。这些数值仅对应通用网页验证域，不是13域中位数。'
 if n==20:return f"n={s['n']}；原始/校准预测平均偏差分别为 {s['predicted']['bias']:.6f}/{s['predicted_calibrated']['bias']:.6f}，RMSE分别为 {s['predicted']['rmse']:.6f}/{s['predicted_calibrated']['rmse']:.6f}。"
 if n==22:return f"展示 {len(s['edges'])} 个域对、{len(s['node_order'])} 个领域，其中 {s.get('n_stable',len(s['edges']))} 个域对的95%区间不跨零；系数范围 {min(e['coefficient'] for e in s['edges']):.4f} 至 {max(e['coefficient'] for e in s['edges']):.4f}。"
 if n==23:return f"17域中有 {sum(x['benefit_score']>0 for x in s['rows'])} 个点估计为正；有 {sum(x['lo']>0 for x in s['rows'])} 个95%区间完全高于0。区间使用 {s['bootstrap_reps']} 次训练行bootstrap。"
 if n==25:return f"三域合计份额固定为 {s['their_fixed_total']:.4%}；{s['grid_count']:,} 个网格点中 {s['feasible_count']:,} 个可行，参考候选J={s['reference_objective']:.6f}。"
 if n==27:return f"成功bootstrap {s['n_success']} 次；{sum(x['ci_low']==0 for x in s['rows'])} 个分量的95%下界为0，参考候选前三域份额合计 {sum(x['p_star'] for x in s['rows'][:3]):.4%}。"
 if n==28:return f"30个起点按目标值保留5位小数组成 {len(s['clusters'])} 个簇，最小/最大目标值分别为 {min(v['objective_round5'] for v in s['clusters']):.5f}/{max(v['objective_round5'] for v in s['clusters']):.5f}。"
 return '统计见同名JSON。'

def write():
 out=C.OUTPUT/'q1';readings=json.loads((C.CACHE/'q1_reading.json').read_text());lines=['# 问题一论文插图说明','', '本问共20张独立图，每张提供PNG（300 dpi）、嵌入字体PDF与同名统计JSON。正文图中文字使用宋体，英文及数字使用Times New Roman；不将解释段落放进绘图区。','', '正式结果仅从 `问题一/results_p12_final_v2` 读取。质量明细按原随机种子和500次bootstrap在 `cache/q1_quality` 恢复；七域 q/CI 与冻结汇总最大差为1.233e-7，在显式1e-6重建容差内。原NumPy/Pandas为1.26.4/2.2.2，当前为2.5.3/3.0.6；原冻结七域汇总与结果文件均未修改。','', '单张重画：`python -m q1.run --id Q1-09`（在论文绘图目录使用 `.venv/bin/python`）。重画会产生新PNG哈希，必须重新送专门审阅者；不要用原哈希的通过记录覆盖新版。','']
 for p in sorted(out.glob('Q1-*.json')):
  m=json.loads(p.read_text());n=int(m['id'].split('-')[1]);sz=m['size_mm'];review=C.PLOT/'reviews'/f"{m['id']}.json";rv=json.loads(review.read_text()) if review.exists() else {};state='待独立目视审阅'
  if rv.get('png_sha256')==m['png_sha256']:state='独立目视审阅通过' if rv.get('status')=='pass' else '审阅要求修改'
  lines += [f"## {m['name']}",'',f"- **文件与尺寸**：`output/q1/{m['name']}.png`、`.pdf`；设计尺寸 {sz[0]:g}×{sz[1]:g} mm（按内容紧凑导出）。" + ('此22指标矩阵建议单独通页放置；不要强缩到双栏宽度。' if n==5 else ''),f"- **来源**：{'；'.join(m['sources'])}。",f"- **图型、编码与图注**：{m['caption']}",f"- **数据给出的结果**：{quantitative(n,m['stats'])}",f"- **解读与可写入正文的说明**：{readings[m['name']]}",f"- **审阅**：{state}；以`reviews/{m['id']}.json`是否对应当前PNG SHA256为准。",'']
 (C.PLOT/'图表说明_Q1.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':write()
