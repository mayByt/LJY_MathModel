"""One-figure-at-a-time paper exports. Usage: python -m q1.run Q1-09."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats as ss
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage,leaves_list
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch,Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm,TwoSlopeNorm,Normalize
from matplotlib.ticker import PercentFormatter,MaxNLocator,LogLocator,NullFormatter
from matplotlib.tri import Triangulation
from common import plot as P
import config as C
from . import data as D
FIGS={}
def register(n):
 def decorate(f):FIGS[n]=f;return f
 return decorate

def source(n):return D.R/n

def finish(fig,n,title,stats,caption,sources,reading):
 name=f'Q1-{n:02d}_{title}'
 P.save(fig,'q1',name,stats=stats,caption=caption,sources=sources)
 doc=C.PLOT/'图表说明_Q1.md'
 records=[]
 for p in sorted((C.OUTPUT/'q1').glob('Q1-*.json')):
  m=json.loads(p.read_text());records.append(m)
 paragraphs=['# 问题一论文插图说明','所有图以真实附件或冻结结果为来源；图内中文宋体、英文及数字 Times New Roman。每图独立 PNG（300 dpi）与嵌入字体 PDF。数字由生成脚本现算，逐图统计及审阅状态保存在同名 JSON 与审阅目录。','正式结果保持只读；质量明细按原配置恢复到独立 cache/q1_quality，并通过七域 q/ci 的 1e-6 重建一致性校验（最大差1.233e-7；原NumPy1.26.4/Pandas2.2.2，本机2.5.3/3.0.6）；正式七域汇总沿冻结值。']
 readingfile=C.CACHE/'q1_reading.json'
 readings=json.loads(readingfile.read_text()) if readingfile.exists() else {}
 readings[name]=reading;readingfile.write_text(json.dumps(readings,ensure_ascii=False,indent=2))
 for m in records:
  s=m['size_mm'];paragraphs+=['',f"## {m['name']}",f"- 文件：`output/q1/{m['name']}.png` / `.pdf`；设计尺寸 {s[0]} × {s[1]} mm。",f"- 数据：{'；'.join(m['sources'])}。",f"- 图注：{m['caption']}",f"- 读法与正文：{readings.get(m['name'],'见图注及同名 JSON。')}",'- 数值与统计口径：同名 JSON 的 `stats` 字段，全部由读取或派生计算写入。','- 审阅：以 review_queue 与独立审阅记录中的当前 PNG SHA256 为准；自动检查不替代逐张目视。']
 doc.write_text('\n'.join(paragraphs)+'\n')
 from .describe import write
 write()

def ydomains(ax,domains,label='领域'):
 ax.set_yticks(range(len(domains)),[D.DOMAIN.get(d,d) for d in domains]);ax.set_ylabel(label);ax.set_ylim(len(domains)-.5,-.5)
def top(ax,n=3):P.legend_top(ax,ncol=n,fontsize=8)
def zerox(ax):ax.axvline(0,c='#b7babc',lw=.7,ls='--',zorder=0)

@register(9)
def domain_quality():
 d=D.csv('quality_domain.csv').query("mapping_type=='quality_anchor'").sort_values('q',ascending=False)
 fig,ax,p=P.figure('q1',(145,93));y=np.arange(len(d))
 ax.errorbar(d.q,y,xerr=[d.q-d.ci_low,d.ci_high-d.q],fmt='o',color=p.line['forest_green'],mfc=p.fill['forest_green'],mec=p.line['forest_green'],ms=5,capsize=2.5,lw=1.2,label='领域估计与95%区间')
 ydomains(ax,d.domain);ax.set_xlabel('质量评分 Q');ax.set_xlim(.40,.652);ax.set_xticks(np.arange(.4,.651,.05));ax.xaxis.set_major_formatter('{x:.2f}')
 for i,row in enumerate(d.itertuples()):ax.text(.646,i,f'{row.q:.3f}',ha='right',va='center',fontsize=8,color=p.ink)
 top(ax,1)
 finish(fig,9,'七域质量区间',{'values':d.to_dict('records'),'interval':'domain-row bootstrap, fixed scoring transform and weights'},'七个原生质量域的稳健质量估计与95%域内bootstrap区间。仅使用 quality_anchor 行；该区间以固定评分规则为条件。',[source('quality_domain.csv')],'C4与通用网页的当前口径质量较高，代码域较低；书籍区间更宽。该评分不是人工绝对真值，不能等同于训练收益。')

@register(13)
def mapping_intervals():
 d=D.csv('domain_mapping.csv').sort_values(['mapping_type','q'],ascending=[True,False]);fig,ax,p=P.figure('q1',(160,131));ys=np.arange(len(d))
 for i,row in enumerate(d.itertuples()):
  inferred=row.mapping_type=='inferred';c=p.line['forest_green'] if not inferred else p.line['indigo']
  if inferred:ax.plot([row.sensitivity_ci_low,row.sensitivity_ci_high],[i,i],color=p.fill['light_green'],lw=6,solid_capstyle='butt',zorder=1)
  ax.plot([row.ci_low,row.ci_high],[i,i],c=c,lw=1.5,zorder=3)
  ax.scatter(row.q,i,s=18,marker='o' if not inferred else 'D',facecolor='white' if inferred else p.fill['forest_green'],edgecolor=c,lw=.8,zorder=4)
 ydomains(ax,d.domain);ax.set_xlabel('映射质量评分 Q');ax.set_xlim(.39,.695);ax.set_xticks(np.arange(.4,.701,.05));ax.xaxis.set_major_formatter('{x:.2f}')
 h=[Line2D([],[],marker='o',color=p.line['forest_green'],mfc=p.fill['forest_green'],lw=1,label='直接/近直接：95%区间'),Line2D([],[],marker='D',color=p.line['indigo'],mfc='white',lw=1,label='推断：条件95%区间'),Line2D([],[],color=p.fill['light_green'],lw=6,label='映射误差敏感性包络')]
 P.legend_top(ax,handles=h,labels=[x.get_label() for x in h],ncol=1)
 finish(fig,13,'跨体系映射双层区间',{'rows':d.to_dict('records'),'n_inferred':int((d.mapping_type=='inferred').sum())},'17域映射的条件bootstrap区间与映射误差敏感性包络。后者不解释为概率置信区间；11个推断域使用同一保守回退质量值。',[source('domain_mapping.csv'),source('mapping_uncertainty_contract.json')],'映射误差远大于条件抽样误差。11个推断域不能据相同的回退值作精细质量排名。')

@register(17)
def model_pareto():
 d=D.csv('mixture_internal_cv.csv');fig,ax,p=P.figure('q1',(140,88))
 labels={'quadratic_ridge':'二阶 Ridge','linear_ridge':'线性 Ridge','elastic_net':'Elastic Net','extra_trees':'Extra Trees','random_forest':'随机森林'}
 shifts={'quadratic_ridge':(8,-25),'elastic_net':(8,3),'linear_ridge':(8,7),'extra_trees':(-10,12),'random_forest':(-8,-15)}
 for row in d.itertuples():
  hi=row.model in ['quadratic_ridge','extra_trees'];ax.scatter(row.normalized_rmse,row.median_spearman,s=46 if hi else 30,marker='D' if row.model=='quadratic_ridge' else 'o',facecolor=p.fill['forest_green'] if hi else '#d8dfdb',edgecolor=p.line['forest_green'] if hi else p.line['base'],zorder=3)
  dx,dy=shifts[row.model];ax.annotate(labels[row.model],(row.normalized_rmse,row.median_spearman),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=8)
 q=d.set_index('model').loc[['quadratic_ridge','extra_trees']];ax.plot(q.normalized_rmse,q.median_spearman,c=p.line['forest_green'],ls='--',lw=1,label='非支配前沿')
 ax.set_xlabel('IQR标准化 NRMSE');ax.set_ylabel('Spearman 相关系数');ax.set_xlim(.390,.450);ax.set_ylim(.797,.877);top(ax,1)
 finish(fig,17,'模型选择Pareto前沿',{'models':d.to_dict('records'),'frontier':['quadratic_ridge','extra_trees']},'内部交叉验证的误差—排序双目标比较。二阶Ridge的NRMSE最低，Extra Trees排序相关最高，构成非支配前沿。',[source('mixture_internal_cv.csv')],'二阶响应面兼顾较低预测误差与可解释性；选择过程仅使用内部交叉验证，不借用外部检验集。')

@register(18)
def ridge_path():
 fig,ax,p=P.figure('q1',(145,87));info={}
 for fname,label,key,ls in [('ridge_path_linear.csv','线性 Ridge','base','--'),('ridge_path_quadratic.csv','二阶 Ridge','forest_green','-')]:
  d=D.csv(fname);best=d.loc[d.normalized_rmse.idxmin()];info[label]={'best':best.to_dict(),'rows':d.to_dict('records')}
  ax.plot(d.alpha,d.normalized_rmse,c=p.line[key],ls=ls,lw=1.3,label=label);ax.scatter(best.alpha,best.normalized_rmse,s=35,marker='D',fc=p.fill[key],ec=p.line[key],zorder=3)
 ax.set_xscale('log');ax.set_xlim(1e-6,1e4);ax.set_xlabel('正则化强度 α');ax.set_ylabel('IQR标准化 NRMSE');ax.set_yscale('log');ax.set_ylim(.33,7);ax.set_yticks([.4,.6,1,2,4,6]);ax.yaxis.set_major_formatter('{x:g}');ax.yaxis.set_minor_formatter(NullFormatter())
 ax.xaxis.set_major_locator(LogLocator(base=10,numticks=6));h=Line2D([],[],color='none',marker='D',mfc='white',mec=p.ink,label='交叉验证最优');handles,labels=ax.get_legend_handles_labels();handles.append(h);labels.append(h.get_label());P.legend_top(ax,handles=handles,labels=labels,ncol=3)
 finish(fig,18,'正则化路径',info,'线性与二阶Ridge在完整41点正则化路径上的交叉验证误差，双轴采用对数尺度。菱形为各自NRMSE最低的强度。',[source('ridge_path_linear.csv'),source('ridge_path_quadratic.csv')],'过强正则化导致显著欠拟合；两种模型的最优强度从同一内部验证设计中确定。完整路径保留全部实际采样点，不制造额外训练轮次。')

@register(19)
def rank_validation():
 a=D.csv('mixture_predictions.csv.gz');target='pile_cc';fig,ax,p=P.figure('q1',(139,94));records={}
 for split,label,key,m in [('test_1m','1M','forest_green','o'),('test_60m','60M','ochre','s'),('test_1B','1B','indigo','^')]:
  d=a[(a.split==split)&(a.loss_domain==target)];xr=ss.rankdata(d.observed)/(len(d)+1);yr=ss.rankdata(d.predicted)/(len(d)+1)
  ax.scatter(xr,yr,s=14 if split!='test_1B' else 25,facecolors=p.fill[key],edgecolors=p.line[key],alpha=.67,linewidths=.35,marker=m,label=label,rasterized=True)
  records[split]={'n':len(d),'spearman':float(ss.spearmanr(d.observed,d.predicted).statistic),'index':d['index'].tolist()}
 ax.plot([0,1],[0,1],c=p.ref,ls='--',lw=.8,label='秩一致');ax.add_patch(Rectangle((0,0),.1,.1,fc='none',ec=p.line['forest_green'],lw=1));ax.set_xlim(-.025,1.025);ax.set_ylim(-.025,1.025);ax.set_aspect('equal');ax.set_xlabel('实测Loss归一化秩');ax.set_ylabel('预测Loss归一化秩');hs,ls=ax.get_legend_handles_labels();hs.append(Patch(fc='none',ec=p.line['forest_green'],label='共同前10%'));ls.append('共同前10%');P.legend_top(ax,handles=hs,labels=ls,ncol=3)
 records['target']=target;records['normalization']='rank/(n+1), lower is better';records['low_corner']='both top 10%'
 finish(fig,19,'跨规模配方秩验证',records,'通用网页验证域上，1M、60M和1B实测配方的预测秩与观测秩。秩以n+1归一化，左下方框为两种排序均处于前10%的区域。',[source('mixture_predictions.csv.gz')],'同尺度与跨尺度下多数配方接近秩一致线。此图检验低Loss配方的筛选能力；不能据此宣称绝对Loss无需校准即可跨规模预测。')

@register(20)
def calibration_ba():
 a=D.csv('mixture_predictions.csv.gz');d=a[(a.split=='test_60m')&(a.loss_domain=='pile_cc')];fig,ax,p=P.figure('q1',(145,89));st={'split':'test_60m','target':'pile_cc','n':len(d)}
 for col,label,key,mark in [('predicted','原始预测','ochre','o'),('predicted_calibrated','跨拟合校准','forest_green','s')]:
  mean=(d.observed+d[col])/2;delta=d[col]-d.observed;bias=delta.mean();sd=delta.std(ddof=1)
  ax.scatter(mean,delta,s=17,fc=p.fill[key],ec=p.line[key],lw=.35,alpha=.60,marker=mark,label=label,rasterized=True)
  ax.hlines(bias,mean.min(),mean.max(),color=p.line[key],lw=1.1)
  st[col]={'bias':float(bias),'sd_difference':float(sd),'rmse':float(np.sqrt(np.mean(delta**2)))}
 ax.axhline(0,c=p.ref,lw=.8,ls='--',label='零偏差');ax.set_xlabel('预测与实测Loss的均值');ax.set_ylabel('预测Loss − 实测Loss');hs,ls=ax.get_legend_handles_labels();hs.append(Line2D([],[],color=p.line['base'],lw=1.1,label='组内平均偏差'));ls.append('组内平均偏差');P.legend_top(ax,handles=hs,labels=ls,ncol=2)
 finish(fig,20,'跨规模绝对误差校准',st,'60M通用网页验证域的Bland–Altman式偏差诊断。实心横线为各组平均偏差，虚线为零偏差。仿射校准采用跨拟合预测。',[source('mixture_predictions.csv.gz')],'未经校准的绝对Loss具有系统偏移，跨拟合校准减小该偏移。该校准使用该规模的折内标签，不能标成完全零样本跨规模预测。')

@register(23)
def local_benefits():
 b=D.bridge();d=D.csv('mixture_directional_effects.csv').set_index('domain').loc[b['domains']];coef=b['coefficient_draws'].astype(float);ref=b['p_ref_point'];g=coef[:,:,:17].copy()
 for k,(i,j) in enumerate(itertools.combinations(range(17),2)):g[:,:,i]+=coef[:,:,17+k]*ref[j];g[:,:,j]+=coef[:,:,17+k]*ref[i]
 draws=-(g-g.mean(axis=2,keepdims=True)).mean(axis=1);lo,hi=np.quantile(draws,[.025,.975],axis=0);d=d.assign(lo=lo,hi=hi).sort_values('benefit_score',ascending=False)
 fig,ax,p=P.figure('q1',(156,131));zerox(ax)
 for i,row in enumerate(d.itertuples()):
  key='ochre' if row.benefit_score>0 else 'indigo';ax.plot([0,row.benefit_score],[i,i],color=p.fill[key],lw=3,zorder=1);ax.plot([row.lo,row.hi],[i,i],c=p.line[key],lw=.7,zorder=2);ax.scatter(row.benefit_score,i,s=21,fc=p.fill[key],ec=p.line[key],zorder=3)
 ydomains(ax,d.index);ax.set_xlabel('局部Loss降低的方向导数');h=[Line2D([],[],c=p.line['ochre'],marker='o',mfc=p.fill['ochre'],lw=1,label='点估计与95%区间'),Line2D([],[],c=p.ref,ls='--',label='零收益')];P.legend_top(ax,handles=h,labels=[x.get_label() for x in h],ncol=2)
 finish(fig,23,'局部配比收益与不确定性',{'rows':d.reset_index().to_dict('records'),'bootstrap_reps':len(draws),'baseline':ref.tolist(),'definition':'negative centered gradient at fixed p_ref, mean over 13 losses'},'平均训练配方处17个领域的切向替换收益，正值表示局部Loss降低。区间来自500次训练配方行bootstrap的冻结响应面系数。',[source('mixture_directional_effects.csv'),source('problem2_bridge_input.npz')],'该图是局部方向导数，不是给该域独立增加训练预算的因果效应。区间跨零的领域不足以支持稳定的有利方向。')

@register(28)
def optimization_clusters():
 d=D.csv('optimization_history.csv');order=np.argsort(d.objective.to_numpy());d=d.iloc[order];clusters=np.round(d.objective.to_numpy(),5);u=np.unique(clusters);fig,ax,p=P.figure('q1',(141,83))
 keys=['forest_green','blue_gray','ochre','indigo','olive_gray'];st=[]
 for i,v in enumerate(u):
  g=d[clusters==v];jitter=(np.arange(len(g))-(len(g)-1)/2)*.042
  ax.scatter(g.objective,np.zeros(len(g))+i+jitter,s=32,fc=p.fill[keys[i%5]],ec=p.line[keys[i%5]],lw=.7)
  st.append({'objective_round5':float(v),'n':len(g),'support_margin_range':[float(g.support_margin.min()),float(g.support_margin.max())]})
 ax.set_yticks(range(len(u)),[f'解簇 {i+1}' for i in range(len(u))]);ax.set_xlabel('标准化综合Loss目标 J');ax.set_ylabel('多起点解簇');ax.invert_yaxis();ax.set_xlim(d.objective.min()-.013,d.objective.max()+.018)
 finish(fig,28,'多起点近优解簇',{'n_starts':len(d),'all_feasible':bool(d.feasible.all()),'clusters':st,'grouping':'objective rounded to 5 decimal places; no claim of equal composition','support_margin':d.support_margin.tolist()},'30个可行起点的最终目标值。纵向按目标值至5位小数分组，组内小幅避让仅用于显示重合点，不表示新的优化迭代或额外目标差异。',[source('optimization_history.csv'),source('optimization_diagnostics.json')],'多起点收敛到多个目标值簇，最佳候选不是唯一已证全局最优。所有支持余量和可行性存于同名统计文件；主体展示真实目标离散，避免将数值容差绘成有意义的边界差异。')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('id',nargs='?',help='One figure ID, e.g. Q1-09');parser.add_argument('--id',dest='named_id');args=parser.parse_args();selected=args.named_id or args.id
 if not selected:parser.error('Specify exactly one figure ID')
 n=int(selected.split('-')[-1]);FIGS[n]()

@register(2)
def text_lengths():
 d=pd.read_csv(D.CACHE/'raw_lengths.csv.gz');order=sorted(d.domain.unique(),key=lambda x:d.loc[d.domain==x,'rps_doc_word_count'].median(),reverse=True);fig,ax,p=P.figure('q1',(148,99));result={}
 for i,dom in enumerate(order):
  vals=d.loc[d.domain==dom,'rps_doc_word_count'].to_numpy();qs=np.quantile(vals,[.005,.0625,.125,.25,.5,.75,.875,.9375,.995]);result[dom]={'n':len(vals),'quantiles':qs.tolist(),'min':float(vals.min()),'max':float(vals.max())};ax.hlines(i,qs[0],qs[-1],color=p.line['forest_green'],ls='--',lw=.7)
  for a,b,h,alpha in [(.0625,.9375,.66,.30),(.125,.875,.47,.55),(.25,.75,.28,.95)]:
   l,r=np.quantile(vals,[a,b]);ax.add_patch(Rectangle((l,i-h/2),r-l,h,fc=p.fill['forest_green'],ec=p.line['forest_green'],alpha=alpha,lw=.7))
  ax.plot(np.median(vals),i,'|',color=p.line['indigo'],ms=10,mew=1.5)
 ax.set_xscale('log');ax.set_xlabel('每篇文本词数');ydomains(ax,order);ax.set_xlim(1.5,6e5);ax.set_xticks([10,100,1000,10000,100000]);ax.xaxis.set_minor_formatter(NullFormatter())
 h=[Patch(facecolor=p.fill['forest_green'],alpha=.95,label='中央50%'),Patch(facecolor=p.fill['forest_green'],alpha=.55,label='中央75%'),Patch(facecolor=p.fill['forest_green'],alpha=.30,label='中央87.5%'),Line2D([],[],color=p.line['forest_green'],ls='--',lw=.7,label='0.5%–99.5%'),Line2D([],[],color=p.line['indigo'],marker='|',ls='none',ms=9,label='中位数')];P.legend_top(ax,handles=h,labels=[v.get_label() for v in h],ncol=3)
 finish(fig,2,'原始文本长度字母值图',result,'去重后261086条质量记录的原始文本词数。嵌套盒展示中央50%、75%、87.5%范围，虚线为0.5%–99.5%分位，横轴为对数尺度。',[D.CACHE/'raw_lengths.csv.gz',D.CACHE/'data_audit_quality.json'],'图书与学术论文的文本明显更长，代码文本的主体较短但尾部很长。质量结构指标需要考虑领域差异，不能简单判定越长越优。')

@register(8)
def sample_remainder_qq():
 d=D.quality();fig,ax,p=P.figure('q1',(130,105));qs=np.linspace(.01,.99,99);out={}
 for dom,key,m in [('arxiv','forest_green','o'),('github','indigo','s')]:
  g=d[d.domain==dom];a=g.loc[g.in_a1,'Q'].to_numpy();b=g.loc[~g.in_a1,'Q'].to_numpy();x=np.quantile(a,qs);y=np.quantile(b,qs);ax.plot(x,y,c=p.line[key],lw=1.2,marker=m,markevery=10,ms=3,mfc=p.fill[key],label=D.DOMAIN[dom]);out[dom]={'n_a1':len(a),'n_remainder':len(b),'quantiles':qs.tolist(),'a1':x.tolist(),'remainder':y.tolist(),'wasserstein':float(ss.wasserstein_distance(a,b))}
 ax.plot([.27,.835],[.27,.835],c=p.ref,ls='--',lw=.8,label='分位一致');ax.set_xlim(.27,.835);ax.set_ylim(.27,.835);ax.set_aspect('equal');ax.set_xlabel('A1抽样集质量分位');ax.set_ylabel('扩展非重合部分质量分位');top(ax,3)
 finish(fig,8,'抽样与扩展质量QQ检验',out,'arXiv与GitHub质量分数的分位对应。扩展部分严格排除与A1重合的样本；每条曲线使用1%至99%分位，不以均值代替分布。',[D.CACHE/'quality_sample.csv.gz',D.CACHE/'verification.json'],'分位曲线与参考线的距离显示抽样和扩展的分布差异。此图不把相互重合的记录误当成独立重复检验。')

@register(10)
def conflict_plane():
 d=D.quality();cols=['block_'+k for k in D.BLOCKS];v=d[cols].to_numpy();x=np.nanmin(v,axis=1);y=np.nanmax(v,axis=1);fig,ax,p=P.figure('q1',(139,108));h=ax.hexbin(x,y,gridsize=52,mincnt=1,cmap=p.cmap_seq(),norm=LogNorm(),linewidths=0,rasterized=True)
 ids=np.flatnonzero(d.conflict.to_numpy());sel=ids[np.linspace(0,len(ids)-1,min(900,len(ids)),dtype=int)]
 ax.scatter(x[sel],y[sel],s=8,fc='none',ec=p.line['ochre'],lw=.3,label='冲突样本（抽样）',rasterized=True)
 ax.axvline(.25,c=p.line['base'],ls='--',lw=.8,label='高低阈值');ax.axhline(.75,c=p.line['base'],ls='--',lw=.8);ax.plot([0,1],[0,1],c=p.ref,ls=':',lw=.7,label='块分相等');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect('equal');ax.set_xlabel('最低语义块得分');ax.set_ylabel('最高语义块得分');P.colorbar(fig,h,ax,label='样本数',shrink=.88);top(ax,2)
 finish(fig,10,'质量冲突相平面',{'n':len(d),'n_conflict':len(ids),'conflict_rate':len(ids)/len(d),'displayed_conflict':len(sel),'thresholds':{'low':.25,'high':.75,'K':'strictly above domain-specific 90th percentile'}},'五语义块最高分与最低分的联合分布。颜色表示全部样本的分箱计数；空心点为确定性抽样显示的冲突样本。冲突同时要求跨越高低阈值和领域内极差90%阈值。',[D.CACHE/'quality_sample.csv.gz'],'同一文本的质量信号可以显著不一致。阈值线仅表示高低门槛；橙色样本还经过各域极差门槛筛选，不能把整个左上象限都称为冲突。')

@register(11)
def huber_correction():
 d=D.quality();g=d[d.conflict].copy();g['delta']=g.Q-g.Q_variant_weighted_no_huber;chosen=[]
 for dom in D.QUALITY_DOMAINS:
  a=g[g.domain==dom].sort_values('delta')
  if len(a):chosen.extend(a.iloc[np.unique(np.round(np.array([.1,.9])*(len(a)-1)).astype(int))].index.tolist())
 s=g.loc[chosen].sort_values('delta');fig,ax,p=P.figure('q1',(148,121));ys=np.arange(len(s));ax.hlines(ys,s.Q_variant_weighted_no_huber,s.Q,color=p.line['forest_green'],lw=1)
 ax.scatter(s.Q_variant_weighted_no_huber,ys,s=24,marker='o',fc='white',ec=p.line['base'],lw=.9,label='块均值');ax.scatter(s.Q,ys,s=28,marker='D',fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.7,label='Huber评分')
 ax.set_yticks(ys,[D.DOMAIN[r.domain]+f' {i+1:02d}' for i,r in enumerate(s.itertuples())]);ax.set_ylim(len(s)-.5,-.5);ax.set_ylabel('代表冲突样本');ax.set_xlabel('综合质量评分 Q');ax.set_xlim(max(0,min(s.Q.min(),s.Q_variant_weighted_no_huber.min())-.05),min(1,max(s.Q.max(),s.Q_variant_weighted_no_huber.max())+.05));top(ax,2)
 finish(fig,11,'冲突消解评分修正',{'selection':'within each domain, nearest rows to 10th/90th percentile in signed correction','n_all_conflict':len(g),'delta_quantiles':g.delta.quantile([0,.1,.5,.9,1]).to_dict(),'selected':s[['domain','id','Q','Q_variant_weighted_no_huber','delta']].to_dict('records')},'各域冲突样本中，在评分修正量10%与90%位置确定性选择代表记录，比较块均值与Huber评分。连接线表示同一文本，不表示时间序列。',[D.CACHE/'quality_sample.csv.gz'],'稳健消解可以向上或向下修正评分，具体方向由冲突的块结构与域内尺度决定。图中样本为按预定分位选择的案例，并非总体平均效应。')

@register(5)
def indicator_correlation():
 d=D.quality();weights=pd.read_csv(D.CACHE/'quality_weights.csv');rng=np.random.default_rng(2026);nmin=int(d.groupby('domain').size().min());bal=pd.concat([g.iloc[rng.choice(len(g),nmin,replace=False)] for _,g in d.groupby('domain',sort=True)]);fields=[];bounds=[]
 for block in D.BLOCKS:
  fs=weights.loc[weights.block==block,'field'].tolist();co=bal[fs].corr(method='spearman').fillna(0).to_numpy();dist=np.clip(1-np.abs(co),0,2);np.fill_diagonal(dist,0)
  from scipy.spatial.distance import squareform
  ix=leaves_list(linkage(squareform(dist,checks=False),method='average')) if len(fs)>1 else [0];fields.extend([fs[i] for i in ix]);bounds.append(len(fields))
 corr=bal[fields].corr(method='spearman').to_numpy();masked=np.ma.masked_where(np.triu(np.ones_like(corr),1).astype(bool),corr);fig,ax,p=P.figure('q1',(166,163));im=ax.imshow(masked,cmap=p.cmap_div(),vmin=-1,vmax=1,interpolation='none');ax.set_xticks(range(22),[D.FIELDS[f] for f in fields],rotation=90,fontsize=7.5);ax.set_yticks(range(22),[D.FIELDS[f] for f in fields],fontsize=7.5);ax.tick_params(length=0);ax.set_xlabel('同向化质量指标');ax.set_ylabel('同向化质量指标');
 for b in bounds[:-1]:ax.plot([-.5,b-.5],[b-.5,b-.5],c='white',lw=1.7);ax.plot([b-.5,b-.5],[b-.5,21.5],c='white',lw=1.7)
 for sp in ax.spines.values():sp.set_visible(False)
 P.colorbar(fig,im,ax,label='Spearman ρ',shrink=.64,pad=.025)
 finish(fig,5,'质量指标相关结构',{'fields':fields,'n_balanced':len(bal),'n_per_domain':nmin,'matrix':corr.tolist(),'clustering':'within semantic blocks, average linkage of 1-|rho|'},'22项同向化质量指标的Spearman相关结构。七域各取171条记录形成等量分层样本，语义块内按绝对相关距离聚类；白色分界表示五个语义块。',[D.CACHE/'quality_sample.csv.gz',D.CACHE/'quality_weights.csv'],'指标存在信息重叠，且并非都沿同一方向变化。此相关图为等量域样本的描述性分析，与正式权重程序的抽样规模不同；不能把相关关系解释为因果。')

@register(6)
def quality_weights():
 d=pd.read_csv(D.CACHE/'quality_weights.csv');keys=['forest_green','blue_gray','indigo','ochre','olive_gray'];order=[]
 for block in D.BLOCKS:order.extend(d[d.block==block].sort_values('total_weight',ascending=False).index)
 d=d.loc[order];fig,ax,p=P.figure('q1',(154,155));y=np.arange(len(d));blockkeys=dict(zip(D.BLOCKS,keys));ax.barh(y,d.total_weight*100,height=.68,color=[p.fill[blockkeys[b]] for b in d.block],edgecolor=[p.line[blockkeys[b]] for b in d.block],linewidth=.6)
 ax.set_yticks(y,[D.FIELDS[f] for f in d.field]);ax.set_ylim(len(d)-.4,-.7);ax.set_xlabel('最终综合权重 / %');ax.set_ylabel('质量指标');ax.set_xlim(0,d.total_weight.max()*100*1.18)
 for i,r in enumerate(d.itertuples()):ax.text(r.total_weight*100+.12,i,f'{r.total_weight*100:.2f}',va='center',fontsize=7.5)
 cum=0
 for block in D.BLOCKS:
  cum+=int((d.block==block).sum())
  if cum<len(d):ax.axhline(cum-.5,c=p.grid,lw=.8)
 hs=[Patch(fc=p.fill[k],ec=p.line[k],label=D.BLOCKS[b]) for b,k in blockkeys.items()];P.legend_top(ax,handles=hs,labels=[x.get_label() for x in hs],ncol=2)
 finish(fig,6,'语义块与质量指标权重',{'rows':d.to_dict('records'),'block_totals':d.groupby('block').total_weight.sum().to_dict(),'layout_change':'hierarchical bar instead of icicle to keep all 22 names legible'},'五个语义块的总权重均为20%，各块内通过稳定CRITIC赋权。条长表示最终综合权重，颜色与分组分别标识语义块。',[D.CACHE/'quality_weights.csv'],'五块等权防止指标数量较多的语义块取得额外优势。采用分组权重条图替代拥挤的冰柱叶片，使22个指标名称与细小权重都能直接读取。')

@register(7)
def quality_raincloud():
 d=D.quality();fig,ax,p=P.figure('q1',(151,115));rng=np.random.default_rng(2026);records={};grid=np.linspace(0,1,300)
 for i,dom in enumerate(D.QUALITY_DOMAINS):
  v=d.loc[d.domain==dom,'Q'].to_numpy();density=ss.gaussian_kde(v,bw_method=.18)(grid);density=density/density.max()*.32;ax.fill_between(grid,i-.10-density,i-.10,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.6,alpha=.8)
  sample=v[rng.choice(len(v),min(200,len(v)),replace=False)];ax.scatter(sample,i+.12+rng.uniform(0,.14,len(sample)),s=3,fc=p.line['forest_green'],alpha=.22,rasterized=True);q=np.quantile(v,[.25,.5,.75]);ax.plot([q[0],q[2]],[i+.11,i+.11],c=p.line['indigo'],lw=2.8);ax.scatter(q[1],i+.11,s=16,fc='white',ec=p.line['indigo'],zorder=3);records[dom]={'n':len(v),'quartiles':q.tolist(),'displayed_points':len(sample)}
 ydomains(ax,D.QUALITY_DOMAINS);ax.set_xlabel('样本质量评分 Q');ax.set_xlim(.1,.91);hs=[Patch(fc=p.fill['forest_green'],ec=p.line['forest_green'],label='全部样本密度'),Line2D([],[],marker='.',ls='none',color=p.line['forest_green'],label='分层抽样记录'),Line2D([],[],lw=3,c=p.line['indigo'],marker='o',mfc='white',ms=3,label='中位数与四分位距')];P.legend_top(ax,handles=hs,labels=[x.get_label() for x in hs],ncol=2)
 finish(fig,7,'七域质量云雨分布',{'domains':records,'kde_bandwidth_factor':.18,'density_grid':[0,1,300]},'七域质量评分的完整核密度、等上限抽样记录及中位数/四分位范围。所有密度均由该域全部记录计算；散点上限为每域200条。',[D.CACHE/'quality_sample.csv.gz'],'领域均值之外仍有广泛的个体差异和分布重叠。各域密度高度独立归一，不能用面积比较样本量；样本量见统计文件。')

@register(15)
def composition_barcode():
 b=D.bridge();v=b['p_train'];dominant=v.argmax(1);effective=1/np.square(v).sum(1);order=np.lexsort((effective,dominant));fig,ax,p=P.figure('q1',(158,121));cmap=p.cmap_seq().copy();cmap.set_bad('white');masked=np.ma.masked_equal(v[order],0);im=ax.imshow(masked,aspect='auto',cmap=cmap,norm=Normalize(0,v.max()),interpolation='none',rasterized=True)
 ax.set_xticks(range(17),[D.DOMAIN[x] for x in b['domains']],rotation=90);ax.set_xlabel('训练领域');ax.set_ylabel('训练配方（排序后）');ax.set_yticks([0,127,255,383,511],[1,128,256,384,512]);P.colorbar(fig,im,ax,label='训练份额',shrink=.85)
 changes=np.flatnonzero(np.diff(dominant[order]))+.5
 for y in changes:ax.axhline(y,c='white',lw=.6)
 ax.legend(handles=[Patch(fc='white',ec='#aab4b0',label='零份额')],loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
 finish(fig,15,'训练配方组成条码',{'shape':v.shape,'zero_fraction':float((v==0).mean()),'row_order':b['train_index'][order].tolist(),'domain_order':b['domains'].tolist(),'sort':'dominant domain, then effective number of domains'},'512个训练配方的17域组成。按主导域及有效领域数排序，白格为精确零份额，颜色表示非零份额。',[source('problem2_bridge_input.npz')],'训练设计具有显著稀疏性和份额集中，许多单纯形边界附近区域存在样本；不能将白格误读为缺失数据。')

@register(16)
def composition_effective():
 v=D.bridge()['p_train'];x=(v>0).sum(1);y=1/np.square(v).sum(1);xb=np.arange(1.5,18.5);yb=np.arange(.75,9.76,.5);h,xe,ye=np.histogram2d(x,y,bins=[xb,yb]);ii,jj=np.nonzero(h);sizes=h[ii,jj];fig,ax,p=P.figure('q1',(137,92));ax.scatter((xe[ii]+xe[ii+1])/2,(ye[jj]+ye[jj+1])/2,s=sizes*6,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.6,alpha=.82)
 ax.plot([1,17],[1,17],c=p.ref,ls='--',lw=.8,label='均匀配比上界');ax.set_xlim(1,18);ax.set_ylim(.5,10);ax.set_xticks(np.arange(2,18,3));ax.set_xlabel('非零领域数');ax.set_ylabel('有效领域数');hs,ls=ax.get_legend_handles_labels()
 for n in [5,15,30]:hs.append(ax.scatter([],[],s=n*6,fc=p.fill['forest_green'],ec=p.line['forest_green'],label=f'{n}个配方'));ls.append(f'{n}个配方')
 P.legend_top(ax,handles=hs,labels=ls,ncol=2)
 finish(fig,16,'领域覆盖与份额集中度',{'active_quantiles':np.quantile(x,[0,.25,.5,.75,1]),'effective_quantiles':np.quantile(y,[0,.25,.5,.75,1]),'effective_definition':'1/sum(p_i^2)','bin_width_y':.5,'bin_counts':h.astype(int)},'配方中非零领域数与逆Simpson有效领域数的联合计数分布。气泡面积与配方计数成正比，有效领域数按0.5宽度分箱。',[source('problem2_bridge_input.npz')],'许多配方虽然覆盖多域，实际份额仍集中在少数域。沿均匀配比上界的距离表现这种集中程度，不能把有效领域数当作新增实际类别。')

@register(22)
def stable_interactions():
 target='pile_cc';d=D.csv('interaction_stability.csv');d=d[(d.loss_domain==target)&((d.ci_low*d.ci_high)>0)].copy();d['abs_coef']=d.coefficient.abs();d=d.nlargest(10,'abs_coef');nodes=sorted(set(d.domain_i)|set(d.domain_j));fig,ax,p=P.figure('q1',(158,118));theta=np.linspace(np.pi/2,np.pi/2+2*np.pi,len(nodes),endpoint=False);pos={n:np.array([np.cos(t),np.sin(t)]) for n,t in zip(nodes,theta)};largest=d.abs_coef.max();drawinfo=[]
 for row in d.itertuples():
  a=pos[row.domain_i];b=pos[row.domain_j];key='indigo' if row.coefficient<0 else 'ochre';lw=.8+2.8*abs(row.coefficient)/largest;ax.plot([a[0],b[0]],[a[1],b[1]],c=p.line[key],lw=lw,ls='-' if row.coefficient<0 else '--',alpha=.72,zorder=1);drawinfo.append({'i':row.domain_i,'j':row.domain_j,'coefficient':row.coefficient,'ci_low':row.ci_low,'ci_high':row.ci_high,'sign_probability':row.sign_probability})
 for n in nodes:
  x,y=pos[n];ax.scatter(x,y,s=50,fc=p.fill['light_green'],ec=p.line['forest_green'],lw=.8,zorder=3);ax.text(x*1.12,y*1.12,D.DOMAIN[n],ha='left' if x>.3 else ('right' if x<-.3 else 'center'),va='center',fontsize=8)
 ax.set_aspect('equal');ax.set_xlim(-1.62,1.62);ax.set_ylim(-1.30,1.42);ax.set_axis_off();hs=[Line2D([],[],c=p.line['indigo'],lw=1.5,label='负二阶系数'),Line2D([],[],c='#818987',lw=3,label='线宽：系数绝对值')];ax.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=3)
 finish(fig,22,'稳定领域交互网络',{'target':target,'selection':'95% coefficient interval excludes zero; top 10 absolute coefficients','edges':drawinfo,'node_order':nodes},'通用网页验证域中，95%系数区间不跨零且绝对值最大的10个二阶域对系数。本次入选10条边均为负系数，以实线表示；线宽按绝对值线性映射。',[source('interaction_stability.csv')],'图示冻结Scheffé响应面中的稳定二阶项，反映相对一次混合平面的偏离。它不是干预因果网络，不代表单独增加任一领域一定改善Loss。')

@register(25)
def supported_ternary():
 b=D.bridge();domains=b['domains'].tolist();names=['uspto_backgrounds','pubmed_abstracts','stackexchange'];ix=[domains.index(d) for d in names];opt=D.csv('optimal_mixture.csv').set_index('domain').loc[domains,'p_star'].to_numpy();total=opt[ix].sum();trip=np.array([(i/160,j/160,1-(i+j)/160) for i in range(161) for j in range(161-i)]);trip=np.vstack([trip,opt[ix]/total]);vs=np.tile(opt,(len(trip),1));vs[:,ix]=trip*total
 nearest=cdist(vs/b['support_scale'],b['p_train']/b['support_scale']).min(1);ok=(nearest<=float(b['support_radius'])+1e-8)&(vs<=b['upper']+1e-10).all(1);xy=P.to_ternary(trip);tri=Triangulation(xy[:,0],xy[:,1]);tri.set_mask((~ok[tri.triangles]).any(1));z=D.objective(vs);fig,ax,p=P.figure('q1',(156,119));ax.add_patch(Polygon(P.TRI,fc='#f5f5f1',ec='#adb6af',hatch='///',lw=0,zorder=0));im=ax.tricontourf(tri,z,levels=60,cmap=p.cmap_seq(),zorder=1);ax.tricontour(tri,z,levels=6,colors=p.line['forest_green'],linewidths=.35,alpha=.65,zorder=2);P.ternary_frame(ax,[D.DOMAIN[n] for n in names],fontsize=8)
 for frac in [.2,.4,.6,.8]:
  label=f'{frac:.0%}';ax.text(1-frac,-.022,label,ha='center',va='top',fontsize=7.5);ax.text(.5*frac-.028,np.sqrt(3)/2*frac,label,ha='right',va='center',fontsize=7.5);ax.text(.5+.5*frac+.028,np.sqrt(3)/2*(1-frac),label,ha='left',va='center',fontsize=7.5)
 xyopt=P.to_ternary((opt[ix]/total)[None,:])[0];ax.scatter(*xyopt,marker='*',s=95,fc=p.fill['ochre'],ec=p.line['ochre'],lw=.7,zorder=6,label='参考近优配方')
 P.colorbar(fig,im,ax,label='标准化综合Loss J',shrink=.7,pad=.04);hs=[Line2D([],[],marker='*',ls='none',ms=9,mfc=p.fill['ochre'],mec=p.line['ochre'],label='参考近优配方'),Patch(fc='#f5f5f1',ec='#adb6af',hatch='///',label='未展示网格区域')];ax.legend(handles=hs,loc='lower center',bbox_to_anchor=(.48,1.01),ncol=1)
 finish(fig,25,'经验支持域内三元响应切片',{'varying_domains':names,'their_fixed_total':float(total),'fixed_other_components':dict(zip(domains,opt.tolist())),'grid_count':len(vs),'feasible_count':int(ok.sum()),'feasible_objective_range':[float(z[ok].min()),float(z[ok].max())],'reference_objective':float(D.objective(opt[None,:])[0]),'grid_step':1/160,'reference_inserted_as_vertex':True,'reference_feasible':bool(abs(opt.sum()-1)<1e-10 and opt.min()>=-1e-10 and (opt<=b['upper']+1e-10).all() and cdist((opt/b['support_scale'])[None,:],b['p_train']/b['support_scale']).min()<=float(b['support_radius'])+1e-8),'reference_support_distance':float(cdist((opt/b['support_scale'])[None,:],b['p_train']/b['support_scale']).min()),'reference_support_margin':float(float(b['support_radius'])-cdist((opt/b['support_scale'])[None,:],b['p_train']/b['support_scale']).min()),'reference_min_upper_margin':float(np.min(b['upper']-opt)),'support_feasibility_tolerance':1e-8,'upper_feasibility_tolerance':1e-10,'mask_rule':'mask whole triangle if any of its vertices fails support or upper check; non-displayed cells may include feasible subregions' },'固定其余14域份额，将专利背景、医学摘要与问答社区的总份额在三元单纯形内重新分配。着色仅包含三个顶点均通过支持距离与份额上界检查的网格三角；灰纹表示未展示区域，包含越界点及跨边界三角中的可行部分。星标独立通过原约束容差检验；三元坐标为三域内部归一比例。',[source('problem2_bridge_input.npz'),source('optimal_mixture.csv')],'支持约束及保守网格遮罩限制了响应面的展示范围，不能将全部灰纹解释为严格不可行。星标位置未移动且未放宽原约束；具体支持距离与余量存于统计文件。本图是17域问题的二维条件切片，不证明全域全局最优。')

@register(27)
def mixture_slab():
 b=D.bridge();arr=np.load(source('optimal_mixture_bootstrap.npz'),allow_pickle=False);draw=arr['p'][arr['success']];d=D.csv('optimal_mixture.csv').set_index('domain').loc[b['domains']];order=np.argsort(-d.p_star.to_numpy());d=d.iloc[order];draw=draw[:,order];fig,ax,p=P.figure('q1',(160,143));fig.set_layout_engine(None);ax.set_position([.17,.12,.56,.79]);right=fig.add_axes([.81,.12,.15,.79],sharey=ax);grid=np.linspace(0,1,320)
 for i,(name,row) in enumerate(d.iterrows()):
  vals=draw[:,i];nonzero=vals[vals>1e-6]
  if len(nonzero)>2 and np.std(nonzero)>1e-8:
   gg=np.linspace(0,row.upper_bound,180);den=ss.gaussian_kde(nonzero,bw_method=.32)(gg);den=den/max(den.max(),1e-12)*.34;ax.fill_between(gg,i-den,i+den,fc=p.fill['light_green'],ec=p.line['forest_green'],lw=.5,alpha=.85)
  ax.plot([row.ci_low,row.ci_high],[i,i],c=p.line['forest_green'],lw=.8,zorder=3);ax.scatter(row.p_star,i,s=24,marker='D',fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.6,zorder=4);ax.plot(row.upper_bound,i,'|',c=p.line['indigo'],ms=9,mew=.9,zorder=4);right.barh(i,row.zero_probability,height=.52,fc=p.fill['blue_gray'],ec=p.line['blue_gray'],lw=.5)
 ydomains(ax,d.index);ax.set_xlim(-.02,1);ax.set_xticks([0,.2,.4,.6,.8,1]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_xlabel('训练份额');right.set_xlim(0,1);right.set_xticks([0,.5,1]);right.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));right.set_xlabel('零份额概率');right.tick_params(axis='y',left=False,labelleft=False);right.spines['left'].set_visible(False)
 hs=[Patch(fc=p.fill['light_green'],ec=p.line['forest_green'],label='非零份额密度'),Line2D([],[],marker='D',color=p.line['forest_green'],mfc=p.fill['forest_green'],lw=.8,label='候选与95%区间'),Line2D([],[],marker='|',color=p.line['indigo'],ls='none',ms=9,label='份额上界')];fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.57,.987),ncol=2,fontsize=8)
 finish(fig,27,'零膨胀近优配比族',{'n_success':len(draw),'rows':d.reset_index().to_dict('records'),'density':'positive draws only, bounded to component upper limit','interval':'all successful draws including exact zeros','zero_threshold':1e-6},'200次成功bootstrap的近优配比族。主轴展示非零份额密度、参考候选及包含零值的95%区间；短竖线为分量上界。右侧条形显示零份额概率。',[source('optimal_mixture.csv'),source('optimal_mixture_bootstrap.npz'),source('problem2_bridge_input.npz')],'所有分量的95%下界均为零。域是否入选与入选后的份额都具有不确定性，单一候选的零份额不能解释为该域应被永久剔除。')

from . import reference_v2
reference_v2.configure(finish)
FIGS=reference_v2.FIGS

if __name__=='__main__':main()
