"""Single-file publication figures for question 2; frozen data only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator, PercentFormatter, ScalarFormatter
from scipy.stats import gaussian_kde, spearmanr
from common import plot as P
import config as C
from q2 import data as D

FIGS={}
def register(k):
 def dec(f): FIGS[k]=f;return f
 return dec

def new(size=(160,102), margins=(.15,.94,.17,.83)):
 fig,ax,pal=P.figure('q2',size,layout=None)
 left,right,bottom,top=margins;fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
 return fig,ax,pal

def top(ax,ncol=3,handles=None,labels=None):
 return P.legend_top(ax,ncol=ncol,handles=handles,labels=labels,pad=.045,fontsize=8,frameon=False)

def src(*names):return [D.path(n) for n in names]
def rawsrc(*keys):return [D.rawpath(k) for k in keys]
def finish(fig,ident,title,stats,caption,sources,reading):
 name=f'Q2-{ident}_{title}'
 issues=P.save(fig,'q2',name,stats=stats,caption=caption,sources=sources)
 docpath=C.OUTPUT/'q2'/f'{name}.json'
 meta=json.loads(docpath.read_text());meta['interpretation']=reading
 docpath.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 write_docs()
 return issues

def write_docs():
    records=[json.loads(p.read_text()) for p in sorted((C.OUTPUT/'q2').glob('Q2-*.json'))]
    text=['# 问题二图表说明','','本批共20张独立图，使用冻结结果 `问题二/results_p12_final_v2`。原始数据、半合成网格、估算与模型计算在图注中区分。中文宋体，英文/数字 Times New Roman，数学字体 STIX；PNG 300 dpi 与嵌入字体 PDF 成对。','','在 `论文绘图` 目录运行：`.venv/bin/python -m q2.run Q2-01` 重画单图，`.venv/bin/python -m q2.run` 重画完整Top20。','','推荐正文组合：证据覆盖（01+15）、经典规律（04+03择一）、质量选择与收益（10+14）、领域互补（19）、资源等价（23+26）、接口边界（27）。24与25适合作资源机制补充；09保留全部450个半合成质量记录，适合附录。','','图中不写长解释。以下图注可用于论文；入稿时应按本节条件保留证据口径。完整精度和逐点统计在每张JSON及 `output/q2/stats.json` 中，不从图上读数。','']
    omit={'states','row_keys','column_keys','domain_labels','domain_slopes','conditional_quantiles','family_RMSE','records'}
    for rec in records:
        values={k:v for k,v in rec['stats'].items() if k not in omit}
        if 'reference' in values and isinstance(values['reference'],dict):
            values['reference']={k:v for k,v in values['reference'].items() if k in ['N_B','D_B','Q','s_Q','Q_A','delta_Q','finite_N_multiplier_minus_1','finite_D_multiplier_minus_1']}
        text += [f"## {rec['name']}",'',f"- 文件：[PNG](<{rec['png']}>) · [PDF](<{rec['pdf']}>) · [完整数值](<{Path(rec['png']).with_suffix('.json')}>)",f"- 画布：{rec['size_mm'][0]} × {rec['size_mm'][1]} mm。",f"- 图注与编码：{rec['caption']}",f"- 解读与条件：{rec.get('interpretation','见图注。')}",'- 数据来源：'+'；'.join(rec['sources']),'- 主要统计：`'+json.dumps(values,ensure_ascii=False,separators=(',',':'))+'`','']
    (C.PLOT/'图表说明_Q2.md').write_text('\n'.join(text))

def use_log(ax,which='both'):
 if which in ('both','x'):
  ax.set_xscale('log');ax.xaxis.set_minor_formatter(NullFormatter())
 if which in ('both','y'):
  ax.set_yscale('log');ax.yaxis.set_minor_formatter(NullFormatter())

@register('01')
def coverage():
 fig,ax,pal=new((160,108),(.13,.95,.15,.77))
 specs=[('B1','B1 训练轨迹','o','slate_teal',8,.48),('B7','B7 半合成主网格','s','forest_green',21,.82),('B8','B8 半合成压力网格','^','ochre',22,.55)]
 sizes={}
 for k,label,marker,c,ss,alpha in specs:
  a=D.raw(k).drop_duplicates(['N_params_B','D_tokens_B']);sizes[k]=len(a)
  ax.scatter(a.N_params_B,a.D_tokens_B,s=ss,marker=marker,c=pal.fill[c],edgecolors=pal.line[c],linewidths=.35,alpha=alpha,label=label)
 ext=pd.concat([D.raw('B4'),D.raw('B5')]).drop_duplicates(['N_params_B','D_tokens_B'])
 ax.scatter(ext.N_params_B,ext.D_tokens_B,s=22,marker='D',facecolors='none',edgecolors=pal.line['indigo'],linewidths=.65,label='B4/B5 跨族与文献')
 large=pd.concat([D.raw('B9'),D.raw('B10')]).query('D_tokens_B > 0').drop_duplicates(['N_params_B','D_tokens_B'])
 ax.scatter(large.N_params_B,large.D_tokens_B,s=21,marker='x',color=pal.line['blue_gray'],linewidths=.65,label='B9/B10 大模型与估算')
 for k,c in [('B1','slate_teal'),('B7','forest_green')]:
  a=D.raw(k);xx=[a.N_params_B.min(),a.N_params_B.max()];yy=[a.D_tokens_B.min(),a.D_tokens_B.max()]
  ax.add_patch(Rectangle((xx[0],yy[0]),xx[1]-xx[0],yy[1]-yy[0],fill=False,ec=pal.line[c],ls=(0,(4,3)),lw=.8,zorder=1))
 use_log(ax);ax.set(xlim=(.045,15000),ylim=(.07,60000),xlabel='参数量 N / 十亿',ylabel='训练数据量 D / 十亿 Token')
 handles,labels=ax.get_legend_handles_labels();handles.append(Line2D([],[],ls=(0,(4,3)),color=pal.line['base'],lw=.8));labels.append('B1/B7 范围')
 top(ax,3,handles,labels)
 finish(fig,'01','证据覆盖地图',{'unique_ND_by_source':sizes,'external_unique_ND':len(ext),'large_unique_positive_ND':len(large),'B9_zero_token_rows':int((D.raw('B9').D_tokens_B==0).sum())},'参数量—数据量空间中的证据覆盖。B1为训练轨迹，B7/B8为半合成网格，B4/B5为跨族及文献记录，B9/B10仅提供大模型位置与估算对照；虚线为B1、B7各自的取值范围。',rawsrc('B1','B4','B5','B7','B8','B9','B10'),'范围矩形不代表内部所有组合均被真实训练覆盖。B9的4条D=0记录不进入对数坐标，B10不增加独立验证证据。')

@register('04')
def collapse():
 fig,ax,pal=new((160,98),(.14,.88,.18,.83));a=D.raw('B1');p=D.cp();n=a.N_params_B.to_numpy();d=a.D_tokens_B.to_numpy();y=a.val_loss.to_numpy()-p['E']-p['A']*n**(-p['alpha']);ok=y>0
 norm=LogNorm(n.min(),n.max());order=np.random.default_rng(2026).permutation(np.flatnonzero(ok));sc=ax.scatter(d[order],y[order],c=n[order],norm=norm,cmap=pal.cmap_seq(),s=13,edgecolors='none',alpha=.78)
 xx=np.geomspace(d.min(),d.max(),300);ax.plot(xx,p['B']*xx**(-p['beta']),color=pal.line['ochre'],lw=1.05,ls='--',label='M0 数据项')
 use_log(ax);ax.set(xlabel='训练数据量 D / 十亿 Token',ylabel='剥离参数项后的可约损失',xlim=(.1,400),ylim=(.2,3))
 cb=P.colorbar(fig,sc,ax,label='参数量 N / 十亿',pad=.035);cb.set_ticks([.1,1,10]);cb.set_ticklabels(['0.1','1','10']);top(ax,1)
 finish(fig,'04','幂律数据坍缩',{'rows':len(a),'nonpositive_transformed_rows':int((~ok).sum()),'beta':p['beta'],'max_abs_transformed_error':float(np.max(np.abs(y-p['B']*d**(-p['beta']))))},'将B1损失扣除不可约项及参数规模项后，八种规模的检查点落在同一数据幂律附近。虚线为冻结M0的数据项，点颜色表示参数量。',rawsrc('B1')+src('classic_scaling_parameters.json'),'这是冻结模型下的结构一致性诊断，并非独立验证；不能将视觉坍缩当作跨模型族普适性的证明。')

@register('10')
def quality_selection():
 fig,ax,pal=new((160,94),(.14,.95,.19,.80));a=D.table('quality_split_family_scores.csv');splits=['leave_N','leave_D','leave_Q','leave_cell'];marks=['o','D','^','s'];cols=['slate_teal','indigo','blue_gray','ochre'];stats={}
 for model,c,m in zip(['GQ1','GQ2','GQ3','GQ4'],cols,marks):
  y=a[a.model==model].set_index('split_type').loc[splits,'family_RMSE'].to_numpy();stats[model]=y.tolist();ax.plot(range(4),y,color=pal.line[c],marker=m,mfc=pal.fill[c],mec=pal.line[c],lw=1.6 if model=='GQ2' else .8,ms=5 if model=='GQ2' else 4,label=model,zorder=4 if model=='GQ2' else 2)
 ax.set(xticks=range(4),xticklabels=['留参数量 N','留数据量 D','留质量 Q','留资源单元 (N,D)'],ylabel='分块验证 RMSE',xlim=(-.2,3.2),ylim=(.06,.138));P.light_grid(ax);top(ax,4)
 scores=D.obj('quality_model_parameters.json')['selection_diagnostics'];finish(fig,'10','质量模型分块选择',{'split_families':splits,'family_RMSE':stats,'GQ2_family_equal_score':scores['score_GQ2'],'family_weight':.25},'四种质量响应模型在留N、留D、留Q和留(N,D)四类切分中的平均RMSE。GQ2在四类切分中均最低，选模总分对四类切分各赋0.25权重。',src('quality_split_family_scores.csv','quality_model_parameters.json'),'曲线连接的是不同验证任务，不是时间趋势。69个折不能直接等权替代四类等权选模。')

@register('14')
def quality_band():
 fig,ax,pal=new((150,96),(.15,.96,.17,.82));a=D.table('quality_model_bootstrap.csv.gz');a=a[a.success];q=np.linspace(.1,1,301);p=D.qp();draw=a.c_Q.to_numpy()[:,None]*(1-q[None,:])**a.nu_Q.to_numpy()[:,None];lo,hi=np.quantile(draw,[.025,.975],axis=0);center=p['c_Q']*(1-q)**p['nu_Q']
 ax.fill_between(q,lo,hi,color=pal.fill['leaf_green'],alpha=.48,label='95% 条件区间');ax.plot(q,center,color=pal.line['indigo'],lw=1.45,label='GQ2 点估计');ax.scatter([1],[0],s=28,marker='o',facecolor='white',edgecolor=pal.line['indigo'],zorder=5,label='Q = 1 锚点');ax.set(xlim=(.08,1.02),ylim=(-.008,.36),xlabel='质量评分 Q',ylabel='质量惩罚 ΔL');top(ax,3)
 ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];gain=p['c_Q']*((1-ref)**p['nu_Q']-(1-ref-.1)**p['nu_Q']);gd=a.c_Q*((1-ref)**a.nu_Q-(1-ref-.1)**a.nu_Q)
 finish(fig,'14','质量惩罚条件区间',{'valid_draws':len(a),'c_Q':p['c_Q'],'nu_Q':p['nu_Q'],'Q_ref':ref,'delta_Q':.1,'gain_at_ref':float(gain),'gain_ci95':np.quantile(gd,[.025,.975]).tolist()},'GQ2质量惩罚随Q变化的中心曲线和95%分位带。区间来自以(N,D)单元为簇的500次既有bootstrap，经典参数固定；Q=1时质量惩罚归零。',src('quality_model_parameters.json','quality_model_bootstrap.csv.gz'),'区间表示当前半合成质量网格与固定经典参数条件下的不确定性，不是跨体系质量标尺的完整预测区间。')

@register('19')
def interactions():
 from matplotlib.colors import SymLogNorm
 fig,ax,pal=new((160,143),(.19,.86,.24,.88));a=D.table('domain_substitution_complementarity.csv');domains=D.obj('problem1_bridge.json')['domains'];short=['arXiv','Freelaw','NIH','PubMed C','Wiki','Math','GitHub','Philpapers','StackEx','Email','Books','Pile CC','Ubuntu','Europarl','HN','PubMed A','USPTO'];idx={d:i for i,d in enumerate(domains)};mat=np.full((17,17),np.nan);unc=[]
 for row in a.itertuples():
  i,j=idx[row.domain_i],idx[row.domain_j];i,j=min(i,j),max(i,j);mat[i,j]=row.standardized_interaction
  if row.relation=='uncertain':unc.append((j,i))
 bound=np.ceil(np.max(np.abs(a.standardized_interaction))/10)*10;norm=SymLogNorm(linthresh=5,vmin=-bound,vmax=bound,base=10)
 im=ax.imshow(np.ma.masked_invalid(mat),cmap=pal.cmap_div(),norm=norm,interpolation='nearest',aspect='equal')
 for x,y in unc:ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='none',edgecolor='#6F7571',hatch='///',linewidth=0))
 ax.set(xticks=range(17),xticklabels=short,yticks=range(17),yticklabels=short);ax.tick_params(axis='x',rotation=60,labelsize=7);ax.tick_params(axis='y',labelsize=7);P.clean_heatmap(ax,lw=.4)
 cb=P.colorbar(fig,im,ax,label='标准化二阶交互',ticks=[-100,-20,-5,0,5,20,100],pad=.04,shrink=.78)
 top(ax,1,[Patch(facecolor='white',edgecolor='#6F7571',hatch='///',lw=0)],['95% 区间跨 0'])
 finish(fig,'19','领域互补矩阵',{'pair_count':len(a),'relations':a.relation.value_counts().to_dict(),'domain_labels':dict(zip(short,domains)),'color_normalization':{'type':'symmetric_log','linear_threshold':5,'bound':bound}},'17个训练领域的136组标准化二阶交互。负值表示相对线性叠加的互补效应；斜线标记95%bootstrap区间跨0的关系，采用对称对数色阶显示不同幅度。',src('domain_substitution_complementarity.csv','problem1_bridge.json'),'75对归类互补、61对不确定，没有稳定识别的替代对。原始系数存在共线与尺度差异，本图为模型响应解释而非因果效应。领域缩写与原字段对应写入单图JSON。')

@register('23')
def isoloss():
 fig,ax,pal=new((155,100),(.14,.96,.18,.76));cp=D.cp();ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];target=float(D.pred(1.,150.,ref));ng=np.geomspace(.07,11.97,600);qs=[.4,ref,.8];cols=['ochre','indigo','forest_green'];styles=['--','-','-.'];counts=[]
 for q,c,ls in zip(qs,cols,styles):
  remain=target-cp['E']-cp['A']*ng**(-cp['alpha'])-D.quality(q);dg=np.full_like(ng,np.nan);ok=remain>0;dg[ok]=(cp['B']/remain[ok])**(1/cp['beta']);ok&=(dg>=10)&(dg<=600);counts.append(int(ok.sum()));ax.plot(ng[ok],dg[ok],color=pal.line[c],lw=1.6,ls=ls,label=f'Q = {q:.3f}' if q==ref else f'Q = {q:.1f}')
 ax.scatter([1],[150],s=35,marker='o',fc='white',ec=pal.line['indigo'],lw=1.0,zorder=5,label='参考状态');ax.plot([.07,11.97,11.97,.07,.07],[10,10,600,600,10],color=pal.line['base'],lw=.6,ls=':',label='B7 数值范围');use_log(ax);ax.set(xlim=(.065,13),ylim=(9,680),xlabel='参数量 N / 十亿',ylabel='训练数据量 D / 十亿 Token');top(ax,3)
 finish(fig,'23','等损失资源替代',{'target_loss':target,'reference':{'N_B':1,'D_B':150,'Q':ref},'Q_scenarios':qs,'valid_curve_nodes':counts,'conditions':{'p':'p_ref','mixture_strength_profile':0,'s_Q':1}},'固定参考配比及主质量映射下，三种质量水平达到同一Loss的N–D组合。空心点为N=1B、D=150B、Q=Qref的参考状态，点线边框为B7半合成网格范围；曲线由冻结模型解析计算。',src('generalized_scaling_parameters.json'),'提高质量使同等损失所需参数量或数据量下降。连续等值线没有新增实测点；仅展示B7数值范围，且N=0.07略低于B1最小N。')

@register('26')
def equivalence():
 fig,ax,pal=new((155,102),(.15,.96,.18,.77));a=D.table('quality_parameter_equivalence.csv');cols={.5:'forest_green',1:'indigo',1.5:'ochre'};stats=[]
 for s,c in cols.items():
  for dq,marker,ls in [(.05,'o','--'),(.1,'D','-')]:
   b=a[(a.s_Q==s)&(a.delta_Q==dq)].sort_values('Q_A');x=100*b.finite_N_multiplier_minus_1;y=100*b.finite_D_multiplier_minus_1
   ax.plot(x,y,color=pal.line[c],lw=.9,ls=ls,alpha=.75);ax.scatter(x,y,s=38,marker=marker,facecolor=pal.fill[c],edgecolor=pal.line[c],lw=.8,zorder=3)
 ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];row=a[(a.s_Q==1)&(a.delta_Q==.1)&np.isclose(a.Q_A,ref)].iloc[0];x=100*row.finite_N_multiplier_minus_1;y=100*row.finite_D_multiplier_minus_1
 ax.scatter([x],[y],s=84,marker='D',facecolor='white',edgecolor=pal.line['indigo'],lw=1.3,zorder=5)
 ax.annotate(f'{x:.1f}%, {y:.1f}%',xy=(x,y),xytext=(x+5,y-15),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6,'color':pal.line['indigo']},ha='left')
 handles=[Line2D([],[],c=pal.line[c],lw=1.5,label=f'sQ = {s:g}') for s,c in cols.items()]+[Line2D([],[],color=pal.line['base'],ls=ls,marker=m,mfc='white',label=f'ΔQ = {dq:.2f}') for dq,m,ls in [(.05,'o','--'),(.1,'D','-')]]+[Line2D([],[],color=pal.line['indigo'],ls='',marker='D',mfc='white',label='参考状态')];top(ax,3,handles,[h.get_label() for h in handles]);ax.set(xlim=(0,73),ylim=(0,112),xlabel='等价参数相对增幅 / %',ylabel='等价数据相对增幅 / %');P.light_grid(ax)
 finish(fig,'26','质量提升有限等价量',{'rows':len(a),'reference':row.to_dict(),'root_status_N':a.root_status_N.value_counts().to_dict(),'root_status_D':a.root_status_D.value_counts().to_dict()},'质量提升ΔQ=0.05或0.10所对应的有限参数与数据扩张。颜色为质量映射斜率sQ，线型和点形为ΔQ，每条短路径按基线质量0.4、Qref、0.8连接；白心菱形为主情景参考点。',src('quality_parameter_equivalence.csv','generalized_scaling_parameters.json'),'主情景ΔQ=0.10对应参数扩张37.30%、数据扩张56.95%。高质量且sQ=1.5时触及映射截断，路径回折。这里比较Loss等价，不是成本等价。')

@register('27')
def physical_gate():
 fig,ax,pal=new((160,148),(.19,.89,.17,.82));a=D.table('generalized_scaling_predictions.csv.gz').copy();a['gap']=a.prediction-a.irreducible_loss_E
 rows=sorted(set(zip(a.N_params_B,a.D_tokens_B,a.Q_A)));cols=[(p,s,f) for p in ['p_ref','p_star_sensitivity'] for s in [.5,1,1.5] for f in [0,.25,.5,1]];ri={r:i for i,r in enumerate(rows)};ci={c:i for i,c in enumerate(cols)};mat=np.full((len(rows),len(cols)),np.nan)
 for r in a.itertuples():mat[ri[(r.N_params_B,r.D_tokens_B,r.Q_A)],ci[(r.p_scenario,r.s_Q,r.mixture_strength_profile)]]=r.gap
 b=np.max(np.abs(mat));im=ax.imshow(mat,aspect='auto',cmap=pal.cmap_div(),norm=TwoSlopeNorm(vmin=-b,vcenter=0,vmax=b),interpolation='nearest');yy,xx=np.where(mat<0);ax.scatter(xx,yy,s=7,c=pal.line['indigo'],marker='o',zorder=3)
 ax.set(xticks=range(24),xticklabels=[f'{c[2]:g}' for c in cols],yticks=range(27),yticklabels=[f'{n:g} / {d:g} / {q:.2f}' for n,d,q in rows],xlabel='配比迁移强度 profile',ylabel='N / D / Q（N、D：十亿）');ax.tick_params(labelsize=7);ax.tick_params(axis='x',rotation=90)
 for k in [3.5,7.5,11.5,15.5,19.5]:ax.axvline(k,color='white',lw=1.2 if k==11.5 else .8)
 for k in [8.5,17.5]:ax.axhline(k,color='white',lw=1)
 for j,(p,s) in enumerate([(p,s) for p in ['p_ref','p_star'] for s in [.5,1,1.5]]):ax.text((4*j+1.5)/24,1.015,f'sQ={s:g}',transform=ax.transAxes,ha='center',fontsize=7)
 ax.text(.25,1.06,'参考配比',transform=ax.transAxes,ha='center',fontsize=8);ax.text(.75,1.06,'近优配比敏感性',transform=ax.transAxes,ha='center',fontsize=8)
 cb=P.colorbar(fig,im,ax,label='预测 Loss − E',pad=.035,shrink=.9)
 top(ax,1,[Line2D([],[],marker='o',lw=0,color=pal.line['indigo'],ms=3)],['低于不可约下界'],)
 # Reposition the symbol legend above the two table headers.
 ax.get_legend().set_bbox_to_anchor((0,1.12))
 finish(fig,'27','配比接口下界缺口',{'rows':len(a),'violations':int((a.gap<0).sum()),'min_prediction':float(a.prediction.min()),'E':float(a.irreducible_loss_E.iloc[0]),'profile_to_lambda_multiplier':float(D.raw('B7').val_loss.quantile(.75)-D.raw('B7').val_loss.quantile(.25)),'row_keys':rows,'column_keys':cols},'广义模型648个固定情景的不可约下界缺口。各列按配比、sQ及profile排列，各行为N–D–Q状态；深点标记预测Loss低于E的52个状态。颜色为有符号缺口，单位与Loss一致。',src('generalized_scaling_predictions.csv.gz','mixture_physical_gate.csv'),'profile为无量纲情景倍数，λ0=profile×B7 Loss IQR=profile×0.482225。52个违规状态均来自近优配比敏感性，故联合配比优化接口被阻断；参考配比主接口不受影响。')

@register('03')
def trajectories():
 fig,ax,pal=new((160,101),(.13,.88,.17,.83));a=D.raw('B1');norm=LogNorm(a.N_params_B.min(),a.N_params_B.max());cmap=LinearSegmentedColormap.from_list('q2_trajectory_seq',pal.cmap_seq()(np.linspace(.23,1,256)))
 for n,g in a.groupby('N_params_B'):
  g=g.sort_values('D_tokens_B');color=cmap(norm(n));ax.scatter(g.D_tokens_B,g.val_loss,s=8,color=color,edgecolors='none',alpha=.6,zorder=2);xx=np.geomspace(g.D_tokens_B.min(),g.D_tokens_B.max(),300);ax.plot(xx,D.l0(n,xx),color=color,lw=.95,zorder=3)
 use_log(ax,'x');ax.set(xlim=(.1,400),ylim=(2,4.9),xlabel='训练数据量 D / 十亿 Token',ylabel='验证损失 Loss')
 cb=P.colorbar(fig,plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax,label='参数量 N / 十亿',pad=.035);cb.set_ticks([.1,1,10]);cb.set_ticklabels(['0.1','1','10'])
 top(ax,2,[Line2D([],[],color=pal.line['base'],marker='o',lw=0,ms=3),Line2D([],[],color=pal.line['base'],lw=1)],['原始检查点','M0 拟合'])
 finish(fig,'03','原始训练曲线族',{'rows':len(a),'scales':sorted(a.N_params_B.unique().tolist()),'loss_range':[float(a.val_loss.min()),float(a.val_loss.max())]},'B1八种参数规模下的原始验证损失轨迹与M0拟合。点为全部1176个训练检查点，实线为冻结模型预测，参数量以同一顺序色阶表示。',rawsrc('B1')+src('classic_scaling_parameters.json'),'同一规模的检查点来自一条轨迹，不是独立实验重复。较大的模型在共同D处Loss较低，但图中不能据此推断同算力最优。')

@register('06')
def parameter_density():
 fig,ax,pal=new((151,109),(.14,.88,.17,.80));a=D.valid_classic();x=(a.alpha.to_numpy()-.34)*1e4;y=(a.beta.to_numpy()-.28)*1e4;kde=gaussian_kde(np.vstack([x,y]));padx=.1*np.ptp(x);pady=.1*np.ptp(y);gx,gy=np.meshgrid(np.linspace(x.min()-padx,x.max()+padx,160),np.linspace(y.min()-pady,y.max()+pady,160));z=kde(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape);z/=z.max();im=ax.contourf(gx,gy,z,levels=np.linspace(0,1,128),cmap=pal.cmap_seq());ax.scatter(x,y,s=9,facecolor='white',edgecolor=pal.line['blue_gray'],lw=.35,alpha=.72,label='有效 bootstrap')
 p=D.cp();ax.scatter([(p['alpha']-.34)*1e4],[(p['beta']-.28)*1e4],s=65,marker='D',facecolor=pal.fill['ochre'],edgecolor=pal.line['ochre'],lw=.8,label='原始点估计',zorder=5);ax.set(xlabel=r'$(\alpha - 0.34) \times 10^4$',ylabel=r'$(\beta - 0.28) \times 10^4$');P.colorbar(fig,im,ax,label='相对抽样密度',pad=.035,ticks=[0,.5,1]);top(ax,2)
 finish(fig,'06','经典参数联合分布',{'valid_draws':len(a),'failed_draws':500-len(a),'alpha_ci95':np.quantile(a.alpha,[.025,.975]).tolist(),'beta_ci95':np.quantile(a.beta,[.025,.975]).tolist(),'alpha_beta_correlation':float(a.alpha.corr(a.beta))},'经典标度律指数α、β的联合抽样分布。白心点为484次有效的规模簇bootstrap，菱形为原始点估计；填色是按最大值归一化的描述性核密度。坐标经平移和放大以显示微小变化。',src('classic_scaling_bootstrap.csv.gz','classic_scaling_parameters.json'),'500次中16次因独立规模簇太少失败。该分布反映当前B1及模型设定的稳定性，不是跨模型族普适指数的可信区域。')

@register('08')
def external_structure():
 fig,ax,pal=new((145,114),(.16,.96,.16,.86));a=D.table('classic_external_validation.csv');spec={'B2':('s','ochre'),'B4':('o','indigo'),'B5':('D','forest_green'),'B10':('^','blue_gray')}
 for r in a.itertuples():
  m,c=spec[r.source];ax.scatter(r.MAE,r.centered_MAE,s=53,marker=m,facecolor='white' if r.source in ('B2','B10') else pal.fill[c],edgecolor=pal.line[c],lw=1,zorder=4)
  ax.annotate(r.source+'（估算）' if r.source=='B10' else r.source,(r.MAE,r.centered_MAE),xytext=(-12,12) if r.source=='B2' else ((8,-14) if r.source=='B5' else ((12,-9) if r.source=='B10' else (8,9))),textcoords='offset points',fontsize=8,ha='right' if r.source=='B2' else 'left')
 xx=np.geomspace(.0003,2,50);ax.plot(xx,xx,color=pal.line['base'],lw=.7,ls='--',label='y = x');use_log(ax);ax.set(xlim=(.0003,2),ylim=(.0003,2),xlabel='绝对 Loss 的 MAE',ylabel='分组中心化 MAE / 对照 MAE');ax.set_aspect('equal',adjustable='box');handles,labels=ax.get_legend_handles_labels();handles.append(Line2D([],[],ls='',marker='s',mfc='white',mec=pal.line['base'],ms=5));labels.append('未中心化对照');top(ax,2,handles,labels)
 finish(fig,'08','外部误差结构诊断',{'records':a.to_dict('records')},'冻结经典标度律的外部误差结构。B4按family、B5按文献source中心化，纵轴为相应分组中心化MAE；B2/B10为空心未中心化对照，纵轴沿用其原MAE，落在对角线上不能证明中心化没有影响。B10另为估算对照，不是独立观测。',src('classic_external_validation.csv'),'B2与B10的导出中心化指标沿用原值，不能把其在对角线上理解为没有族间偏移。B10接近原点也不能当作独立大模型观测验证。')

@register('09')
def quality_increment_matrix():
 fig,ax,pal=new((156,166),(.20,.88,.11,.90));a=D.raw('B7');anchor=a[a.Q_score==1][['N_params_B','D_tokens_B','val_loss']].rename(columns={'val_loss':'anchor'});a=a.merge(anchor,on=['N_params_B','D_tokens_B']);a['increment']=a.val_loss-a.anchor;piv=a.pivot(index=['N_params_B','D_tokens_B'],columns='Q_score',values='increment').sort_index();mat=piv.to_numpy();b=max(abs(mat.min()),abs(mat.max()));im=ax.imshow(mat,aspect='auto',cmap=pal.cmap_div(),norm=TwoSlopeNorm(vmin=-b,vcenter=0,vmax=b),interpolation='nearest');ax.set(xticks=range(10),xticklabels=[f'{x:.1f}' for x in piv.columns],yticks=range(45),yticklabels=[f'{n:g} / {d:g}' for n,d in piv.index],xlabel='质量评分 Q',ylabel='N / D（十亿）');ax.tick_params(axis='y',labelsize=7)
 for j in np.arange(4.5,44,5):ax.axhline(j,color='white',lw=1.4)
 P.colorbar(fig,im,ax,label='Loss(Q) − Loss(1)',pad=.035,shrink=.85)
 finish(fig,'09','质量增量响应矩阵',{'resource_cells':len(piv),'Q_levels':piv.columns.tolist(),'increment_range':[float(mat.min()),float(mat.max())],'negative_increment_count':int((mat<0).sum())},'B7半合成网格中每个N–D单元相对Q=1的损失增量。每五行属于同一参数规模，十列为离散质量水平，白色为零增量。',rawsrc('B7'),'Q=1中心化仅用于描述同单元的响应，不进入留出验证预测器。原始响应总体随Q下降，但局部噪声导致少数负增量；不把数据修整为严格单调。')

@register('12')
def quality_calibration():
 fig,ax,pal=new((156,102),(.14,.96,.17,.82));a=D.cv_predictions();summ=[]
 for field,c,dx,mark,label in [('val_loss','forest_green',-.014,'o','半合成值：中位数与 IQR'),('predicted','indigo',.014,'D','留出预测：中位数与 IQR')]:
  g=a.groupby('Q_score')[field].quantile([.25,.5,.75]).unstack();x=g.index.to_numpy()+dx;y=g[.5].to_numpy();ax.errorbar(x,y,yerr=np.vstack([y-g[.25],g[.75]-y]),fmt=mark,color=pal.line[c],mfc=pal.fill[c],mec=pal.line[c],ms=4,capsize=2,lw=1,label=label);summ.append({'field':field,'quantiles':g.reset_index().to_dict('records')})
 ax.set(xlim=(.05,1.05),ylim=(2.02,3.35),xticks=np.arange(.1,1.01,.1),xlabel='质量评分 Q',ylabel='验证损失 Loss');top(ax,2)
 finish(fig,'12','离散质量条件分布',{'prediction_rows':len(a),'selection':'GQ2 / leave_cell','conditional_quantiles':summ},'GQ2留(N,D)验证中，各离散质量水平的半合成Loss与留出预测分布。点为中位数，竖线为资源单元之间的四分位区间，左右微移仅区分两种统计量。',src('quality_block_predictions.csv.gz'),'每个Q档包含45个不同N,D资源单元，IQR表示资源异质性，不是预测置信区间。横轴位置仅为可读性微移，不改变原始质量档。')

@register('13')
def residual_rain():
 fig,ax,pal=new((159,118),(.18,.96,.17,.75));a=D.table('quality_block_predictions.csv.gz');a=a[a.model=='GQ2'];splits=['leave_N','leave_D','leave_Q','leave_cell'];labels=['留 N','留 D','留 Q','留 (N,D)'];colors=['blue_gray','forest_green','leaf_green','indigo'];summ=[]
 for i,(s,c) in enumerate(zip(splits,colors)):
  vals=a[a.split_type==s].residual.to_numpy();kde=gaussian_kde(vals);h=kde.factor*vals.std();xx=np.linspace(vals.min()-2*h,vals.max()+2*h,250);den=kde(xx);den=den/den.max()*.30;ax.fill_between(xx,i+.05,i+.05+den,color=pal.fill[c],alpha=.65,edgecolor=pal.line[c],lw=.6)
  jitter=np.random.default_rng(2026+i).uniform(-.18,-.035,len(vals));ax.scatter(vals,i+jitter,s=4,c=pal.fill[c],alpha=.6,edgecolors='none');q1,med,q3=np.quantile(vals,[.25,.5,.75]);ax.plot([q1,q3],[i-.255,i-.255],color=pal.line[c],lw=2.4);ax.scatter([med],[i-.255],s=19,facecolor='white',edgecolor=pal.line[c],lw=.8,zorder=4);summ.append({'split_type':s,'n':len(vals),'median':med,'q25':q1,'q75':q3})
 ax.axvline(0,color=pal.line['base'],ls='--',lw=.65);ax.set(yticks=range(4),yticklabels=labels,ylim=(-.5,3.5),xlabel='预测 Loss − 半合成 Loss',xlim=(-.37,.26));ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
 handles=[Patch(fc=pal.fill['base'],ec=pal.line['base'],alpha=.6,label='核密度'),Line2D([],[],marker='o',ls='',color=pal.fill['base'],ms=3,label='逐点残差'),Line2D([],[],marker='o',mfc='white',color=pal.line['base'],lw=2,label='中位数与 IQR'),Line2D([],[],color=pal.line['base'],ls='--',lw=.65,label='零残差')];top(ax,2,handles,[h.get_label() for h in handles])
 finish(fig,'13','分块残差云雨分布',{'selection':'GQ2','split_summaries':summ},'GQ2在四种留出切分下的有符号残差分布。上半密度、下方逐点及中位数/IQR同时展示形状、尾部和中心位置；残差为预测减半合成Loss。',src('quality_block_predictions.csv.gz'),'每类包含450个留出预测，同一个原始样本在四种切分中重复出现；不同切分不能混池当1800个独立观测。密度是描述性平滑。')

@register('15')
def direction_swarm():
 fig,ax,pal=new((154,85),(.15,.96,.22,.81));summ=[]
 for i,(key,c) in enumerate([('B7','forest_green'),('B8','ochre')]):
  vals=np.array([spearmanr(g.Q_score,g.val_loss).statistic for _,g in D.raw(key).groupby(['N_params_B','D_tokens_B'])]);order=np.argsort(vals);ys=np.zeros(len(vals));bins={}
  # Deterministic density-informed stacking, while preserving every correlation.
  for j in order:
   b=int(round(vals[j]/.012));k=bins.get(b,0);bins[b]=k+1;ys[j]=(0 if k==0 else ((k+1)//2)*(.026 if k%2 else -.026))
  maxabs=np.max(np.abs(ys));ys=ys/max(1,maxabs/.28);ax.scatter(vals,i+ys,s=16,facecolor=pal.fill[c],edgecolor=pal.line[c],lw=.4,alpha=.82)
  med=float(np.median(vals));ax.plot([med,med],[i-.35,i+.35],color=pal.line[c],lw=1.4);summ.append({'source':key,'cells':len(vals),'rho_median':med,'rho_min':float(vals.min()),'rho_max':float(vals.max()),'positive':int((vals>0).sum()),'negative':int((vals<0).sum())})
 ax.axvline(0,color=pal.line['base'],ls='--',lw=.65);ax.set(xticks=[-1,-.5,0,.5,1],xlim=(-1.07,1.07),ylim=(-.52,1.52),yticks=[0,1],yticklabels=['B7 主质量网格','B8 压力网格'],xlabel='单元内 Spearman ρ(Q, Loss)');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
 handles=[Line2D([],[],marker='o',ls='',color=pal.line['base'],mfc=pal.fill['base'],label='一个 (N,D) 单元'),Line2D([],[],color=pal.line['base'],marker='|',ms=9,lw=0,label='中位数'),Line2D([],[],color=pal.line['base'],ls='--',lw=.65,label='零相关')];top(ax,3,handles,[h.get_label() for h in handles])
 finish(fig,'15','质量响应方向冲突',{'summaries':summ},'B7与B8各N–D单元内部Q和Loss的Spearman相关。每点一个资源单元，竖短线为中位数；纵向堆叠只用于分开重合点，横向相关值未经改动。',rawsrc('B7','B8'),'B7的45个单元均为负相关，B8的150个单元均为正相关。两者均为半合成资料，图说明输入支持关系冲突，不证明真实训练机制发生反转。')

@register('16')
def floor_boundary():
 fig,ax,pal=new((158,110),(.13,.87,.20,.78));a=D.raw('B8');ns=np.sort(a.N_params_B.unique());ds=np.sort(a.D_tokens_B.unique());ni={v:i for i,v in enumerate(ns)};di={v:i for i,v in enumerate(ds)};mat=np.full((len(ds),len(ns)),np.nan);nof=[];incomplete=[]
 for (n,d),g in a.groupby(['N_params_B','D_tokens_B']):
  floor=g[np.isclose(g.val_loss,.5)];i,j=di[d],ni[n]
  if len(floor):mat[i,j]=floor.Q_score.max()
  else:nof.append((j,i))
  if g.Q_score.nunique()<12:incomplete.append((j,i))
 cmap=pal.cmap_seq().copy();cmap.set_bad('#FFFFFF');im=ax.imshow(np.ma.masked_invalid(mat),origin='lower',aspect='auto',vmin=.05,vmax=float(np.nanmax(mat)),cmap=cmap,interpolation='nearest')
 for x,y in nof:ax.scatter(x,y,s=18,marker='o',fc='white',ec=pal.line['base'],lw=.55)
 for x,y in incomplete:ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,hatch='///',edgecolor='#969E99',lw=0))
 ax.set(xticks=range(len(ns)),xticklabels=[f'{n:g}' for n in ns],yticks=range(len(ds)),yticklabels=[f'{d:g}' for d in ds],xlabel='参数量 N / 十亿',ylabel='训练数据量 D / 十亿 Token');ax.tick_params(axis='x',rotation=45,labelsize=7.5)
 P.colorbar(fig,im,ax,label='Loss = 0.5 的最高 Q',pad=.035,ticks=[.05,.1,.2,.3,.4]);handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=pal.line['base'],label='未触及地板'),Patch(fc='white',ec='#969E99',hatch='///',label='Q 网格不完整')];top(ax,2,handles,[h.get_label() for h in handles])
 finish(fig,'16','截断地板边界',{'cells':len(ns)*len(ds),'floor_rows':int(np.isclose(a.val_loss,.5).sum()),'floor_cells':int(np.isfinite(mat).sum()),'no_floor_cells':len(nof),'incomplete_cells':len(incomplete),'finite_threshold_range':[float(np.nanmin(mat)),float(np.nanmax(mat))]},'B8各离散N–D单元中Loss恰为0.5时的最高Q。空心点表示从未触及该地板，斜线表示仅有部分Q格点；轴按原始离散水平等距排列。',rawsrc('B8'),'颜色是已给定离散Q网格中的阈值，不是连续物理边界。362条记录触地板，不能用平滑插值创造未观测的阈值。')

@register('18')
def transfer_profiles():
 fig,ax,pal=new((158,102),(.14,.95,.18,.80));a=D.table('mixture_transfer_parameters.csv');scales=['1M','60M','1B'];domains=a[a.role=='domain'].pivot(index='loss_domain',columns='scale',values='slope').reindex(columns=scales);comp=a[a.role=='composite'].set_index('scale').loc[scales,'slope'];color=pal.line['blue_gray']
 for d,row in domains.iterrows():ax.plot(range(3),row.to_numpy(),color=color,lw=.75,alpha=.38,marker='o',ms=2.8,mfc=pal.fill['blue_gray'],mec=color)
 ax.plot(range(3),comp,color=pal.line['indigo'],lw=1.7,marker='D',ms=5,mfc=pal.fill['indigo'],label='综合响应');ax.axhline(0,color=pal.line['base'],ls='--',lw=.65,label='零迁移斜率');ax.set(xticks=range(3),xticklabels=scales,xlim=(-.12,2.12),xlabel='检验模型规模',ylabel='R(p) → 标准化 Loss 的斜率')
 handles,labels=ax.get_legend_handles_labels();handles.insert(0,Line2D([],[],color=color,alpha=.45,lw=.8,marker='o',ms=3));labels.insert(0,'13 个验证域');top(ax,3,handles,labels)
 finish(fig,'18','配比响应跨规模路径',{'domain_count':len(domains),'domain_slopes':domains.reset_index().to_dict('records'),'composite_slopes':comp.to_dict(),'composite_1B_vs_1M':float(comp['1B']/comp['1M'])},'问题一归一化配比响应R(p)在1M、60M、1B检验集上的迁移斜率。细线逐一保留13个验证域，粗菱形线为综合响应，虚线为零斜率；横轴为离散检验尺度。',src('mixture_transfer_parameters.csv'),'综合方向为正但强度随规模减弱，1B综合斜率约为1M的31.24%。单域存在方向变化，不应把该图解释为A/B绝对Loss已经对齐。')

@register('21')
def mixture_distribution():
 fig,ax,pal=new((154,90),(.13,.96,.18,.78));r=D.rstar_draws();kde=gaussian_kde(r);bw=kde.factor*r.std();xx=np.linspace(r.min()-2*bw,r.max()+2*bw,500);dd=kde(xx);lo,med,hi=np.quantile(r,[.025,.5,.975]);point=D.obj('problem1_bridge.json')['R_star'];ax.fill_between(xx,0,dd,color=pal.fill['blue_gray'],alpha=.48,label='固定近优配比的抽样密度');ax.plot(xx,dd,color=pal.line['blue_gray'],lw=1)
 ax.axvline(0,color=pal.line['ochre'],ls='--',lw=1,label='参考响应 R = 0');ax.plot([lo,hi],[-.028,-.028],color=pal.line['indigo'],lw=3,label='95% 分位区间');ax.scatter([med],[-.028],marker='o',s=26,fc='white',ec=pal.line['indigo'],zorder=4,label='抽样中位数');ax.scatter([point],[0],marker='D',s=35,fc=pal.fill['ochre'],ec=pal.line['ochre'],zorder=4,label='原始点估计')
 ax.set(xlim=(xx.min(),xx.max()),ylim=(-.06,dd.max()*1.07),xlabel='近优配比响应 R(p*)',ylabel='概率密度');ax.yaxis.set_major_locator(MaxNLocator(4));top(ax,3)
 finish(fig,'21','近优配比响应不确定性',{'valid_draws':len(r),'R_star_point':point,'median':med,'ci95':[lo,hi],'nonnegative_draw_fraction':float(np.mean(r>=0))},'固定问题一近优配比p*，通过500组既有Scheffé重拟合系数、参考配比与归一化尺度计算R(p*)分布。底部线段为95%分位区间，白点为中位数，菱形为原始点估计，虚线为参考配比零响应。',src('problem1_mixture_response_bootstrap.npz','problem1_bridge.json'),'区间跨0，因此p*只作为敏感性情景，不是稳定优于参考配比的最终决策。没有把不同sQ或配比强度混池。')

@register('24')
def elasticity_ternary():
 fig,ax,pal=new((157,124),(.06,.87,.12,.86));a=D.table('elasticity_grid.csv');a=a[(a.p_scenario=='p_ref')&(a.mixture_strength_profile==0)].copy();v=a[['epsilon_N','epsilon_D','epsilon_Q']].to_numpy();shares=v/v.sum(1,keepdims=True);xy=P.to_ternary(shares);norm=LogNorm(a.N_params_B.min(),a.N_params_B.max());cmap=LinearSegmentedColormap.from_list('q2_ternary_seq',pal.cmap_seq()(np.linspace(.2,1,256)));P.ternary_frame(ax,['参数弹性份额','数据弹性份额','质量弹性份额'],fontsize=9)
 for sq,mark in [(.5,'o'),(1,'D'),(1.5,'^')]:
  mask=np.isclose(a.s_Q,sq);ax.scatter(xy[mask,0],xy[mask,1],c=a.loc[mask,'N_params_B'],cmap=cmap,norm=norm,marker=mark,s=31,edgecolor=pal.line['base'],lw=.35,alpha=.9,label=f'sQ = {sq:g}')
 # Label three share-grid values once per edge; these are ternary fractions.
 for val in [.2,.4,.6,.8]:
  pt=P.to_ternary(np.array([[1-val,0,val]]))[0];ax.text(pt[0],-.023,f'{val:.1f}',ha='center',va='top',fontsize=7,color='#69716D')
  left=P.to_ternary(np.array([[val,1-val,0]]))[0];ax.text(left[0]-.028,left[1],f'{val:.1f}',ha='right',va='center',fontsize=7,color='#69716D')
  right=P.to_ternary(np.array([[0,val,1-val]]))[0];ax.text(right[0]+.028,right[1],f'{val:.1f}',ha='left',va='center',fontsize=7,color='#69716D')
 cb=P.colorbar(fig,plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax,label='参数量 N / 十亿',pad=.015,shrink=.72,ticks=[.07,1,11.97]);cb.set_ticklabels(['0.07','1','11.97']);handles=[Line2D([],[],ls='',marker=m,mfc='white',mec=pal.line['base'],ms=5,label=f'sQ = {s:g}') for s,m in [(.5,'o'),(1,'D'),(1.5,'^')]];top(ax,3,handles,[h.get_label() for h in handles])
 winners=shares.argmax(1);counts={str(sq):{name:int(np.sum((a.s_Q.to_numpy()==sq)&(winners==i))) for i,name in enumerate(['N','D','Q'])} for sq in [.5,1,1.5]}
 finish(fig,'24','资源弹性组成',{'scenario_rows':len(a),'conditions':{'p':'p_ref','profile':0},'row_sum_error':float(np.max(np.abs(shares.sum(1)-1))),'dominance_counts':counts,'normalization':'epsilon_x / (epsilon_N+epsilon_D+epsilon_Q)','states':a[['N_params_B','D_tokens_B','Q_A','s_Q','epsilon_N','epsilon_D','epsilon_Q']].to_dict('records')},'固定参考配比且无配比迁移项时的N、D、Q总Loss弹性份额。每个点为一个N–D–Q状态，形状区分质量映射斜率sQ，颜色表示参数规模；三份额经归一化后和为1。',src('elasticity_grid.csv'),'这是相对弹性组成，不是原始弹性或预算比例。改变sQ会改变质量主导的状态数；三元图不加入有下界违规的近优配比情景。')

@register('25')
def improvement_field():
 fig,ax,pal=new((160,99),(.14,.87,.18,.79));p=D.cp();qp=D.qp();x=np.linspace(np.log10(.070542),np.log10(11.965825),180);y=np.linspace(np.log10(.1),np.log10(.9),140);X,Y=np.meshgrid(x,y);N,Q=10**X,10**Y;gN=p['alpha']*p['A']*N**(-p['alpha']);gQ=Q*qp['c_Q']*qp['nu_Q']*(1-Q)**(qp['nu_Q']-1);ratio=gQ/gN
 field_cmap=LinearSegmentedColormap.from_list('q2_field_light',pal.cmap_seq()(np.linspace(0,.72,256)));im=ax.pcolormesh(X,Y,ratio,cmap=field_cmap,norm=LogNorm(vmin=ratio.min(),vmax=ratio.max()),shading='gouraud',rasterized=False);cont=ax.contour(X,Y,ratio,levels=[1],colors=[pal.line['ochre']],linestyles='--',linewidths=1)
 xs=np.linspace(x.min()+.08,x.max()-.08,13);ys=np.linspace(y.min()+.05,y.max()-.10,6);XX,YY=np.meshgrid(xs,ys);nn,qq=10**XX,10**YY;u=p['alpha']*p['A']*nn**(-p['alpha']);vv=qq*qp['c_Q']*qp['nu_Q']*(1-qq)**(qp['nu_Q']-1);mag=np.sqrt(u*u+vv*vv);ax.quiver(XX,YY,u/mag,vv/mag,angles='xy',scale_units='xy',scale=15,color=pal.line['indigo'],width=.0032,headwidth=3.2,headlength=4,zorder=4)
 ax.set(xticks=np.log10([.1,.3,1,3,10]),xticklabels=['0.1','0.3','1','3','10'],yticks=np.log10([.1,.2,.4,.6,.9]),yticklabels=['0.1','0.2','0.4','0.6','0.9'],xlabel='参数量 N / 十亿（对数坐标）',ylabel='质量评分 Q（对数坐标）',xlim=(x.min(),x.max()),ylim=(y.min(),y.max()));ax.set_aspect('equal',adjustable='box')
 cb=P.colorbar(fig,im,ax,label='质量弹性 / 参数弹性',pad=.035,match_axes=True,width=.025);cb.set_ticks([.2,.5,1,2,5]);cb.set_ticklabels(['0.2','0.5','1','2','5']);cb.ax.yaxis.set_minor_formatter(NullFormatter());handles=[Line2D([],[],color=pal.line['indigo'],marker='>',lw=1,label='局部降 Loss 方向'),Line2D([],[],color=pal.line['ochre'],ls='--',label='两类弹性相等')];top(ax,2,handles,[h.get_label() for h in handles])
 finish(fig,'25','资源改进方向场',{'N_B_range':[float(10**x.min()),float(10**x.max())],'Q_range':[.1,.9],'D_B':150,'ratio_range':[float(ratio.min()),float(ratio.max())],'arrow_contract':'unit negative gradient in (log10 N, log10 Q), equal data aspect, length normalized','conditions':{'p':'p_ref','profile':0,'s_Q':1}},'固定D=150B、参考配比和主质量映射时，N–Q对数平面上的局部降Loss方向。箭头由两个对数资源导数归一化，长度只表示方向；底色为质量/参数弹性比，虚线为比值1。未计成本，不代表等预算最优方向。',src('generalized_scaling_parameters.json'),'两坐标均为无量纲相对变化尺度且数据纵横等比例，因此箭头可比较相对增幅下的局部效用；图未包含任何成本权重，不能读成“同等经费最优投入方向”。')
