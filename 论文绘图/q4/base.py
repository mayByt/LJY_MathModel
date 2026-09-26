"""Question 4: frozen evidence, calibrated links and uncertainty figures."""
from functools import lru_cache
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter,MaxNLocator,FuncFormatter
from scipy.special import expit,logit
from scipy.stats import gaussian_kde
from common import plot as P
from style import theme
import config as C
sys.path.insert(0,str(C.REPO/'问题四/src'))
from problem4.io import model_family

R=C.RESULTS['q4'];RAW=C.RAW/'C_efficiency_evolution'
TARGETS=['Average','IFEval','BBH','MATH','GPQA','MUSR','MMLU_PRO']
MAPPING=json.loads((R/'loss_benchmark_mapping.json').read_text())
@lru_cache(None)
def read(name):return pd.read_csv(R/name)
def linked(x):return (100+2*MAPPING['epsilon'])*expit(np.asarray(x))-MAPPING['epsilon']
def save(fig,number,title,caption,data=None,sources=(),stats=None):
    ident=f'Q4-{number:02d}';out=C.OUTPUT/'q4'/'data';out.mkdir(parents=True,exist_ok=True)
    if data is not None:data.to_csv(out/f'{ident}.csv',index=False)
    meta=stats or {}
    if data is not None:meta.update({'table':str(out/f'{ident}.csv'),'rows':len(data)})
    return P.save(fig,'q4',f'{ident}_{title}',stats=meta,caption=caption,sources=[R/s for s in sources])
def task_unique():
    d=read('task_level_aggregates.csv').dropna(subset=['official_bbh']).copy()
    assert len(d)==d.normalized_core.nunique()
    return d
def subtask_unique():
    d=read('task_level_aggregates.csv').sort_values('model_directory').drop_duplicates('normalized_core')
    t=read('c8_subtask_normalization.csv.gz').query("benchmark == 'BBH'").merge(d[['model_directory','normalized_core','model_name_json']],on='model_directory',validate='many_to_one')
    t['family']=t.model_name_json.map(model_family);return t
def interval(ax,row,y,color,center='median'):
    ax.plot([row.p025,row.p975],[y,y],c=color,lw=.85)
    ax.plot([row.p10,row.p90],[y,y],c=color,lw=4.6,solid_capstyle='round')
    ax.scatter(row[center],y,s=22,c=color,ec='white',lw=.45,zorder=4)
def interval_legend(ax,p,include_center=True):
    h=[Line2D([],[],color=p.line['base'],lw=4.6,label='80%区间'),Line2D([],[],color=p.line['base'],lw=.85,label='95%区间')]
    if include_center:h.insert(0,Line2D([],[],color=p.line['base'],marker='o',ls='',label='中心值'))
    P.legend_top(ax,handles=h,labels=[x.get_label() for x in h],ncol=len(h))

def f04():
    d=read('frontier_training_panel.csv');params=json.loads((R/'dynamic_frontier_parameters.json').read_text());xx=np.linspace(d.log10_compute.min()-.15,d.log10_compute.max()+.15,400)
    yy=linked(params['beta0']+params['beta_x_standardized_log10_compute']*((xx-params['x_mean'])/params['x_std'])+params['states'][-1])
    bs=read('frontier_bootstrap_draws.csv.gz').query('tau == 0.9 and fit_success == True');arr=linked(bs.beta0.to_numpy()[:,None]+bs.beta_x.to_numpy()[:,None]*(xx[None,:]-bs.x_mean.to_numpy()[:,None])/bs.x_std.to_numpy()[:,None]+bs.state_last.to_numpy()[:,None]);lo,hi=np.quantile(arr,[.1,.9],axis=0)
    fig,ax,p=P.figure('q4',(148,96));ax.fill_between(xx,lo,hi,fc=p.fill['blue_gray'],alpha=.23,label='80%条件区间');ax.plot(xx,yy,c=p.line['indigo'],lw=1.7,label='0.90条件前沿');ax.scatter(d.log10_compute,d['Average ⬆️'],s=27,c=p.fill['forest_green'],ec='white',lw=.45,zorder=3,label='基础模型')
    upper=d.dropna(subset=['c4_compute_upper']);upper=upper[upper.c4_compute_upper>0]
    for n,(_,r) in enumerate(upper.iterrows()):
        ux=np.log10(r.c4_compute_upper);ax.plot([r.log10_compute,ux],[r['Average ⬆️']]*2,c=p.line['forest_green'],lw=.8);ax.scatter(ux,r['Average ⬆️'],marker='|',c=p.line['forest_green'],s=38,label='单侧算力上界' if n==0 else None)
    ax.set(xlabel='训练算力 / lg(FLOPs)',ylabel='综合能力 / 分',ylim=(0,max(hi.max(),d['Average ⬆️'].max())+3));P.legend_top(ax,ncol=2)
    save(fig,4,'训练算力与能力前沿','41条正式pretrained记录；曲线固定2025-01状态，阴影为冻结bootstrap在相同状态末端的80%条件区间。仅2条有单侧算力上界的记录画短横线和竖端点，其余缺失不补造误差棒。观察性关联，不作因果解释。',pd.DataFrame({'log10_compute':xx,'frontier':yy,'p10':lo,'p90':hi}),['frontier_training_panel.csv','dynamic_frontier_parameters.json','frontier_bootstrap_draws.csv.gz'],{'observed_rows':len(d),'available_upper_bounds':len(upper),'conditional_month':'2025-01'})

def f06():
    d=task_unique();x=np.sort(d.bbh_abs_diff.to_numpy());y=np.arange(1,len(x)+1)/len(x);fig,ax,p=P.figure('q4',(142,88));ax.step(x,y,where='post',c=p.line['forest_green'],lw=1.5);ax.axvline(1e-6,c=p.line['ochre'],ls='--',lw=.8,label='精确容差');exact=(x<=1e-6).mean();ax.scatter(1e-6,exact,c=p.fill['ochre'],ec=p.line['ochre'],s=27,zorder=4);ax.annotate(f'{exact:.1%}',xy=(1e-6,exact),xytext=(8e-5,exact-.018),fontsize=8,arrowprops={'arrowstyle':'-','color':p.ref,'lw':.6})
    ax.set(xscale='symlog',xlabel='BBH 复现绝对误差 / 分',ylabel='累计比例',ylim=(.94,1.003));ax.set_xscale('symlog',linthresh=1e-6);ax.set_xlim(-2e-7,max(x)*1.5);ax.yaxis.set_major_formatter(PercentFormatter(1));P.legend_top(ax,ncol=1)
    save(fig,6,'BBH复现误差分布','1786个唯一可比模型的复现绝对误差ECDF，精确容差1e-6分。纵轴聚焦94%–100%累计尾段；1736模型精确复现，零误差未转成非零伪值。',pd.DataFrame({'absolute_error':x,'ecdf':y}),['task_level_aggregates.csv'],{'n':len(x),'exact_count':int((x<=1e-6).sum()),'tolerance':1e-6})

def f10():
    d=pd.read_csv(RAW/'loss_benchmark_bridge_expanded.csv').dropna(subset=['Val_Loss','LB_Average']).copy();d['family']=d.Model.map(model_family);eps=MAPPING['epsilon'];x=np.log(d.Val_Loss.to_numpy());y=logit((d.LB_Average.to_numpy()+eps)/(100+2*eps));w=np.where(d.Loss_Comparability.str.lower().str.startswith('high'),1.,MAPPING['medium_weight']);grid=np.linspace(d.Val_Loss.min(),d.Val_Loss.max(),300)
    def fit(weights):
        X=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(X*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0];slope=min(float(coef[1]),0);intercept=np.average(y-slope*x,weights=weights);return linked(intercept+slope*np.log(grid)),intercept,slope
    central,intercept,slope=fit(w);rng=np.random.default_rng(2026);families=np.sort(d.family.unique());pred=[]
    for _ in range(1000):
        sample=rng.choice(families,len(families),replace=True);counts=pd.Series(sample).value_counts();fw=d.family.map(counts).fillna(0).to_numpy(float);pred.append(fit(w*fw)[0])
    lo,hi=np.quantile(pred,[.025,.975],axis=0);fig,ax,p=P.figure('q4',(147,94));ax.fill_between(grid,lo,hi,fc=p.fill['blue_gray'],alpha=.22,label='95%条件区间');ax.plot(grid,central,c=p.line['indigo'],lw=1.5,label='线性桥接')
    high=w==1;ax.scatter(d.loc[~high,'Val_Loss'],d.loc[~high,'LB_Average'],s=24,marker='o',facecolor='white',ec=p.line['forest_green'],lw=.65,label='中可比性',zorder=3);ax.scatter(d.loc[high,'Val_Loss'],d.loc[high,'LB_Average'],s=34,marker='D',c=p.fill['ochre'],ec=p.line['ochre'],lw=.65,label='高可比性',zorder=4)
    ax.set(xlabel='验证 Loss',ylabel='综合能力 / 分',xlim=(1.60,2.89),ylim=(-1,55));P.legend_top(ax,ncol=2)
    save(fig,10,'Loss与综合能力桥接','C6的75条记录，按既有选定线性模型与权重（High=1、Medium=0.25）重建。95%条件带另按来源家族重采样1000次、seed=2026，固定模型选择；它是成图补算的参数带，不替代正式折外误差或联合预测区间。正逆链接使用V2的(100+2ε)expit−ε。',pd.DataFrame({'loss':grid,'center':central,'p025':lo,'p975':hi}),[RAW/'loss_benchmark_bridge_expanded.csv','loss_benchmark_mapping.json'],{'intercept':intercept,'slope':slope,'groups':len(families),'n':len(d),'bootstrap':1000,'seed':2026,'high':int(high.sum())})

def f16():
    main=read('loss_benchmark_predictions.csv').query("target == 'Average' and context_length == 4096 and quality_cost_type == 'exponential'").sort_values('budget_FLOPs');u=read('loss_benchmark_uncertainty.csv');records=[];fig,ax,p=P.figure('q4',(153,105));labels=[];mode_names={'upstream_only':'上游','bridge_only':'桥接','joint':'联合'};mode_keys={'upstream_only':'forest_green','bridge_only':'blue_gray','joint':'indigo'};i=0
    for _,r in main.iterrows():
        for mode in mode_names:
            v=u[(u.scenario_index==r.scenario_index)&(u.target=='Average')&(u.uncertainty_mode==mode)].iloc[0];interval(ax,v,i,p.line[mode_keys[mode]]);labels.append(f'1e{int(np.log10(r.budget_FLOPs))} · {mode_names[mode]}');records.append(v.to_dict()|{'budget_FLOPs':r.budget_FLOPs});i+=1
    ax.set(yticks=range(i),yticklabels=labels,xlabel='综合能力 / 分',ylabel='预算 / FLOPs · 不确定性来源');ax.invert_yaxis();interval_legend(ax,p)
    save(fig,16,'桥接不确定性来源','指数成本、上下文4096的三预算情景。分别展示upstream_only、bridge_only、joint经验区间；点为各模式中位数。区间宽度不可相加，不将该图解读为方差可加分解。',pd.DataFrame(records),['loss_benchmark_predictions.csv','loss_benchmark_uncertainty.csv'])

def f18():
    d=read('frontier_bootstrap_draws.csv.gz').query('tau == 0.9 and fit_success == True');center=read('contribution_decomposition.csv').query("estimate == 'central'").iloc[0];fig,old,p=P.figure('q4',(158,81));old.remove();axs=fig.subplots(1,2);records=[]
    for ax,field,key,lab in zip(axs,['scale_score','tech_score'],['blue_gray','ochre'],['规模关联贡献 / 分','非规模残余 / 分']):
        vals=d[field].dropna().to_numpy();lo,med,hi=np.quantile(vals,[.025,.5,.975]);xx=np.linspace(vals.min(),vals.max(),350);yy=gaussian_kde(vals)(xx);yy=.45*yy/yy.max();ax.fill_between(xx,0,yy,color=p.fill[key],alpha=.65,lw=.7,ec=p.line[key]);rngx=np.random.default_rng(2026);sel=np.arange(len(vals))[::3];ax.scatter(vals[sel],-.08+rngx.uniform(-.04,.04,len(sel)),s=3,c=p.line[key],alpha=.20,rasterized=True)
        ax.plot([lo,hi],[-.24,-.24],c=p.line[key],lw=2.8);ax.scatter(med,-.24,c=p.line[key],s=25,zorder=3);ax.scatter(center[field],-.24,marker='D',facecolor='white',ec=p.line['indigo'],s=32,zorder=4);ax.axvline(0,c=p.ref,lw=.7,ls='--');ax.set(yticks=[],xlabel=lab,ylim=(-.36,.56));ax.spines['left'].set_visible(False);ax.xaxis.set_major_locator(MaxNLocator(4));records.append({'field':field,'n':len(vals),'p025':lo,'median':med,'p975':hi,'center':center[field]})
    fig.legend(handles=[Line2D([],[],color=p.line['base'],lw=2.8,marker='o',label='中位数与95%区间'),Line2D([],[],color=p.line['indigo'],ls='',marker='D',mfc='white',label='中心估计')],loc='outside upper center',ncol=2)
    save(fig,18,'规模与非规模贡献分布','2024-06至2025-01，τ=0.90有效bootstrap；两面板使用各自横轴范围，不可比较面积尺度。散点只沿无量纲展示轴微抖，x值不改；区间与密度来自完整抽样。非规模残余被强正则压近零且区间跨零，不证明技术效应不存在。',pd.DataFrame(records),['frontier_bootstrap_draws.csv.gz','contribution_decomposition.csv'])

def f22():
    d=read('frontier_forecast.csv').query('tau == 0.9').sort_values(['horizon_months','scenario_factor'],ascending=[True,False]);fig,ax,p=P.figure('q4',(148,94));labels=[];keys={1.:'indigo',.5:'blue_gray',.25:'forest_green'};names={1.:'维持',.5:'减半',.25:'四分之一'}
    for i,(_,r) in enumerate(d.iterrows()):interval(ax,r,i,p.line[keys[r.scenario_factor]],'forecast_score');labels.append(f'{int(r.horizon_months)}个月 · {names[r.scenario_factor]}')
    ax.set(yticks=range(len(d)),yticklabels=labels,xlabel='综合能力前沿 / 分',ylabel='期限与算力增速',xlim=(28,100));ax.invert_yaxis();ax.axhline(2.5,color=p.grid,lw=.7);interval_legend(ax,p)
    save(fig,22,'增速放缓情景前沿','以2025-03-13为起点，12/24个月分别对应2026-03/2027-03。τ=0.90；中心采用重排后的forecast_score，粗细线为80%/95%经验区间。只有两个离散时点，不插补逐月轨迹。',d,['frontier_forecast.csv'],{'origin':'2025-03-13','tau':.9})

def f01():
    d=read('open_model_panel.csv');flags=pd.DataFrame({'权重核实':d.open_weight_verified.astype(bool),'算力可用':np.isfinite(d.log10_compute),'基础模型':d.type_group.eq('pretrained'),'严格许可':d.strict_unrestricted.astype(bool)});counts=flags.value_counts().sort_values(ascending=False);combos=[tuple(x) for x in counts.index];fig,old,p=P.figure('q4',(160,105),layout=None);old.remove();gs=fig.add_gridspec(2,1,height_ratios=[3,1.5],left=.18,right=.98,bottom=.12,top=.96,hspace=.06);ax=fig.add_subplot(gs[0]);mx=fig.add_subplot(gs[1],sharex=ax);x=np.arange(len(counts));ax.bar(x,counts.to_numpy(),color=p.fill['forest_green'],alpha=.83,width=.68);ax.set_yscale('log');ax.set_ylabel('模型数量');ax.tick_params(axis='x',bottom=False,labelbottom=False)
    for i,c in enumerate(combos):
        ys=np.flatnonzero(c);mx.scatter(np.repeat(i,4),range(4),c=p.grid,s=20)
        if len(ys):mx.plot([i,i],[ys.min(),ys.max()],c=p.line['indigo'],lw=1);mx.scatter(np.repeat(i,len(ys)),ys,c=p.line['indigo'],s=24)
    mx.set(yticks=range(4),yticklabels=flags.columns,xticks=[],xlabel='证据条件的互斥交集',ylim=(3.5,-.5));mx.tick_params(axis='y',length=0);mx.spines['bottom'].set_visible(False);mx.spines['left'].set_visible(False);ax.legend(handles=[Line2D([],[],marker='o',ls='',color=p.line['indigo'],label='满足'),Line2D([],[],marker='o',ls='',color=p.grid,label='不满足')],loc='upper right',ncol=2)
    tab=pd.DataFrame(combos,columns=flags.columns).assign(count=counts.to_numpy());save(fig,1,'开源证据交集','4,546条去重模型记录；每列是四个布尔条件的一个互斥交集，上方数量用对数轴。暗点=满足、淡点=不满足；四项不等于正式前沿的完整筛选规则。',tab,['open_model_panel.csv'],{'total':len(d)})

def f02():
    d=read('open_model_panel.csv').copy();d['month']=pd.to_datetime(d.submission_date,errors='coerce').dt.strftime('%Y-%m');d=d.dropna(subset=['month']);d['有算力']=np.isfinite(d.log10_compute);d['有权重证据']=d.open_weight_verified.astype(bool);d['有许可证']=d['Hub License'].notna();d['分数完整']=d.complete_score.astype(bool);cols=['有算力','有权重证据','有许可证','分数完整'];g=d.groupby('month')[cols].mean();fig,ax,p=P.figure('q4',(148,95));im=ax.imshow(g.to_numpy().T,aspect='auto',vmin=0,vmax=1,cmap=p.cmap_seq());ax.set(yticks=range(4),yticklabels=cols,xticks=range(len(g)),xticklabels=g.index,xlabel='提交月份');ax.tick_params(axis='x',rotation=45);P.clean_heatmap(ax,lw=.9);cb=P.colorbar(fig,im,ax,label='可用比例',pad=.035);cb.ax.yaxis.set_major_formatter(PercentFormatter(1));save(fig,2,'时间与证据覆盖','按提交月份计算各字段可用比例。月份行数存表，不将缺失当0分；权重核实是证据状态，不等于模型实际上不开放。',g.assign(n=d.groupby('month').size()).reset_index(),['open_model_panel.csv'])

def f03():
    d=read('open_model_panel.csv');types=['pretrained','posttrained','merge'];names=['基础模型','后训练模型','合并模型'];fig,ax,p=P.figure('q4',(137,81));records=[]
    for i,(t,lab,key) in enumerate(zip(types,names,['forest_green','blue_gray','indigo'])):
        v=d.loc[d.type_group.eq(t),'Average ⬆️'];q=np.quantile(v,[.1,.25,.5,.75,.9]);ax.plot([q[0],q[4]],[i,i],c=p.line[key],lw=1);ax.plot([q[1],q[3]],[i,i],c=p.fill[key],lw=8,solid_capstyle='round');ax.scatter(q[2],i,s=35,c=p.line[key],ec='white',lw=.5,zorder=3);records.append({'type':t,'n':len(v),'p10':q[0],'p25':q[1],'median':q[2],'p75':q[3],'p90':q[4]})
    ax.set(yticks=range(3),yticklabels=names,xlabel='综合能力 / 分',ylim=(-.55,2.55));ax.invert_yaxis();P.legend_top(ax,handles=[Line2D([],[],c=p.line['base'],marker='o',ls='',label='中位数'),Line2D([],[],c=p.fill['base'],lw=8,label='25%–75%'),Line2D([],[],c=p.line['base'],lw=1,label='10%–90%')],labels=['中位数','25%–75%','10%–90%'],ncol=3)
    save(fig,3,'模型类型分位分布','基础/后训练/合并类型的能力分位数；区间是观测分布分位范围，不是均值置信区间。未调整规模与家族，不将组差异解释为后训练因果收益。',pd.DataFrame(records),['open_model_panel.csv'])

def f07():
    d=task_unique();mean=(d.official_bbh+d.bbh_normalized_equal_pct)/2;diff=d.bbh_normalized_equal_pct-d.official_bbh;fig,ax,p=P.figure('q4',(142,92));ax.scatter(mean,diff,s=11,c=p.fill['forest_green'],ec='none',alpha=.45,rasterized=True);bias=float(diff.mean());sd=float(diff.std());ax.axhline(bias,c=p.line['indigo'],lw=1,label='平均偏差');ax.axhline(bias-1.96*sd,c=p.line['ochre'],ls='--',lw=.8,label='均值 ± 1.96 SD');ax.axhline(bias+1.96*sd,c=p.line['ochre'],ls='--',lw=.8);ax.set(xlabel='官方与复现均值 / 分',ylabel='复现 − 官方 / 分');P.legend_top(ax,ncol=2)
    save(fig,7,'BBH聚合一致性','1786唯一可比模型，Bland–Altman一致性诊断；差值零膨胀，均值±1.96SD仅为描述性参考，不宣称误差服从正态或覆盖95%。',pd.DataFrame({'mean':mean,'difference':diff}),['task_level_aggregates.csv'],{'bias':bias,'sd':sd})

def f08():
    d=subtask_unique();g=d.groupby('task')[['raw_score','normalized_score']].mean().sort_values('normalized_score');fig,ax,p=P.figure('q4',(160,140));ys=np.arange(len(g));ax.hlines(ys,g.normalized_score*100,g.raw_score*100,color=p.ref,lw=1);ax.scatter(g.raw_score*100,ys,c=p.fill['base'],marker='s',s=20,label='原始分');ax.scatter(g.normalized_score*100,ys,c=p.fill['forest_green'],ec=p.line['forest_green'],s=24,label='下界校正');labs=[x.replace('leaderboard_bbh_','').replace('_',' ') for x in g.index];ax.set(yticks=ys,yticklabels=labs,xlabel='任务得分 / 分');ax.tick_params(axis='y',labelsize=7.5);P.legend_top(ax,ncol=2)
    save(fig,8,'任务随机下界校正','对规范化模型标识去重后，在每个BBH子任务内对模型等权平均，再将0–1分数乘100。连接同一任务的原始与随机下界校正分，避免把两类得分混作同一量。',g.reset_index(),['c8_subtask_normalization.csv.gz','task_level_aggregates.csv'])

def f09():
    d=subtask_unique();top=d.groupby('family').normalized_core.nunique().nlargest(8).index;d=d[d.family.isin(top)];g=d.groupby(['task','family']).normalized_score.mean().unstack().reindex(columns=top);g=g.loc[g.mean(axis=1).sort_values().index];fig,ax,p=P.figure('q4',(165,147));im=ax.imshow(g.to_numpy()*100,cmap=p.cmap_seq(),vmin=0,vmax=100,aspect='auto');labs=[x.replace('leaderboard_bbh_','').replace('_',' ') for x in g.index];ax.set(xticks=range(len(top)),xticklabels=top,yticks=range(len(g)),yticklabels=labs,xlabel='模型家族');ax.tick_params(axis='x',rotation=45);ax.tick_params(axis='y',labelsize=7.5);P.clean_heatmap(ax,lw=.5);P.colorbar(fig,im,ax,label='下界校正分',pad=.03)
    save(fig,9,'家族任务能力指纹','模型去重后选模型数最多的8个家族；每任务内按模型等权平均，不调整规模。任务排序按所选家族平均分，仅描述能力结构。家族按原model_family命名规则推断，未作为因果分组。',g.reset_index(),['c8_subtask_normalization.csv.gz','task_level_aggregates.csv'],{'family_model_counts':d.groupby('family').normalized_core.nunique().to_dict()})

def f11():
    d=read('bridge_cv_results.csv');models=['constant','linear','spline','isotonic'];names={'constant':'常数','linear':'线性','spline':'样条','isotonic':'保序'};keys=['base','indigo','forest_green','ochre'];marks=['s','o','^','D'];fig,ax,p=P.figure('q4',(150,106))
    for j,(m,key,mark) in enumerate(zip(models,keys,marks)):
        v=d[d.model==m].set_index('target_name').loc[TARGETS];yy=np.arange(7)+(j-1.5)*.16;ax.errorbar(v.cv_mae,yy,xerr=v.cv_mae_se,fmt=mark,color=p.line[key],mfc=p.fill[key],ms=3.8,capsize=1.5,lw=.65,label=names[m])
    ax.set(yticks=range(7),yticklabels=[t.replace('_','-') for t in TARGETS],xlabel='折外 MAE / 分',ylabel='评测目标');ax.invert_yaxis();P.legend_top(ax,ncol=4)
    save(fig,11,'七目标桥接模型选择','各目标四种候选桥接模型的留来源家族验证MAE及折均值标准误；标准误用于描述折间变动，不是独立同分布样本的显著性检验。',d,['bridge_cv_results.csv'])

def f12():
    d=read('bridge_cv_predictions.csv.gz').query("target_name == 'Average' and model == 'linear'").copy();d['error']=d.prediction-d.actual;families=d.groupby('held_out_family').error.median().sort_values().index;fig,ax,p=P.figure('q4',(151,113));rng=np.random.default_rng(2026)
    for i,fam in enumerate(families):
        v=d[d.held_out_family==fam].error;ax.scatter(v,np.repeat(i,len(v))+rng.uniform(-.12,.12,len(v)),s=18,c=p.fill['forest_green'],ec=p.line['forest_green'],lw=.3,alpha=.72);ax.plot([v.median(),v.median()],[i-.22,i+.22],c=p.line['indigo'],lw=1.3)
    ax.axvline(0,c=p.ref,ls='--',lw=.7,label='零误差');ax.set(yticks=range(len(families)),yticklabels=families,xlabel='预测 − 观测 / 分',ylabel='留出家族');ax.invert_yaxis();P.legend_top(ax,handles=[Line2D([],[],c=p.line['indigo'],marker='|',ls='',ms=7,label='家族中位数'),Line2D([],[],c=p.ref,ls='--',label='零误差')],labels=['家族中位数','零误差'],ncol=2)
    save(fig,12,'留家族误差结构','Average线性桥接的折外误差，每点保留真实误差，抖动仅在家族展示轴；竖短线是家族误差中位数。图内分布不以可比性权重重新复制样本。',d,['bridge_cv_predictions.csv.gz'])

def f13():
    d=read('bridge_cv_results.csv').query('selected == True').set_index('target_name').loc[TARGETS];fig,ax,p=P.figure('q4',(144,93));xs=np.arange(7);ax.scatter(xs-.10,d.coverage80,c=p.fill['forest_green'],ec=p.line['forest_green'],s=35,label='名义80%',marker='o');ax.scatter(xs+.10,d.coverage95,c=p.fill['indigo'],ec=p.line['indigo'],s=34,label='名义95%',marker='D');ax.axhline(.8,c=p.line['forest_green'],ls='--',lw=.8);ax.axhline(.95,c=p.line['indigo'],ls=':',lw=.8);ax.set(xticks=xs,xticklabels=[x.replace('_','-') for x in TARGETS],ylabel='折外实际覆盖率',xlabel='评测目标',ylim=(.48,1.01));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='x',rotation=30);P.legend_top(ax,ncol=2)
    save(fig,13,'桥接区间覆盖率','每个目标的选中桥接模型，点为折外实际覆盖，虚线/点线分别是名义80%/95%。覆盖不足如实保留，不能把正式区间称为校准完美。',d.reset_index(),['bridge_cv_results.csv'])

def f14():
    d=read('loss_benchmark_predictions.csv').query("target == 'Average'").copy();fig,ax,p=P.figure('q4',(146,99));keymap={'high_supported':('forest_green','o','高可比范围'),'expanded_supported':('blue_gray','s','扩展范围'),'out_of_bridge_range':('ochre','^','范围外')}
    for status,(key,mark,label) in keymap.items():
        v=d[d.bridge_support==status];ax.errorbar(v.predicted_loss,v['median'],yerr=[v['median']-v.p10,v.p90-v['median']],fmt=mark,color=p.line[key],mfc=p.fill[key],ms=4,lw=.65,alpha=.8,label=label)
    ax.set(xlabel='上游最优 Loss',ylabel='映射综合能力 / 分');hh,ll=ax.get_legend_handles_labels();hh.append(Line2D([],[],c=p.line['base'],marker='|',lw=.8,label='80%区间'));ll.append('80%区间');P.legend_top(ax,handles=hh,labels=ll,ncol=2)
    save(fig,14,'最优Loss到能力映射','45个上游情景的Average联合传播中位数与80%经验区间。形状区分桥接范围，重合结果不人为抖动到不同Loss；此支持标签不等于上游N-D-Q全部观测支持。',d,['loss_benchmark_predictions.csv'])

def f15():
    d=read('loss_benchmark_predictions.csv').query("target == 'Average'");costs=['exponential','power','logarithmic'];names={'exponential':'指数','power':'幂函数','logarithmic':'对数'};contexts=sorted(d.context_length.unique());budgets=sorted(d.budget_FLOPs.unique());code={'high_supported':0,'expanded_supported':1,'out_of_bridge_range':2};arr=[];labels=[]
    for cost in costs:
        for context in contexts:
            v=d[(d.quality_cost_type==cost)&(d.context_length==context)].set_index('budget_FLOPs').loc[budgets];arr.append([code[x] for x in v.bridge_support]);labels.append(f'{names[cost]} · {context:,}')
    fig,ax,p=P.figure('q4',(144,119));cm=ListedColormap([p.fill['forest_green'],p.fill['blue_gray'],p.fill['ochre']]);ax.imshow(arr,cmap=cm,vmin=-.5,vmax=2.5,aspect='auto');ax.set(xticks=range(3),xticklabels=['1e19','1e22','1e24'],yticks=range(15),yticklabels=labels,xlabel='算力预算 / FLOPs',ylabel='成本类型与上下文 / Token');P.clean_heatmap(ax,lw=1.2);P.legend_top(ax,handles=[Patch(fc=p.fill[k],label=l) for k,l in [('forest_green','高可比范围'),('blue_gray','扩展范围'),('ochre','范围外')]],labels=['高可比范围','扩展范围','范围外'],ncol=3)
    save(fig,15,'能力映射证据地图','Average桥接支持范围的45情景地图；三类标签来自冻结bridge_support。高可比范围只代表Loss处于高等级桥接区间，仍受上游模型与质量支持边界限制。',d[['budget_FLOPs','context_length','quality_cost_type','bridge_support','final_evidence_tier']],['loss_benchmark_predictions.csv'])

def f20():
    d=read('c3_historical_sensitivity.csv');v=d.query("analysis == 'leave_one_family_out'").sort_values('year_coef');bs=d.query("analysis == 'bootstrap'").year_coef;lo,hi=np.quantile(bs,[.025,.975]);center=d.query("analysis == 'central'").year_coef.iloc[0];fig,ax,p=P.figure('q4',(145,117));ax.axvspan(lo,hi,color=p.fill['blue_gray'],alpha=.16,label='整体95%区间');ax.scatter(v.year_coef,range(len(v)),c=p.fill['forest_green'],ec=p.line['forest_green'],s=28,label='留一族系数');ax.axvline(0,c=p.ref,ls='--',lw=.7,label='零系数');ax.axvline(center,c=p.line['indigo'],ls=':',lw=.8,label='整体中心');ax.set(yticks=range(len(v)),yticklabels=v.excluded_family,xlabel='年系数 / (logit/年)',ylabel='排除家族');ax.invert_yaxis();P.legend_top(ax,ncol=2)
    save(fig,20,'历史方向留族敏感性','C3独立26条历史记录，点为排除各家族的年系数；浅带为整体bootstrap95%分位区间，不是每个留族点自己的区间。来源异质，不与现代前沿拼接成长趋势。',v,['c3_historical_sensitivity.csv'],{'global_p025':lo,'global_p975':hi,'central':center})

def f21():
    d=read('compute_growth_monthly.csv');summ=json.loads((R/'compute_growth_summary.json').read_text());dt=pd.to_datetime(d.month);end=dt.max();x=d.month_index.to_numpy();central=summ['endpoint_log10_compute']+summ['monthly_log10_slope']*(x-x.max());bs=read('compute_growth_bootstrap.csv.gz').query('fit_success == True');arr=bs.endpoint_log10_compute.to_numpy()[:,None]+bs.monthly_log10_slope.to_numpy()[:,None]*(x-x.max());lo,hi=np.quantile(arr,[.1,.9],axis=0);fig,ax,p=P.figure('q4',(150,94));ax.fill_between(dt,lo,hi,color=p.fill['blue_gray'],alpha=.22,label='80%趋势区间');ax.step(dt,d.log10_compute_q90,where='mid',c=p.line['forest_green'],lw=1,label='月度0.90算力');ax.scatter(dt,d.log10_compute_q90,c=p.fill['forest_green'],s=17,ec=p.line['forest_green'],lw=.3);ax.plot(dt,central,c=p.line['indigo'],lw=1.5,label='稳健趋势');ax.scatter([dt.iloc[-1]],[summ['endpoint_log10_compute']],marker='D',c=p.fill['ochre'],ec=p.line['ochre'],s=34,zorder=4,label='趋势端点');ax.set(xlabel='发布日期月份',ylabel='训练算力 / lg(FLOPs)');ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));P.legend_top(ax,ncol=2)
    save(fig,21,'月度算力前沿趋势','25个月的月度0.90算力与冻结稳健趋势；趋势端点与末月原始值分别显示，80%区间由算力趋势bootstrap传播。时间口径是C4发布日期，不与排行榜提交时间混同。',d.assign(trend=central,p10=lo,p90=hi),['compute_growth_monthly.csv','compute_growth_summary.json','compute_growth_bootstrap.csv.gz'])

def f23():
    d=read('frontier_forecast_draws.csv.gz').query('tau == 0.9');fig,ax,p=P.figure('q4',(145,95));records=[]
    styles={12:'-',24:'--'}
    for horizon in [12,24]:
        v=d[d.horizon_months==horizon].pivot(index='draw_id',columns='scenario_factor',values='forecast_score').dropna()
        for factor,key,lab in [(.5,'blue_gray','减半'),(.25,'forest_green','四分之一')]:
            vals=np.sort((v[1.]-v[factor]).to_numpy());cdf=np.arange(1,len(vals)+1)/len(vals);ax.step(vals,cdf,where='post',c=p.line[key],ls=styles[horizon],lw=1.4,label=f'{horizon}个月 · {lab}');records.append(pd.DataFrame({'horizon':horizon,'slowdown':factor,'loss_of_score':vals,'ecdf':cdf}))
    ax.axvline(0,c=p.ref,lw=.7,ls=':');ax.set(xlabel='相对维持增速的能力差 / 分',ylabel='累计概率',ylim=(0,1.01));ax.yaxis.set_major_formatter(PercentFormatter(1));P.legend_top(ax,ncol=2)
    save(fig,23,'算力减速的配对代价','固定同一draw_id、τ=0.90和期限，计算维持增速预测减去减速预测。ECDF来自配对联合抽样，不能用两个独立预测区间端点相减代替。',pd.concat(records),['frontier_forecast_draws.csv.gz'])

FIGS={1:f01,2:f02,3:f03,4:f04,6:f06,7:f07,8:f08,9:f09,10:f10,11:f11,12:f12,13:f13,14:f14,15:f15,16:f16,18:f18,20:f20,21:f21,22:f22,23:f23}
def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if not args:raise SystemExit('请指定单张图号，例如 python -m q4.run Q4-04')
    for name in args:FIGS[int(name.split('-')[-1])]()
if __name__=='__main__':main()
