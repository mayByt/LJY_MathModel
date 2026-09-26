"""Question 3: twenty figures from frozen results, one figure per invocation."""
from pathlib import Path
from functools import lru_cache
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter, LogLocator, NullFormatter, MaxNLocator, FuncFormatter
from common import plot as P
from style import theme
import config as C

R=C.RESULTS['q3']
CFG=json.loads((C.REPO/'问题三/config/problem3_v1.json').read_text())
COSTS=['exponential','power','logarithmic']
LABELS={'exponential':'指数型','power':'幂函数型','logarithmic':'对数型'}
KEYS={'exponential':'forest_green','power':'blue_gray','logarithmic':'ochre'}
@lru_cache(None)
def read(name):return pd.read_csv(R/name)
def main_rows():return read('optimal_allocations.csv').query("support_mode == 'operational_extended' and solver_success == True")
def path(cost='exponential',context=4096):return read('budget_paths.csv.gz').query('quality_cost_type == @cost and context_length == @context').sort_values('budget_FLOPs')
def anchor(cost='exponential',context=4096,budget=1e19):
    d=main_rows();return d[(d.quality_cost_type==cost)&(d.context_length==context)&(d.budget_FLOPs==budget)].iloc[0]
def qcost(q,name):
    s=CFG['cost_parameters'][name];a,b=s['amplitude'],s['shape']
    return a*({'exponential':lambda:np.exp(b*q),'power':lambda:q**b,'logarithmic':lambda:np.log1p(b*q)}[name]())
def landscape(cost='exponential',res=200):
    a=anchor(cost);ns=np.geomspace(.08,1.3,res);qs=np.linspace(a.Q0,.999999,res)
    nn,qq=np.meshgrid(ns,qs);dd=1e19/((6+CFG['eta']*4096)*nn*1e9+qcost(qq,cost)-qcost(a.Q0,cost))/1e9
    loss=a.param_E+a.param_A*nn**(-a.param_alpha)+a.param_B*dd**(-a.param_beta)+a.param_c_Q*(1-qq)**a.param_nu_Q
    mask=(dd<.1)|(dd>36000);loss[mask]=np.nan
    return a,ns,qs,nn,qq,dd,loss
def save(fig,number,title,caption,data=None,sources=(),stats=None):
    ident=f'Q3-{number:02d}';out=C.OUTPUT/'q3'/'data';out.mkdir(parents=True,exist_ok=True)
    if data is not None:
        data.to_csv(out/f'{ident}.csv',index=False)
    meta=stats or {}
    if data is not None:meta.update({'table':str(out/f'{ident}.csv'),'rows':len(data)})
    return P.save(fig,'q3',f'{ident}_{title}',stats=meta,caption=caption,sources=[R/s for s in sources])
def b_label(v):
    return f'1e{int(np.log10(v))}'
def log_budget(ax):
    ax.set_xscale('log');ax.set_xlim(1e19,1e24);ax.xaxis.set_major_locator(LogLocator(base=10,numticks=6));ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel('算力预算 / FLOPs')

def f03():
    fig,ax,p=P.figure('q3',(142,88));contexts=np.array(CFG['contexts']);xx=np.geomspace(contexts.min(),contexts.max(),400);ratio=CFG['eta']*xx/6
    ax.fill_between(xx,0,ratio,color=p.fill['attention'],alpha=.16)
    ax.plot(xx,ratio,color=p.line['attention'],lw=1.6)
    ax.scatter(contexts,CFG['eta']*contexts/6,s=36,c=p.fill['attention'],edgecolor=p.line['attention'],zorder=4,label='C7 上下文')
    ax.axhline(1,color=p.ref,lw=.8,ls='--',label='等成本')
    critical=6/CFG['eta'];ax.axvline(critical,color=p.line['indigo'],lw=.85,ls=':')
    ax.annotate('30,000',xy=(critical,1),xytext=(critical*1.65,.38),arrowprops={'arrowstyle':'-','color':p.ref,'lw':.6},fontsize=8)
    ax.set(xscale='log',yscale='log',xlabel='上下文长度 / Token',ylabel='注意力 / 训练成本',ylim=(.035,7),xlim=(1500,190000))
    P.legend_top(ax,ncol=2)
    save(fig,3,'注意力成本临界','注意力与训练成本之比为 ηL/6；五点来自C7，临界长度为30,000 Token。关系来自题设成本代理，不是拟合测量。',pd.DataFrame({'context':contexts,'cost_ratio':CFG['eta']*contexts/6}),['c7_context_audit.csv'],{'eta':CFG['eta'],'critical':critical})

def f05():
    fig,old,p=P.figure('q3',(160,106),layout=None);old.remove();axs=fig.subplots(1,3,sharex=True,sharey=True);fig.subplots_adjust(left=.105,right=.97,bottom=.31,top=.84,wspace=.17);frames=[]
    for ax,cost in zip(axs,COSTS):
        a,ns,qs,nn,qq,dd,ll=landscape(cost);delta=ll-a.predicted_loss
        levels=np.linspace(0,.6,257);im=ax.contourf(nn,qq,np.clip(delta,0,.6),levels=levels,cmap=p.cmap_seq(),extend='max')
        ax.contour(nn,qq,delta,levels=[.02,.05,.1,.2,.4],colors=[p.line['forest_green']],linewidths=.5,alpha=.65)
        ax.scatter(a.N_B,a.Q_A,s=40,facecolor='white',edgecolor=p.line['indigo'],marker='D',zorder=5,clip_on=False)
        ax.set(xscale='log',xlim=(.08,1.3),ylim=(a.Q0,1.012),xlabel='N / B');ax.set_title(LABELS[cost],fontsize=9,pad=7)
        ax.set_xticks([.1,.3,1],labels=['0.1','0.3','1']);ax.xaxis.set_minor_formatter(NullFormatter())
        frames.append(pd.DataFrame({'cost':cost,'N_B':nn.ravel(),'Q':qq.ravel(),'D_B':dd.ravel(),'delta_loss':delta.ravel()}))
    axs[0].set_ylabel('质量 Q');cax=fig.add_axes([.26,.14,.58,.027]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.15,.30,.45,.60]);cb.set_label('相对最优 Loss 增量');cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([],[],marker='D',mfc='white',mec=p.line['indigo'],ls='',label='中心最优解')],loc='upper center',bbox_to_anchor=(.54,.997),ncol=1)
    save(fig,5,'等预算损失地形','预算10^19 FLOPs、上下文4096。解析消去D；三成本共享网格、坐标和色条，等高线表示0.02/0.05/0.10/0.20/0.40 Loss增量。菱形是冻结中心解。连续色面是模型评估，不是新增训练观测。',pd.concat(frames),['optimal_allocations.csv'])

def f07():
    fig,ax,p=P.figure('q3',(148,106));frames=[]
    for cost in COSTS:
        d=path(cost);color=p.line[KEYS[cost]];ax.plot(d.N_B,d.D_B,color=color,lw=1.5,label=LABELS[cost])
        for qstate,mark in [('lower','s'),('inside','o'),('upper','^')]:
            mask={'lower':d.Q_A<=d.Q0+1e-5,'upper':d.Q_A>=.9999,'inside':(d.Q_A>d.Q0+1e-5)&(d.Q_A<.9999)}[qstate]
            ss=d[mask].iloc[::5];ax.scatter(ss.N_B,ss.D_B,s=19,marker=mark,facecolor='white',edgecolor=color,zorder=3)
        for i in [13,39]:
            ax.annotate('',xy=(d.N_B.iloc[i+2],d.D_B.iloc[i+2]),xytext=(d.N_B.iloc[i],d.D_B.iloc[i]),arrowprops={'arrowstyle':'->','color':color,'lw':1})
        frames.append(d)
    bound=CFG['support_modes']['strict_joint'];ax.add_patch(Rectangle((bound['N_min_B'],bound['D_min_B']),bound['N_max_B']-bound['N_min_B'],bound['D_max_B']-bound['D_min_B'],fc=p.fill['light_green'],alpha=.16,ec=p.line['leaf_green'],lw=.7,zorder=0))
    ax.set(xscale='log',yscale='log',xlabel='参数量 N / B',ylabel='训练数据 D / B Token')
    h,l=ax.get_legend_handles_labels();h += [Line2D([],[],marker=m,ls='',color=p.ink,mfc='white',label=lab) for m,lab in [('s','质量下界'),('o','内点'),('^','质量上界')]];l += ['质量下界','内点','质量上界'];h.append(Patch(fc=p.fill['light_green'],alpha=.35,ec=p.line['leaf_green'],label='N–D支持范围'));l.append('N–D支持范围')
    P.legend_top(ax,handles=h,labels=l,ncol=3)
    save(fig,7,'最优配置路径','上下文4096；每条路径61个预算节点，箭头指预算增加。淡绿矩形仅是N-D共同支持范围，质量Q不在此标签覆盖内；三种点形显示约束状态，路径为模型条件最优解。',pd.concat(frames),['budget_paths.csv.gz'])

def f08():
    d=path();fig,ax,p=P.figure('q3',(148,90));cols=['share_train','share_quality','share_attention'];ax.stackplot(d.budget_FLOPs,*[d[c] for c in cols],colors=[p.fill[k] for k in ['train','quality','attention']],labels=['训练','提质','注意力'],alpha=.86,linewidth=.5,edgecolor='white')
    log_budget(ax);ax.set(ylim=(0,1),ylabel='预算份额');ax.yaxis.set_major_formatter(PercentFormatter(1));P.legend_top(ax,ncol=3)
    save(fig,8,'预算组成路径','指数型提质成本、上下文4096；三项预算份额来自冻结路径。份额下降不等于绝对投入下降。',d[['budget_FLOPs']+cols],['budget_paths.csv.gz'])

def f10():
    d=read('bootstrap_intervals.csv').query('context_length == 4096');fig,ax,p=P.figure('q3',(150,96));ys=[];labs=[];frames=[]
    for i,(_,r) in enumerate(d.sort_values(['budget_FLOPs','quality_cost_type']).iterrows()):
        key=KEYS[r.quality_cost_type];ax.plot([r.N_B_q025,r.N_B_q975],[i,i],color=p.line[key],lw=1.4);ax.scatter(r.N_B_median,i,s=24,c=p.fill[key],ec=p.line[key],zorder=3)
        a=anchor(r.quality_cost_type,budget=r.budget_FLOPs);ax.scatter(a.N_B,i,s=42,marker='D',facecolor='none',ec=p.line['indigo'],zorder=4)
        ys.append(i);labs.append(f'{LABELS[r.quality_cost_type]} · '+b_label(r.budget_FLOPs))
    ax.set(xscale='log',yticks=ys,yticklabels=labs,xlabel='最优参数量 N / B',ylabel='成本类型与预算 / FLOPs');ax.invert_yaxis()
    h=[Line2D([],[],color=p.line['forest_green'],marker='o',label='中位数与95%区间'),Line2D([],[],color=p.line['indigo'],marker='D',mfc='none',ls='',label='中心解')];P.legend_top(ax,handles=h,labels=[x.get_label() for x in h],ncol=2)
    save(fig,10,'三档配置不确定性','上下文4096；每情景500组上游参数传播得到N的条件分位区间，菱形是冻结中心解。中高预算区间很窄，未为可见性人工加宽。',d,['bootstrap_intervals.csv','optimal_allocations.csv'])

def f11():
    d=main_rows();contexts=CFG['contexts'];rows=[];labels=[]
    for cost in COSTS:
        for budget in CFG['budgets']:
            v=d[(d.quality_cost_type==cost)&(d.budget_FLOPs==budget)].set_index('context_length').loc[contexts].predicted_loss;rows.append((v-v.iloc[0]).to_numpy());labels.append(f'{LABELS[cost]} · '+b_label(budget))
    arr=np.array(rows);fig,ax,p=P.figure('q3',(160,105));im=ax.imshow(arr,cmap=p.cmap_seq(),aspect='auto',vmin=0,interpolation='none');ax.set(xticks=range(5),xticklabels=['2,048','4,096','8,192','32,768','131,072'],yticks=range(9),yticklabels=labels,xlabel='上下文长度 / Token',ylabel='成本类型与预算 / FLOPs')
    P.clean_heatmap(ax,lw=1.4);P.colorbar(fig,im,ax,label='Loss 增量',pad=.04)
    for i in range(9):
        for j in range(5):ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',fontsize=7.5,color='white' if arr[i,j]>.58*arr.max() else p.ink)
    save(fig,11,'上下文损失惩罚','相对同成本、同预算的2048 Token情景计算Loss增量；格点为已求解情景，不作格点间插值。',pd.DataFrame(arr,index=labels,columns=contexts).reset_index(),['optimal_allocations.csv'])

def f12():
    d=main_rows().query('context_length == 4096').sort_values(['budget_FLOPs','quality_cost_type']);fig,ax,p=P.figure('q3',(150,98));labels=[]
    for i,(_,r) in enumerate(d.iterrows()):
        ax.plot([r.predicted_loss,r.baseline_Q0_loss],[i,i],c=p.ref,lw=1.3);ax.scatter(r.baseline_Q0_loss,i,c=p.fill['base'],s=28,marker='s');ax.scatter(r.predicted_loss,i,c=p.fill['quality'],ec=p.line['quality'],s=34)
        labels.append(LABELS[r.quality_cost_type]+' · '+b_label(r.budget_FLOPs))
    ax.set(yticks=range(len(d)),yticklabels=labels,xlabel='预测 Loss',ylabel='成本类型与预算 / FLOPs');ax.invert_yaxis();P.legend_top(ax,handles=[Line2D([],[],marker='s',color=p.fill['base'],ls='',label='基线质量'),Line2D([],[],marker='o',color=p.fill['quality'],ls='',label='联合提质')],labels=['基线质量','联合提质'],ncol=2)
    save(fig,12,'提质收益配对','上下文4096；同预算比较固定基线质量和联合优化提质后的Loss。重合点代表没有提质收益；模型等价不代表新增训练实验。',d[['budget_FLOPs','quality_cost_type','baseline_Q0_loss','predicted_loss']],['optimal_allocations.csv'])

def f14():
    d=read('bootstrap_optimal_allocations.csv.gz').query('context_length == 4096 and budget_FLOPs == 1e19');fig,ax,p=P.figure('q3',(142,88))
    for cost in COSTS:
        vals=np.sort(d[d.quality_cost_type==cost].Q_A.to_numpy());x=np.r_[vals[0],vals];y=np.arange(len(x))/len(vals);ax.step(x,y,where='post',color=p.line[KEYS[cost]],lw=1.5,label=LABELS[cost]);ax.scatter([vals[0],vals[-1]],[0,1],color=p.fill[KEYS[cost]],s=20,zorder=4)
    ax.set(xlabel='最优质量 Q',ylabel='经验累计比例',xlim=(.51,1.025),ylim=(-.02,1.04));ax.yaxis.set_major_formatter(PercentFormatter(1));P.legend_top(ax,ncol=3)
    save(fig,14,'质量边界点质量','预算10^19 FLOPs、上下文4096；各成本500组抽样。用ECDF保留质量下界和上界处的点质量，不用核密度抹平受约束解。',d[['draw_id','quality_cost_type','Q_A','Q0']],['bootstrap_optimal_allocations.csv.gz'])

def f15():
    d=read('bootstrap_optimal_allocations.csv.gz').query("context_length == 4096 and budget_FLOPs == 1e19 and quality_cost_type == 'exponential'");a=anchor();fig,ax,p=P.figure('q3',(138,100));im=ax.hexbin(d.N_B,d.D_B,gridsize=23,mincnt=1,cmap=p.cmap_seq(),xscale='log',yscale='log',linewidths=.15);ax.scatter(a.N_B,a.D_B,marker='D',s=38,facecolor='white',edgecolor=p.line['indigo'],label='中心解',zorder=3)
    ax.set(xlabel='最优参数量 N / B',ylabel='最优数据量 D / B Token');P.colorbar(fig,im,ax,label='抽样数',pad=.025);P.legend_top(ax,ncol=1);ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.3f}'));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'));ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
    save(fig,15,'资源配置联合分布','指数成本、上下文4096、预算10^19 FLOPs；500组具有相同draw_id的N-D联合配置，六边格颜色是抽样计数。',d[['draw_id','N_B','D_B']],['bootstrap_optimal_allocations.csv.gz'])

def f17():
    center=read('transition_points.csv').query("transition_class == 'primary'").sort_values(['quality_cost_type','context_length','event_type']);bs=read('transition_bootstrap.csv.gz').query("record_type == 'event' and transition_class == 'primary'");fig,ax,p=P.figure('q3',(160,125));records=[];labels=[]
    for i,(_,r) in enumerate(center.iterrows()):
        v=bs[(bs.quality_cost_type==r.quality_cost_type)&(bs.context_length==r.context_length)&(bs.event_type==r.event_type)].budget_mid
        lo,med,hi=np.quantile(v,[.025,.5,.975]);key=KEYS[r.quality_cost_type];ax.plot([lo,hi],[i,i],c=p.fill[key],alpha=.65,lw=6,solid_capstyle='round');ax.plot([r.budget_lower,r.budget_upper],[i,i],c=p.line['indigo'],lw=1.6);ax.scatter(r.budget_mid,i,c=p.line['indigo'],s=15,zorder=3)
        event='启动' if ('activation' in r.event_type or 'start' in r.event_type) else '饱和';labels.append(f'{LABELS[r.quality_cost_type]} · {r.context_length:,} · {event}');records.append({'cost':r.quality_cost_type,'context':r.context_length,'event':r.event_type,'center':r.budget_mid,'loc_low':r.budget_lower,'loc_high':r.budget_upper,'p025':lo,'p975':hi,'probability':r.stability_probability,'draws_with_event':len(v)})
    ax.set(xscale='log',yticks=range(len(center)),yticklabels=labels,xlabel='转移预算 / FLOPs');ax.invert_yaxis();P.legend_top(ax,handles=[Line2D([],[],color=p.fill['forest_green'],lw=6,label='95%条件区间'),Line2D([],[],color=p.line['indigo'],lw=1.5,marker='o',label='中心定位区间')],labels=['95%条件区间','中心定位区间'],ncol=2)
    save(fig,17,'结构转移双区间','主要质量转移；浅粗带为200组选中上游draw经局部格点近似优化得到的2.5–97.5%事件条件分位范围，深线为中心数值定位区间。事件未出现的draw未被强填，出现概率存于数据表。',pd.DataFrame(records),['transition_points.csv','transition_bootstrap.csv.gz'])

def f18():
    d=read('budget_paths.csv.gz');contexts=CFG['contexts'];mat=[];labs=[];bs=None
    for cost in COSTS:
        for ctx in contexts:
            v=path(cost,ctx);bs=v.budget_FLOPs.to_numpy();mat.append(np.where(v.Q_A<=v.Q0+1e-5,0,np.where(v.Q_A>=.9999,2,1)));labs.append(f'{LABELS[cost]} · {ctx:,}')
    arr=np.array(mat);fig,ax,p=P.figure('q3',(160,113));cm=ListedColormap([p.fill['base2'],p.fill['leaf_green'],p.fill['indigo']]);ax.imshow(arr,aspect='auto',cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),extent=(np.log10(bs[0]),np.log10(bs[-1]),len(labs)-.5,-.5),interpolation='nearest');ax.set(yticks=range(len(labs)),yticklabels=labs,xticks=range(19,25),xticklabels=[rf'$10^{{{n}}}$' for n in range(19,25)],xlabel='算力预算 / FLOPs',ylabel='成本类型与上下文 / Token')
    for sep in [4.5,9.5]:ax.axhline(sep,color='white',lw=2)
    ax.tick_params(axis='y',length=0);P.legend_top(ax,handles=[Patch(fc=p.fill[k],label=l) for k,l in [('base2','质量下界'),('leaf_green','内点'),('indigo','质量上界')]],labels=['质量下界','内点','质量上界'],ncol=3)
    save(fig,18,'质量约束状态','15条路径各61个预算节点；颜色为冻结解的质量下界、内点或上界状态。对数成本从研究预算下界已处上界，并不意味着更小预算不存在转移。',d[['budget_FLOPs','quality_cost_type','context_length','Q_A','Q0','active_set']],['budget_paths.csv.gz'])

def f21():
    d=read('sensitivity_summary.csv').query("context_length == 4096 and budget_FLOPs == 1e19 and quality_cost_type == 'exponential' and solver_success == True").copy();d=d[~d.dimension.eq('support')].sort_values('loss_change');fig,ax,p=P.figure('q3',(155,103));names={'quality_mapping_slope':'质量映射斜率','quality_cost_amplitude':'提质成本幅度','quality_cost_shape':'提质成本形状','eta':'注意力系数','quality_cap':'质量上限','quality_empirical_cap':'经验质量上限','cost_amplitude':'提质成本幅度倍数','cost_shape':'提质成本形状倍数','attention_eta':'注意力系数倍数'}
    for i,(_,r) in enumerate(d.iterrows()):
        k='neg' if r.loss_change<0 else 'pos';ax.plot([0,r.loss_change],[i,i],color=p.fill[k],lw=5,solid_capstyle='round');ax.scatter(r.loss_change,i,c=p.line[k],s=25,zorder=3)
    labels=[names.get(r.dimension,r.dimension)+' · '+str(r.setting) for _,r in d.iterrows()];ax.set(yticks=range(len(d)),yticklabels=labels,xlabel='Loss 变化');ax.invert_yaxis();P.zero_line(ax,axis='x');save(fig,21,'关键假设敏感性','指数成本、上下文4096、预算10^19 FLOPs。只使用solver_success=True的敏感性结果，相对基准Loss变化；正负是损失方向，不是实验因果效应。setting保持登记参数语义，成本倍数另见随图表。',d,['sensitivity_summary.csv'])

def f23():
    a=anchor(budget=1e22);d=read('domain_token_allocations.csv');d=d[d.scenario_id==a.scenario_id].sort_values('D_i_B');b=read('bootstrap_intervals.csv').query("context_length == 4096 and budget_FLOPs == 1e22 and quality_cost_type == 'exponential'").iloc[0];fig,ax,p=P.figure('q3',(148,126))
    lo=d.p_ref*b.D_B_q025;hi=d.p_ref*b.D_B_q975;ax.errorbar(d.D_i_B,np.arange(len(d)),xerr=[d.D_i_B-lo,hi-d.D_i_B],fmt='o',color=p.line['D'],mfc=p.fill['D'],capsize=2,ms=4,lw=.8,label='固定配比95%区间');ax.set(yticks=range(len(d)),yticklabels=d.domain,xscale='log',xlabel='领域数据量 / B Token',ylabel='训练领域');P.legend_top(ax,ncol=1)
    tab=d.assign(p025=lo,p975=hi);save(fig,23,'领域Token配额','指数成本、上下文4096、预算10^22 FLOPs。17域共享固定p_ref，区间只传播D的不确定性，不含配比估计误差。',tab,['domain_token_allocations.csv','bootstrap_intervals.csv'])

def f25():
    d=read('optimal_allocations.csv').query("support_mode == 'strict_joint'").copy();labels=[];arr=[]
    for cost in COSTS:
        for ctx in CFG['contexts']:
            row=d[(d.quality_cost_type==cost)&(d.context_length==ctx)].set_index('budget_FLOPs').loc[CFG['budgets']];arr.append(np.where(row.solver_success,0,np.where(row.N_B.isna(),1,2)));labels.append(f'{LABELS[cost]} · {ctx:,}')
    arr=np.array(arr);fig,ax,p=P.figure('q3',(141,117));cm=ListedColormap([p.fill['light_green'],p.fill['base2'],p.fill['ochre']]);ax.imshow(arr,aspect='auto',cmap=cm,vmin=-.5,vmax=2.5);ax.set(xticks=range(3),xticklabels=[rf'$10^{{{int(np.log10(x))}}}$' for x in CFG['budgets']],yticks=range(15),yticklabels=labels,xlabel='算力预算 / FLOPs',ylabel='成本类型与上下文 / Token');P.clean_heatmap(ax,lw=1.7)
    symbols={0:'○',1:'×',2:'△'}
    for i in range(15):
        for j in range(3):ax.text(j,i,symbols[arr[i,j]],ha='center',va='center',fontsize=9)
    P.legend_top(ax,handles=[Patch(fc=p.fill[k],label=l) for k,l in [('light_green','成功'),('base2','不可行'),('ochre','求解失败')]],labels=['成功','不可行','求解失败'],ncol=3)
    save(fig,25,'严格域求解状态','严格N-D域共45个情景，按solver_success及数值缺失区分成功、数学不可行和数值求解失败。失败记录即使带数值也不作为最优解。',d[['quality_cost_type','context_length','budget_FLOPs','solver_success','solver_message','N_B']],['optimal_allocations.csv'],{'counts':{str(k):int((arr==k).sum()) for k in range(3)}})

def f26():
    d=path();fig,ax,p=P.figure('q3',(148,88));ratioN=d.marginal_N/d.marginal_D;ratioQ=d.marginal_Q/d.marginal_D;ax.plot(d.budget_FLOPs,ratioN,c=p.line['N'],lw=1.6,label='参数 / 数据');ax.plot(d.budget_FLOPs,ratioQ,c=p.line['Q'],lw=1.6,ls='--',label='质量 / 数据');ax.axhline(1,c=p.ref,ls=':',lw=.7,label='边际相等');upper=d[d.Q_A>=.9999];
    if len(upper):ax.axvspan(upper.budget_FLOPs.min(),1e24,color=p.fill['light_green'],alpha=.22,label='质量上界')
    log_budget(ax);ax.set(yscale='log',ylabel='单位算力边际收益比');P.legend_top(ax,ncol=2)
    save(fig,26,'边际资源竞争','指数成本、上下文4096；N和Q的单位算力边际收益相对D归一化。质量内点才要求三者相等，质量上界后偏离1不代表求解错误。',d[['budget_FLOPs','marginal_N','marginal_D','marginal_Q','active_set']],['budget_paths.csv.gz'])

def f01():
    d=read('c7_context_audit.csv');fig,ax,p=P.figure('q3',(136,76));ctx=CFG['contexts'];counts=[]
    for i,n in enumerate(ctx):
        count=int((d.max_position_embeddings==n).sum());counts.append(count);ax.scatter(np.repeat(i,count),np.arange(count)+1,s=27,color=p.fill['forest_green'],edgecolor=p.line['forest_green'],linewidth=.5)
    ax.set(xticks=range(5),xticklabels=[f'{n:,}' for n in ctx],xlabel='上下文长度 / Token',ylabel='模型数量',ylim=(0,max(counts)+1.4));ax.yaxis.set_major_locator(MaxNLocator(integer=True));save(fig,1,'C7上下文分布','C7全部45条模型记录，一个圆点代表一个模型；五档上下文是附件支持的离散情景，不拟合连续密度。',pd.DataFrame({'context':ctx,'models':counts}),['c7_context_audit.csv'])

def f02():
    a=anchor();q=np.linspace(a.Q0,.999999,400);fig,ax,p=P.figure('q3',(140,85));records=[]
    for cost in COSTS:
        delta=(qcost(q,cost)-qcost(a.Q0,cost))/1e9;ax.plot(q,delta,color=p.line[KEYS[cost]],lw=1.6,label=LABELS[cost]);records.append(pd.DataFrame({'cost':cost,'Q':q,'increment_FLOPs_per_token':delta*1e9}))
    ax.set(xlabel='质量 Q',ylabel='提质增量成本 / (GFLOPs/Token)',xlim=(a.Q0,1));P.legend_top(ax,ncol=3);save(fig,2,'质量提升成本曲线','三种题设成本函数，相对冻结Q0的增量。单位为每Token的GFLOPs；曲线是题设成本代理，不是经验拟合。',pd.concat(records),['optimal_allocations.csv'])

def f04():
    fig,ax,p=P.figure('q3',(140,88));ctx=np.geomspace(2048,131072,400);b=CFG['support_modes']['strict_joint'];cmin=(6+CFG['eta']*ctx)*b['N_min_B']*b['D_min_B']*1e18;ax.fill_between(ctx,1e18,cmin,color=p.fill['base2'],alpha=.30,label='严格域不可行');ax.plot(ctx,cmin,c=p.line['indigo'],lw=1.6,label='最小可行预算')
    for budget,mark in zip(CFG['budgets'],['o','s','^']):ax.scatter(CFG['contexts'],np.repeat(budget,5),marker=mark,s=25,facecolor='white',edgecolor=p.line['forest_green'],zorder=4)
    ax.set(xscale='log',yscale='log',xlabel='上下文长度 / Token',ylabel='算力预算 / FLOPs',xlim=(1800,155000),ylim=(2e18,2e24));P.legend_top(ax,ncol=2);save(fig,4,'严格域可行预算边界','严格域N_min、D_min、基线质量Q0给出最低成本。15个空心符号为三预算×五上下文的题设格点；不可行区域与数值求解失败不是同一概念。',pd.DataFrame({'context':ctx,'minimum_budget':cmin}),[C.REPO/'问题三/config/problem3_v1.json'],{'N_min_B':b['N_min_B'],'D_min_B':b['D_min_B']})

def f06():
    fig,ax,p=P.figure('q3',(145,90));records=[]
    for cost in COSTS:
        a,ns,qs,nn,qq,dd,ll=landscape(cost,400);profile=np.nanmin(ll,axis=0)-a.predicted_loss;ax.plot(ns,np.maximum(profile,0),color=p.line[KEYS[cost]],lw=1.6,label=LABELS[cost]);records.append(pd.DataFrame({'cost':cost,'N_B':ns,'profile_delta_loss':profile}))
    ax.axhline(.01,color=p.ref,ls='--',lw=.8,label='ΔLoss = 0.01');ax.set(xscale='log',xlabel='参数量 N / B',ylabel='剖面 Loss 增量',ylim=(-.004,.15),xlim=(.1,1.1));ax.set_xticks([.1,.2,.3,.4,.6,1.],labels=['0.1','0.2','0.3','0.4','0.6','1']);ax.xaxis.set_minor_formatter(NullFormatter());P.legend_top(ax,ncol=2)
    save(fig,6,'近优配置剖面','预算10^19 FLOPs、上下文4096。每个N在Q网格上求最小Loss并减去冻结中心最优值；0.01为预先指定的决策容许增量，不是统计区间。',pd.concat(records),['optimal_allocations.csv'])

def f20():
    d=read('transition_bootstrap.csv.gz').query("record_type == 'event' and quality_cost_type == 'power' and context_length == 4096 and transition_class == 'primary'");v=d.pivot(index='draw_id',columns='event_type',values='budget_mid');names=list(v.columns);start=next((x for x in names if ('activation' in x or 'start' in x)),None);end='quality_saturation'
    if start is None:raise ValueError(names)
    pair=v[[start,end]].dropna();cnt=pair.groupby([start,end]).size().reset_index(name='draws');fig,ax,p=P.figure('q3',(139,97));sc=ax.scatter(cnt[start],cnt[end],s=20+cnt.draws*8,c=cnt.draws,cmap=p.cmap_seq(),edgecolor=p.line['forest_green'],linewidth=.5);P.colorbar(fig,sc,ax,label='配对抽样数',pad=.035);ax.set(xscale='log',yscale='log',xlabel='质量启动预算 / FLOPs',ylabel='质量饱和预算 / FLOPs');ax.set_xticks(sorted(cnt[start].unique()));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1e19:.2f}'));ax.set_xlabel('质量启动预算 / (10¹⁹ FLOPs)');ax.set_yticks(np.geomspace(cnt[end].min(),cnt[end].max(),4));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1e20:.2f}'));ax.set_ylabel('质量饱和预算 / (1e20 FLOPs)');ax.set_xlabel('质量启动预算 / (1e19 FLOPs)');ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
    save(fig,20,'质量启动与饱和配对','幂函数成本、上下文4096。按同一draw_id配对质量启动与饱和事件；面积与颜色均表示重复格点计数。使用既有200组局部格点近似，不添加随机抖动或平滑密度。',cnt,['transition_bootstrap.csv.gz'],{'paired_draws':len(pair),'start_event':start})

FIGS={1:f01,2:f02,3:f03,4:f04,5:f05,6:f06,7:f07,8:f08,10:f10,11:f11,12:f12,14:f14,15:f15,17:f17,18:f18,20:f20,21:f21,23:f23,25:f25,26:f26}
def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if not args:raise SystemExit('请指定单张图号，例如 python -m q3.run Q3-03')
    for name in args:
        number=int(name.split('-')[-1]);FIGS[number]()
if __name__=='__main__':main()
