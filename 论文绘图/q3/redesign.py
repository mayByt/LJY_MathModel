"""Reference V2: richer comparisons, light fills, readable data hierarchy."""
from .base import *
from style import palettes as PL
from matplotlib.colors import LogNorm
from matplotlib import patheffects
from scipy.interpolate import PchipInterpolator

def soft(p,key):return PL.SOFT[key]
def format_axes(ax,x=True,y=True):
    if y:P.light_grid(ax,'y')
    if ax.get_xscale()=='log':ax.xaxis.set_minor_formatter(NullFormatter())
    if ax.get_yscale()=='log':ax.yaxis.set_minor_formatter(NullFormatter())
def budget_text(v):return '10'+str(int(np.log10(v))).translate(str.maketrans('0123456789','⁰¹²³⁴⁵⁶⁷⁸⁹'))
def cost_line(ax,x,y,cost,mark=True):
    ls,m=PL.COST_STYLE[cost];return ax.plot(x,y,c=PL.COST[cost],lw=1.6,ls=ls,marker=m if mark else None,ms=4.3,markevery=max(1,len(x)//11),mfc=PL.SOFT[KEYS[cost]],mec=PL.COST[cost],mew=.7,label=LABELS[cost])[0]

def v03():
    ctx=np.array(CFG['contexts']);xx=np.geomspace(1200,210000,500);rr=CFG['eta']*xx/6;share=rr/(1+rr);fig,ax,p=P.figure('q3',(136,86));ax.fill_between(xx,0,share,color=soft(p,'attention'),alpha=.75);ax.fill_between(xx,share,1,color=soft(p,'train'),alpha=.38)
    ax.plot(xx,share,c=p.line['attention'],lw=1.7);ys=(CFG['eta']*ctx/6)/(1+CFG['eta']*ctx/6);ax.scatter(ctx,ys,s=45,c=p.fill['attention'],ec=p.line['attention'],lw=.8,zorder=4);ax.axhline(.5,c=p.ref,ls='--',lw=.8);ax.axvline(30000,c=p.line['indigo'],ls=':',lw=.9)
    ax.annotate('30,000',(30000,.5),xytext=(6800,.79),fontsize=8,color=p.ink,arrowprops={'arrowstyle':'-','color':p.ref,'lw':.7});ax.set(xscale='log',xlim=(1200,210000),ylim=(0,1),xlabel='上下文长度 L / Token',ylabel='注意力 / (训练 + 注意力)');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xticks([2048,8192,32768,131072],labels=['2k','8k','32k','128k']);P.light_grid(ax,'y');P.legend_top(ax,handles=[Patch(fc=soft(p,'attention'),label='注意力'),Patch(fc=soft(p,'train'),label='基础训练'),Line2D([],[],c=p.line['attention'],marker='o',ls='',label='C7 档位')],labels=['注意力','基础训练','C7 档位'],ncol=3)
    save(fig,3,'注意力成本临界','图型调整：成本比直线改为其等价份额S曲线，使长上下文挤压预算更易读。ηL/(6+ηL)达到50%时对应L=30,000；不含质量处理成本，五点来自C7。',pd.DataFrame({'context':xx,'attention_train_ratio':rr,'attention_share':share}),['c7_context_audit.csv'],{'eta':CFG['eta'],'critical':6/CFG['eta'],'visual_reference':'gmgc2026F/Q3-02'})

def v05():
    fig,axs,p=P.facets('q3',1,3,(160,78),sharey=True,wspace=.015);frames=[];images=[];floor=1e-5;maximum=.3
    for ax,cost in zip(axs[0],COSTS):
        a,ns,qs,nn,qq,dd,ll=landscape(cost,420);delta=np.maximum(ll-a.predicted_loss,0);ridx=np.nanargmin(ll,axis=1)
        im=ax.pcolormesh(ns,qs,np.maximum(delta,floor),shading='auto',cmap=p.cmap_seq().reversed(),norm=LogNorm(floor,maximum),rasterized=True);images.append(im)
        ax.contour(ns,qs,delta,levels=[.0001,.001,.01,.05,.1],colors=[p.line['slate_teal']],linewidths=.65,alpha=.9)
        ax.plot(ns[ridx],qs,c=p.ink,ls='--',lw=1.25);ax.scatter(a.N_B,a.Q_A,s=95,marker='*',facecolor='white',edgecolor=p.ink,linewidth=1.15,zorder=6,clip_on=False)
        P.facet_title(ax,LABELS[cost]+' · Q*='+f'{a.Q_A:.2f}');ax.set(xscale='log',xlim=(.08,1.3),ylim=(a.Q0,1.006),xlabel='参数量 N / B');ax.set_xticks([.1,1.],labels=['0.1','1']);ax.xaxis.set_minor_formatter(NullFormatter());frames.append(pd.DataFrame({'cost':cost,'N_B':nn.ravel(),'Q':qq.ravel(),'D_B':dd.ravel(),'delta_loss':delta.ravel()}))
    axs[0,0].set_ylabel('数据质量 Q');cb=fig.colorbar(images[0],ax=axs.ravel().tolist(),fraction=.022,pad=.015,aspect=28,ticks=[1e-5,1e-4,1e-3,1e-2,.1],extend='max');cb.set_label('Loss 增量（对数）');cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([],[],c=p.ink,ls='--',lw=1.25,label='内层最优路径'),Line2D([],[],marker='*',ls='',mfc='white',mec=p.ink,ms=8,label='最优解'),Line2D([],[],c=p.line['slate_teal'],lw=.65,label='等 Loss 增量')],loc='outside upper center',ncol=3)
    save(fig,5,'等预算损失地形','预算10^19 FLOPs、上下文4096，解析消去D；参考图的对数色标凸显近最优谷底，星形仍为本项目冻结解。色图下限1e-5仅用于显示，原始增量完整保存在表中；虚线为逐Q网格内最优N，不宣称全局认证。',pd.concat(frames),['optimal_allocations.csv'],{'visual_reference':'gmgc2026F/Q3-09','color_floor':floor,'color_max':maximum})

def v08():
    fig,axs,p=P.facets('q3',1,3,(160,79),sharex=True,sharey=True,wspace=.02);frames=[]
    for ax,cost in zip(axs[0],COSTS):
        d=path(cost);x=d.budget_FLOPs.to_numpy();bottom=np.zeros(len(d))
        for col,key in [('share_train','train'),('share_quality','quality'),('share_attention','attention')]:
            top=bottom+d[col].to_numpy();ax.fill_between(x,bottom,top,color=soft(p,key),ec=p.line[key],lw=.65,alpha=.93);bottom=top
        events=read('transition_points.csv').query("transition_class == 'primary' and context_length == 4096 and quality_cost_type == @cost")
        for c in events.budget_mid:ax.axvline(c,c=p.ink,lw=.8,ls=':')
        log_budget(ax);ax.set_ylim(0,1);ax.set_xticks([1e19,1e21,1e24]);P.facet_title(ax,LABELS[cost]);ax.yaxis.set_major_formatter(PercentFormatter(1));frames.append(d)
    axs[0,0].set_ylabel('算力预算份额');fig.legend(handles=[Patch(fc=soft(p,k),ec=p.line[k],label=l) for k,l in [('train','基础训练'),('quality','质量提升'),('attention','长文本注意力')]]+[Line2D([],[],c=p.ink,lw=.8,ls=':',label='质量转移点')],loc='outside upper center',ncol=4)
    save(fig,8,'预算组成路径','图型调整：将指数成本单条组成图扩为三种成本共享轴分面。上下文4096，竖虚线为冻结主要质量事件；大面积浅填充、深色边界。份额下降不等于绝对投入减少。',pd.concat(frames),['budget_paths.csv.gz','transition_points.csv'],{'visual_reference':'gmgc2026F/Q3-19'})

def v10():
    d=read('bootstrap_intervals.csv').query('context_length == 4096');order=[(c,b) for c in COSTS for b in CFG['budgets']];fig,axs,p=P.facets('q3',1,4,(160,104),sharey=True,wspace=.015)
    fields=[('N_B','参数量 N / B',True),('D_B','训练数据 D / B Token',True),('Q_A','数据质量 Q',False),('share_quality','提质预算份额',False)]
    for ax,(field,label,lg) in zip(axs[0],fields):
        for i,(cost,budget) in enumerate(order):
            r=d[(d.quality_cost_type==cost)&(d.budget_FLOPs==budget)].iloc[0];a=anchor(cost,budget=budget);key=KEYS[cost];ax.plot([r[field+'_q025'],r[field+'_q975']],[i,i],c=PL.COST[cost],lw=2.1,solid_capstyle='round');ax.scatter(r[field+'_median'],i,s=28,c=p.fill[key],ec=PL.COST[cost],lw=.7,zorder=5);ax.scatter(a[field],i,s=60,marker='D',facecolor='none',ec=PL.COST[cost],lw=.9,zorder=4)
        ax.set_xlabel(label);ax.set_ylim(8.65,-.65);ax.set_yticks(range(9));ax.grid(axis='x',color=p.grid,lw=.45)
        if lg:ax.set_xscale('log');ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlim(d[field+'_q025'].min()/1.5,d[field+'_q975'].max()*1.4)
        if field=='share_quality':ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlim(-.03,.42)
        if field=='Q_A':ax.set_xlim(.5,1.04);ax.set_xticks([.6,.8,1.])
    axs[0,0].set_yticklabels([LABELS[c]+' · '+budget_text(b) for c,b in order]);fig.legend(handles=[Line2D([],[],color=p.line['base'],marker='o',lw=2,label='中位数与95%区间'),Line2D([],[],color=p.line['base'],mfc='white',marker='D',ls='',label='中心解')]+[Line2D([],[],c=PL.COST[c],lw=2,label=LABELS[c]) for c in COSTS],loc='outside upper center',ncol=3)
    save(fig,10,'三档配置不确定性','图型调整：不再只画N，把N、D、Q与提质份额四个联动维度作为共享行的小多图。上下文4096；区间为每情景500组上游参数传播条件分位数；中高预算窄区间没有人为加宽。',d,['bootstrap_intervals.csv','optimal_allocations.csv'],{'visual_reference':'gmgc2026F/Q3-10'})

def v11():
    d=main_rows();contexts=CFG['contexts'];fig,axs,p=P.facets('q3',1,3,(160,66),sharey=True,wspace=.035);frames=[];maxv=0
    for ax,budget in zip(axs[0],CFG['budgets']):
        arr=[]
        for cost in COSTS:
            v=d[(d.quality_cost_type==cost)&(d.budget_FLOPs==budget)].set_index('context_length').loc[contexts].predicted_loss;arr.append((v-v.iloc[0]).to_numpy())
        arr=np.array(arr);im=ax.imshow(arr,cmap=p.cmap_seq_soft(),aspect='auto',vmin=0,vmax=.36);P.clean_heatmap(ax,lw=1.1);ax.set(xticks=range(5),xticklabels=['2k','4k','8k','32k','128k'],yticks=range(3),yticklabels=[LABELS[c] for c in COSTS],xlabel='上下文 / Token');P.facet_title(ax,'预算 '+budget_text(budget)+' FLOPs')
        for i in range(3):
            for j in range(5):ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',fontsize=7.7,color=p.ink)
        frames.append(pd.DataFrame(arr,index=COSTS,columns=contexts).reset_index().assign(budget=budget))
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.023,pad=.018);cb.set_label('Loss 增量');cb.outline.set_visible(False)
    save(fig,11,'上下文损失惩罚','同预算同成本下相对2048上下文的损失增量。三预算分面共享0–0.36柔和色标，文字呈现精确值，格点之间不插值。',pd.concat(frames),['optimal_allocations.csv'],{'visual_reference':'gmgc2026F/Q3-34'})

def v02():
    a=anchor();q=np.linspace(a.Q0,.999999,350);fig,axs,p=P.facets('q3',1,2,(152,76),sharex=True);rows=[]
    for cost in COSTS:
        delta=(qcost(q,cost)-qcost(a.Q0,cost))/1e9;s=CFG['cost_parameters'][cost];der={'exponential':lambda:s['amplitude']*s['shape']*np.exp(s['shape']*q),'power':lambda:s['amplitude']*s['shape']*q**(s['shape']-1),'logarithmic':lambda:s['amplitude']*s['shape']/(1+s['shape']*q)}[cost]()/1e9
        cost_line(axs[0,0],q,delta,cost);cost_line(axs[0,1],q,der,cost);rows.append(pd.DataFrame({'cost':cost,'Q':q,'increment_GFLOPs_per_token':delta,'derivative_GFLOPs_per_token':der}))
    for ax,label in zip(axs[0],['增量成本 / (GFLOPs/Token)','边际成本 / (GFLOPs/Token)']):ax.set(xlabel='质量 Q',ylabel=label,xlim=(a.Q0,1.));P.light_grid(ax,'y')
    fig.legend(handles=[Line2D([],[],color=PL.COST[c],ls=PL.COST_STYLE[c][0],marker=PL.COST_STYLE[c][1],label=LABELS[c]) for c in COSTS],loc='outside upper center',ncol=3)
    save(fig,2,'质量提升成本曲线','三种题设提质成本的增量与对Q的导数。以固定Q0为基线，每Token单位；是解析成本代理，曲线上的标记只为区分，不表示实验点。',pd.concat(rows),['optimal_allocations.csv',C.REPO/'问题三/config/problem3_v1.json'],{'visual_reference':'gmgc2026F/Q3-05a,b'})

def v07():
    fig,ax,p=P.figure('q3',(139,105));frames=[]
    bound=CFG['support_modes']['strict_joint'];ax.add_patch(Rectangle((bound['N_min_B'],bound['D_min_B']),bound['N_max_B']-bound['N_min_B'],bound['D_max_B']-bound['D_min_B'],fc=soft(p,'forest_green'),alpha=.45,ec=p.line['forest_green'],lw=.7,zorder=0))
    for cost in COSTS:
        d=path(cost);cost_line(ax,d.N_B,d.D_B,cost);frames.append(d)
        for b in CFG['budgets']:
            a=anchor(cost,budget=b);ax.scatter(a.N_B,a.D_B,s=47,marker=PL.COST_STYLE[cost][1],fc=p.fill[KEYS[cost]],ec=PL.COST[cost],zorder=4)
    for budget,offset in zip(CFG['budgets'],[(-24,-20),(-30,12),(-18,14)]):
        aa=anchor('exponential',budget=budget);ax.annotate(budget_text(budget),(aa.N_B,aa.D_B),xytext=offset,textcoords='offset points',fontsize=8,ha='right',color=p.ink,arrowprops={'arrowstyle':'-','color':p.ref,'lw':.6})
    ax.set(xscale='log',yscale='log',xlabel='参数量 N / B',ylabel='训练数据 D / B Token');P.light_grid(ax,'y');h,l=ax.get_legend_handles_labels();h.append(Patch(fc=soft(p,'forest_green'),ec=p.line['forest_green'],label='N–D共同范围'));l.append('N–D共同范围');P.legend_top(ax,handles=h,labels=l,ncol=2)
    save(fig,7,'最优配置路径','上下文4096；三条冻结预算路径以线型和点形区分，较大点为三档预算锚点，路径从左下到右上随预算增大。淡框仅表示N-D共同支持范围，不包括Q。',pd.concat(frames),['budget_paths.csv.gz','optimal_allocations.csv'],{'visual_reference':'gmgc2026F/Q3-16'})

def v06():
    fig,ax,p=P.figure('q3',(129,81));records=[]
    for cost in COSTS:
        a,ns,qs,nn,qq,dd,ll=landscape(cost,400);profile=np.nanmin(ll,axis=0)-a.predicted_loss;cost_line(ax,ns,np.maximum(profile,0),cost,False);ax.scatter(a.N_B,0,s=50,marker=PL.COST_STYLE[cost][1],fc='white',ec=PL.COST[cost],zorder=4,clip_on=False);records.append(pd.DataFrame({'cost':cost,'N_B':ns,'profile_delta_loss':profile}))
    ax.axhspan(0,.01,color=soft(p,'forest_green'),alpha=.45);ax.axhline(.01,c=p.ref,ls='--',lw=.8);ax.set(xscale='log',xlabel='参数量 N / B',ylabel='剖面 Loss 增量',ylim=(-.008,.15),xlim=(.1,1.1));ax.set_xticks([.1,.2,.3,.5,1.],labels=['0.1','0.2','0.3','0.5','1']);ax.xaxis.set_minor_formatter(NullFormatter());P.light_grid(ax,'y');h,l=ax.get_legend_handles_labels();h.append(Patch(fc=soft(p,'forest_green'),label='ΔLoss ≤ 0.01'));l.append('ΔLoss ≤ 0.01');P.legend_top(ax,handles=h,labels=l,ncol=2)
    save(fig,6,'近优配置剖面','低预算10^19、上下文4096。按Q网格最小化后的剖面与三种中心解，浅带表示预先声明的0.01容许Loss增量，不是统计区间。',pd.concat(records),['optimal_allocations.csv'])

def v12():
    d=main_rows().query('context_length == 4096').sort_values(['quality_cost_type','budget_FLOPs']);fig,axs,p=P.facets('q3',1,2,(150,100),sharey=True,wspace=.02);labels=[];rows=[]
    for i,(cost,budget) in enumerate([(c,b) for c in COSTS for b in CFG['budgets']]):
        r=d[(d.quality_cost_type==cost)&(d.budget_FLOPs==budget)].iloc[0];gain=r.baseline_Q0_loss-r.predicted_loss;col=PL.COST[cost];axs[0,0].plot([r.predicted_loss,r.baseline_Q0_loss],[i,i],c=p.ref,lw=1.4);axs[0,0].scatter(r.baseline_Q0_loss,i,s=31,marker='s',fc='white',ec=col);axs[0,0].scatter(r.predicted_loss,i,s=38,c=col);axs[0,1].barh(i,gain,fc=PL.SOFT[KEYS[cost]],ec=col,lw=.8,height=.53);labels.append(LABELS[cost]+' · '+budget_text(budget));rows.append(r.to_dict()|{'loss_gain':gain})
    for ax in axs[0]:ax.set(yticks=range(9),ylim=(8.65,-.65));ax.grid(axis='x',color=p.grid,lw=.45)
    axs[0,0].set_yticklabels(labels);axs[0,0].set_xlabel('预测 Loss');axs[0,1].set_xlabel('提质收益 ΔLoss');fig.legend(handles=[Line2D([],[],marker='s',mfc='white',mec=p.ink,ls='',label='基线质量'),Line2D([],[],marker='o',color=p.ink,ls='',label='联合优化')],loc='outside upper center',ncol=2)
    save(fig,12,'提质收益配对','上下文4096；左侧同预算基线与联合优化配对，右侧精确损失收益补足点距离难以比较的问题。零收益如实为零，条件结果不外推为独立训练收益。',pd.DataFrame(rows),['optimal_allocations.csv'])

def v17():
    center=read('transition_points.csv').query("transition_class == 'primary'").sort_values(['quality_cost_type','context_length','event_type']);bs=read('transition_bootstrap.csv.gz').query("record_type == 'event' and transition_class == 'primary'");fig,ax,p=P.figure('q3',(160,110));records=[];labels=[]
    for i,(_,r) in enumerate(center.iterrows()):
        v=bs[(bs.quality_cost_type==r.quality_cost_type)&(bs.context_length==r.context_length)&(bs.event_type==r.event_type)].budget_mid;lo,med,hi=np.quantile(v,[.025,.5,.975]);col=PL.COST[r.quality_cost_type];ax.plot([lo,hi],[i,i],c=PL.SOFT[KEYS[r.quality_cost_type]],lw=7,solid_capstyle='round');ax.plot([r.budget_lower,r.budget_upper],[i,i],c=col,lw=1.8);ax.scatter(r.budget_mid,i,c=col,s=26,zorder=4);event='启动' if 'start' in r.event_type else '饱和';labels.append(f'{LABELS[r.quality_cost_type]} · {int(r.context_length)//1024}k · {event}');records.append({'cost':r.quality_cost_type,'context':r.context_length,'event':event,'center':r.budget_mid,'p025':lo,'p975':hi,'probability':r.stability_probability})
    ax.set(xscale='log',yticks=range(len(center)),yticklabels=labels,xlabel='转移预算 / FLOPs',ylim=(len(center)-.4,-.6));ax.grid(axis='x',color=p.grid,lw=.45);ax.xaxis.set_minor_formatter(NullFormatter());P.legend_top(ax,handles=[Line2D([],[],c=PL.SOFT['forest_green'],lw=7,label='95%条件范围'),Line2D([],[],c=PL.COST['exponential'],marker='o',lw=1.8,label='中心与定位范围')],labels=['95%条件范围','中心与定位范围'],ncol=2)
    save(fig,17,'结构转移双区间','主要质量事件：浅宽条是既有200组局部格点近似优化的事件条件95%分位范围，深线/点为冻结中心定位范围与点估计。不是将数值定位宽度当置信区间。',pd.DataFrame(records),['transition_points.csv','transition_bootstrap.csv.gz'])

def v18():
    d=read('budget_paths.csv.gz');fig,axs,p=P.facets('q3',1,3,(160,72),sharex=True,sharey=True,wspace=.03);cm=ListedColormap([soft(p,'slate_teal'),p.fill['light_green'],p.fill['forest_green']]);labels=[f'{v//1024}k' for v in CFG['contexts']]
    for ax,cost in zip(axs[0],COSTS):
        arr=[]
        for ctx in CFG['contexts']:
            v=path(cost,ctx);arr.append(np.where(v.Q_A<=v.Q0+1e-5,0,np.where(v.Q_A>=.9999,2,1)))
        ax.imshow(arr,aspect='auto',cmap=cm,vmin=-.5,vmax=2.5,extent=(19-5/120,24+5/120,4.5,-.5),interpolation='nearest');ax.set(xlim=(19,24),yticks=range(5),yticklabels=labels,xticks=[19,21,24],xticklabels=['10¹⁹','10²¹','10²⁴'],xlabel='预算 / FLOPs');P.facet_title(ax,LABELS[cost]);ax.tick_params(length=0)
        for yy in [.5,1.5,2.5,3.5]:ax.axhline(yy,c='white',lw=1)
    axs[0,0].set_ylabel('上下文 / Token');fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip(cm.colors,['质量下界','内点投入','质量上界'])],loc='outside upper center',ncol=3)
    save(fig,18,'质量约束状态','三成本分面、五上下文共享预算轴。离散色标直接表示冻结路径的质量约束状态；不将离散路径插值为连续观测，也不把起点已饱和解释为从未转移。',d[['budget_FLOPs','quality_cost_type','context_length','Q_A','Q0','active_set']],['budget_paths.csv.gz'])

def v25():
    d=read('optimal_allocations.csv').query("support_mode == 'strict_joint'");fig,axs,p=P.facets('q3',1,3,(155,74),sharey=True,wspace=.03);cm=ListedColormap([soft(p,'forest_green'),soft(p,'slate_teal'),soft(p,'ochre')]);symbols={0:'○',1:'×',2:'△'};records=[]
    for ax,cost in zip(axs[0],COSTS):
        arr=[]
        for ctx in CFG['contexts']:
            v=d[(d.quality_cost_type==cost)&(d.context_length==ctx)].set_index('budget_FLOPs').loc[CFG['budgets']];arr.append(np.where(v.solver_success,0,np.where(v.N_B.isna(),1,2)))
        arr=np.array(arr);ax.imshow(arr,aspect='auto',cmap=cm,vmin=-.5,vmax=2.5);ax.set(xticks=range(3),xticklabels=['10¹⁹','10²²','10²⁴'],yticks=range(5),yticklabels=[f'{v//1024}k' for v in CFG['contexts']],xlabel='预算 / FLOPs');P.clean_heatmap(ax,lw=1.5);P.facet_title(ax,LABELS[cost])
        for i in range(5):
            for j in range(3):ax.text(j,i,symbols[arr[i,j]],fontsize=11,ha='center',va='center',color=p.line['forest_green'] if arr[i,j]==0 else p.line['ochre'])
    axs[0,0].set_ylabel('上下文 / Token');fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip(cm.colors,['成功','预算不可行','数值失败'])],loc='outside upper center',ncol=3)
    save(fig,25,'严格域求解状态','三成本、五上下文和三预算的严格域结果。33成功、3不可行、9有数值但求解失败，失败值不作最优解展示。',d[['quality_cost_type','context_length','budget_FLOPs','solver_success','solver_message','N_B']],['optimal_allocations.csv'])

# Remaining figures receive explicit reference-driven layout/mark upgrades below.
def v01():
    d=read('c7_context_audit.csv');ctx=CFG['contexts'];cnt=[int((d.max_position_embeddings==x).sum()) for x in ctx];fig,ax,p=P.figure('q3',(118,79));keys=['light_green','leaf_green','forest_green','blue_gray','indigo']
    for i,(n,key) in enumerate(zip(cnt,keys)):
        ax.scatter(np.repeat(i,n),np.arange(n)+1,s=32,fc=p.fill[key],ec=p.line[key],lw=.65);ax.text(i,n+1.05,str(n),ha='center',va='bottom',fontsize=8,color=p.line[key])
    ax.set(xticks=range(5),xticklabels=['2k','4k','8k','32k','128k'],xlabel='上下文长度 / Token',ylabel='模型数量',ylim=(0,max(cnt)+2.5));ax.yaxis.set_major_locator(MaxNLocator(integer=True));P.light_grid(ax,'y');save(fig,1,'C7上下文分布','C7全部45条记录，一点一模型。上下文档位使用相邻深浅色并由横轴区分，顶端数值为模型数；不作连续密度。',pd.DataFrame({'context':ctx,'models':cnt}),['c7_context_audit.csv'])

def v04():
    fig,ax,p=P.figure('q3',(133,84));ctx=np.geomspace(2048,131072,400);b=CFG['support_modes']['strict_joint'];cmin=(6+CFG['eta']*ctx)*b['N_min_B']*b['D_min_B']*1e18;ax.fill_between(ctx,2e18,cmin,color=soft(p,'slate_teal'),alpha=.7);ax.plot(ctx,cmin,c=p.line['indigo'],lw=1.7,label='最小可行预算');ax.axhline(1e19,c=p.line['ochre'],lw=1.1,ls='--',label='低预算 10¹⁹');ax.scatter(CFG['contexts'],(6+CFG['eta']*np.array(CFG['contexts']))*b['N_min_B']*b['D_min_B']*1e18,s=38,fc=p.fill['indigo'],ec='white',lw=.6,zorder=3)
    ax.set(xscale='log',yscale='log',xlabel='上下文长度 / Token',ylabel='最低算力预算 / FLOPs',xlim=(1800,155000),ylim=(2e18,4e19));ax.set_xticks(CFG['contexts'],labels=['2k','4k','8k','32k','128k']);P.light_grid(ax,'y');h,l=ax.get_legend_handles_labels();h.append(Patch(fc=soft(p,'slate_teal'),label='预算不可行'));l.append('预算不可行');P.legend_top(ax,handles=h,labels=l,ncol=2)
    save(fig,4,'严格域可行预算边界','图型调整：聚焦发生可行性变化的低预算区域；10^22/10^24预算整体高于曲线，图注说明而不让两个远离边界点撑出大片空白。边界为(6+ηL)N_minD_min。',pd.DataFrame({'context':ctx,'minimum_budget':cmin}),[C.REPO/'问题三/config/problem3_v1.json'])

def v14():
    d=read('bootstrap_optimal_allocations.csv.gz').query('context_length == 4096 and budget_FLOPs == 1e19');fig,ax,p=P.figure('q3',(126,85))
    for cost in COSTS:
        vals=np.sort(d[d.quality_cost_type==cost].Q_A);xs=np.r_[vals[0],vals];ys=np.arange(len(xs))/len(vals);ls,m=PL.COST_STYLE[cost];ax.step(xs,ys,where='post',c=PL.COST[cost],ls=ls,lw=1.7,label=LABELS[cost]);ax.scatter([vals[0],vals[-1]],[0,1],s=35,marker=m,fc='white',ec=PL.COST[cost],zorder=4)
    ax.set(xlabel='最优质量 Q',ylabel='经验累计比例',xlim=(.51,1.025),ylim=(-.035,1.035));ax.yaxis.set_major_formatter(PercentFormatter(1));P.light_grid(ax,'y');P.legend_top(ax,ncol=3);save(fig,14,'质量边界点质量','低预算10^19、上下文4096，每成本500抽样。保留原始ECDF和质量边界跳跃；线型区分成本，不用平滑线制造连续概率。',d[['draw_id','quality_cost_type','Q_A','Q0']],['bootstrap_optimal_allocations.csv.gz'])

def v15():
    d=read('bootstrap_optimal_allocations.csv.gz').query("context_length == 4096 and budget_FLOPs == 1e19 and quality_cost_type == 'exponential'");a=anchor();fig,ax,p=P.figure('q3',(126,96));im=ax.hexbin(d.N_B,d.D_B,gridsize=25,mincnt=1,cmap=p.cmap_seq(),linewidths=.3,edgecolors='white');ax.scatter(a.N_B,a.D_B,marker='*',s=95,fc='white',ec=p.ink,lw=1,zorder=4,label='中心最优解');ax.set(xlabel='参数量 N / B',ylabel='训练数据 D / B Token');ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.3f}'));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'));ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));P.colorbar(fig,im,ax,label='抽样数',pad=.025);P.legend_top(ax,ncol=1)
    save(fig,15,'资源配置联合分布','指数成本、低预算10^19、上下文4096；500配对draw。窄范围改为普通连续数值轴，六边格白边提高结构可读性，星形是冻结中心解。',d[['draw_id','N_B','D_B']],['bootstrap_optimal_allocations.csv.gz'])

def v21():
    d=read('sensitivity_summary.csv').query("context_length == 4096 and budget_FLOPs == 1e19 and quality_cost_type == 'exponential' and solver_success == True").copy();d=d[~d.dimension.eq('support')].sort_values('loss_change');names={'quality_mapping_slope':'质量映射斜率','quality_empirical_cap':'经验质量上限','cost_amplitude':'成本幅度倍数','cost_shape':'成本形状倍数','attention_eta':'注意力系数倍数'};fig,ax,p=P.figure('q3',(141,97));yy=np.arange(len(d));colors=[p.line['neg'] if x<0 else p.line['pos'] for x in d.loss_change]
    for i,(v,col) in enumerate(zip(d.loss_change,colors)):ax.barh(i,v,height=.56,fc=PL.tint(col,.48),ec=col,lw=.75);ax.scatter(v,i,s=17,fc=col,zorder=3)
    ax.set(yticks=yy,yticklabels=[names[r.dimension]+' · '+str(r.setting) for _,r in d.iterrows()],xlabel='预测 Loss 变化');ax.invert_yaxis();P.zero_line(ax,axis='x');ax.grid(axis='x',color=p.grid,lw=.45);save(fig,21,'关键假设敏感性','指数成本、上下文4096、预算10^19；成功敏感性解相对主情景，横条真实表示损失变化，正负沿全局编码。',d,['sensitivity_summary.csv'])

def v23():
    a=anchor(budget=1e22);d=read('domain_token_allocations.csv');d=d[d.scenario_id==a.scenario_id].sort_values('D_i_B',ascending=False);b=read('bootstrap_intervals.csv').query("context_length == 4096 and budget_FLOPs == 1e22 and quality_cost_type == 'exponential'").iloc[0];fig,ax,p=P.figure('q3',(150,118));y=np.arange(len(d));lo=d.p_ref*b.D_B_q025;hi=d.p_ref*b.D_B_q975;ax.hlines(y,0,d.D_i_B,color=soft(p,'D'),lw=4.4);ax.errorbar(d.D_i_B,y,xerr=[d.D_i_B-lo,hi-d.D_i_B],fmt='o',color=p.line['D'],mfc=p.fill['D'],capsize=2,ms=4.8,lw=1.1,label='固定配比95%区间')
    ax.set(yticks=y,yticklabels=d.domain,xlabel='领域训练数据 / B Token',ylim=(len(d)-.45,-.65));ax.set_xlim(0,d.D_i_B.max()*1.18);ax.grid(axis='x',color=p.grid,lw=.45)
    for i,v in enumerate(d.D_i_B):ax.text(d.D_i_B.max()*1.155,i,f'{v:.2f}',ha='right',va='center',fontsize=7.7,color=p.ink)
    P.legend_top(ax,ncol=1);save(fig,23,'领域Token配额','指数成本、上下文4096、预算10^22；线段长度、点与右侧精确量共同表示17域Token配额。固定p_ref，仅传播D不确定性，非领域份额演化。',d.assign(p025=lo,p975=hi),['domain_token_allocations.csv','bootstrap_intervals.csv'])

def v26():
    fig,axs,p=P.facets('q3',1,3,(160,76),sharex=True,sharey=True,wspace=.02);frames=[]
    for ax,cost in zip(axs[0],COSTS):
        d=path(cost);x=d.budget_FLOPs;ax.plot(x,d.marginal_N/d.marginal_D,c=p.line['N'],lw=1.5,label='参数 / 数据');ax.plot(x,d.marginal_Q/d.marginal_D,c=p.line['Q'],lw=1.5,ls='--',label='质量 / 数据');ax.axhline(1,c=p.ref,ls=':',lw=.8);upper=d[d.Q_A>=.9999]
        if len(upper):ax.axvspan(upper.budget_FLOPs.min(),1e24,fc=soft(p,'quality'),alpha=.5)
        log_budget(ax);ax.set_yscale('log');ax.set_xticks([1e19,1e21,1e24]);P.facet_title(ax,LABELS[cost]);P.light_grid(ax,'y');frames.append(d)
    axs[0,0].set_ylabel('单位算力边际收益比');fig.legend(handles=[Line2D([],[],c=p.line['N'],lw=1.5,label='参数 / 数据'),Line2D([],[],c=p.line['Q'],lw=1.5,ls='--',label='质量 / 数据'),Patch(fc=soft(p,'quality'),label='质量上界'),Line2D([],[],c=p.ref,ls=':',label='边际相等')],loc='outside upper center',ncol=4)
    save(fig,26,'边际资源竞争','三成本共享上下文4096和预算轴。质量边界处不要求边际收益相等，浅底显示已到质量上界；多情景分面揭示差异，不再仅展示一个成本。',pd.concat(frames),['budget_paths.csv.gz'])

def v20():
    d=read('transition_bootstrap.csv.gz').query("record_type == 'event' and quality_cost_type == 'power' and context_length == 4096 and transition_class == 'primary'");v=d.pivot(index='draw_id',columns='event_type',values='budget_mid');start='quality_investment_start';end='quality_saturation';pair=v[[start,end]].dropna();cnt=pair.groupby([start,end]).size().reset_index(name='draws');fig,ax,p=P.figure('q3',(123,94));sc=ax.scatter(cnt[start]/1e19,cnt[end]/1e20,s=26,c=cnt.draws,cmap=p.cmap_seq(),ec=p.line['forest_green'],lw=.55,zorder=3);ax.set(xlabel='启动预算 / 10¹⁹ FLOPs',ylabel='饱和预算 / 10²⁰ FLOPs',xlim=(1.01,1.70));ax.set_xticks(sorted(cnt[start].unique()/1e19));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'));ax.yaxis.set_major_locator(MaxNLocator(5));P.light_grid(ax,'y');P.colorbar(fig,sc,ax,label='配对抽样数',pad=.025);save(fig,20,'质量启动与饱和配对','幂函数成本、上下文4096。同draw的启动与饱和事件按离散预算格点聚合；点面积固定，仅色深表示频数，避免低预算处点面积相互遮挡。未抖动真实预算，也未把离散格点平滑成密度。',cnt,['transition_bootstrap.csv.gz'],{'paired_draws':len(pair)})

FIGS={1:v01,2:v02,3:v03,4:v04,5:v05,6:v06,7:v07,8:v08,10:v10,11:v11,12:v12,14:v14,15:v15,17:v17,18:v18,20:v20,21:v21,23:v23,25:v25,26:v26}
