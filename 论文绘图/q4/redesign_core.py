"""Reference-aligned core charts; all numerical evidence stays local and frozen."""
from .base import *
from style import palettes as PL
from matplotlib.ticker import NullFormatter

def soft(key):return PL.SOFT[key]
def r_interval(ax,r,y,p,key,center='median'):
    ax.plot([r.p025,r.p975],[y,y],c=p.line[key],lw=1.05,zorder=2)
    ax.plot([r.p10,r.p90],[y,y],c=soft(key),lw=8,solid_capstyle='round',zorder=1)
    ax.scatter(r[center],y,s=43,fc=p.fill[key],ec=p.line[key],lw=.8,zorder=4)
def r_legend(fig,p):
    fig.legend(handles=[Line2D([],[],c=p.line['forest_green'],marker='o',ls='',label='中心值'),Line2D([],[],c=soft('forest_green'),lw=8,label='80%区间'),Line2D([],[],c=p.line['forest_green'],lw=1.05,label='95%区间')],loc='outside upper center',ncol=3)

def v04():
    d=read('frontier_training_panel.csv');m=json.loads((R/'dynamic_frontier_parameters.json').read_text());xx=np.linspace(d.log10_compute.min()-.15,d.log10_compute.max()+.15,350);center=linked(m['beta0']+m['beta_x_standardized_log10_compute']*((xx-m['x_mean'])/m['x_std'])+m['states'][-1]);bs=read('frontier_bootstrap_draws.csv.gz').query('tau == .9 and fit_success == True');draw=linked(bs.beta0.to_numpy()[:,None]+bs.beta_x.to_numpy()[:,None]*(xx-bs.x_mean.to_numpy()[:,None])/bs.x_std.to_numpy()[:,None]+bs.state_last.to_numpy()[:,None]);lo95,lo80,hi80,hi95=np.quantile(draw,[.025,.1,.9,.975],axis=0)
    fig,ax,p=P.figure('q4',(145,92));ax.fill_between(xx,lo95,hi95,fc=soft('blue_gray'),alpha=.42,label='95%条件区间');ax.fill_between(xx,lo80,hi80,fc=p.fill['blue_gray'],alpha=.34,label='80%条件区间');ax.plot(xx,center,c=p.line['indigo'],lw=1.7,label='0.90前沿');ax.scatter(d.log10_compute,d['Average ⬆️'],s=42,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.7,zorder=4,label='基础模型')
    upper=d.dropna(subset=['c4_compute_upper']);upper=upper[upper.c4_compute_upper>0]
    for i,(_,r) in enumerate(upper.iterrows()):
        ux=np.log10(r.c4_compute_upper);ax.plot([r.log10_compute,ux],[r['Average ⬆️']]*2,c=p.line['forest_green'],lw=1);ax.scatter(ux,r['Average ⬆️'],marker='|',c=p.line['forest_green'],s=48,label='算力上界' if i==0 else None)
    ax.set(xlabel='训练算力 / lg(FLOPs)',ylabel='综合能力 / 分',xlim=(xx.min(),xx.max()),ylim=(0,5*np.ceil(hi95.max()/5)));P.light_grid(ax,'y');P.legend_top(ax,ncol=3)
    save(fig,4,'训练算力与能力前沿','参考的点—条件曲线—双层区间组织，仍为41条本项目正式基础模型。固定2025-01状态，80%/95%由冻结bootstrap计算；两条单侧上界有据才画，不新增测量。条件曲线并非因果效应。',pd.DataFrame({'log10_compute':xx,'center':center,'p025':lo95,'p10':lo80,'p90':hi80,'p975':hi95}),['frontier_training_panel.csv','dynamic_frontier_parameters.json','frontier_bootstrap_draws.csv.gz'],{'n':len(d),'month':'2025-01','visual_reference':'gmgc2026F/Q4-前沿条件图'})

def v06():
    d=task_unique();x=np.sort(d.bbh_abs_diff);tail=x[x>1e-6];ecdf=np.arange(1,len(x)+1)/len(x);fig,axs,p=P.facets('q4',1,2,(151,82),wspace=.045);ax=axs[0,0];ax.step(x,ecdf,where='post',c=p.line['forest_green'],lw=1.7);ax.axvline(1e-6,c=p.line['ochre'],ls='--',lw=.9);exact=float((x<=1e-6).mean());ax.scatter(1e-6,exact,s=42,fc=p.fill['ochre'],ec=p.line['ochre'],zorder=3);ax.text(2e-4,.946,f'{exact:.1%} 精确复现',fontsize=8,color=p.ink);ax.set_xscale('symlog',linthresh=1e-6);ax.set(xlabel='绝对误差 / 分',ylabel='累计比例',ylim=(.94,1.001),xlim=(-2e-7,20));ax.yaxis.set_major_formatter(PercentFormatter(1));P.facet_title(ax,f'全部模型 · n={len(x):,}');P.light_grid(ax,'y')
    ax=axs[0,1];bins=np.geomspace(tail.min()*.8,tail.max()*1.2,13);ax.hist(tail,bins=bins,fc=soft('ochre'),ec=p.line['ochre'],lw=.7);ax.set(xscale='log',xlabel='绝对误差 / 分',ylabel='模型数');ax.yaxis.set_major_locator(MaxNLocator(integer=True));P.light_grid(ax,'y');P.facet_title(ax,f'未精确复现 · n={len(tail)}')
    fig.legend(handles=[Line2D([],[],c=p.line['forest_green'],lw=1.7,label='累计误差分布'),Line2D([],[],c=p.line['ochre'],ls='--',lw=.9,label='精确容差 10⁻⁶分')],loc='outside upper center',ncol=2)
    save(fig,6,'BBH复现误差分布','左图为1786唯一可比模型的ECDF尾段（94%–100%），阈值为1e-6分；右图仅展示不满足精确复现的真实误差。用整体比例与尾部形态两个相关层次组织，无伪造零值或平滑。',pd.DataFrame({'absolute_error':x,'ecdf':ecdf}),['task_level_aggregates.csv'],{'n':len(x),'exact':int((x<=1e-6).sum()),'non_exact':len(tail)})

def v07():
    all_d=read('task_level_aggregates.csv');fig,axs,p=P.facets('q4',1,2,(153,83),wspace=.055);tables=[]
    for ax,off,repro,title,key in zip(axs[0],['official_bbh','official_math'],['bbh_normalized_equal_pct','math_weighted_pct'],['BBH · 正式复现','MATH · 审计敏感性'],['forest_green','ochre']):
        d=all_d.dropna(subset=[off,repro]).sort_values('model_directory').drop_duplicates('normalized_core');mean=(d[off]+d[repro])/2;diff=d[repro]-d[off];bias=diff.mean();sd=diff.std();ax.scatter(mean,diff,s=13,fc=p.fill[key],ec='none',alpha=.43,rasterized=True);ax.axhline(bias,c=p.line[key],lw=1.5);ax.axhline(bias+1.96*sd,c=p.line['base'],lw=.85,ls='--');ax.axhline(bias-1.96*sd,c=p.line['base'],lw=.85,ls='--');ax.set(xlabel='官方与复现均值 / 分',ylabel='复现 − 官方 / 分');P.facet_title(ax,title);P.light_grid(ax,'y');tables.append(pd.DataFrame({'benchmark':title,'mean':mean,'difference':diff}))
    fig.legend(handles=[Line2D([],[],c=p.line['base'],lw=1.5,label='平均偏差'),Line2D([],[],c=p.line['base'],ls='--',lw=.85,label='均值 ± 1.96 SD')],loc='outside upper center',ncol=2)
    save(fig,7,'BBH聚合一致性','图型调整：增加MATH审计对照分面，避免单看BBH无法理解为何MATH不承担主证据。每面板独立差值范围，均值±1.96SD仅为描述性参考；误差零膨胀，不宣称正态95%覆盖。',pd.concat(tables),['task_level_aggregates.csv'])

def bridge_reconstruction():
    d=pd.read_csv(RAW/'loss_benchmark_bridge_expanded.csv').dropna(subset=['Val_Loss','LB_Average']).copy();d['family']=d.Model.map(model_family);eps=MAPPING['epsilon'];x=np.log(d.Val_Loss.to_numpy());y=logit((d.LB_Average.to_numpy()+eps)/(100+2*eps));w=np.where(d.Loss_Comparability.str.lower().str.startswith('high'),1.,MAPPING['medium_weight']);grid=np.linspace(d.Val_Loss.min(),d.Val_Loss.max(),350)
    def fit(weights,where=grid):
        X=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(X*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0];slope=min(float(coef[1]),0);intercept=np.average(y-slope*x,weights=weights);return linked(intercept+slope*np.log(where))
    center=fit(w);pred=[];rng=np.random.default_rng(2026);fam=np.sort(d.family.unique())
    for _ in range(1000):
        counts=pd.Series(rng.choice(fam,len(fam),replace=True)).value_counts();fw=d.family.map(counts).fillna(0).to_numpy(float);pred.append(fit(w*fw))
    lo,hi=np.quantile(pred,[.025,.975],axis=0);d['fitted']=fit(w,d.Val_Loss);d['residual']=d.LB_Average-d.fitted;d['high']=w==1
    return d,grid,center,lo,hi

def v10():
    d,grid,center,lo,hi=bridge_reconstruction();fig,axs,p=P.facets('q4',1,2,(158,84),wspace=.055);ax=axs[0,0];ax.fill_between(grid,lo,hi,fc=soft('blue_gray'),alpha=.8);ax.plot(grid,center,c=p.line['indigo'],lw=1.7)
    for flag,key,m,label in [(False,'forest_green','o','中可比性'),(True,'ochre','D','高可比性')]:
        v=d[d.high==flag];ax.scatter(v.Val_Loss,v.LB_Average,s=34,marker=m,fc=p.fill[key] if flag else 'white',ec=p.line[key],lw=.8,label=label,zorder=4);axs[0,1].scatter(v.Val_Loss,v.residual,s=34,marker=m,fc=p.fill[key] if flag else 'white',ec=p.line[key],lw=.8)
    ax.set(xlabel='验证 Loss',ylabel='综合能力 / 分',xlim=(1.60,2.90));axs[0,1].set(xlabel='验证 Loss',ylabel='观测 − 拟合 / 分',xlim=(1.60,2.90));axs[0,1].axhline(0,c=p.ref,ls='--',lw=.9)
    for ax in axs[0]:P.light_grid(ax,'y')
    h,l=axs[0,0].get_legend_handles_labels();h.extend([Line2D([],[],c=p.line['indigo'],lw=1.7,label='线性桥接'),Patch(fc=soft('blue_gray'),label='95%条件带')]);l+=['线性桥接','95%条件带'];fig.legend(handles=h,labels=l,loc='outside upper center',ncol=4)
    save(fig,10,'Loss与综合能力桥接','以参考的拟合/残差相关面板组织C6。原模型选择与High=1、Medium=0.25权重不变；参数带在绘图区按来源家族补算1000次seed2026，固定模型选择，不等于正式预测区间。右侧为样本内诊断，不标成折外验证。',pd.DataFrame({'loss':grid,'center':center,'p025':lo,'p975':hi}),[RAW/'loss_benchmark_bridge_expanded.csv','loss_benchmark_mapping.json'],{'n':len(d),'bootstrap':1000,'visual_reference':'gmgc2026F/Q4桥接与一致性图'})

def v16():
    main=read('loss_benchmark_predictions.csv').query("target == 'Average' and context_length == 4096 and quality_cost_type == 'exponential'").sort_values('budget_FLOPs');u=read('loss_benchmark_uncertainty.csv');fig,axs,p=P.facets('q4',1,2,(155,102),sharey=True,wspace=.03);labels=[];rows=[];i=0
    for _,r in main.iterrows():
        for mode,key,name in [('upstream_only','forest_green','上游'),('bridge_only','blue_gray','桥接'),('joint','indigo','联合')]:
            v=u[(u.scenario_index==r.scenario_index)&(u.target=='Average')&(u.uncertainty_mode==mode)].iloc[0];r_interval(axs[0,0],v,i,p,key);axs[0,1].barh(i,v.width95,height=.54,fc=soft(key),ec=p.line[key],lw=.75);labels.append('10'+str(int(np.log10(r.budget_FLOPs))).translate(str.maketrans('0123456789','⁰¹²³⁴⁵⁶⁷⁸⁹'))+' · '+name);rows.append(v.to_dict()|{'budget':r.budget_FLOPs});i+=1
    for ax in axs[0]:ax.set(yticks=range(i),ylim=(i-.4,-.65));ax.grid(axis='x',color=p.grid,lw=.45)
    axs[0,0].set_yticklabels(labels);axs[0,0].set_xlabel('综合能力 / 分');axs[0,1].set_xlabel('95%区间宽度 / 分');r_legend(fig,p)
    save(fig,16,'桥接不确定性来源','指数成本、上下文4096的三预算情景。左并列三来源区间，右列出其95%宽度以比较大小；不将区间堆叠或将宽度解释为可加方差贡献。点是各模式中位数。',pd.DataFrame(rows),['loss_benchmark_predictions.csv','loss_benchmark_uncertainty.csv'],{'visual_reference':'gmgc2026F/Q4-40'})

def v18():
    d=read('frontier_bootstrap_draws.csv.gz').query('tau == .9 and fit_success == True');center=read('contribution_decomposition.csv').query("estimate == 'central'").iloc[0];fig,axs,p=P.facets('q4',1,2,(152,87),wspace=.065);rows=[]
    for ax,field,key,label in zip(axs[0],['scale_score','tech_score'],['blue_gray','ochre'],['规模关联贡献 / 分','非规模残余 / 分']):
        vals=d[field].dropna().to_numpy();q=np.quantile(vals,[.025,.1,.25,.5,.75,.9,.975]);xx=np.linspace(vals.min(),vals.max(),400);yy=gaussian_kde(vals)(xx);yy=.52*yy/yy.max();ax.fill_between(xx,0,yy,fc=soft(key),ec=p.line[key],lw=1.5);rng=np.random.default_rng(2026);sel=np.arange(len(vals))[::3];ax.scatter(vals[sel],-.12+rng.uniform(-.045,.045,len(sel)),s=7,fc=p.fill[key],ec='white',lw=.2,alpha=.65,rasterized=True);ax.plot([q[0],q[-1]],[-.30]*2,c=p.line[key],lw=1);ax.plot([q[1],q[-2]],[-.30]*2,c=soft(key),lw=7,solid_capstyle='round');ax.scatter(q[3],-.30,s=35,fc=p.fill[key],ec=p.line[key],lw=.8,zorder=4);ax.scatter(center[field],-.30,s=63,marker='D',fc='none',ec=p.ink,lw=.9,zorder=5);ax.axvline(0,c=p.ref,ls='--',lw=.8);ax.set(yticks=[],xlabel=label,ylim=(-.40,.61));ax.spines['left'].set_visible(False);ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='x',color=p.grid,lw=.45);rows.append({'field':field,'center':center[field],'p025':q[0],'p10':q[1],'median':q[3],'p90':q[-2],'p975':q[-1]})
    fig.legend(handles=[Patch(fc=soft('blue_gray'),ec=p.line['blue_gray'],label='经验分布'),Line2D([],[],c=soft('blue_gray'),lw=7,label='80%区间'),Line2D([],[],c=p.line['blue_gray'],lw=1,marker='o',label='中位数与95%区间'),Line2D([],[],c=p.ink,marker='D',mfc='none',ls='',label='中心估计')],loc='outside upper center',ncol=2)
    save(fig,18,'规模与非规模贡献分布','2024-06至2025-01、τ=0.90。两个分布各用自己的横轴尺度，配对draw未拆散；密度/抽样点/双区间提高可读性。非规模残余强正则压近零且跨零，不证明技术进步不存在；观察性分解不作因果份额。',pd.DataFrame(rows),['frontier_bootstrap_draws.csv.gz','contribution_decomposition.csv'],{'visual_reference':'gmgc2026F/云雨图'})

def v22():
    d=read('frontier_forecast.csv').query('tau == .9');draws=read('frontier_forecast_draws.csv.gz').query('tau == .9');fig,axs,p=P.facets('q4',1,2,(158,88),sharex=True,sharey=True,wspace=.025);keys={1.:'indigo',.5:'blue_gray',.25:'forest_green'};names={1.:'维持增速',.5:'增速减半',.25:'增速四分之一'}
    for ax,h in zip(axs[0],[12,24]):
        for i,factor in enumerate([1.,.5,.25]):
            r=d[(d.horizon_months==h)&(d.scenario_factor==factor)].iloc[0];vals=draws[(draws.horizon_months==h)&(draws.scenario_factor==factor)].forecast_score.to_numpy();key=keys[factor];xx=np.linspace(vals.min(),vals.max(),280);yy=gaussian_kde(vals)(xx);yy=.24*yy/yy.max();ax.fill_between(xx,i-.13-yy,i-.13,fc=soft(key),ec=p.line[key],lw=.9,alpha=.9);r_interval(ax,r,i+.13,p,key,'forecast_score')
        ax.set(yticks=range(3),yticklabels=[names[f] for f in [1.,.5,.25]],xlabel='综合能力前沿 / 分',xlim=(20,102),ylim=(2.55,-.65));ax.grid(axis='x',color=p.grid,lw=.45);P.facet_title(ax,f'{h}个月 · '+('2026-03' if h==12 else '2027-03'))
    fig.legend(handles=[Patch(fc=soft('forest_green'),ec=p.line['forest_green'],label='预测经验分布'),Line2D([],[],c=soft('forest_green'),lw=8,label='80%区间'),Line2D([],[],c=p.line['forest_green'],lw=1,marker='o',label='中心与95%区间')],loc='outside upper center',ncol=3)
    save(fig,22,'增速放缓情景前沿','参考预测分位/密度层次，将12/24个月分面，每情景展示已保存的真实预测抽样分布与80%/95%区间；中心用重排值。起点仍是2025-03-13，不虚构连续月份扇带。',d,['frontier_forecast.csv','frontier_forecast_draws.csv.gz'],{'origin':'2025-03-13','tau':.9,'visual_reference':'gmgc2026F/Q4-41'})

def v20():
    d=read('c3_historical_sensitivity.csv');v=d.query("analysis == 'leave_one_family_out'").sort_values('year_coef');bs=d.query("analysis == 'bootstrap'").year_coef;lo,hi=np.quantile(bs,[.025,.975]);center=d.query("analysis == 'central'").year_coef.iloc[0];fig,axs,p=P.facets('q4',1,2,(155,111),wspace=.02);ax=axs[0,0];ax.axvspan(lo,hi,fc=soft('blue_gray'),alpha=.52);ax.scatter(v.year_coef,range(len(v)),s=39,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.7);ax.axvline(0,c=p.ref,ls='--',lw=.8);ax.axvline(center,c=p.line['indigo'],ls=':',lw=1);ax.set(yticks=range(len(v)),yticklabels=v.excluded_family,xlabel='留族年系数 / (logit/年)',ylim=(len(v)-.4,-.6));ax.grid(axis='x',color=p.grid,lw=.45)
    ax=axs[0,1];xx=np.linspace(bs.min(),bs.max(),400);den=gaussian_kde(bs)(xx);ax.fill_between(xx,0,den,fc=soft('blue_gray'),ec=p.line['blue_gray'],lw=1.5);ax.axvline(0,c=p.ref,ls='--',lw=.8);ax.axvline(center,c=p.line['indigo'],ls=':',lw=1);ax.set(xlabel='整体年系数 / (logit/年)',ylabel='概率密度');P.light_grid(ax,'y');fig.legend(handles=[Patch(fc=soft('blue_gray'),label='整体95%范围'),Line2D([],[],c=p.line['indigo'],ls=':',label='整体中心'),Line2D([],[],c=p.ref,ls='--',label='零系数')],loc='outside upper center',ncol=3)
    save(fig,20,'历史方向留族敏感性','C3的26条异质历史记录，左图为15个留族点与整体bootstrap区间，右图为整体bootstrap分布。整体区间不是每个留族点的CI；方向不稳定，不与现代前沿拼接。',v,['c3_historical_sensitivity.csv'],{'global_p025':lo,'global_p975':hi,'central':center})

def v21():
    d=read('compute_growth_monthly.csv');summ=json.loads((R/'compute_growth_summary.json').read_text());dt=pd.to_datetime(d.month);xx=d.month_index.to_numpy();trend=summ['endpoint_log10_compute']+summ['monthly_log10_slope']*(xx-xx.max());bs=read('compute_growth_bootstrap.csv.gz').query('fit_success == True');arr=bs.endpoint_log10_compute.to_numpy()[:,None]+bs.monthly_log10_slope.to_numpy()[:,None]*(xx-xx.max());lo,hi=np.quantile(arr,[.1,.9],axis=0);fig,axs,p=P.facets('q4',2,1,(145,106),sharex=True,hspace=.015,height_ratios=[3,1]);ax=axs[0,0];ax.fill_between(dt,lo,hi,fc=soft('blue_gray'),alpha=.65,label='80%趋势区间');ax.step(dt,d.log10_compute_q90,where='mid',c=p.line['forest_green'],lw=1.3,label='月度0.90算力');ax.scatter(dt,d.log10_compute_q90,s=26,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.5,zorder=3);ax.plot(dt,trend,c=p.line['indigo'],lw=1.7,label='稳健趋势');ax.scatter(dt.iloc[-1],summ['endpoint_log10_compute'],s=65,marker='D',fc=p.fill['ochre'],ec=p.line['ochre'],zorder=4,label='趋势端点');ax.set_ylabel('训练算力 / lg(FLOPs)');P.light_grid(ax,'y');P.legend_top(ax,ncol=2)
    ax=axs[1,0];ax.bar(dt,d.model_families,width=18,fc=soft('forest_green'),ec=p.line['forest_green'],lw=.5);ax.set(xlabel='发布日期月份',ylabel='家族数');ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=3));P.light_grid(ax,'y');ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    save(fig,21,'月度算力前沿趋势','上方25个月算力前沿、趋势与80%传播区间，下方对齐显示每月家族支持数；趋势端点与截止月原始值分别显示。时间沿C4发布日期，不能与提交日期混同。',d.assign(trend=trend,p10=lo,p90=hi),['compute_growth_monthly.csv','compute_growth_summary.json','compute_growth_bootstrap.csv.gz'],{'visual_reference':'gmgc2026F/Q4-34,36'})

def v23():
    d=read('frontier_forecast_draws.csv.gz').query('tau == .9');fig,axs,p=P.facets('q4',1,2,(152,84),sharex=True,sharey=True,wspace=.03);records=[]
    for ax,h in zip(axs[0],[12,24]):
        v=d[d.horizon_months==h].pivot(index='draw_id',columns='scenario_factor',values='forecast_score').dropna()
        for factor,key,label,ls in [(.5,'blue_gray','增速减半','-'),(.25,'forest_green','增速四分之一','--')]:
            x=np.sort((v[1.]-v[factor]).to_numpy());cdf=np.arange(1,len(x)+1)/len(x);ax.step(x,cdf,where='post',c=p.line[key],lw=1.7,ls=ls,label=label);med=np.median(x);ax.scatter(med,.5,s=42,fc=p.fill[key],ec=p.line[key],zorder=3);ax.plot([med,med],[0,.5],c=p.line[key],lw=.7,ls=':');records.append(pd.DataFrame({'horizon':h,'factor':factor,'loss_of_score':x,'ecdf':cdf}))
        ax.set(xlabel='维持增速 − 放缓情景 / 分',ylim=(0,1.015));ax.yaxis.set_major_formatter(PercentFormatter(1));P.light_grid(ax,'y');P.facet_title(ax,f'{h}个月')
    axs[0,0].set_ylabel('累计概率');h,l=axs[0,0].get_legend_handles_labels();h.append(Line2D([],[],c=p.ink,marker='o',ls='',label='中位差'));l.append('中位差');fig.legend(handles=h,labels=l,loc='outside upper center',ncol=3)
    save(fig,23,'算力减速的配对代价','期限分面，固定同draw和τ=.9，以维持增速预测减去放缓情景预测（正值表示放缓更低）；点/短虚线标示中位差。仍为真实经验分布，不把共享不确定性的两个独立区间相减。',pd.concat(records),['frontier_forecast_draws.csv.gz'])

FIGS={4:v04,6:v06,7:v07,10:v10,16:v16,18:v18,20:v20,21:v21,22:v22,23:v23}
