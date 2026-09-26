"""Q4 evidence figures redesigned against the user's publication reference."""
from .base import *
from style import palettes as PL
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize,PowerNorm
FIGS={}
def reg(n):
 def f(fn):FIGS[n]=fn;return fn
 return f
def soft(key):return PL.SOFT[key]
def grid(ax,axis='x'):P.light_grid(ax,axis)
TYPES=[('pretrained','基础模型','forest_green'),('posttrained','后训练模型','blue_gray'),('merge','合并模型','indigo')]
TASK_GROUPS={
 '逻辑推断':['logical_deduction_three_objects','logical_deduction_five_objects','logical_deduction_seven_objects','formal_fallacies','boolean_expressions','web_of_lies','causal_judgement'],
 '状态追踪':['tracking_shuffled_objects_three_objects','tracking_shuffled_objects_five_objects','tracking_shuffled_objects_seven_objects','temporal_sequences','date_understanding','navigate'],
 '对象结构':['reasoning_about_colored_objects','geometric_shapes','object_counting','penguins_in_a_table'],
 '语言语义':['disambiguation_qa','hyperbaton','snarks','ruin_names','salient_translation_error_detection'],
 '常识推荐':['sports_understanding','movie_recommendation']}
LABELS={'logical_deduction_three_objects':'逻辑推断·3','logical_deduction_five_objects':'逻辑推断·5','logical_deduction_seven_objects':'逻辑推断·7','formal_fallacies':'形式谬误','boolean_expressions':'布尔表达式','web_of_lies':'真假陈述','causal_judgement':'因果判断','tracking_shuffled_objects_three_objects':'打乱追踪·3','tracking_shuffled_objects_five_objects':'打乱追踪·5','tracking_shuffled_objects_seven_objects':'打乱追踪·7','temporal_sequences':'时序关系','date_understanding':'日期理解','navigate':'路径导航','reasoning_about_colored_objects':'颜色对象推理','geometric_shapes':'几何形状','object_counting':'对象计数','penguins_in_a_table':'表格推理','disambiguation_qa':'语义消歧','hyperbaton':'词序判断','snarks':'反讽判断','ruin_names':'名称变换','salient_translation_error_detection':'翻译错误检测','sports_understanding':'体育常识','movie_recommendation':'电影推荐'}
TASK_ORDER=[x for xs in TASK_GROUPS.values() for x in xs]
def short_task(s):return s.replace('leaderboard_bbh_','')
def boundaries(ax,side=None):
 pos=0
 for group,tasks in TASK_GROUPS.items():
  end=pos+len(tasks)
  if end<24:ax.axhline(end-.5,c='#d2dad2',lw=1.15)
  if side is not None:P.bracket(ax,pos-.25,end-.75,at=side,text=group,orientation='v',side='left',fontsize=7.5,text_offset_pt=3)
  pos=end

def main_selected():
 cv=read('bridge_cv_results.csv');return cv[cv.selected].set_index('target_name').loc[TARGETS]

@reg(1)
def evidence_upset():
 d=read('open_model_panel.csv');flags=pd.DataFrame({'权重核实':d.open_weight_verified.astype(bool),'算力可用':np.isfinite(d.log10_compute),'基础模型':d.type_group.eq('pretrained'),'严格许可':d.strict_unrestricted.astype(bool)});counts=flags.value_counts().sort_values(ascending=False);combos=[tuple(v) for v in counts.index];fig,old,p=P.figure('q4',(151,96),layout=None);old.remove();gs=fig.add_gridspec(2,2,left=.17,right=.98,bottom=.09,top=.87,width_ratios=[1.35,4],height_ratios=[2.5,1.55],wspace=.13,hspace=.10);topax=fig.add_subplot(gs[0,1]);matrix=fig.add_subplot(gs[1,1],sharex=topax);sets=fig.add_subplot(gs[1,0],sharey=matrix);x=np.arange(len(counts));colors=[p.fill['forest_green'] if c[0] and c[1] else soft('base') for c in combos];edges=[p.line['forest_green'] if c[0] and c[1] else p.line['base'] for c in combos];topax.bar(x,counts.to_numpy(),fc='white',width=.66)
 for i,n in enumerate(counts):topax.add_patch(Rectangle((i-.33,0),.66,n,fc=colors[i],ec=edges[i],lw=.75));topax.text(i,n*1.16,f'{n:,}',ha='center',va='bottom',fontsize=7.5)
 topax.set_yscale('log');topax.set_ylim(.8,9000);topax.set_ylabel('交集模型数');topax.tick_params(axis='x',bottom=False,labelbottom=False);grid(topax,'y');matrix.set_ylim(3.55,-.55);matrix.set_xticks([]);matrix.tick_params(axis='y',left=False,labelleft=False)
 for j in range(4):matrix.axhspan(j-.5,j+.5,fc='#f4f6f2' if j%2==0 else 'white',zorder=0)
 for i,c in enumerate(combos):
  yy=np.flatnonzero(c);matrix.scatter(np.repeat(i,4),range(4),s=24,fc='#dde4dd',ec='none',zorder=2)
  if len(yy):matrix.plot([i,i],[yy.min(),yy.max()],c=edges[i],lw=1.5,zorder=3);matrix.scatter(np.repeat(i,len(yy)),yy,s=30,c=edges[i],ec='white',lw=.25,zorder=4)
 total=flags.sum().to_numpy();sets.barh(range(4),total,fc=soft('base'),ec=p.line['base'],lw=.6,height=.5);sets.invert_xaxis();sets.set_yticks(range(4),flags.columns,fontsize=8);sets.set_xlim(max(total)*1.3,0);sets.set_xticks([]);sets.tick_params(axis='y',length=0)
 for j,n in enumerate(total):sets.text(n+max(total)*.045,j,f'{n:,}',ha='right',va='center',fontsize=8,color=p.line['base'])
 for a in [sets,matrix]:
  for sp in a.spines.values():sp.set_visible(False)
 sets.set_xlabel('集合规模');matrix.set_xlabel('证据条件的互斥交集');hs=[Patch(fc=p.fill['forest_green'],ec=p.line['forest_green'],label='权重与算力证据同时具备'),Patch(fc=soft('base'),ec=p.line['base'],label='其余交集')];fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.58,.995),ncol=1)
 table=pd.DataFrame(combos,columns=flags.columns).assign(count=counts.to_numpy());save(fig,1,'开源证据交集','四项证据条件的互斥交集及各集合边际规模。深点表示满足，淡点表示不满足；上方为对数计数，左侧为各条件总量。四个条件不是正式前沿的完整筛选流程。',table,['open_model_panel.csv'],{'total':len(d),'marginal_counts':flags.sum().to_dict(),'visual_reference':'gmgc2026F/Q4-02'})

@reg(2)
def evidence_months():
 d=read('open_model_panel.csv').copy();d['month']=pd.to_datetime(d.submission_date,errors='coerce').dt.strftime('%Y-%m');d=d.dropna(subset=['month']);d['权重核实']=d.open_weight_verified.astype(bool);d['算力可用']=np.isfinite(d.log10_compute);d['许可字段可用']=d['Hub License'].notna();d['分数完整']=d.complete_score.astype(bool);fields=['权重核实','算力可用','许可字段可用','分数完整'];g=d.groupby('month')[fields].mean();n=d.groupby('month').size().reindex(g.index);fig,axes,p=P.facets('q4',2,1,(151,94),sharex=True,hspace=.025,gridspec_kw={'height_ratios':[1,2.1]});count,ax=axes[:,0];count.bar(range(len(g)),n,fc=soft('blue_gray'),ec=p.line['blue_gray'],width=.65,lw=.75);count.set_ylabel('模型数');count.tick_params(axis='x',bottom=False,labelbottom=False);grid(count,'y');im=ax.imshow(g.to_numpy().T,cmap=p.cmap_seq_soft(),vmin=0,vmax=1,aspect='auto',interpolation='none');ax.set_yticks(range(4),fields);ax.set_xticks(range(len(g)),g.index,rotation=35,ha='right');ax.set_xlabel('提交月份');P.clean_heatmap(ax,lw=1)
 for j in range(len(g)):
  for i in range(4):
   value=float(g.iloc[j,i]);label='0%' if value==0 else ('100%' if value==1 else (f'{value:.1%}' if value<.1 else f'{value:.0%}'));ax.text(j,i,label,ha='center',va='center',fontsize=7.5,color=P.text_color_on(im.cmap(value)))
 cb=fig.colorbar(im,ax=list(axes[:,0]),fraction=.026,pad=.018,shrink=.68);cb.set_label('证据可用率');cb.ax.yaxis.set_major_formatter(PercentFormatter(1));cb.outline.set_visible(False)
 save(fig,2,'时间与证据覆盖','各提交月份证据可用率及顶部实际模型数。格内为可用比例而非模型能力；权重核实是当前证据状态，不等价于实际不开放。不存在的日期不插补为零。',g.assign(n=n).reset_index(),['open_model_panel.csv'],{'months':len(g),'total':len(d),'visual_reference':'gmgc2026F/evidence matrix hierarchy'})

@reg(3)
def score_distributions():
 d=read('open_model_panel.csv');fields=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO'];fig,axes,p=P.facets('q4',3,2,(160,149),sharex=True,sharey=True,wspace=.08,hspace=.11);records=[];rng=np.random.default_rng(2026)
 for ax,field in zip(axes.flat,fields):
  for i,(kind,name,key) in enumerate(TYPES):
   v=d.loc[d.type_group.eq(kind),field].dropna().to_numpy();y=2-i;positive=v[v>0];xx=np.linspace(0,100,320);den=gaussian_kde(positive,bw_method=.19)(xx);den=den/den.max()*.32;lo,hi=np.quantile(positive,[.001,.999]);keep=xx<=min(100,hi+3);ax.fill_between(xx[keep],y+.06,y+.06+den[keep],fc=soft(key),lw=0);ax.plot(xx[keep],y+.06+den[keep],c=p.line[key],lw=1.3);take=v[rng.choice(len(v),min(160,len(v)),replace=False)];ax.scatter(take,y-.16+rng.uniform(-.07,.07,len(take)),s=3.5,fc=p.line[key],ec='white',lw=.08,alpha=.35,rasterized=True);qs=np.quantile(v,[.1,.25,.5,.75,.9]);ax.plot(qs[[0,4]],[y,y],c=p.line[key],lw=.9);ax.plot(qs[[1,3]],[y,y],c=p.line[key],lw=3.2);ax.scatter(qs[2],y,s=17,fc='white',ec=p.line[key],lw=.9,zorder=4);zero=float((v==0).mean());ax.plot([0,0],[y+.06,y+.06+zero*.8],c=p.line['ochre'],lw=2.4,zorder=5);ax.text(1.02,y,('0%' if zero==0 else ('<0.1%' if zero<.001 else f'{zero:.1%}')),transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7.5,color=p.line['ochre']);records.append({'task':field,'type':kind,'n':len(v),'p10':qs[0],'p25':qs[1],'median':qs[2],'p75':qs[3],'p90':qs[4],'zero_rate':zero})
  ax.set_yticks(range(3),[v[1] for v in TYPES[::-1]]);ax.set_xlim(-2,100);ax.set_ylim(-.37,2.53);ax.set_xticks([0,25,50,75,100]);P.facet_title(ax,field);ax.text(1.02,2.42,'零分率',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7.5,color=p.line['ochre']);grid(ax)
 for ax in axes[-1]:ax.set_xlabel('任务得分 / 分')
 hs=[Patch(fc=soft(key),ec=p.line[key],label=f'{name}（n={(d.type_group==kind).sum():,}）') for kind,name,key in TYPES];hs += [Line2D([],[],marker='o',mfc='white',mec=p.line['base'],c=p.line['base'],lw=3,label='中位数与四分位距'),Line2D([],[],ls='',marker='|',ms=9,mew=2,c=p.line['ochre'],label='零分率（右列）')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,3,'模型类型分位分布','六项真实评测分布按基础、后训练和合并模型分面比较。密度仅由正分记录构造，零分率另以原点竖线及右侧百分数表示；分位数与散点包括零分。分位范围不是均值CI，组间差异未控制规模或家族。',pd.DataFrame(records),['open_model_panel.csv'],{'excluded_other':int((~d.type_group.isin([x[0] for x in TYPES])).sum()),'visual_reference':'gmgc2026F/Q4-06; expanded to three observed model types'})

@reg(8)
def task_normalization():
 d=subtask_unique().copy();d['short']=d.task.map(short_task);g=d.groupby('short').agg(raw_mean=('raw_score','mean'),normalized_mean=('normalized_score','mean'),chance_mean=('random_lower_bound','mean'),chance_min=('random_lower_bound','min'),chance_max=('random_lower_bound','max'),n=('normalized_core','nunique')).loc[TASK_ORDER];fig,axes,p=P.facets('q4',1,2,(160,153),sharey=True,wspace=.05,gridspec_kw={'width_ratios':[3.4,1]});ax,right=axes[0];y=np.arange(24);ax.hlines(y,g.normalized_mean*100,g.raw_mean*100,colors=p.line['forest_green'],lw=1.25);ax.scatter(g.raw_mean*100,y,s=29,marker='s',fc='white',ec=p.line['base'],lw=1,zorder=3);ax.scatter(g.normalized_mean*100,y,s=30,fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.6,zorder=4);right.barh(y,g.chance_mean*100,height=.55,fc=soft('ochre'),ec=p.line['ochre'],lw=.75);right.hlines(y,g.chance_min*100,g.chance_max*100,colors=p.line['ochre'],lw=1.3);ax.set_yticks(y,[LABELS[t] for t in TASK_ORDER],fontsize=8);ax.set_ylim(23.55,-.55);ax.set_xlabel('跨模型平均任务分 / 分');right.set_xlabel('随机下界 / %');right.set_xlim(0,60);right.set_xticks([0,25,50]);grid(ax);grid(right);boundaries(ax,side=-.27);boundaries(right)
 hs=[Line2D([],[],ls='',marker='s',mfc='white',mec=p.line['base'],ms=5,label='原始得分'),Line2D([],[],ls='',marker='o',mfc=p.fill['forest_green'],mec=p.line['forest_green'],ms=5,label='随机下界校正')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,8,'任务随机下界校正','24个BBH子任务按语义分组，每任务对1854个去重模型等权平均。左为原始与下界校正得分，右给实际随机下界；对象计数存在两种下界，细横线保留其范围，未假定所有版本一致。',g.reset_index(),['c8_subtask_normalization.csv.gz','task_level_aggregates.csv'],{'models':int(d.normalized_core.nunique()),'groups':TASK_GROUPS,'visual_reference':'gmgc2026F/Q4-11 grouped task labels'})

@reg(9)
def family_fingerprint():
 d=subtask_unique().copy();d['short']=d.task.map(short_task);counts=d.groupby('family').normalized_core.nunique().nlargest(8);top=counts.index;g=d[d.family.isin(top)].groupby(['short','family']).normalized_score.mean().unstack().reindex(index=TASK_ORDER,columns=top)*100;fig,ax,p=P.figure('q4',(160,154));im=ax.imshow(g.to_numpy(),cmap=p.cmap_seq(),vmin=0,vmax=70,aspect='auto',interpolation='none');ax.set_xticks(range(8),[f'{f}\n(n={counts[f]})' for f in top],rotation=35,ha='right',fontsize=8);ax.set_yticks(range(24),[LABELS[t] for t in TASK_ORDER],fontsize=8);ax.set_xlabel('模型家族（去重模型数）');P.clean_heatmap(ax,lw=.6);boundaries(ax,side=-.22);P.colorbar(fig,im,ax,label='随机下界校正分',pad=.03,shrink=.83,ticks=list(range(0,71,10)))
 save(fig,9,'家族任务能力指纹','去重模型最多的8个家族在24个BBH子任务上的平均随机下界校正分。任务按可解释语义组排列，家族列保留真实支持数；为保留有效色阶，色标显示0–70分（原评分尺度仍0–100分）；仅描述能力结构，未调整参数量或训练方式。',g.reset_index(),['c8_subtask_normalization.csv.gz','task_level_aggregates.csv'],{'family_model_counts':counts.to_dict(),'groups':TASK_GROUPS,'visual_reference':'gmgc2026F/Q4-11 task matrix hierarchy'})

@reg(11)
def model_selection():
 d=read('bridge_cv_results.csv');models=['constant','linear','spline','isotonic'];names=['常数','线性','样条','保序'];keys=['base','indigo','forest_green','ochre'];marks=['s','o','^','D'];fig,ax,p=P.figure('q4',(149,109));selected=[]
 for i in range(7):
  if i%2==0:ax.axhspan(i-.45,i+.45,fc='#f5f7f2',zorder=0)
 for j,(model,name,key,m) in enumerate(zip(models,names,keys,marks)):
  v=d[d.model==model].set_index('target_name').loc[TARGETS];yy=np.arange(7)+(j-1.5)*.17;ax.errorbar(v.cv_mae,yy,xerr=v.cv_mae_se,fmt=m,c=p.line[key],mfc=soft(key),ms=4.7,mew=.8,capsize=2,lw=.95,label=name,zorder=3)
  chosen=v.selected.to_numpy(bool);ax.scatter(v.cv_mae.to_numpy()[chosen],yy[chosen],s=98,marker='o',fc='none',ec=p.ink,lw=1.2,zorder=4);selected+=v[chosen].reset_index().to_dict('records')
 ax.set_yticks(range(7),[x.replace('_','-') for x in TARGETS]);ax.set_ylim(6.58,-.62);ax.set_xlabel('留来源家族折外 MAE / 分');ax.set_ylabel('评测目标');grid(ax);hs,ls=ax.get_legend_handles_labels();hs.append(Line2D([],[],ls='',marker='o',ms=7,mfc='none',mec=p.ink,label='冻结选中规格'));ls.append('冻结选中规格');fig.legend(hs,ls,loc='outside upper center',ncol=3)
 save(fig,11,'七目标桥接模型选择','每个目标的四种候选桥接模型及折均值标准误，黑色外圈标出冻结选中规格。标准误描述折间变动，不作为独立样本显著性检验；选中常数意味着该目标的桥接关系较弱。',d,['bridge_cv_results.csv'],{'selected_models':selected,'visual_reference':'gmgc2026F model selection with selected-specification markers'})

@reg(12)
def family_errors():
 chosen=MAPPING['targets']['Average']['selected_model'];d=read('bridge_cv_predictions.csv.gz').query("target_name=='Average'");d=d[d.model==chosen].copy();d['error']=d.prediction-d.actual;order=d.groupby('held_out_family').error.median().sort_values().index;fig,ax,p=P.figure('q4',(142,111));rng=np.random.default_rng(2026);records=[]
 for i,f in enumerate(order):
  v=d[d.held_out_family==f].error.to_numpy();q=np.quantile(v,[.1,.25,.5,.75,.9]);ax.plot(q[[0,4]],[i,i],c=p.line['blue_gray'],lw=1.2);ax.plot(q[[1,3]],[i,i],c=soft('blue_gray'),lw=7,solid_capstyle='butt');ax.scatter(v,i+rng.uniform(-.14,.14,len(v)),s=22,fc=p.fill['blue_gray'],ec='white',lw=.45,alpha=.75,zorder=3);ax.plot([q[2],q[2]],[i-.22,i+.22],c=p.line['indigo'],lw=1.7,zorder=4);ax.text(1.02,i,str(len(v)),transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8,color=p.line['base']);records.append({'family':f,'n':len(v),'p10':q[0],'p25':q[1],'median':q[2],'p75':q[3],'p90':q[4]})
 ax.text(1.02,-.58,'n',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8);ax.set_yticks(range(len(order)),order);ax.set_ylim(len(order)-.55,-.75);ax.set_xlabel('预测 − 观测 / 分');ax.set_ylabel('留出家族');ax.axvline(0,c=p.ref,ls='--',lw=1);grid(ax);hs=[Line2D([],[],ls='',marker='o',ms=4,mfc=p.fill['blue_gray'],mec='white',label='真实折外记录'),Line2D([],[],c=soft('blue_gray'),lw=7,label='25%–75%分位'),Line2D([],[],c=p.line['blue_gray'],lw=1.2,label='10%–90%分位'),Line2D([],[],ls='',marker='|',ms=8,mew=1.5,c=p.line['indigo'],label='家族中位数')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,12,'留家族误差结构','Average选中线性桥接的全部折外有符号误差，按留出家族组织；新增真实分位范围与n，以区分单点家族和多点分布。抖动只在家族显示轴，点不按可比性权重复制。',d,['bridge_cv_predictions.csv.gz'],{'family_quantiles':records,'selected_model':chosen,'visual_reference':'gmgc2026F cloud-and-interval layering'})

@reg(13)
def interval_coverage():
 d=main_selected();fig,ax,p=P.figure('q4',(134,100));ys=np.arange(7)
 for nominal,col,key,m,off in [(.8,'coverage80','forest_green','o',-.14),(.95,'coverage95','indigo','D',.14)]:
  ax.axvline(nominal,c=p.line[key],ls='--',lw=.85,alpha=.5)
  for i,r in enumerate(d.itertuples()):
   actual=getattr(r,col);y=i+off;ax.plot([actual,nominal],[y,y],c=p.line[key],lw=1.2);ax.scatter(nominal,y,s=37,marker=m,fc='white',ec=p.line[key],lw=1.1,zorder=3);ax.scatter(actual,y,s=37,marker=m,fc=p.fill[key],ec=p.line[key],lw=.7,zorder=4)
 ax.set_yticks(ys,[x.replace('_','-') for x in TARGETS]);ax.set_ylim(6.55,-.6);ax.set_xlim(.48,1.015);ax.set_xlabel('区间覆盖率');ax.xaxis.set_major_formatter(PercentFormatter(1));grid(ax);hs=[Line2D([],[],ls='',marker='o',ms=5,mfc=p.fill['forest_green'],mec=p.line['forest_green'],label='80%区间：实际'),Line2D([],[],ls='',marker='D',ms=5,mfc=p.fill['indigo'],mec=p.line['indigo'],label='95%区间：实际'),Line2D([],[],ls='',marker='o',ms=5,mfc='white',mec=p.line['base'],label='空心：名义覆盖')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,13,'桥接区间覆盖率','选中桥接模型的折外实际覆盖率（实心）与名义80%/95%覆盖率（空心）逐目标配对。连接线表示覆盖缺口；未套用独立二项Wilson区间，因为正式覆盖使用可比性权重和留族结构。',d.reset_index(),['bridge_cv_results.csv'],{'visual_reference':'gmgc2026F/Q4-43; paired nominal/actual instead of unsupported Wilson errors'})

SUPPORT={'high_supported':('forest_green','o','高可比范围'),'expanded_supported':('blue_gray','s','扩展范围'),'out_of_bridge_range':('ochre','^','范围外')}
COST_NAMES={'exponential':'指数成本','power':'幂函数成本','logarithmic':'对数成本'}
@reg(14)
def loss_to_ability():
 d=read('loss_benchmark_predictions.csv').query("target=='Average'");fig,axes,p=P.facets('q4',1,3,(160,88),sharex=True,sharey=True,wspace=.035)
 for ax,cost in zip(axes[0],COST_NAMES):
  g=d[d.quality_cost_type==cost]
  for status,(key,m,label) in SUPPORT.items():
   v=g[g.bridge_support==status];other=v[v.context_length!=4096];main=v[v.context_length==4096];ax.scatter(other.predicted_loss,other['median'],s=22,marker=m,fc='white',ec=p.line[key],lw=.9,alpha=.7,zorder=3)
   for r in main.itertuples():ax.plot([r.predicted_loss]*2,[r.p025,r.p975],c=p.line[key],lw=.9);ax.plot([r.predicted_loss]*2,[r.p10,r.p90],c=p.line[key],lw=3.2);ax.scatter(r.predicted_loss,r.median,s=41,marker=m,fc=p.fill[key],ec=p.line[key],lw=.6,zorder=4)
  P.facet_title(ax,COST_NAMES[cost]);ax.set_xlabel('上游最优Loss');grid(ax,'both')
 axes[0,0].set_ylabel('映射综合能力 / 分');axes[0,0].set_ylim(0,5*np.ceil(max(d['median'].max(),d[d.context_length==4096].p975.max())/5));hs=[Line2D([],[],ls='',marker=m,mfc=p.fill[k],mec=p.line[k],ms=5,label=lab) for k,m,lab in SUPPORT.values()];hs+=[Line2D([],[],ls='',marker='o',mfc='white',mec=p.line['base'],ms=4,label='其余上下文'),Line2D([],[],c=p.line['base'],marker='o',mfc=p.fill['base'],lw=3,label='4096：中位与80%/95%区间')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,14,'最优Loss到能力映射','三成本类型下45个真实上游情景。空心点保留其他上下文，重点展示4096 Token的联合传播中位数及80%/95%区间；形状与颜色编码冻结桥接支持范围，不人为抖动Loss。',d,['loss_benchmark_predictions.csv'],{'highlight_context':4096,'visual_reference':'gmgc2026F bridge scatter and uncertainty layering'})

@reg(15)
def evidence_map():
 d=read('loss_benchmark_predictions.csv').query("target=='Average'");contexts=sorted(d.context_length.unique());budgets=sorted(d.budget_FLOPs.unique());costs=list(COST_NAMES);fig,ax,p=P.figure('q4',(156,132));labels=[];rows=[]
 for i,cost in enumerate(costs):
  for j,ctx in enumerate(contexts):
   y=i*len(contexts)+j;labels.append(f'{ctx:,}');g=d[(d.quality_cost_type==cost)&(d.context_length==ctx)].set_index('budget_FLOPs')
   for k,budget in enumerate(budgets):
    r=g.loc[budget];key,m,name=SUPPORT[r.bridge_support];base=k*1.25;ax.add_patch(Rectangle((base-.025,y-.43),1.05,.86,fc=soft(key),ec='white',lw=.8,alpha=.45,zorder=0));ax.plot(base+np.array([r.p025,r.p975])/100,[y,y],c=p.line[key],lw=.75,zorder=2);ax.plot(base+np.array([r.p10,r.p90])/100,[y,y],c=p.line[key],lw=2.8,zorder=3);ax.scatter(base+r['median']/100,y,s=23,marker=m,fc=p.fill[key],ec='white',lw=.35,zorder=4);rows.append(r.to_dict()|{'budget_FLOPs':budget,'quality_cost_type':cost,'context_length':ctx})
 for k,budget in enumerate(budgets):
  base=k*1.25
  for frac in [0,.5,1]:ax.axvline(base+frac,c=p.grid,lw=.5,zorder=1)
  label={1e19:'10¹⁹ FLOPs',1e22:'10²² FLOPs',1e24:'10²⁴ FLOPs'}[budget];P.bracket(ax,base-.02,base+1.02,at=1.025,text=label,fontsize=8,text_offset_pt=3)
 ax.set_yticks(range(15),labels);ax.set_ylim(14.55,-.65);ax.set_xlim(-.055,3.55);ax.set_xticks([k*1.25+v for k in range(3) for v in [0,.5,1]],['0','50','100']*3);ax.set_xlabel('各预算列内的综合能力 / 分');ax.set_ylabel('上下文 / Token')
 for i,cost in enumerate(costs):P.bracket(ax,i*5-.32,i*5+4.32,at=-.20,text=COST_NAMES[cost].replace('成本',''),orientation='v',side='left',fontsize=8,text_offset_pt=3)
 hs=[Patch(fc=soft(k),ec=p.line[k],label=lab) for k,m,lab in SUPPORT.values()]+[Line2D([],[],c=p.line['base'],marker='o',lw=2.8,ms=3,label='中位数与80%/95%区间')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 save(fig,15,'能力映射证据地图','45情景的桥接证据地图，每个预算列都有独立且一致的0–100能力刻度；浅背景为支持等级，内嵌点及双层线为真实联合传播中位数与区间。支持标签仅描述Loss桥接范围，不表示上游N-D-Q均处于实测支持域。',pd.DataFrame(rows),['loss_benchmark_predictions.csv'],{'budgets':budgets,'contexts':contexts,'column_axis':'each budget column independently maps scores 0 to100','visual_reference':'gmgc2026F evidence maps enriched with actual scores and uncertainty'})
