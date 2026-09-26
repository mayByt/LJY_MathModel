"""Reference-led Q1 redesign: real data and frozen estimates, richer visual hierarchy."""
from __future__ import annotations
import sys,itertools,json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch,Polygon
from matplotlib.colors import Normalize,LogNorm,TwoSlopeNorm,PowerNorm,BoundaryNorm
from matplotlib.ticker import PercentFormatter,MaxNLocator,NullFormatter,FixedLocator
from matplotlib.tri import Triangulation
from scipy import stats as ss
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage,leaves_list
from common import plot as P
from style import palettes as PL
import config as C
from . import data as D
FIGS={}
finish=None
def configure(fn):
 global finish
 finish=fn
def register(n):
 def inner(f):FIGS[n]=f;return f
 return inner
def src(n):return D.R/n
def grid(ax,axis='x'):P.light_grid(ax,axis)
def top(ax,handles=None,ncol=2):
 if handles:P.legend_top(ax,handles=handles,labels=[h.get_label() for h in handles],ncol=ncol,columnspacing=1.2)
 else:P.legend_top(ax,ncol=ncol,columnspacing=1.2)
def domcolor(d):return PL.Q_DOMAIN.get(d,PL.fill['forest_green']),PL.Q_DOMAIN_LINE.get(d,PL.line['forest_green'])
def q_order():return D.csv('quality_domain.csv').query("mapping_type=='quality_anchor'").sort_values('q',ascending=False).domain.tolist()
def huber(v):
 sys.path.insert(0,str(C.REPO/'问题一/src'))
 from problem1.common import huber_mean_1d
 return huber_mean_1d(np.asarray(v),c=1.345)

def count_col(ax,rows,top_y):
 ax.text(1.025,top_y,'文档数',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8.5,color='#627069')
 for y,n in rows:ax.text(1.025,y,f'{n:,}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8.5,color='#627069')

@register(7)
def quality_raincloud():
 d=D.quality();order=q_order();fig,ax,p=P.figure('q1',(145,123));rng=np.random.default_rng(2026);records={};xs=np.linspace(0,1,450);counts=[]
 for i,dom in enumerate(order):
  v=d.loc[d.domain==dom,'Q'].to_numpy();y=len(order)-1-i;fc,lc=domcolor(dom);den=ss.gaussian_kde(v,bw_method='scott')(xs);den=den/den.max()*.40;lo,hi=np.quantile(v,[.001,.999]);mask=(xs>=max(0,lo-.025))&(xs<=min(1,hi+.025));ax.fill_between(xs[mask],y+.10,y+.10+den[mask],fc=PL.SOFT[fc],lw=0,zorder=2);ax.plot(xs[mask],y+.10+den[mask],c=lc,lw=1.5,zorder=3)
  q=np.quantile(v,[.1,.25,.5,.75,.9]);ax.plot([q[0],q[4]],[y+.015,y+.015],c=lc,lw=1.25,solid_capstyle='round');ax.add_patch(Rectangle((q[1],y-.045),q[3]-q[1],.12,fc='white',ec=lc,lw=1.1,zorder=5));ax.plot([q[2],q[2]],[y-.045,y+.075],c=lc,lw=2,zorder=6)
  sampled=v[rng.choice(len(v),min(260,len(v)),replace=False)];ax.scatter(sampled,y-.21+rng.uniform(-.10,.10,len(sampled)),s=6,fc=lc,ec='white',lw=.15,alpha=.40,rasterized=True,zorder=2);counts.append((y,len(v)));records[dom]={'n':len(v),'quartiles':q[[1,2,3]].tolist(),'P10':q[0],'P90':q[4],'displayed_points':len(sampled)}
 ax.set_yticks(range(7),[D.DOMAIN[x] for x in order[::-1]]);ax.tick_params(axis='y',length=0);ax.set_xlim(.10,.92);ax.set_ylim(-.45,6.65);ax.set_xlabel('综合质量评分 Q');grid(ax);count_col(ax,counts,6.57)
 hs=[Patch(fc=PL.SOFT['slate_teal'],ec=p.line['base'],label='密度'),Patch(fc='white',ec=p.ink,label='四分位距与中位数'),Line2D([],[],c=p.ink,lw=1.25,label='10%–90%分位'),Line2D([],[],ls='',marker='o',ms=3,c=p.line['base'],label='抽样文档（每域≤260）')];top(ax,hs,2)
 finish(fig,7,'七域质量云雨分布',{'domains':records,'kde_bandwidth':'scott','density_grid':[0,1,450],'reference_visual':'gmgc2026F/Q1-11; own scoring data retained'},'七域质量评分的全量核密度、10%–90%范围、四分位距与中位数，附每域最多260条文档散点及全量计数。各域使用固定原色与浅填充，密度高度单独归一。',[D.CACHE/'quality_sample.csv.gz'],'领域内部仍有广泛的质量差异和分布重叠。密度由全部记录计算，文档计数在右侧独立给出；密度面积不代表样本量。新版参照云雨图的层次组织，未读取参考项目的质量数值。')

@register(9)
def quality_forest():
 d=D.quality();official=D.csv('quality_domain.csv').query("mapping_type=='quality_anchor'").sort_values('q',ascending=False);fig,ax,p=P.figure('q1',(145,106));rows=[];counts=[]
 for i,r in enumerate(official.itertuples()):
  dom=r.domain;y=6-i;g=d[d.domain==dom];v=g.Q.to_numpy();quant=np.quantile(v,[.1,.9]);fc,lc=domcolor(dom);ax.plot(quant,[y,y],c=PL.SOFT[fc],lw=8,solid_capstyle='butt',zorder=2);ax.plot([r.ci_low,r.ci_high],[y,y],c=p.ink,lw=1.65,zorder=4);ax.scatter(r.q,y,s=46,fc=lc,ec='white',lw=.5,zorder=5)
  a1=g.loc[g.in_a1,'Q'].to_numpy();a=huber(a1);ax.scatter(a,y+.23,s=33,marker='D',fc='white',ec=lc,lw=1.2,zorder=5);new=g.loc[~g.in_a1,'Q'].to_numpy();remainder=huber(new) if len(new) else None
  if len(new):ax.scatter(remainder,y-.23,s=39,marker='^',fc=p.fill['ochre'],ec=p.line['ochre'],lw=.8,zorder=5)
  counts.append((y,len(v)));rows.append({'domain':dom,'q':r.q,'ci_low':r.ci_low,'ci_high':r.ci_high,'n':len(v),'p10':quant[0],'p90':quant[1],'a1_huber':a,'remainder_huber':remainder,'n_remainder':len(new)})
 ax.set_yticks(range(7),[D.DOMAIN[x] for x in official.domain.tolist()[::-1]]);ax.tick_params(axis='y',length=0);ax.set_ylim(-.5,6.66);ax.set_xlim(.25,.83);ax.set_xlabel('综合质量评分 Q');grid(ax);count_col(ax,counts,6.57)
 hs=[Line2D([],[],c=PL.SOFT['forest_green'],lw=8,label='10%–90%分位'),Line2D([],[],ls='',marker='o',ms=5,c=p.line['forest_green'],label='去重域稳健评分'),Line2D([],[],c=p.ink,lw=1.5,label='评分95%区间'),Line2D([],[],ls='',marker='D',ms=5,mfc='white',mec=p.line['forest_green'],label='A1稳健评分'),Line2D([],[],ls='',marker='^',ms=5,mfc=p.fill['ochre'],mec=p.line['ochre'],label='扩展新增稳健评分')];top(ax,hs,2)
 finish(fig,9,'七域质量区间',{'values':rows,'interval':'frozen domain-row bootstrap; distribution and split scores from reconstructed own records','reference_visual':'gmgc2026F/Q1-13; Huber aggregation retained instead of reference arithmetic mean'},'七域去重质量的10%–90%分布范围与冻结稳健评分/95%bootstrap区间，同时展示A1与扩展非重合记录的同口径Huber聚合。右侧为全量文档数；新增扩展仅存在于arXiv与GitHub。',[src('quality_domain.csv'),D.CACHE/'quality_sample.csv.gz',D.CACHE/'verification.json'],'狭窄的域级区间与宽阔的文档分布是不同层次的不确定性。正式七域评分及区间保持冻结值；抽样与新增部分按原Huber规则聚合，避免将参考项目的算术均值或质量定义混入本项目。')

@register(2)
def lengths():
 d=pd.read_csv(D.CACHE/'raw_lengths.csv.gz');order=sorted(d.domain.unique(),key=lambda x:d.loc[d.domain==x,'rps_doc_word_count'].median(),reverse=True);fig,ax,p=P.figure('q1',(143,100));result={};counts=[]
 for i,dom in enumerate(order):
  y=6-i;v=d.loc[d.domain==dom,'rps_doc_word_count'].to_numpy();fc,lc=domcolor(dom);q=np.quantile(v,[.005,.0625,.125,.25,.5,.75,.875,.9375,.995]);result[dom]={'n':len(v),'quantiles':q.tolist(),'min':float(v.min()),'max':float(v.max())};ax.plot([q[0],q[-1]],[y,y],c=lc,lw=1.1,ls='--')
  for a,b,h,alpha in [(.0625,.9375,.56,.4),(.125,.875,.4,.65),(.25,.75,.24,1)]:
   lo,hi=np.quantile(v,[a,b]);ax.add_patch(Rectangle((lo,y-h/2),hi-lo,h,fc=PL.SOFT[fc],ec=lc,lw=1,alpha=alpha,zorder=2))
  ax.plot(q[4],y,marker='|',c=lc,ms=10,mew=2,zorder=4);counts.append((y,len(v)))
 ax.set_xscale('log');ax.set_xlim(2,6e5);ax.set_xticks([10,100,1000,10000,100000]);ax.xaxis.set_minor_formatter(NullFormatter());ax.set_yticks(range(7),[D.DOMAIN[x] for x in order[::-1]]);ax.tick_params(axis='y',length=0);ax.set_ylim(-.55,6.67);ax.set_xlabel('每篇文本词数');grid(ax);count_col(ax,counts,6.56)
 hs=[Patch(fc=PL.SOFT['forest_green'],ec=p.line['forest_green'],label='中央50% / 75% / 87.5%'),Line2D([],[],c=p.line['forest_green'],ls='--',label='0.5%–99.5%分位'),Line2D([],[],marker='|',ls='',ms=9,c=p.line['forest_green'],label='中位数')];top(ax,hs,2)
 finish(fig,2,'原始文本长度字母值图',result,'A1–A3去重后原始词数的多层分位范围；横轴对数尺度，固定域色表示样本域，右侧列出全量文档数。',[D.CACHE/'raw_lengths.csv.gz',D.CACHE/'data_audit_quality.json'],'域内长尾与跨域数量级差异同时存在。嵌套范围保留完整分布层次，长度不能直接作为跨领域单调质量标准。')

@register(8)
def quantile_comparison():
 d=D.quality();fig,axes,p=P.facets('q1',1,2,(160,83),wspace=.12);left,right=axes[0];q=np.linspace(.01,.99,99);out={}
 for dom,m in [('arxiv','o'),('github','s')]:
  g=d[d.domain==dom];a=g[g.in_a1].Q.to_numpy();b=g[~g.in_a1].Q.to_numpy();x=np.quantile(a,q);y=np.quantile(b,q);fc,lc=domcolor(dom);left.plot(x,y,c=lc,lw=1.6,marker=m,markevery=10,ms=4,mfc=PL.SOFT[fc],mec=lc,label=D.DOMAIN[dom]);right.plot(q*100,y-x,c=lc,lw=1.6,marker=m,markevery=10,ms=3.5,mfc=PL.SOFT[fc],mec=lc);out[dom]={'n_a1':len(a),'n_remainder':len(b),'quantiles':q.tolist(),'a1':x.tolist(),'remainder':y.tolist(),'wasserstein':float(ss.wasserstein_distance(a,b))}
 left.plot([.27,.835],[.27,.835],c=p.ref,ls='--',lw=1.1,label='分位一致');left.set_xlim(.27,.835);left.set_ylim(.27,.835);left.set_aspect('equal');left.set_xlabel('A1质量分位');left.set_ylabel('扩展新增质量分位');right.axhline(0,c=p.ref,ls='--',lw=1.1);right.set_xlabel('分位位置 / %');right.set_ylabel('扩展新增 − A1');right.set_xlim(0,100);right.set_xticks([0,25,50,75,100]);grid(left,'both');grid(right,'both');P.facet_title(left,'分位对应');P.facet_title(right,'分位差异');hs,ls=left.get_legend_handles_labels();fig.legend(hs,ls,loc='outside upper center',ncol=3)
 finish(fig,8,'抽样与扩展质量QQ检验',out,'抽样与扩展非重合部分的分位对应及分位差异。右侧直接呈现扩展新增−A1，灰虚线分别为分位一致线与零差线，曲线均由1%–99%实测分位计算。',[D.CACHE/'quality_sample.csv.gz',D.CACHE/'verification.json'],'QQ图总体接近对角线，但细小尾部差异需要在分位差图中读取。通过联动两种编码，避免“曲线看似重合”遮蔽可检验差异。')

@register(10)
def conflict_panels():
 d=D.quality();cols=['block_'+k for k in D.BLOCKS];v=d[cols].to_numpy();x=v.min(1);y=v.max(1);fig,axes,p=P.facets('q1',1,2,(160,85),sharex=True,sharey=True,wspace=.06);panels=[(d.in_a1.to_numpy(),'A1抽样'),(~d.in_a1.to_numpy(),'扩展新增')];allmax=0;objs=[];stats=[]
 for ax,(mask,label) in zip(axes[0],panels):
  h=ax.hexbin(x[mask],y[mask],gridsize=37,mincnt=1,cmap=p.cmap_seq_soft(),norm=LogNorm(1,1200),linewidths=0,rasterized=True);objs.append(h);ids=np.flatnonzero(mask&d.conflict.to_numpy());sel=ids[np.linspace(0,len(ids)-1,min(240,len(ids)),dtype=int)];ax.scatter(x[sel],y[sel],s=11,fc=p.fill['ochre'],ec='white',lw=.25,alpha=.65,rasterized=True);ax.axvline(.25,c=p.line['base'],ls='--',lw=1);ax.axhline(.75,c=p.line['base'],ls='--',lw=1);ax.plot([0,.75],[0,.75],c=p.ref,ls=':',lw=.9);ax.set_xlim(0,.75);ax.set_ylim(.1,1.025);ax.set_xlabel('最低语义块得分');P.facet_title(ax,f'{label} · n={mask.sum():,}');stats.append({'group':label,'n':int(mask.sum()),'n_conflict':len(ids),'displayed_conflict':len(sel)})
 axes[0,0].set_ylabel('最高语义块得分');common_norm=LogNorm(1,max(float(h.get_array().max()) for h in objs))
 for h in objs:h.set_norm(common_norm)
 cb=fig.colorbar(objs[0],ax=list(axes[0]),fraction=.036,pad=.025,shrink=.82);cb.set_label('分箱文档数');cb.outline.set_visible(False)
 hs=[Line2D([],[],ls='',marker='o',ms=4,mfc=p.fill['ochre'],mec='white',label='冲突记录（抽样）'),Line2D([],[],c=p.line['base'],ls='--',label='高低阈值'),Line2D([],[],c=p.ref,ls=':',label='块分相等')];fig.legend(handles=hs,loc='outside upper center',ncol=3)
 total=int(d.conflict.sum());finish(fig,10,'质量冲突相平面',{'n':len(d),'n_conflict':total,'conflict_rate':total/len(d),'displayed_conflict':sum(s['displayed_conflict'] for s in stats),'panels':stats,'thresholds':{'low':.25,'high':.75,'K':'strictly above domain-specific 90th percentile'}},'A1抽样与扩展新增记录的质量冲突相平面对照，共享坐标和计数色标。橙点为实际触发冲突的记录子样本；判定同时使用高低门槛和域内极差90%门槛。',[D.CACHE/'quality_sample.csv.gz'],'分面使扩展集是否保留主要冲突形态可以直接比较。计数色标一致，样本数量差异不会被每面单独归一化掩盖；橙色点仅用于定位冲突，不承担总体计数。')

@register(11)
def correction_pairs():
 d=D.quality();g=d[d.conflict].copy();g['delta']=g.Q-g.Q_variant_weighted_no_huber;chosen=[]
 for dom in D.QUALITY_DOMAINS:
  a=g[g.domain==dom].sort_values('delta');chosen.extend(a.iloc[np.unique(np.round(np.array([.1,.9])*(len(a)-1)).astype(int))].index.tolist())
 s=g.loc[chosen].sort_values('delta');fig,axes,p=P.facets('q1',1,2,(160,119),sharey=True,wspace=.08,gridspec_kw={'width_ratios':[2.15,1]});ax,right=axes[0];ys=np.arange(len(s))
 for i,row in enumerate(s.itertuples()):
  fc,lc=domcolor(row.domain);ax.plot([row.Q_variant_weighted_no_huber,row.Q],[i,i],c=lc,lw=1.55);ax.scatter(row.Q_variant_weighted_no_huber,i,s=31,fc='white',ec=p.line['base'],lw=1,zorder=3);ax.scatter(row.Q,i,s=35,marker='D',fc=fc,ec=lc,lw=.6,zorder=4);key='pos' if row.delta>=0 else 'neg';right.barh(i,row.delta,height=.5,fc=PL.SOFT[key],ec=p.line[key],lw=.75)
 ax.set_yticks(ys,[D.DOMAIN[r.domain]+f' {i+1:02d}' for i,r in enumerate(s.itertuples())]);ax.set_ylim(len(s)-.5,-.6);ax.set_xlabel('综合质量评分 Q');ax.set_ylabel('代表冲突样本');right.set_xlabel('修正 ΔQ');right.axvline(0,c=p.ref,lw=1,ls='--');grid(ax);grid(right);P.facet_title(ax,'同一文档的评分变化');P.facet_title(right,'有符号修正量');hs=[Line2D([],[],ls='',marker='o',mfc='white',mec=p.line['base'],ms=5,label='块均值'),Line2D([],[],ls='',marker='D',mfc=p.fill['forest_green'],mec=p.line['forest_green'],ms=5,label='Huber评分')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,11,'冲突消解评分修正',{'selection':'domain-specific 10th/90th percentile of signed correction','n_all_conflict':len(g),'delta_quantiles':g.delta.quantile([0,.1,.5,.9,1]).to_dict(),'selected':s[['domain','id','Q','Q_variant_weighted_no_huber','delta']].to_dict('records')},'每域按有符号评分修正量10%与90%位置选择真实冲突记录，左侧连接块均值与Huber评分，右侧给出对应ΔQ；橙色为正修正，靛色为负修正。',[D.CACHE/'quality_sample.csv.gz'],'同一评分前后的对应关系与修正幅度同时可读。样本由预定分位规则选择，并非总体平均效应或只挑选最显著变化。')

@register(13)
def mapping_support():
 d=D.csv('domain_mapping.csv');d['type_order']=d.mapping_type.map({'direct':0,'near_direct':1,'inferred':2});d=d.sort_values(['type_order','q'],ascending=[True,False]);fig,axes,p=P.facets('q1',1,2,(160,133),sharey=True,wspace=.08,gridspec_kw={'width_ratios':[3,1.25]});ax,right=axes[0]
 for i,row in enumerate(d.itertuples()):
  inf=row.mapping_type=='inferred';lc=p.line['indigo'] if inf else p.line['forest_green'];ax.plot([row.sensitivity_ci_low,row.sensitivity_ci_high],[i,i],c=PL.SOFT['forest_green'] if inf else 'white',lw=7,solid_capstyle='butt',zorder=1);ax.plot([row.ci_low,row.ci_high],[i,i],c=lc,lw=1.6,zorder=3);ax.scatter(row.q,i,s=25,marker='D' if inf else 'o',fc='white' if inf else p.fill['forest_green'],ec=lc,lw=.9,zorder=4);right.plot([10,row.sample_rows],[i,i],c=PL.SOFT['blue_gray'],lw=2);right.scatter(row.sample_rows,i,s=28,marker='^' if row.low_support_flag else 'o',fc=p.fill['ochre'] if row.low_support_flag else p.fill['blue_gray'],ec=p.line['ochre'] if row.low_support_flag else p.line['blue_gray'],lw=.7)
 ax.set_yticks(range(17),[D.DOMAIN[v] for v in d.domain]);ax.set_ylim(16.6,-.65);ax.set_xlim(.39,.70);ax.set_xlabel('映射质量 Q');right.set_xscale('log');right.set_xlim(10,7e4);right.set_xticks([10,100,1000,10000]);right.set_xlabel('映射文档数');grid(ax);grid(right);right.axvline(100,c=p.ref,ls='--',lw=.8)
 for lo,hi,label in [(0,2,'直接'),(3,5,'近直接'),(6,16,'推断')]:
  P.bracket(ax,lo-.32,hi+.32,at=-.23,text=label,orientation='v',side='left',fontsize=8,text_offset_pt=3)
 for y in [2.5,5.5]:ax.axhline(y,c=p.grid,lw=.9);right.axhline(y,c=p.grid,lw=.9)
 hs=[Line2D([],[],c=p.line['forest_green'],marker='o',ms=4,lw=1.5,label='条件95%区间'),Line2D([],[],c=PL.SOFT['forest_green'],lw=7,label='映射敏感性包络'),Line2D([],[],ls='',marker='^',mfc=p.fill['ochre'],mec=p.line['ochre'],ms=5,label='低支持域（n<100）')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,13,'跨体系映射双层区间',{'rows':d.drop(columns='type_order').to_dict('records'),'n_inferred':int((d.mapping_type=='inferred').sum())},'分组展示直接、近直接与推断映射。左侧将条件bootstrap区间与非概率的映射误差包络分层，右侧展示各域原文映射支持量，三角标记不足100条的域。',[src('domain_mapping.csv'),src('mapping_uncertainty_contract.json')],'推断域共享保守回退值，条件抽样区间不能代表未识别的映射误差。右侧把样本支持不足和模型误差两种不同局限分开，避免只看窄区间得到过强结论。')

@register(17)
def pareto():
 d=D.csv('mixture_internal_cv.csv');fig,ax,p=P.figure('q1',(111,78));labels={'quadratic_ridge':'二阶Ridge','linear_ridge':'线性Ridge','elastic_net':'Elastic Net','extra_trees':'Extra Trees','random_forest':'随机森林'};offset={'quadratic_ridge':(5,-20),'elastic_net':(8,5),'linear_ridge':(5,8),'extra_trees':(-4,12),'random_forest':(-4,-16)}
 for r in d.itertuples():
  front=r.model in ['quadratic_ridge','extra_trees'];ax.scatter(r.normalized_rmse,r.median_spearman,s=56 if front else 38,marker='D' if r.model=='quadratic_ridge' else 'o',fc=p.fill['forest_green'] if front else PL.SOFT['base'],ec=p.line['forest_green'] if front else p.line['base'],lw=1,zorder=4);dx,dy=offset[r.model];ax.annotate(labels[r.model],(r.normalized_rmse,r.median_spearman),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=8,bbox={'fc':'white','ec':'none','alpha':.9,'pad':.2})
 frontier=d.set_index('model').loc[['quadratic_ridge','extra_trees']];ax.plot(frontier.normalized_rmse,frontier.median_spearman,c=p.line['forest_green'],lw=1.5,ls='--',label='非支配前沿');ax.set_xlim(.39,.449);ax.set_ylim(.798,.88);ax.set_xlabel('IQR标准化 NRMSE');ax.set_ylabel('Spearman 相关系数');grid(ax,'both');top(ax,ncol=1)
 finish(fig,17,'模型选择Pareto前沿',{'models':d.to_dict('records'),'frontier':['quadratic_ridge','extra_trees'],'design':'compact 111x78 mm; prominent frontier and labeled actual models'},'五种候选模型内部交叉验证的误差与排序权衡。紧凑构图与突出前沿避免稀疏点图占用过大版面，所有位置仍为原始验证指标。',[src('mixture_internal_cv.csv')],'二阶Ridge以较低误差入选，Extra Trees排序相关更高；两者构成非支配前沿，其余模型被至少一个前沿模型支配。')

@register(18)
def regularization():
 fig,axes,p=P.facets('q1',1,2,(154,80),wspace=.14,gridspec_kw={'width_ratios':[1.25,1]});info={};hs=[]
 for fname,label,key,ls in [('ridge_path_linear.csv','线性Ridge','base','--'),('ridge_path_quadratic.csv','二阶Ridge','forest_green','-')]:
  d=D.csv(fname);best=d.loc[d.normalized_rmse.idxmin()];info[label]={'best':best.to_dict(),'rows':d.to_dict('records')}
  for ax in axes[0]:ax.plot(d.alpha,d.normalized_rmse,c=p.line[key],ls=ls,lw=1.7);ax.scatter(best.alpha,best.normalized_rmse,s=35,marker='D',fc=p.fill[key],ec=p.line[key],lw=.8,zorder=4);ax.set_xscale('log');ax.set_xlabel('正则化强度 α');grid(ax,'both')
  hs.append(Line2D([],[],c=p.line[key],ls=ls,lw=1.6,label=label))
 q=D.csv('ridge_path_quadratic.csv');near=q[q.normalized_rmse<=q.normalized_rmse.min()*1.01];axes[0,0].axvspan(near.alpha.min(),near.alpha.max(),fc=PL.SOFT['leaf_green'],alpha=.7,zorder=0);axes[0,0].set_xlim(.001,10);axes[0,0].set_ylim(.385,.52);axes[0,0].set_ylabel('IQR标准化 NRMSE');P.facet_title(axes[0,0],'近优区间');axes[0,1].set_xlim(1e-6,1e4);axes[0,1].set_yscale('log');axes[0,1].set_ylim(.33,7);axes[0,1].set_yticks([.4,.6,1,2,4,6]);axes[0,1].yaxis.set_major_formatter('{x:g}');axes[0,1].yaxis.set_minor_formatter(NullFormatter());P.facet_title(axes[0,1],'完整路径')
 hs += [Patch(fc=PL.SOFT['leaf_green'],label='二阶误差≤最小值×1.01'),Line2D([],[],ls='',marker='D',mfc='white',mec=p.ink,ms=4,label='最优强度')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,18,'正则化路径',info,'共享实际41点正则化路径，左侧放大近优强度区间，右侧保留全范围并使用对数误差轴。浅带表示二阶Ridge误差不超过最小值1%的采样强度跨度。',[src('ridge_path_linear.csv'),src('ridge_path_quadratic.csv')],'局部视图让两种模型的有效差异可读，全范围视图交代过强正则化的退化。阴影不是参数置信区间，仅是预定义误差容许范围。')

@register(19)
def ranks():
 a=D.csv('mixture_predictions.csv.gz');fig,axes,p=P.facets('q1',1,3,(160,73),sharex=True,sharey=True,wspace=.035);out={}
 for ax,(split,label) in zip(axes[0],[('test_1m','1M'),('test_60m','60M'),('test_1B','1B')]):
  d=a[(a.split==split)&(a.loss_domain=='pile_cc')];x=ss.rankdata(d.observed)/(len(d)+1);y=ss.rankdata(d.predicted)/(len(d)+1);rho=ss.spearmanr(d.observed,d.predicted).statistic;ax.add_patch(Rectangle((0,0),.1,.1,fc=PL.SOFT['leaf_green'],ec=p.line['forest_green'],lw=.9,zorder=1));ax.plot([0,1],[0,1],c=p.ref,ls='--',lw=1,zorder=0);ax.scatter(x,y,s=19,fc=p.fill['forest_green'],ec='white',lw=.35,alpha=.70,zorder=3,rasterized=True);ax.set_xlim(-.03,1.03);ax.set_ylim(-.03,1.03);ax.set_aspect('equal');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_xlabel('实测归一化秩');P.facet_title(ax,f'{label} · n={len(d)}');ax.text(.98,.035,f'ρ = {rho:.3f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.5});grid(ax,'both');out[split]={'n':len(d),'spearman':float(rho),'index':d['index'].tolist()}
 axes[0,0].set_ylabel('预测归一化秩');hs=[Line2D([],[],marker='o',ls='',ms=4,mfc=p.fill['forest_green'],mec='white',label='实测配方'),Line2D([],[],c=p.ref,ls='--',label='秩一致'),Patch(fc=PL.SOFT['leaf_green'],ec=p.line['forest_green'],label='共同前10%')];fig.legend(handles=hs,loc='outside upper center',ncol=3);out.update(target='pile_cc',normalization='rank/(n+1)',low_corner='both top10%')
 finish(fig,19,'跨规模配方秩验证',out,'通用网页验证域的1M、60M和1B实测配方排序分面，使用统一归一化秩尺度。每面标记真实配方数与Spearman相关，低秩区域高亮共同前10%。',[src('mixture_predictions.csv.gz')],'分面避免三种规模散点相互覆盖，使各规模的离群配方和排序偏离可分别观察。这仍是排序检验，不是绝对Loss的无校准跨规模验证。')

@register(20)
def calibration():
 a=D.csv('mixture_predictions.csv.gz');fig,axes,p=P.facets('q1',1,2,(158,84),sharey=True,wspace=.08);panelstats={}
 for ax,(split,label) in zip(axes[0],[('test_60m','60M'),('test_1B','1B')]):
  d=a[(a.split==split)&(a.loss_domain=='pile_cc')];st={'split':split,'target':'pile_cc','n':len(d)}
  for col,name,key,m in [('predicted','原始预测','ochre','o'),('predicted_calibrated','跨拟合校准','forest_green','s')]:
   avg=(d.observed+d[col])/2;delta=d[col]-d.observed;bias=delta.mean();ax.scatter(avg,delta,s=20,fc=p.fill[key],ec='white',lw=.35,alpha=.72,marker=m,rasterized=True);ax.hlines(bias,avg.min(),avg.max(),colors=p.line[key],lw=1.55);st[col]={'bias':float(bias),'sd_difference':float(delta.std()),'rmse':float(np.sqrt(np.mean(delta**2)))}
  ax.axhline(0,c=p.ref,ls='--',lw=1);ax.set_xlabel('预测与实测Loss均值');P.facet_title(ax,f'{label} · n={len(d)}');grid(ax,'both');panelstats[split]=st
 axes[0,0].set_ylabel('预测Loss − 实测Loss');hs=[Line2D([],[],marker='o',ls='',ms=4,mfc=p.fill['ochre'],mec='white',label='原始预测'),Line2D([],[],marker='s',ls='',ms=4,mfc=p.fill['forest_green'],mec='white',label='跨拟合校准'),Line2D([],[],c=p.line['base'],lw=1.5,label='组内平均偏差'),Line2D([],[],c=p.ref,ls='--',label='零偏差')];fig.legend(handles=hs,loc='outside upper center',ncol=2);st=panelstats['test_60m'];st['panels']=panelstats.copy();st['panels']['test_60m']={k:v for k,v in st.items() if k!='panels'}
 finish(fig,20,'跨规模绝对误差校准',st,'60M与1B通用网页验证域的绝对Loss偏移，共享偏差轴。原始与跨拟合校准结果以不同标记表示，实线为组内平均偏差，虚线为零偏差。',[src('mixture_predictions.csv.gz')],'随模型规模变化，原始绝对Loss偏移并不能由排序相关良好而消除。跨拟合仿射校准使用目标规模折内标签，因此不能称为完全零样本预测。')

FAMILIES={'学术':['arxiv','nih_exporter','pubmed_central','pubmed_abstracts','philpapers'],'代码':['github','dm_mathematics'],'社区':['stackexchange','hackernews','ubuntu_irc','enron_emails'],'网页':['pile_cc'],'百科':['wikipedia_en','gutenberg_pg_19'],'法律':['freelaw','europarl','uspto_backgrounds']}
MIX_ORDER=[d for ds in FAMILIES.values() for d in ds]
BLOCK_SHORT={'education':'教育','expression':'表达','reasoning':'推理','noise':'噪声','structure':'结构'}
BLOCK_KEYS=dict(zip(D.BLOCKS,['forest_green','blue_gray','indigo','ochre','olive_gray']))
def balanced_quality():
 d=D.quality();rng=np.random.default_rng(2026);n=int(d.groupby('domain').size().min());return pd.concat([g.iloc[rng.choice(len(g),n,replace=False)] for _,g in d.groupby('domain',sort=True)],ignore_index=True),n

@register(5)
def correlations():
 bal,n=balanced_quality();w=pd.read_csv(D.CACHE/'quality_weights.csv');fields=[];groups=[]
 from scipy.spatial.distance import squareform
 for block in D.BLOCKS:
  fs=w.loc[w.block==block,'field'].tolist();co=bal[fs].corr('spearman').fillna(0).to_numpy();dist=np.clip(1-abs(co),0,2);np.fill_diagonal(dist,0);idx=leaves_list(linkage(squareform(dist,checks=False),method='average'));start=len(fields);fields += [fs[k] for k in idx];groups.append((start,len(fields),block))
 raw=bal[fields].corr('spearman').to_numpy();ranks=bal[fields].rank(pct=True);resid=ranks-ranks.groupby(bal.domain).transform('mean');adjusted=resid.corr().to_numpy();matrix=np.where(np.tril(np.ones_like(raw),-1),raw,adjusted);np.fill_diagonal(matrix,np.nan);fig,ax,p=P.figure('q1',(162,149));cmap=p.cmap_div(soft=True).copy();cmap.set_bad('#fafbf7');im=ax.imshow(matrix,cmap=cmap,vmin=-1,vmax=1,interpolation='none');ax.set_xticks(range(22),[D.FIELDS[x] for x in fields],rotation=90,fontsize=7.5);ax.set_yticks(range(22),[D.FIELDS[x] for x in fields],fontsize=7.5);ax.tick_params(length=0);P.clean_heatmap(ax,lw=.35)
 for start,end,block in groups:
  ax.add_patch(Rectangle((start-.5,start-.5),end-start,end-start,fc='none',ec=p.line[BLOCK_KEYS[block]],lw=1.2));P.bracket(ax,start-.4,end-.6,at=1.025,text=BLOCK_SHORT[block],orientation='h',color=p.line[BLOCK_KEYS[block]],fontsize=8,text_offset_pt=2)
 P.colorbar(fig,im,ax,label='相关系数 ρ',shrink=.78,pad=.025);ax.set_xlabel('同向化质量指标');ax.set_ylabel('同向化质量指标');hs=[Patch(fc='white',ec=p.line['base'],label='下三角：总体相关'),Patch(fc=PL.SOFT['base'],ec=p.line['base'],label='上三角：控制领域差异')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,5,'质量指标相关结构',{'fields':fields,'n_balanced':len(bal),'n_per_domain':n,'matrix':raw.tolist(),'domain_adjusted_matrix':adjusted.tolist(),'clustering':'within semantic blocks; rank residuals remove domain-specific means'},'以七域等量样本对照总体Spearman相关（下三角）与去除领域均值差异后的秩残差相关（上三角）；边框及顶部分组对应五语义块。两半共用相关色标，主对角不赋相关值。',[D.CACHE/'quality_sample.csv.gz',D.CACHE/'quality_weights.csv'],'新版填补空白半矩阵并区分跨域差异与域内关联，帮助判断表面高相关是否主要来自领域结构。控制域差异为描述性残差化，不建立因果关系。')

@register(6)
def weights_association():
 d=pd.read_csv(D.CACHE/'quality_weights.csv');order=[]
 for block in D.BLOCKS:order.extend(d[d.block==block].sort_values('total_weight',ascending=False).index)
 d=d.loc[order].copy();bal,n=balanced_quality();corr=bal[d.field.tolist()+['Q']].corr('spearman')['Q'];d['rho_Q']=d.field.map(corr);fig,axes,p=P.facets('q1',1,2,(158,151),sharey=True,wspace=.08,gridspec_kw={'width_ratios':[1.35,1]});ax,right=axes[0]
 for i,r in enumerate(d.itertuples()):
  key=BLOCK_KEYS[r.block];ax.barh(i,r.total_weight*100,height=.62,fc=PL.SOFT[key],ec=p.line[key],lw=.85);ax.text(r.total_weight*100+.12,i,f'{r.total_weight*100:.2f}',va='center',fontsize=7.5);right.plot([0,r.rho_Q],[i,i],c=PL.SOFT['base'],lw=2.4);right.scatter(r.rho_Q,i,s=25,fc='white',ec=p.line['base'],lw=1,zorder=3)
 ax.set_yticks(range(22),[D.FIELDS[f] for f in d.field],fontsize=7.5);ax.set_ylim(21.6,-.6);ax.set_xlim(0,10.5);ax.set_xlabel('综合权重 / %');right.set_xlim(-.55,1);right.set_xticks([-.5,0,.5,1]);right.set_xlabel('与Q的Spearman ρ');right.axvline(0,c=p.ref,ls='--',lw=.9);grid(ax);grid(right);P.facet_title(ax,'模型赋权');P.facet_title(right,'评分关联')
 cum=0
 for block in D.BLOCKS:
  cum+=int((d.block==block).sum())
  if cum<22:ax.axhline(cum-.5,c=p.grid,lw=1);right.axhline(cum-.5,c=p.grid,lw=1)
 hs=[Patch(fc=PL.SOFT[k],ec=p.line[k],label=D.BLOCKS[b]) for b,k in BLOCK_KEYS.items()];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,6,'语义块与质量指标权重',{'rows':d.to_dict('records'),'block_totals':d.groupby('block').total_weight.sum().to_dict(),'association_n_per_domain':n,'association_n':len(bal),'layout_change':'weights and observed score association in aligned columns'},'左侧为稳定CRITIC综合权重，五块各占20%；右侧为同一指标与综合评分Q的Spearman关联，使用七域等量样本。对齐行与分块帮助比较名义赋权和实际评分结构。',[D.CACHE/'quality_weights.csv',D.CACHE/'quality_sample.csv.gz'],'权重和相关性是不同量：前者是模型设定中的分配，后者是当前数据上的联合变化。评分自身由这些指标构造，因此右侧不能当作外部验证或因果重要性。')

@register(15)
def composition_design():
 b=D.bridge();ix=[list(b['domains']).index(x) for x in MIX_ORDER];v=b['p_train'][:,ix];dom=v.argmax(1);eff=1/(v*v).sum(1);order=np.lexsort((eff,dom));fig,axes,p=P.facets('q1',1,2,(160,126),sharey=True,wspace=.045,gridspec_kw={'width_ratios':[5.5,1]});ax,side=axes[0];cmap=p.cmap_seq_soft().copy();cmap.set_bad('white');im=ax.imshow(np.ma.masked_equal(v[order],0),aspect='auto',cmap=cmap,norm=PowerNorm(.5,vmin=0,vmax=1),interpolation='none',rasterized=True);ax.set_xticks(range(17),[D.DOMAIN[x] for x in MIX_ORDER],rotation=90,fontsize=8);ax.set_xlabel('训练领域');ax.set_ylabel('训练配方（排序后）');ax.set_yticks([0,127,255,383,511],[1,128,256,384,512]);side.scatter(eff[order],np.arange(512),s=5,fc=p.line['forest_green'],ec='white',lw=.1,alpha=.45,rasterized=True);side.set_xlim(.7,9.5);side.set_xticks([1,5,9]);side.set_xlabel('有效域数');side.spines['left'].set_visible(False);side.tick_params(axis='y',left=False,labelleft=False);grid(side)
 start=0
 for fam,ds in FAMILIES.items():
  end=start+len(ds);P.bracket(ax,start-.4,end-.6,1.015,fam,fontsize=7.5,text_offset_pt=2);start=end
 for yy in np.flatnonzero(np.diff(dom[order]))+.5:ax.axhline(yy,c='white',lw=.8)
 cb=fig.colorbar(im,ax=list(axes[0]),fraction=.027,pad=.024,ticks=[0,.01,.05,.2,.5,1],shrink=.82);cb.set_label('训练份额');cb.outline.set_visible(False);cb.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));fig.legend(handles=[Patch(fc='white',ec=p.ref,label='精确零份额')],loc='outside upper left')
 finish(fig,15,'训练配方组成条码',{'shape':v.shape,'zero_fraction':float((v==0).mean()),'row_order':b['train_index'][order].tolist(),'domain_order':MIX_ORDER,'effective_domain_count':eff[order].tolist(),'sort':'dominant domain in semantic group order, then effective domain count','color_norm':'PowerNorm gamma=0.5'},'训练配方按语义分组与主导领域排序；平方根颜色标度保留低份额结构，白格为精确零。右侧与每个配方对齐的点表示逆Simpson有效领域数，顶端括号给出域群层级。',[src('problem2_bridge_input.npz')],'矩阵呈现稀疏组成，侧栏区分域覆盖和份额均衡。配方行按设计属性排序，不是训练时间轴；侧栏每点对应一条真实配方。')

@register(16)
def effective_domains():
 v=D.bridge()['p_train'];x=(v>0).sum(1);y=1/(v*v).sum(1);h,xe,ye=np.histogram2d(x,y,bins=[np.arange(1.5,18.5),np.arange(.75,9.76,.5)]);ii,jj=np.nonzero(h);fig,axes,p=P.facets('q1',2,1,(143,113),sharex=True,hspace=.02,gridspec_kw={'height_ratios':[1,3.3]});count,ax=axes[:,0];rows=[]
 for k in sorted(set(x)):
  vals=y[x==k];qs=np.quantile(vals,[.1,.5,.9]);rows.append({'active':int(k),'n':len(vals),'q10':qs[0],'median':qs[1],'q90':qs[2]})
 frame=pd.DataFrame(rows);count.bar(frame.active,frame.n,fc=PL.SOFT['forest_green'],ec=p.line['forest_green'],lw=.7,width=.65);count.set_ylabel('配方数');count.tick_params(axis='x',bottom=False,labelbottom=False);grid(count,'y');ax.fill_between(frame.active,frame.q10,frame.q90,fc=PL.SOFT['leaf_green'],alpha=.60,zorder=1);ax.plot(frame.active,frame['median'],c=p.line['forest_green'],lw=1.7,zorder=2);ax.scatter((xe[ii]+xe[ii+1])/2,(ye[jj]+ye[jj+1])/2,s=h[ii,jj]*5.5,fc=p.fill['forest_green'],ec='white',lw=.55,alpha=.68,zorder=3);ax.plot([1,17],[1,17],c=p.ref,ls='--',lw=1);ax.set_xlim(1,18);ax.set_ylim(.7,10);ax.set_xticks([2,5,8,11,14,17]);ax.set_xlabel('非零领域数');ax.set_ylabel('有效领域数');grid(ax,'both');hs=[Line2D([],[],c=p.line['forest_green'],lw=1.7,label='条件中位数'),Patch(fc=PL.SOFT['leaf_green'],label='条件10%–90%分位'),Line2D([],[],c=p.ref,ls='--',label='均匀配比上界')]
 for n in [5,20]:hs.append(Line2D([],[],ls='',marker='o',ms=np.sqrt(n*5.5),mfc=p.fill['forest_green'],mec='white',label=f'{n}个配方'))
 fig.legend(handles=hs,loc='outside upper center',ncol=3)
 finish(fig,16,'领域覆盖与份额集中度',{'active_quantiles':np.quantile(x,[0,.25,.5,.75,1]),'effective_quantiles':np.quantile(y,[0,.25,.5,.75,1]),'effective_definition':'1/sum(p_i^2)','bin_width_y':.5,'bin_counts':h.astype(int),'conditional_summaries':rows},'非零域数与有效域数的联合计数，加上条件中位数、10%–90%分位及顶部样本量。气泡面积与分箱配方数成正比，分位带只连接实际存在的整数域数。',[src('problem2_bridge_input.npz')],'配方包含更多领域时有效域数总体上升，但离均匀配比上界仍有明显距离。顶部计数说明条件统计的支持量，分位带不是估计误差置信区间。')

@register(22)
def interactions_matrix():
 target='pile_cc';d=D.csv('interaction_stability.csv');d=d[d.loss_domain==target].copy();idx={x:i for i,x in enumerate(MIX_ORDER)};mat=np.full((17,17),np.nan);stable=np.zeros((17,17),bool)
 for r in d.itertuples():
  i,j=idx[r.domain_i],idx[r.domain_j];mat[i,j]=mat[j,i]=r.coefficient;stable[i,j]=stable[j,i]=(r.ci_low*r.ci_high>0)
 vmax=float(np.nanmax(abs(mat)));from matplotlib.colors import SymLogNorm
 fig,ax,p=P.figure('q1',(160,151));cmap=p.cmap_div(soft=True).copy();cmap.set_bad('#f8faf6');im=ax.imshow(mat,cmap=cmap,norm=SymLogNorm(linthresh=1,linscale=1,vmin=-vmax,vmax=vmax),interpolation='none');ys,xs=np.nonzero(stable);ax.scatter(xs,ys,s=7,c=p.ink,ec='white',lw=.12);ax.set_xticks(range(17),[D.DOMAIN[x] for x in MIX_ORDER],rotation=90,fontsize=8);ax.set_yticks(range(17),[D.DOMAIN[x] for x in MIX_ORDER],fontsize=8);ax.set_xlabel('训练领域');ax.set_ylabel('训练领域');P.clean_heatmap(ax,lw=.45);start=0
 for fam,ds in FAMILIES.items():
  end=start+len(ds);ax.add_patch(Rectangle((start-.5,start-.5),len(ds),len(ds),fc='none',ec=p.line['base'],lw=1));P.bracket(ax,start-.35,end-.65,1.02,fam,fontsize=8,text_offset_pt=2);start=end
 cb=P.colorbar(fig,im,ax,label='二阶系数',shrink=.78,pad=.025,ticks=[-10,-3,-1,0,1,3,10]);fig.legend(handles=[Line2D([],[],ls='',marker='o',ms=3,c=p.ink,label='95%区间不跨0')],loc='outside upper center');edge=d.to_dict('records')
 finish(fig,22,'稳定领域交互网络',{'target':target,'edges':edge,'node_order':MIX_ORDER,'n_stable':int((d.ci_low*d.ci_high>0).sum()),'layout_change':'replace top10 hub-star network with complete symmetric interaction matrix','color_norm':'symmetric log, linear within +/-1'},'通用网页验证域的完整136个二阶域对关系，以对称矩阵代替仅展示10条边的星状网络。点标记95%系数区间不跨零的域对，主对角无二阶自交互，颜色在±1内线性、外侧对称对数。',[src('interaction_stability.csv')],'完整矩阵保留较弱和不稳定的组合关系，避免Top10都连接单一枢纽而掩盖整体结构。对称位置展示同一个域对，并非两次独立实验；系数关系不构成因果网络。')

@register(23)
def benefits_uncertainty():
 b=D.bridge();d=D.csv('mixture_directional_effects.csv').set_index('domain').loc[b['domains']];c=b['coefficient_draws'].astype(float);ref=b['p_ref_point'];g=c[:,:,:17].copy()
 for t,(i,j) in enumerate(itertools.combinations(range(17),2)):g[:,:,i]+=c[:,:,17+t]*ref[j];g[:,:,j]+=c[:,:,17+t]*ref[i]
 draws=-(g-g.mean(2,keepdims=True)).mean(1);lo,hi=np.quantile(draws,[.025,.975],axis=0);d=d.assign(lo=lo,hi=hi,prob_positive=(draws>0).mean(0));order=np.argsort(-d.benefit_score.to_numpy());d=d.iloc[order];draws=draws[:,order];fig,axes,p=P.facets('q1',1,2,(158,132),sharey=True,wspace=.055,gridspec_kw={'width_ratios':[3.8,1]});ax,right=axes[0]
 for i,r in enumerate(d.itertuples()):
  v=draws[:,i];key='pos' if r.benefit_score>=0 else 'neg';xx=np.linspace(np.quantile(v,.005),np.quantile(v,.995),150);den=ss.gaussian_kde(v)(xx);den=den/den.max()*.31;ax.fill_between(xx,i-den,i+den,fc=PL.SOFT[key],ec=p.line[key],lw=.65,alpha=.72);ax.plot([r.lo,r.hi],[i,i],c=p.line[key],lw=1.05);ax.scatter(r.benefit_score,i,s=24,fc=p.fill[key],ec='white',lw=.5,zorder=4);right.barh(i,r.prob_positive,height=.55,fc=PL.SOFT['ochre'],ec=p.line['ochre'],lw=.8)
 ax.set_yticks(range(17),[D.DOMAIN[x] for x in d.index]);ax.set_ylim(16.6,-.6);ax.set_xlabel('局部Loss降低方向导数');ax.axvline(0,c=p.ref,ls='--',lw=1);right.axvline(.5,c=p.ref,ls='--',lw=1);right.set_xlim(0,1.03);right.set_xticks([0,.5,1]);right.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));right.set_xlabel('正收益概率');grid(ax);hs=[Patch(fc=PL.SOFT['base'],ec=p.line['base'],label='bootstrap分布'),Line2D([],[],c=p.line['base'],marker='o',mfc='white',lw=1,label='点估计与95%区间')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,23,'局部配比收益与不确定性',{'rows':d.reset_index().to_dict('records'),'bootstrap_reps':len(draws),'baseline':ref.tolist(),'definition':'negative centered gradient at fixed p_ref, average over13 losses','right_probability':'fraction of bootstrap directional effects above0'},'平均训练配方处的切向替换收益及500次冻结系数bootstrap分布，正/负点值用橙/靛区分。右侧展示正收益概率，虚线为50%，左侧虚线为零收益。',[src('mixture_directional_effects.csv'),src('problem2_bridge_input.npz')],'分布形状与正收益概率补充区间信息：局部点估计为正不等于方向稳定。收益是固定参照组成附近的模型方向导数，不是追加训练预算的干预效应。')

@register(25)
def response_slice():
 b=D.bridge();domains=list(b['domains']);names=['dm_mathematics','ubuntu_irc','enron_emails'];ix=[domains.index(x) for x in names];ref=b['p_ref_point'];total=ref[ix].sum();trip=np.array([(i/150,j/150,1-(i+j)/150) for i in range(151) for j in range(151-i)]);trip=np.vstack([trip,ref[ix]/total]);vs=np.tile(ref,(len(trip),1));vs[:,ix]=trip*total;dist=cdist(vs/b['support_scale'],b['p_train']/b['support_scale']).min(1);ok=(dist<=b['support_radius']+1e-8)&(vs<=b['upper']+1e-10).all(1);z=D.objective(vs);base=float(D.objective(ref[None,:])[0]);delta=(z-base)*100;xy=P.to_ternary(trip);tri=Triangulation(xy[:,0],xy[:,1]);tri.set_mask((~ok[tri.triangles]).any(1));best=int(np.flatnonzero(ok)[np.argmin(z[ok])]);fig,ax,p=P.figure('q1',(145,117));ax.add_patch(Polygon(P.TRI,fc='#f5f7f2',ec='#aeb7ae',hatch='///',lw=0,zorder=0));norm=TwoSlopeNorm(0,vmin=float(delta[ok].min()),vmax=float(delta[ok].max()));im=ax.tricontourf(tri,delta,levels=70,cmap=p.cmap_div(soft=True),norm=norm);ax.tricontour(tri,delta,levels=[-4,-2,0,2,4,6],colors=p.line['base'],linewidths=.6);ax.tricontour(tri,delta,levels=[0],colors=p.ink,linewidths=1.3);P.ternary_frame(ax,[D.DOMAIN[x] for x in names],fontsize=8.5)
 for q in [.2,.4,.6,.8]:
  ax.text(1-q,-.022,f'{q:.0%}',ha='center',va='top',fontsize=7.5);ax.text(.5*q-.026,np.sqrt(3)/2*q,f'{q:.0%}',ha='right',va='center',fontsize=7.5);ax.text(.5+.5*q+.026,np.sqrt(3)/2*(1-q),f'{q:.0%}',ha='left',va='center',fontsize=7.5)
 refxy=P.to_ternary((ref[ix]/total)[None,:])[0];ax.scatter(*refxy,s=55,fc='white',ec=p.ink,lw=1.3,zorder=5);ax.scatter(*xy[best],s=100,marker='*',fc=p.fill['ochre'],ec=p.line['ochre'],lw=.8,zorder=6);P.colorbar(fig,im,ax,label='相对参照配方的 ΔJ / 10⁻²',shrink=.76,pad=.035,ticks=[-6,-3,0,3,6]);hs=[Line2D([],[],ls='',marker='o',mfc='white',mec=p.ink,ms=5,label='训练平均配方'),Line2D([],[],ls='',marker='*',mfc=p.fill['ochre'],mec=p.line['ochre'],ms=8,label='切片内最低网格点'),Patch(fc='#f5f7f2',ec='#aeb7ae',hatch='///',label='未展示网格区域'),Line2D([],[],c=p.ink,lw=1.3,label='与参照目标相同')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,25,'经验支持域内三元响应切片',{'varying_domains':names,'their_fixed_total':float(total),'fixed_other_components':dict(zip(domains,ref.tolist())),'grid_count':len(vs),'feasible_count':int(ok.sum()),'feasible_objective_range':[float(z[ok].min()),float(z[ok].max())],'reference_objective':base,'grid_step':1/150,'reference_feasible':bool(ok[-1]),'reference_support_distance':float(dist[-1]),'reference_support_margin':float(b['support_radius']-dist[-1]),'reference_min_upper_margin':float(np.min(b['upper']-ref)),'minimum_grid_objective':float(z[best]),'minimum_grid_composition':vs[best].tolist(),'mask_rule':'triangle omitted if any vertex fails support/upper check','design_change':'p_ref slice of three locally beneficial domains, instead of p_star top-share slice'},'固定训练平均配方其余14域，将数学推理、技术对话和商务邮件的总份额重新分配。颜色为相对参照J变化，粗零等值线与白参照点帮助定位改善区；星标只表示可行网格最低值。灰纹包括越界和跨边界三角中未展示的区域。',[src('problem2_bridge_input.npz')],'新版以有意义的参照差异和清晰零等值线呈现局部组合效应，并保留支持约束。该切片的最低网格点不是正式17维近优配方，更不是全局最优证明。')

@register(27)
def mixture_uncertainty():
 b=D.bridge();a=np.load(src('optimal_mixture_bootstrap.npz'),allow_pickle=False);draw=a['p'][a['success']];d=D.csv('optimal_mixture.csv').set_index('domain').loc[b['domains']].copy();d['p_ref']=b['p_ref_point'];order=np.argsort(-d.p_star.to_numpy());d=d.iloc[order];draw=draw[:,order];fig,axes,p=P.facets('q1',1,2,(160,140),sharey=True,wspace=.05,gridspec_kw={'width_ratios':[4,1.2]});ax,right=axes[0]
 for i,r in enumerate(d.itertuples()):
  v=draw[:,i];nonzero=v[v>1e-6];xx=np.linspace(0,min(r.upper_bound,nonzero.max()+.025),160);den=ss.gaussian_kde(nonzero,bw_method=.3)(xx);den=den/den.max()*.33;ax.fill_between(xx,i-den,i+den,fc=PL.SOFT['forest_green'],ec=p.line['forest_green'],lw=.75,alpha=.85);ax.plot([r.ci_low,r.ci_high],[i,i],c=p.line['forest_green'],lw=1.1,zorder=3);ax.scatter(r.p_ref,i+.12,s=26,fc='white',ec=p.line['base'],lw=1,zorder=4);ax.scatter(r.p_star,i-.03,s=31,marker='D',fc=p.fill['forest_green'],ec=p.line['forest_green'],lw=.8,zorder=5);right.barh(i,r.zero_probability,height=.56,fc=PL.SOFT['blue_gray'],ec=p.line['blue_gray'],lw=.8)
 ax.set_yticks(range(17),[D.DOMAIN[x] for x in d.index]);ax.set_ylim(16.6,-.6);ax.set_xlim(-.022,.86);ax.set_xticks([0,.2,.4,.6,.8]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_xlabel('训练份额');right.set_xlim(0,1);right.set_xticks([0,.5,1]);right.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));right.set_xlabel('零份额概率');grid(ax);right.axvline(.5,c=p.ref,ls='--',lw=.85);hs=[Patch(fc=PL.SOFT['forest_green'],ec=p.line['forest_green'],label='非零bootstrap分布'),Line2D([],[],marker='D',c=p.line['forest_green'],mfc=p.fill['forest_green'],lw=1.1,ms=4,label='候选与95%区间'),Line2D([],[],ls='',marker='o',mfc='white',mec=p.line['base'],ms=4,label='训练平均配方')];fig.legend(handles=hs,loc='outside upper center',ncol=2)
 finish(fig,27,'零膨胀近优配比族',{'n_success':len(draw),'rows':d.reset_index().to_dict('records'),'density':'positive bootstrap draws only','interval':'all successful draws including zeros','zero_threshold':1e-6,'reference':'mean of512 training compositions','max_draw':float(draw.max())},'训练平均配方、正式近优参考候选与200次成功bootstrap配比族对照。条件非零密度和包含零的95%区间在同轴展示，右侧为零份额概率；横轴覆盖所有bootstrap份额。',[src('optimal_mixture.csv'),src('optimal_mixture_bootstrap.npz'),src('problem2_bridge_input.npz')],'增加训练参照后可同时判断“建议如何调配”与“建议有多稳定”。所有17域区间下界均为零，因此单一候选不代表域的永久入选或剔除。')

@register(28)
def solution_clusters():
 d=D.csv('optimization_history.csv');objectives=np.round(d.objective.to_numpy(),5);unique=np.sort(np.unique(objectives));b=D.bridge();ix=[list(b['domains']).index(x) for x in MIX_ORDER];composition=[];summary=[];fig,axes,p=P.facets('q1',1,2,(160,100),sharey=True,wspace=.05,gridspec_kw={'width_ratios':[1.4,3.7]});ax,mat=axes[0]
 for i,v in enumerate(unique):
  g=d[objectives==v];yy=i+(np.arange(len(g))-(len(g)-1)/2)*.045;valid=g.feasible.to_numpy(bool);ax.scatter(g.loc[valid,'objective'],yy[valid],s=31,fc=p.fill['forest_green'] if i==0 else PL.SOFT['base'],ec=p.line['forest_green'] if i==0 else p.line['base'],lw=.8);ax.scatter(g.loc[~valid,'objective'],yy[~valid],s=48,marker='x',c=p.line['ochre'],linewidths=1.3,zorder=5);composition.append(g[[f'p_{j}' for j in ix]].mean().to_numpy());summary.append({'objective_round5':float(v),'n':len(g),'support_margin_range':[float(g.support_margin.min()),float(g.support_margin.max())]})
 m=np.array(composition);im=mat.imshow(m,aspect='auto',cmap=p.cmap_seq_soft(),norm=PowerNorm(.6,vmin=0,vmax=max(.3,m.max())),interpolation='none');ax.set_yticks(range(len(unique)),[f'{i+1}（{s["n"]}）' for i,s in enumerate(summary)]);ax.set_ylim(len(unique)-.5,-.55);ax.set_ylabel('终态簇（起点数）');ax.set_xlabel('目标 J');ax.set_xticks([4.77,4.84,4.92]);ax.set_xlim(4.75,4.94);grid(ax);mat.set_xticks(range(17),[D.DOMAIN[x] for x in MIX_ORDER],rotation=90,fontsize=7.5);mat.set_xlabel('终态平均训练份额');mat.tick_params(axis='y',left=False,labelleft=False);P.facet_title(ax,'最终目标');P.facet_title(mat,'组成差异');cb=fig.colorbar(im,ax=list(axes[0]),fraction=.028,pad=.02,shrink=.8);cb.set_label('训练份额');cb.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));cb.outline.set_visible(False)
 fig.legend(handles=[Line2D([],[],ls='',marker='o',mfc=PL.SOFT['base'],mec=p.line['base'],label='可行终态'),Line2D([],[],ls='',marker='x',c=p.line['ochre'],label='未通过可行判据')],loc='outside upper center',ncol=2)
 finish(fig,28,'多起点优化终态',{'n_feasible':int(d.feasible.sum()),'n_failed_feasibility':int((~d.feasible).sum()),'infeasible_start_ids':d.loc[~d.feasible,'start_id'].tolist(),'composition_scope':'all terminal states including two infeasible endpoints; descriptive only','n_starts':len(d),'all_feasible':bool(d.feasible.all()),'clusters':summary,'grouping':'objective rounded5 decimals','support_margin':d.support_margin.tolist(),'cluster_mean_composition':m.tolist(),'domain_order':MIX_ORDER},'30次优化终态按最终目标保留5位小数分簇，其中28次满足可行判据、2次未通过；叉号明确区分后者（起点11、14）。右侧为簇内全部终态组成的描述性平均，包含这两条失败终态，不代表可行决策。括号为起点数，纵向避让仅防点重叠。',[src('optimization_history.csv'),src('optimization_diagnostics.json')],'相近的目标值可能对应不同的资源组成，新版同时呈现目标与配方模式。簇均值只用于描述，不被当作重新验证过可行的决策点；最优簇不能证明全局唯一性。')
