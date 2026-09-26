"""Reference-v2 Q2: reorganized scientific figures, frozen LJY data only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, SymLogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
from scipy.stats import gaussian_kde, spearmanr
from common import plot as P
from style.palettes import SOFT
import config as C
from q2 import data as D
from q2.figures import finish, write_docs

FIGS={}
REF=Path('/Users/sd3/Desktop/project/gmgc2026F/绘图/output/q2')
def reg(k):
    def inner(f):FIGS[k]=f;return f
    return inner

def one(size=(160,100), margins=(.14,.95,.16,.84)):
    fig,ax,pal=P.figure('q2',size,layout=None)
    l,r,b,t=margins;fig.subplots_adjust(left=l,right=r,bottom=b,top=t)
    return fig,ax,pal

def facets(r,c,size=(160,104),sharex=False,sharey=False,ratios=None,margins=(.11,.97,.16,.83),wspace=.25,hspace=.5):
    kw={'gridspec_kw':{'width_ratios':ratios}} if ratios else {}
    fig,axs,pal=P.facets('q2',r,c,size,sharex=sharex,sharey=sharey,**kw)
    fig.set_layout_engine(None);l,rr,b,t=margins;fig.subplots_adjust(left=l,right=rr,bottom=b,top=t,wspace=wspace,hspace=hspace)
    return fig,axs,pal

def legend(fig,handles,ncol=3,y=.99):
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,y),ncol=ncol,frameon=False,fontsize=8,columnspacing=1.2,handlelength=1.8)

def lhandle(pal,key,label,ls='-',marker=None):
    return Line2D([],[],color=pal.line[key],lw=1.5,ls=ls,marker=marker,ms=4.7,mfc=SOFT[key],mec=pal.line[key],label=label)
def src(*fs):return [D.path(f) for f in fs]
def rawsrc(*ks):return [D.rawpath(k) for k in ks]
def save(fig,k,title,stats,caption,sources,reading,reference,change):
    stats={**stats,'visual_reference':str(REF/reference),'design_change':change}
    finish(fig,k,title,stats,caption,sources,reading)
    # Design rationale is prose in the guide, not extra text inside the figure.
    p=C.PLOT/'图表说明_Q2.md'
    text=p.read_text();text=text.replace('本批共20张独立图','本批按用户指定参考重新设计，共20张独立图')
    p.write_text(text)

def logticks(ax,axis,ticks):
    setter=ax.set_xticks if axis=='x' else ax.set_yticks
    setter(ticks,[f'{v:g}' if .01<=v<1e4 else f'$10^{{{int(np.log10(v))}}}$' for v in ticks])
    (ax.xaxis if axis=='x' else ax.yaxis).set_minor_formatter(NullFormatter())

@reg('01')
def evidence():
    fig,ax,pal=one((160,110),(.12,.97,.15,.86));b1,b7,b8=D.raw('B1'),D.raw('B7'),D.raw('B8')
    def box(df,key,alpha=.6,fill=True):
        n0,n1=df.N_params_B.min(),df.N_params_B.max();d0,d1=df.D_tokens_B.min(),df.D_tokens_B.max()
        ax.add_patch(Rectangle((d0,n0),d1-d0,n1-n0,fc=SOFT[key] if fill else 'none',alpha=alpha,ec=pal.line[key],lw=1.1,ls='--' if not fill else '-',zorder=0))
    box(b8,'ochre',.20);box(b7,'leaf_green',.70)
    for n,g in b1.groupby('N_params_B'):
        ax.plot([g.D_tokens_B.min(),g.D_tokens_B.max()],[n,n],color=pal.line['blue_gray'],lw=1.55,zorder=2)
    b4,b5=D.raw('B4'),D.raw('B5');large=D.raw('B9').query('D_tokens_B>0')
    ax.scatter(b4.D_tokens_B,b4.N_params_B,s=21,c=pal.fill['blue_gray'],ec='white',lw=.5,zorder=3)
    ax.scatter(b5.D_tokens_B,b5.N_params_B,s=26,marker='^',c=SOFT['ochre'],ec=pal.line['ochre'],lw=.7,zorder=3)
    # Large-scale metadata are summarized by their genuine point cloud, not a fake filled experiment domain.
    ax.scatter(large.D_tokens_B,large.N_params_B,s=17,marker='D',fc='none',ec=pal.line['indigo'],lw=.65,alpha=.70,zorder=2)
    ax.set_xscale('log');ax.set_yscale('log');ax.set(xlim=(.08,60000),ylim=(.045,22000),xlabel='训练数据量 D / 十亿 Token',ylabel='参数量 N / 十亿')
    ax.set_xticks(10.**np.arange(-1,5),[f'$10^{{{e}}}$' for e in range(-1,5)]);ax.set_yticks(10.**np.arange(-1,5),[f'$10^{{{e}}}$' for e in range(-1,5)]);ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(.16,22,'B1 训练轨迹',color=pal.line['blue_gray'],fontsize=8.5)
    ax.annotate('B7 质量网格',xy=(10,.26),xytext=(.6,.26),color=pal.line['forest_green'],fontsize=8.5,va='center',arrowprops={'arrowstyle':'-|>','color':pal.line['forest_green'],'lw':.8})
    ax.text(5,930,'B8 压力网格',color=pal.line['ochre'],fontsize=8.5)
    ax.text(1000,14000,'B9 大模型元数据',color=pal.line['indigo'],fontsize=8.5)
    hs=[lhandle(pal,'blue_gray','B1 轨迹'),Patch(fc=SOFT['leaf_green'],ec=pal.line['leaf_green'],label='B7 半合成范围'),Patch(fc=SOFT['ochre'],ec=pal.line['ochre'],label='B8 半合成范围'),Line2D([],[],ls='',marker='o',ms=5,mfc=pal.fill['blue_gray'],mec='white',label='B4 跨族记录'),Line2D([],[],ls='',marker='^',ms=5,mfc=SOFT['ochre'],mec=pal.line['ochre'],label='B5 文献记录'),Line2D([],[],ls='',marker='D',ms=4.5,mfc='white',mec=pal.line['indigo'],label='B9 元数据位置')]
    legend(fig,hs,3)
    save(fig,'01','证据覆盖地图',{'B1_rows':len(b1),'B7_rows':len(b7),'B8_rows':len(b8),'B4_rows':len(b4),'B5_rows':len(b5),'B9_positive_D_rows':len(large),'B9_zero_D_rows':int((D.raw('B9').D_tokens_B==0).sum())},'N–D证据覆盖图。B1以八条真实训练轨迹的取值范围呈现，B7/B8半合成网格以范围区域呈现；B4、B5和B9保留原始稀疏记录。范围内部不代表全部组合均有实验观测。',rawsrc('B1','B4','B5','B7','B8','B9'),'B9无Loss且四行D=0未进入对数图；B10沿同类大模型支持域提供估算Loss，不新增独立证据。','Q2-01_证据地图.png','密集重叠格点 → 轨迹、支持区域和稀疏来源点，保留完整稀疏来源而避免重复Q点叠画。')

@reg('19')
def interaction_matrix():
    order=['arxiv','pubmed_central','pubmed_abstracts','nih_exporter','philpapers','github','dm_mathematics','stackexchange','hackernews','ubuntu_irc','enron_emails','pile_cc','wikipedia_en','gutenberg_pg_19','freelaw','uspto_backgrounds','europarl'];labels=['论文','全文','摘要','科研','哲学','代码','数学','问答','资讯','对话','邮件','网页','百科','图书','法律','专利','议会']
    a=D.table('domain_substitution_complementarity.csv');idx={d:i for i,d in enumerate(order)};mat=np.full((16,16),np.nan);stable=[]
    for r in a.itertuples():
        i,j=sorted([idx[r.domain_i],idx[r.domain_j]])
        mat[j-1,i]=r.standardized_interaction
        if r.ci_high<0:stable.append((i,j-1))
    fig,ax,pal=one((160,136),(.13,.88,.17,.84));lim=float(np.ceil(np.nanmax(np.abs(mat))/10)*10);norm=SymLogNorm(linthresh=5,vmin=-lim,vmax=lim)
    im=ax.imshow(np.ma.masked_invalid(mat),cmap=pal.cmap_div(soft=True),norm=norm,interpolation='nearest');ax.set(xticks=range(16),xticklabels=labels[:-1],yticks=range(16),yticklabels=labels[1:]);ax.tick_params(labelsize=7.5,length=0);P.clean_heatmap(ax,lw=.6)
    ax.scatter([p[0] for p in stable],[p[1] for p in stable],s=9,c=pal.ink,marker='o',zorder=3)
    for b in [5,8,11,14]:
        ax.plot([-.5,b-.5],[b-1.5,b-1.5],color=pal.line['base'],lw=1.1);ax.plot([b-.5,b-.5],[b-1.5,15.5],color=pal.line['base'],lw=1.1)
    P.legend_top(ax,handles=[Line2D([],[],ls='',marker='o',color=pal.ink,ms=3,label='95% 区间低于 0'),Line2D([],[],color=pal.line['base'],lw=1.1,label='语义领域分组')],labels=['95% 区间低于 0','语义领域分组'],ncol=2,pad=.04,fontsize=8)
    cb=P.colorbar(fig,im,ax,label='标准化二阶交互',pad=.04,shrink=.77);cb.set_ticks([-100,-20,-5,0,5,20,100]);cb.set_ticklabels(['−100','−20','−5','0','5','20','100'])
    save(fig,'19','领域互补矩阵',{'pairs':len(a),'relations':a.relation.value_counts().to_dict(),'domain_abbreviations':dict(zip(labels,order)),'color_normalization':'symmetric_log, linear threshold=5','semantic_groups':[5,3,3,3,3]},'按学术、技术、交流、综合和政法领域排序的136组二阶交互。点标记95%bootstrap区间完全低于0的互补关系；未标点的格子为不确定关系，不能解释为已验证替代。色阶为对称对数。',src('domain_substitution_complementarity.csv','problem1_bridge.json'),'负向交互表示当前Scheffé模型中相对线性叠加的额外Loss下降；不作因果解释。75对互补、61对不确定。','Q2-40_领域互补矩阵.png','密集斜线的上三角 → 语义分组下三角、柔色填充和显著互补点标记；中文短名映射保存在统计JSON。')

@reg('23')
def iso_shift():
    fig,axs,pal=facets(1,3,(160,90),sharex=True,sharey=True,margins=(.10,.98,.16,.87),wspace=.10);p=D.cp();ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];levels=[2.35,float(D.pred(1,150,ref)),2.8];ns=np.geomspace(.07,11.97,500);rows=[]
    for ax,sq in zip(axs[0],[.5,1,1.5]):
        for target in levels:
            for dq,ls,key in [(0,'-','blue_gray'),(.1,'--','forest_green')]:
                rem=target-p['E']-p['A']*ns**(-p['alpha'])-D.quality(ref+dq,sq);ds=np.full_like(ns,np.nan);ok=rem>0;ds[ok]=(p['B']/rem[ok])**(1/p['beta']);ok&=(ds>=10)&(ds<=600);ax.plot(ds[ok],ns[ok],ls=ls,color=pal.line[key],lw=1.5)
            dlabel=260;rem=target-p['E']-p['B']*dlabel**(-p['beta'])-D.quality(ref,sq)
            if rem>0:
                nlabel=(p['A']/rem)**(1/p['alpha'])
                if .075<nlabel<10:ax.text(dlabel,nlabel*1.17,f'{target:.2f}',fontsize=7.5,ha='left',va='bottom',color=pal.line['blue_gray'],bbox={'fc':'white','ec':'none','pad':.5,'alpha':.9})
        # At the central reference loss, mark the parameter-saving distance for ΔQ=.1.
        target=float(D.pred(1,150,ref,sq));rem=target-p['E']-p['B']*150**(-p['beta'])-D.quality(ref+.1,sq);n2=float((p['A']/rem)**(1/p['alpha']))
        ax.plot([150,150],[n2,1],color=pal.ink,lw=1.3);ax.scatter([150,150],[n2,1],s=24,c=[pal.fill['forest_green'],pal.fill['blue_gray']],ec='white',lw=.7,zorder=5);rows.append({'s_Q':sq,'same_loss_parameter_ratio':n2})
        ax.set_xscale('log');ax.set_yscale('log');ax.set(xlim=(8,700),ylim=(.065,15));logticks(ax,'x',[10,100,600]);logticks(ax,'y',[.1,1,10]);P.facet_title(ax,f'sQ = {sq:g}',fontsize=9);P.light_grid(ax,'both')
    fig.supxlabel('训练数据量 D / 十亿 Token',y=.025,fontsize=9);axs[0,0].set_ylabel('参数量 N / 十亿')
    legend(fig,[lhandle(pal,'blue_gray','Q = Qref'),lhandle(pal,'forest_green','Q = Qref + 0.1','--'),Line2D([],[],color=pal.ink,lw=1.3,marker='o',mfc='white',label='D = 150B 的等效变化')],3)
    save(fig,'23','等损失资源替代',{'loss_levels':levels,'quality_reference':ref,'parameter_saving_at_reference':rows,'conditions':{'p':'p_ref','profile':0}},'三种质量映射斜率下，基线质量与提高0.1后的同Loss等值线。线旁数字为Loss；D=150B的连接点表示在参考Loss下可减少的参数量。曲线由本项目冻结模型解析计算。',src('generalized_scaling_parameters.json'),'分面只改变sQ，固定pref及零配比迁移。连接点表达提质后的参数节省，和保持原质量而扩大参数的有限等价量不是同一百分数；不混用两个方向。','Q2-27_等值线平移图.png','单目标三条线 → 共享轴的三个sQ分面、多Loss配对等值线及等效距离。')

@reg('03')
def training_facets():
    a=D.raw('B1');cv=D.table('classic_scaling_cv.csv').query("model=='M0'").set_index('held_out_N')
    fig,axs,pal=facets(2,4,(160,118),sharex=True,sharey=True,margins=(.10,.98,.15,.85),wspace=.12,hspace=.38)
    for ax,(n,g) in zip(axs.flat,a.groupby('N_params_B')):
        g=g.sort_values('D_tokens_B');xx=np.geomspace(g.D_tokens_B.min(),g.D_tokens_B.max(),200)
        ax.plot(xx,D.l0(n,xx),lw=1.5,color=pal.line['blue_gray']);ax.scatter(g.D_tokens_B.iloc[::3],g.val_loss.iloc[::3],s=12,c=SOFT['blue_gray'],ec=pal.line['blue_gray'],lw=.35,zorder=3)
        ax.set_xscale('log');ax.set(xlim=(.1,400),ylim=(2,4.9));logticks(ax,'x',[.1,3,100]);ax.set_yticks([2,3,4]);P.light_grid(ax);P.facet_title(ax,f'N = {n:.2g}B',fontsize=8.5)
        ax.text(.97,.94,f'RMSE\n{cv.loc[n,"RMSE"]*1e4:.1f}×10⁻⁴',transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color=pal.line['base'])
    fig.supxlabel('训练数据量 D / 十亿 Token',y=.03,fontsize=9);fig.supylabel('验证损失 Loss',x=.025,fontsize=9)
    legend(fig,[lhandle(pal,'blue_gray','M0 冻结拟合'),Line2D([],[],ls='',marker='o',ms=4,mfc=SOFT['blue_gray'],mec=pal.line['blue_gray'],label='B1 检查点（每 3 个显示 1 个）')],2)
    save(fig,'03','原始训练曲线族',{'source_rows':len(a),'display_stride':3,'scales':sorted(a.N_params_B.unique().tolist())},'八个参数规模共享坐标的小多图。曲线为M0冻结拟合，点按每三个检查点取一个等间距显示；各分面给出独立的留一规模RMSE。',rawsrc('B1')+src('classic_scaling_parameters.json','classic_scaling_cv.csv'),'图上的点抽稀只为显示，拟合和验证仍使用全部1176行；RMSE来自留一规模验证，不能误当作各面内拟合误差。','Q2-04_B1训练轨迹与拟合.png','八条叠线 → 八个共享坐标分面，增加真实留一规模误差层，解决同色线聚集。')

@reg('04')
def collapse_residual():
    a=D.raw('B1');p=D.cp();fig,axs,pal=facets(2,1,(160,113),sharex=True,margins=(.14,.88,.15,.88),hspace=.20);up,down=axs[:,0];cmap=LinearSegmentedColormap.from_list('q2_n', [SOFT['blue_gray'],pal.fill['blue_gray'],pal.line['indigo']]);norm=LogNorm(a.N_params_B.min(),a.N_params_B.max());rows=[]
    for n,g in a.groupby('N_params_B'):
        g=g.sort_values('D_tokens_B');d=g.D_tokens_B.to_numpy();trans=g.val_loss.to_numpy()-p['E']-p['A']*n**(-p['alpha']);expected=p['B']*d**(-p['beta']);c=cmap(norm(n));up.scatter(d,trans,s=10,fc=c,ec='white',lw=.2,alpha=.6);down.plot(d,(trans/expected-1)*100,color=c,lw=.9,alpha=.8);rows.append({'N_B':n,'max_relative_deviation_percent':float(np.max(np.abs((trans/expected-1)*100)))})
    xx=np.geomspace(a.D_tokens_B.min(),a.D_tokens_B.max(),300);up.plot(xx,p['B']*xx**(-p['beta']),color=pal.line['ochre'],lw=1.6,ls='--');up.set_yscale('log');logticks(up,'y',[.25,.5,1,2]);up.set_ylabel('剥离参数项后的损失');down.axhline(0,color=pal.line['base'],lw=.8,ls='--');down.set_ylabel('相对偏离 / %');down.set_xlabel('训练数据量 D / 十亿 Token')
    for ax in [up,down]:ax.set_xscale('log');ax.set_xlim(.1,400);P.light_grid(ax)
    logticks(down,'x',[.1,1,10,100]);cb=P.colorbar(fig,plt.cm.ScalarMappable(norm=norm,cmap=cmap),up,label='N / 十亿',pad=.03,shrink=.95,match_axes=True,width=.025);cb.set_ticks([.1,1,10]);cb.set_ticklabels(['0.1','1','10'])
    legend(fig,[lhandle(pal,'ochre','M0 数据幂律','--'),lhandle(pal,'base','零相对偏离','--')],2)
    save(fig,'04','幂律数据坍缩',{'rows':len(a),'beta':p['beta'],'scale_deviation':rows},'上方为剥离不可约项与参数项后的八规模数据坍缩，下方为各规模相对冻结数据幂律的百分比偏离。色阶沿用参数规模，零线用于识别微小系统偏差。',rawsrc('B1')+src('classic_scaling_parameters.json'),'下方放大残差使近乎完全重叠的数据仍可检查；这是结构诊断，不新增独立验证证据。','Q2-04_B1训练轨迹与拟合.png','单条坍缩线 → 共享D轴的结构与相对残差两层，显示原图隐藏的规模差异。')

@reg('10')
def blocked_selection():
    a=D.table('quality_model_comparison.csv');summ=D.obj('quality_model_parameters.json')['selection_diagnostics'];models=['GQ1','GQ2','GQ3','GQ4'];splits=['leave_N','leave_D','leave_Q','leave_cell'];names=['留 N','留 D','留 Q','留 (N,D)'];cols=['slate_teal','forest_green','blue_gray','ochre']
    fig,axs,pal=facets(1,4,(160,92),sharex=True,sharey=True,margins=(.10,.98,.16,.87),wspace=.20)
    stats=[]
    for ax,s,title in zip(axs[0],splits,names):
        for i,(m,key) in enumerate(zip(models,cols)):
            vals=a[(a.model==m)&(a.split_type==s)].RMSE.to_numpy();j=np.random.default_rng(i+2026).uniform(-.15,.15,len(vals));lo,hi=np.quantile(vals,[.25,.75]);mean=vals.mean();ax.plot([lo,hi],[i,i],color=SOFT[key],lw=9,solid_capstyle='butt');ax.scatter(vals,i+j,s=11,c=pal.fill[key],ec='white',lw=.3,alpha=.7);ax.scatter([mean],[i],s=32 if m=='GQ2' else 26,marker='D',fc=pal.line[key],ec='white',lw=.7,zorder=5);stats.append({'split':s,'model':m,'mean':mean,'folds':len(vals)})
        P.facet_title(ax,f'{title}（{len(a[(a.model=="GQ2")&(a.split_type==s)])} 折）',fontsize=8);ax.set(yticks=range(4),yticklabels=models,xlim=(0,float(np.ceil(a.RMSE.max()/.05)*.05)),ylim=(3.5,-.5),xticks=[.05,.15,.25]);P.light_grid(ax,'x')
    fig.supxlabel('分块验证 RMSE',y=.035,fontsize=9)
    legend(fig,[Line2D([],[],ls='',marker='o',mfc=SOFT['base'],mec='white',ms=4,label='逐折误差'),Line2D([],[],color=SOFT['base'],lw=7,label='折间 IQR'),Line2D([],[],ls='',marker='D',color=pal.line['forest_green'],ms=5,label='家族均值')],3)
    save(fig,'10','质量模型分块选择',{'fold_summaries':stats,'GQ2_equal_family_score':summ['score_GQ2'],'family_weight':.25},'四类切分的逐折RMSE、四分位范围和家族均值并列展示。GQ2在四类家族均值上均最低，正式选模对四类各赋0.25权重。',src('quality_model_comparison.csv','quality_model_parameters.json'),'区间展示折间离散性，不是均值的95%置信区间；不能因leave_cell有45折而提高它的选模权重。','Q2-18_质量形式1SE选择图.png','四条均值折线 → 四类验证任务的分布分面，同时展示个体折、IQR和选模所用均值。')

@reg('14')
def penalty_gain():
    a=D.table('quality_model_bootstrap.csv.gz').query('success');p=D.qp();ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];q=np.linspace(.1,1,251);draw=a.c_Q.to_numpy()[:,None]*(1-q)**a.nu_Q.to_numpy()[:,None];lo,hi=np.quantile(draw,[.025,.975],axis=0)
    fig,axs,pal=facets(1,2,(160,88),ratios=[1.4,1],margins=(.12,.97,.19,.83),wspace=.34);ax,eff=axs[0]
    ax.fill_between(q,lo,hi,color=SOFT['forest_green'],alpha=.95);ax.plot(q,p['c_Q']*(1-q)**p['nu_Q'],color=pal.line['forest_green'],lw=1.6);ax.set(xlim=(.08,1.02),ylim=(-.01,.36),xlabel='质量评分 Q',ylabel='质量惩罚 ΔL');P.light_grid(ax)
    gains=[]
    for i,dq in enumerate([.05,.1]):
        vals=a.c_Q*((1-ref)**a.nu_Q-(1-ref-dq)**a.nu_Q);l,m,h=np.quantile(vals,[.025,.5,.975]);point=p['c_Q']*((1-ref)**p['nu_Q']-(1-ref-dq)**p['nu_Q']);eff.plot([l,h],[i,i],lw=7,color=SOFT['forest_green'],solid_capstyle='butt');eff.plot([l,h],[i,i],lw=1.4,color=pal.line['forest_green']);eff.scatter([point],[i],s=40,marker='D',fc=pal.fill['forest_green'],ec='white',zorder=4);gains.append({'delta_Q':dq,'point':float(point),'CI95':[float(l),float(h)]})
    eff.set(yticks=[0,1],yticklabels=['ΔQ = 0.05','ΔQ = 0.10'],ylim=(-.65,1.65),xlabel='Loss 改善量',xlim=(.012,.043));P.light_grid(eff,'x');P.facet_title(eff,f'Qref = {ref:.3f}',fontsize=8.5)
    legend(fig,[lhandle(pal,'forest_green','GQ2 点估计',marker='D'),Patch(fc=SOFT['forest_green'],label='95% 条件区间')],2)
    save(fig,'14','质量惩罚条件区间',{'draws':len(a),'Q_ref':ref,'gain_intervals':gains},'GQ2质量惩罚曲线及参考质量处有限提质收益。带和横线为既有500次(N,D)簇bootstrap的95%分位区间，经典参数固定；菱形为原始点估计。',src('quality_model_parameters.json','quality_model_bootstrap.csv.gz','generalized_scaling_parameters.json'),'右侧把曲线转成ΔQ=0.05/0.10的直接收益，区间未包含跨体系映射不确定性。','Q2-35_等效倍数不确定性阶梯.png','仅曲线与带 → 连续响应及有限收益的两种互补统计层，避免大幅单薄三角带。')

@reg('06')
def joint_marginals():
    a=D.valid_classic();x=(a.alpha-.34)*1e4;y=(a.beta-.28)*1e4;p=D.cp();fig,ax,pal=one((135,115),(.16,.77,.16,.72));topax=fig.add_axes([.16,.75,.61,.12],sharex=ax);right=fig.add_axes([.79,.16,.12,.56],sharey=ax)
    ax.scatter(x,y,s=16,fc=SOFT['blue_gray'],ec=pal.line['blue_gray'],lw=.4,alpha=.65);ax.scatter([(p['alpha']-.34)*1e4],[(p['beta']-.28)*1e4],marker='+',s=115,c=pal.ink,lw=2,zorder=5)
    topax.hist(x,bins=22,color=SOFT['blue_gray'],ec='white',lw=.5);right.hist(y,bins=22,orientation='horizontal',color=SOFT['blue_gray'],ec='white',lw=.5)
    for value in np.quantile(x,[.025,.975]):topax.axvline(value,color=pal.line['blue_gray'],ls='--',lw=.9)
    for value in np.quantile(y,[.025,.975]):right.axhline(value,color=pal.line['blue_gray'],ls='--',lw=.9)
    for a2 in [topax,right]:
        for spine in a2.spines.values():spine.set_visible(False)
        a2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set(xlabel='(α − 0.34) × 10⁴',ylabel='(β − 0.28) × 10⁴');P.light_grid(ax,'both')
    legend(fig,[Line2D([],[],ls='',marker='o',ms=4,mfc=SOFT['blue_gray'],mec=pal.line['blue_gray'],label='484 次有效抽样'),Line2D([],[],ls='',marker='+',ms=7,mew=1.5,color=pal.ink,label='原始点估计'),lhandle(pal,'blue_gray','边际 95% 分位','--')],2)
    save(fig,'06','经典参数联合分布',{'valid_draws':len(a),'failed_draws':500-len(a),'alpha_ci95':np.quantile(a.alpha,[.025,.975]).tolist(),'beta_ci95':np.quantile(a.beta,[.025,.975]).tolist(),'correlation':float(a.alpha.corr(a.beta))},'经典参数α和β的规模簇bootstrap联合点云及边际分布。虚线仅标各参数的95%分位，十字是原始点估计；坐标经过平移与放大以显示估计波动。',src('classic_scaling_bootstrap.csv.gz','classic_scaling_parameters.json'),'不绘制依赖正态假设的联合置信椭圆；16次失效抽样未混入分布。','Q2-19_质量系数联合分布.png','整幅深色密度背景 → 白底联合点云、两条边际直方图和明确边际分位。')

@reg('08')
def external_facets():
    D.BFILES['B2']='cerebras_training_log.csv';fig,axs,pal=facets(2,2,(160,137),margins=(.11,.98,.14,.87),wspace=.26,hspace=.36);metrics=D.table('classic_external_validation.csv').set_index('source');spec=[('B2','半合成','forest_green','o'),('B4','跨族记录','blue_gray','o'),('B5','文献记录','ochre','^'),('B10','估算对照','indigo','D')];result=[]
    for ax,(key,title,c,marker) in zip(axs.flat,spec):
        a=D.raw(key);pred=D.l0(a.N_params_B.to_numpy(),a.D_tokens_B.to_numpy());obs=a.val_loss.to_numpy();low=min(obs.min(),pred.min());high=max(obs.max(),pred.max());pad=.05*(high-low);ax.plot([low-pad,high+pad],[low-pad,high+pad],c=pal.line['base'],ls='--',lw=.8)
        ax.scatter(obs,pred,s=11 if len(a)>500 else 26,marker=marker,fc='none' if key=='B10' else SOFT[c],ec=pal.line[c],lw=.5,alpha=.60 if len(a)>500 else .86)
        ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad));ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));P.light_grid(ax,'both');P.facet_title(ax,f'{key} {title}（n = {len(a)}）',fontsize=8.5)
        err=float(metrics.loc[key,"MAE"]);txt=f'MAE = {err*1e4:.1f}×10⁻⁴' if err<.001 else f'MAE = {err:.3f}'
        if key in ['B4','B5']:txt+=f'\n中心化 = {metrics.loc[key,"centered_MAE"]:.3f}'
        ax.text(.04,.95,txt,transform=ax.transAxes,fontsize=7.5,ha='left',va='top',bbox={'fc':'white','ec':'none','alpha':.90,'pad':1.5});result.append({'source':key,'n':len(a),'recalculated_MAE':float(np.mean(abs(obs-pred)))})
    fig.supxlabel('来源表中的 Loss',y=.025,fontsize=9);fig.supylabel('冻结经典项预测 Loss',x=.02,fontsize=9);legend(fig,[lhandle(pal,'base','y = x','--')],1)
    save(fig,'08','外部误差结构诊断',{'recalculated_errors':result,'reported':metrics.reset_index().to_dict('records')},'冻结经典项的四来源预测诊断。B2为半合成、B4为跨族、B5为文献、B10为空心估算对照，各面单独保留来源的Loss范围与误差指标。B4按family、B5按source中心化；B2/B10不报告中心化改善。',rawsrc('B2','B4','B5','B10')+src('classic_scaling_parameters.json','classic_external_validation.csv'),'B10的接近对角线不能构成独立大模型验证。中心化前后指标仅比较同一来源，不把不同族的绝对Loss视为统一标准。','Q2-11_外部检验对角线图.png','四个汇总小点 → 各来源真实逐点预测分面，汇总MAE作为辅助统计，解决空白过多且口径不明。')

@reg('09')
def quality_matrix_facets():
    a=D.raw('B7');base=a.query('Q_score==1')[['N_params_B','D_tokens_B','val_loss']].rename(columns={'val_loss':'anchor'});a=a.merge(base,on=['N_params_B','D_tokens_B']);a['delta']=a.val_loss-a.anchor
    fig,axs,pal=facets(3,3,(160,140),sharex=True,sharey=True,margins=(.11,.89,.12,.91),wspace=.12,hspace=.43);lim=max(abs(a.delta.min()),abs(a.delta.max()));norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    for ax,(n,g) in zip(axs.flat,a.groupby('N_params_B')):
        m=g.pivot(index='D_tokens_B',columns='Q_score',values='delta');im=ax.imshow(m,aspect='auto',origin='lower',cmap=pal.cmap_div(soft=True),norm=norm,interpolation='nearest');ax.set(xticks=[0,4,9],xticklabels=['0.1','0.5','1.0'],yticks=range(5),yticklabels=[f'{v:g}' for v in m.index]);P.clean_heatmap(ax,lw=.4);P.facet_title(ax,f'N = {n:g}B',fontsize=8.5)
    cax=fig.add_axes([.915,.19,.017,.62]);cb=fig.colorbar(im,cax=cax);cb.outline.set_visible(False);cb.set_label('Loss(Q) − Loss(1)');fig.supxlabel('质量评分 Q',y=.025,fontsize=9);fig.supylabel('训练数据量 D / 十亿 Token',x=.025,fontsize=9)
    save(fig,'09','质量增量响应矩阵',{'cells':45,'rows':len(a),'delta_range':[float(a.delta.min()),float(a.delta.max())],'negative_delta_rows':int((a.delta<0).sum())},'B7半合成质量增量按九个参数规模分面，每面为五个D水平与十个Q水平的原始网格。所有面使用同一有符号色阶，相对同单元Q=1中心化。',rawsrc('B7'),'没有跨格平滑或强行单调化；负增量保留。Q=1中心化只用于描述，未进入留出验证预测器。','Q2-15_组内中心化质量响应束.png','45行长条热图 → 九规模共享D/Q轴的小矩阵，清楚区分参数与数据两个层级。')

@reg('12')
def conditional_calibration():
    a=D.cv_predictions();fig,axs,pal=facets(1,3,(160,84),sharex=True,sharey=True,margins=(.11,.98,.19,.86),wspace=.12);Ns=[.07,1,11.97];stats=[]
    for ax,n in zip(axs[0],Ns):
        g=a[np.isclose(a.N_params_B,n)]
        for f,key,marker in [('val_loss','forest_green','o'),('predicted','blue_gray','D')]:
            q=g.groupby('Q_score')[f].quantile([.25,.5,.75]).unstack();x=q.index.to_numpy();ax.fill_between(x,q[.25],q[.75],color=SOFT[key],alpha=.40);ax.plot(x,q[.5],color=pal.line[key],marker=marker,ms=4,mfc=SOFT[key],mec='white',mew=.6,lw=1.4);stats.append({'N':n,'field':f,'quantiles':q.reset_index().to_dict('records')})
        P.facet_title(ax,f'N = {n:g}B',fontsize=8.5);ax.set(xlim=(.08,1.02),xticks=[.1,.5,1]);P.light_grid(ax)
    axs[0,0].set_ylabel('验证损失 Loss');fig.supxlabel('质量评分 Q',y=.035,fontsize=9)
    legend(fig,[lhandle(pal,'forest_green','半合成中位数',marker='o'),lhandle(pal,'blue_gray','留出预测中位数',marker='D'),Patch(fc=SOFT['base'],label='D 单元间 IQR')],3)
    save(fig,'12','离散质量条件分布',{'selection':'GQ2 leave_cell','display_N':Ns,'display_rows':sum(np.isclose(a.N_params_B,n).sum() for n in Ns),'conditional_quantiles':stats},'GQ2留资源单元预测在小、中、大三个代表参数规模下的质量响应。曲线为五个D单元的中位数，浅带为其四分位范围；半合成值与严格留出预测使用相同分组。',src('quality_block_predictions.csv.gz'),'仅取覆盖规模范围的三个N水平作条件检查，不把150条展示记录说成全450条；IQR是不同D的异质性，不是预测置信区间。','Q2-15_组内中心化质量响应束.png','合并所有N的10档误差棒 → 三规模共享轴的成对条件响应与IQR带，使收缩和偏移可定位。')

@reg('13')
def residual_clouds():
    a=D.table('quality_block_predictions.csv.gz').query("model=='GQ2'");fig,ax,pal=one((160,106),(.13,.95,.16,.83));splits=['leave_N','leave_D','leave_Q','leave_cell'];labels=['留 N','留 D','留 Q','留 (N,D)'];keys=['blue_gray','forest_green','leaf_green','ochre'];stats=[]
    for i,(s,key) in enumerate(zip(splits,keys)):
        vals=a[a.split_type==s].residual.to_numpy();kde=gaussian_kde(vals);h=kde.factor*vals.std();yy=np.linspace(vals.min()-2*h,vals.max()+2*h,250);den=kde(yy);den=den/den.max()*.30;ax.fill_betweenx(yy,i-.02-den,i-.02,color=SOFT[key],ec=pal.line[key],lw=.8);jit=np.random.default_rng(2026+i).uniform(.06,.25,len(vals));ax.scatter(i+jit,vals,s=6,fc=pal.fill[key],ec='white',lw=.2,alpha=.45);q1,med,q3=np.quantile(vals,[.25,.5,.75]);ax.plot([i,i],[q1,q3],color=pal.line[key],lw=5);ax.scatter([i],[med],s=28,fc='white',ec=pal.line[key],lw=1,zorder=4);stats.append({'split':s,'n':len(vals),'median':float(med),'q25':float(q1),'q75':float(q3),'MAE':float(np.mean(abs(vals)))})
    ax.axhline(0,color=pal.line['base'],ls='--',lw=.8);ax.set(xticks=range(4),xticklabels=labels,xlim=(-.6,3.5),ylabel='预测 Loss − 半合成 Loss');P.light_grid(ax)
    legend(fig,[Patch(fc=SOFT['base'],ec=pal.line['base'],label='逐点残差与核密度'),Line2D([],[],c=pal.line['base'],lw=4,marker='o',mfc='white',label='中位数与 IQR'),lhandle(pal,'base','零残差','--')],3)
    save(fig,'13','分块残差云雨分布',{'split_summaries':stats,'raw_rows_per_split':450},'四类留出切分的GQ2残差云雨图。半密度、逐点和中位数/IQR在同一竖向损失尺度上比较，450个逐点残差在各切分中分别保留。',src('quality_block_predictions.csv.gz'),'四列是同一原始样本的不同验证切分，不能合并为1800个独立观测；只作残差分布诊断。','../q4/Q4-06_六任务零膨胀云雨图.png','横向长条分布 → 四列紧凑竖向云雨和共同零线，标记白边并增强中位/IQR层次。')

@reg('15')
def direction_examples():
    fig,axs,pal=facets(2,2,(160,113),margins=(.11,.98,.15,.87),wspace=.23,hspace=.48);stats=[]
    for ax,n in zip(axs.flat,[.07,1,6.9]):
        for key,col in [('B7','forest_green'),('B8','ochre')]:
            g=D.raw(key);g=g[np.isclose(g.N_params_B,n)&np.isclose(g.D_tokens_B,150)].sort_values('Q_score');yy=g.val_loss-g.val_loss.mean();ax.plot(g.Q_score,yy,color=pal.line[col],marker='o' if key=='B7' else 's',mfc=SOFT[col],mec='white',ms=4,mew=.6,lw=1.4)
        ax.axhline(0,color=pal.line['base'],ls='--',lw=.7);ax.set(xlim=(.02,1.03),ylim=(-1.15,1.15),xticks=[.1,.5,1],ylabel='组内中心化 Loss');P.facet_title(ax,f'N = {n:g}B，D = 150B',fontsize=8.5);P.light_grid(ax)
    ax=axs[1,1]
    for i,(key,col) in enumerate([('B7','forest_green'),('B8','ochre')]):
        vals=np.array([spearmanr(g.Q_score,g.val_loss).statistic for _,g in D.raw(key).groupby(['N_params_B','D_tokens_B'])]);jit=np.random.default_rng(2026+i).uniform(-.18,.18,len(vals));ax.scatter(vals,i+jit,s=13,fc=pal.fill[col],ec='white',lw=.4,alpha=.70);ax.plot([np.median(vals)]*2,[i-.3,i+.3],color=pal.line[col],lw=1.6);stats.append({'source':key,'n':len(vals),'median_rho':float(np.median(vals)),'positive':int((vals>0).sum()),'negative':int((vals<0).sum())})
    ax.axvline(0,c=pal.line['base'],ls='--',lw=.7);ax.set(xlim=(-1.07,1.07),xticks=[-1,0,1],ylim=(-.5,1.5),yticks=[0,1],yticklabels=['B7','B8'],xlabel='单元内 Spearman ρ');P.facet_title(ax,'全部 N–D 单元',fontsize=8.5)
    axs[1,0].set_xlabel('质量评分 Q');legend(fig,[lhandle(pal,'forest_green','B7 半合成',marker='o'),lhandle(pal,'ochre','B8 半合成',marker='s'),Line2D([],[],ls='',marker='|',ms=9,c=pal.line['base'],label='相关中位数'),lhandle(pal,'base','零参考','--')],4)
    save(fig,'15','质量响应方向冲突',{'summary':stats,'representative_cells':{'N_B':[.07,1,6.9],'D_B':150}},'三个代表N–D单元的原始质量响应经组内中心化后比较，右下以全部单元相关系数验证方向。B7全部45个单元负相关，B8全部150个单元正相关。',rawsrc('B7','B8'),'代表曲线只展示原始存在的Q水平，没有补齐B8缺失档位。两类均为半合成资料，方向冲突用于压力测试而非真实机制反转结论。','Q2-15_组内中心化质量响应束.png','仅两团相关点 → 三个原始条件响应+全量相关汇总，让方向冲突可以追溯到真实表格值。')

@reg('16')
def floor_thresholds():
    a=D.raw('B8');ns=np.sort(a.N_params_B.unique());ds=np.sort(a.D_tokens_B.unique());mat=np.full((len(ds),len(ns)),np.nan);incomplete=[]
    for (n,d),g in a.groupby(['N_params_B','D_tokens_B']):
        floor=g[np.isclose(g.val_loss,.5)];i=np.searchsorted(ds,d);j=np.searchsorted(ns,n)
        if len(floor):mat[i,j]=floor.Q_score.max()
        if g.Q_score.nunique()<12:incomplete.append((j,i))
    fig,ax,pal=one((160,111),(.12,.88,.17,.84));cmap=pal.cmap_seq_soft().copy();cmap.set_bad('white');im=ax.imshow(np.ma.masked_invalid(mat),origin='lower',aspect='auto',cmap=cmap,vmin=.05,vmax=.4,interpolation='nearest')
    for i in range(len(ds)):
        for j in range(len(ns)):
            v=mat[i,j];ax.text(j,i,'—' if not np.isfinite(v) else f'{v:.2f}'.rstrip('0'),ha='center',va='center',fontsize=7.5,color=pal.ink)
    for x,y in incomplete:ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,hatch='///',ec='#AAB5AC',lw=0))
    ax.axvline(8.5,color='white',lw=2.5);ax.set(xticks=range(len(ns)),xticklabels=[f'{x:g}' for x in ns],yticks=range(len(ds)),yticklabels=[f'{x:g}' for x in ds],xlabel='参数量 N / 十亿',ylabel='训练数据量 D / 十亿 Token');ax.tick_params(axis='x',rotation=45,labelsize=7.5);P.bracket(ax,-.45,8.45,1.035,'校准规模',fontsize=8);P.bracket(ax,8.55,14.45,1.035,'外推规模',fontsize=8)
    cb=P.colorbar(fig,im,ax,label='Loss = 0.5 的最高 Q',pad=.03);cb.set_ticks([.05,.1,.2,.3,.4]);cb.set_ticklabels(['0.05','0.1','0.2','0.3','0.4'])
    legend(fig,[Line2D([],[],lw=0,marker='_',ms=7,color=pal.ink,label='未触及地板'),Patch(fc='white',ec='#AAB5AC',hatch='///',label='Q 档不完整')],2)
    save(fig,'16','截断地板边界',{'floor_records':int(np.isclose(a.val_loss,.5).sum()),'threshold_cells':int(np.isfinite(mat).sum()),'incomplete_cells':len(incomplete),'threshold_range':[float(np.nanmin(mat)),float(np.nanmax(mat))]},'B8半合成补充集的离散资源格点Loss=0.5地板阈值。格内为已记录的最高触底Q，横线表示未触底；斜线是Q档位不完整，顶部区分校准和外推规模。',rawsrc('B8'),'数值、类别间隔和格点均按原始离散表格保留，未把边界连续化。362条触底记录不是正常连续Loss尾部。','Q2-24_B8下限堆积直方图.png','无数值整片浅色网格 → 校准/外推分组、逐格阈值和未触底状态可读的柔色矩阵。')

@reg('18')
def transfer_dumbbell():
    a=D.table('mixture_transfer_parameters.csv');scales=['1M','60M','1B'];piv=a.pivot(index='loss_domain',columns='scale',values='slope');order=['__composite__']+[x for x in piv.index if x!='__composite__'];aliases={'__composite__':'综合响应','arxiv':'论文','freelaw':'法律','pubmed_central':'医学全文','wikipedia_en':'百科','dm_mathematics':'数学','github':'代码','stackexchange':'问答','gutenberg_pg_19':'图书','pile_cc':'网页','ubuntu_irc':'技术对话','hackernews':'技术资讯','pubmed_abstracts':'医学摘要','uspto_backgrounds':'专利'}
    fig,ax,pal=one((145,130),(.20,.96,.13,.88));ax.axhspan(-.5,.5,fc=SOFT['leaf_green'],alpha=.4);keys=['blue_gray','forest_green','ochre'];marks=['o','s','D']
    for i,d in enumerate(order):
        vals=piv.loc[d,scales].to_numpy();ax.plot([min(vals),max(vals)],[i,i],color=SOFT['base'],lw=2)
        for val,key,m in zip(vals,keys,marks):ax.scatter(val,i,s=39 if i==0 else 28,marker=m,fc=SOFT[key],ec=pal.line[key],lw=1,zorder=4)
    ax.axvline(0,color=pal.line['base'],lw=.8,ls='--');ax.set(yticks=range(len(order)),yticklabels=[aliases.get(d,d) for d in order],ylim=(len(order)-.5,-.5),xlabel='R(p) → 标准化 Loss 的迁移斜率');P.light_grid(ax,'x');legend(fig,[Line2D([],[],ls='',marker=m,mfc=SOFT[k],mec=pal.line[k],ms=5,label=s) for k,s,m in zip(keys,scales,marks)]+[Line2D([],[],color=SOFT['base'],lw=2,label='同域范围'),lhandle(pal,'base','零斜率','--')],5)
    save(fig,'18','配比响应跨规模路径',{'slopes':piv.loc[order].reset_index().to_dict('records'),'composite_1B_to_1M':float(piv.loc['__composite__','1B']/piv.loc['__composite__','1M'])},'13个验证域及综合响应在三种检验规模下的迁移斜率。每行连接同一验证域的斜率范围，三种点形代表1M、60M、1B；综合行用浅底区分。',src('mixture_transfer_parameters.csv'),'点是冻结稳健回归系数，没有给它们增加未经计算的置信区间；跨规模幅度不可解释为A/B绝对Loss强度已识别。','Q2-13_B4分族森林图.png','13条交叉路径 → 按领域对齐的三规模配对点图，保留全部领域身份和综合行。')

@reg('21')
def response_distribution():
    r=D.rstar_draws();lo,med,hi=np.quantile(r,[.025,.5,.975]);point=D.obj('problem1_bridge.json')['R_star'];fig,axs,pal=facets(1,2,(160,85),margins=(.11,.98,.19,.85),wspace=.27);ax,ecdf=axs[0];bins=np.linspace(r.min()-.15,r.max()+.15,27);ax.hist(r,bins=bins,density=True,color=SOFT['blue_gray'],ec='white',lw=.5);grid=np.linspace(bins[0],bins[-1],300);ax.plot(grid,gaussian_kde(r)(grid),color=pal.line['blue_gray'],lw=1.5);ax.axvline(point,color=pal.line['ochre'],lw=1.2,ls='-.');ax.axvline(0,color=pal.line['base'],lw=.8,ls='--');ax.set(xlabel='R(p*)',ylabel='概率密度');P.light_grid(ax)
    rr=np.sort(r);ecdf.step(rr,np.arange(1,len(r)+1)/len(r),where='post',color=pal.line['forest_green'],lw=1.6);ecdf.axvspan(lo,hi,fc=SOFT['forest_green'],alpha=.32);ecdf.axvline(0,c=pal.line['base'],lw=.8,ls='--');prob=float(np.mean(r<0));ecdf.scatter([0],[prob],s=33,c=pal.fill['forest_green'],ec='white',zorder=4);ecdf.set(xlabel='R(p*)',ylabel='经验累计比例',ylim=(0,1.04));P.light_grid(ecdf);ecdf.text(.04,.92,f'R < 0：{prob:.1%}',transform=ecdf.transAxes,fontsize=8,va='top',color=pal.line['forest_green'])
    legend(fig,[lhandle(pal,'ochre','原始点估计','-.'),lhandle(pal,'base','R = 0','--'),Patch(fc=SOFT['forest_green'],label='95% 分位范围')],3)
    save(fig,'21','近优配比响应不确定性',{'valid_draws':len(r),'R_point':point,'median':med,'ci95':[lo,hi],'empirical_fraction_R_below_zero':prob},'固定近优配比p*的500次既有响应抽样分布及经验累计曲线。右侧浅带为95%分位范围，R<0百分比为抽样中负响应的经验占比，不是后验优胜概率。',src('problem1_mixture_response_bootstrap.npz','problem1_bridge.json'),'95%区间仍跨0，近优配比仅作敏感性情景。没有混入sQ或迁移强度情景的广义bootstrap。','Q2-19_质量系数联合分布.png','单一巨大密度面 → 频数形态与ECDF的互补分布视图，区间跨0与尾部概率均可读。')

@reg('24')
def ternary_facets():
    a=D.table('elasticity_grid.csv').query("p_scenario=='p_ref' and mixture_strength_profile==0");fig,axs,pal=facets(1,3,(160,70),margins=(.03,.99,.12,.81),wspace=.03);ns=sorted(a.N_params_B.unique());keys=['blue_gray','forest_green','ochre'];marks=['o','D','^'];counts={}
    for ax,sq in zip(axs[0],[.5,1,1.5]):
        g=a[np.isclose(a.s_Q,sq)];v=g[['epsilon_N','epsilon_D','epsilon_Q']].to_numpy();v=v/v.sum(1,keepdims=True);xy=P.to_ternary(v);ax.set_xticks([]);ax.set_yticks([]);P.ternary_frame(ax,['εN','εD','εQ'],grid=(.25,.5,.75),fontsize=9)
        for n,key,m in zip(ns,keys,marks):
            keep=np.isclose(g.N_params_B,n);ax.scatter(xy[keep,0],xy[keep,1],s=32,marker=m,fc=SOFT[key],ec=pal.line[key],lw=.8)
        for val in [.25,.5,.75]:
            pts=P.to_ternary(np.array([[1-val,0,val],[val,1-val,0],[0,val,1-val]]));ax.text(pts[0,0],-.026,f'{val:g}',ha='center',va='top',fontsize=7.5);ax.text(pts[1,0]-.032,pts[1,1],f'{val:g}',ha='right',va='center',fontsize=7.5);ax.text(pts[2,0]+.032,pts[2,1],f'{val:g}',ha='left',va='center',fontsize=7.5)
        P.facet_title(ax,f'sQ = {sq:g}',fontsize=8.5);counts[str(sq)]={lab:int(np.sum(v.argmax(1)==j)) for j,lab in enumerate(['N','D','Q'])}
    legend(fig,[lhandle(pal,key,f'N = {n:g}B',marker=m) for n,key,m in zip(ns,keys,marks)],3)
    save(fig,'24','资源弹性组成',{'rows':len(a),'dominance_counts':counts,'normalization':'epsilon / sum(epsilon_N,epsilon_D,epsilon_Q)','p':'p_ref','profile':0},'三种sQ情景的总Loss弹性份额三元图。每面27个N–D–Q状态，三种参数规模以颜色和点形共同区分；每点三份额之和为1，三边均给分位刻度。',src('elasticity_grid.csv'),'εN、εD、εQ在此表示各自弹性除以三者和的份额，不是原始弹性，也不是预算份额。','../q3/Q3-21_成本份额三元图.png','81点单一三元图 → 三种映射强度的对齐三元分面，N用点形冗余编码，降低遮挡。')

@reg('25')
def elasticity_phases():
    p=D.cp();qp=D.qp();ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];xs=np.linspace(np.log10(.070542),np.log10(11.965825),200);qs=np.linspace(.1,.9,150);X,Q=np.meshgrid(xs,qs);N=10**X;maps=[]
    for sq in [.5,1,1.5]:
        raw=ref+sq*(Q-ref);eff=np.clip(raw,1e-6,1);interior=(raw>1e-6)&(raw<1);gN=p['alpha']*p['A']*N**(-p['alpha']);gQ=np.zeros_like(Q);gQ[interior]=Q[interior]*qp['c_Q']*qp['nu_Q']*(1-eff[interior])**(qp['nu_Q']-1)*sq;loss=D.pred(N,150,Q,sq);maps.append(((gQ-gN)/loss,~interior))
    limit=max(np.max(np.abs(v)) for v,_ in maps);norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);fig,axs,pal=facets(1,3,(160,87),sharex=True,sharey=True,margins=(.10,.88,.20,.84),wspace=.11)
    for ax,sq,(z,clipped) in zip(axs[0],[.5,1,1.5],maps):
        im=ax.pcolormesh(X,Q,z,norm=norm,cmap=pal.cmap_div(soft=True),shading='gouraud');ax.contour(X,Q,np.where(clipped,np.nan,z),levels=[0],colors=[pal.ink],linewidths=1.6)
        if clipped.any():ax.contourf(X,Q,clipped.astype(float),levels=[.5,1.5],colors='none',hatches=['///'])
        ax.set(xticks=np.log10([.1,1,10]),xticklabels=['0.1','1','10'],yticks=[.1,.3,.5,.7,.9]);P.facet_title(ax,f'sQ = {sq:g}',fontsize=8.5)
    axs[0,0].set_ylabel('质量评分 Q');fig.supxlabel('参数量 N / 十亿（对数坐标）',y=.025,fontsize=9);cax=fig.add_axes([.915,.23,.017,.57]);cb=fig.colorbar(im,cax=cax);cb.outline.set_visible(False);cb.set_label('εQ − εN')
    legend(fig,[Line2D([],[],c=pal.ink,lw=1.6,label='两类弹性相等'),Patch(fc='white',ec=pal.line['base'],hatch='///',label='质量映射触及截断')],2)
    save(fig,'25','资源改进方向场',{'s_Q':[.5,1,1.5],'difference_range_by_scenario':[[float(v.min()),float(v.max())] for v,_ in maps],'fixed_D_B':150,'p':'p_ref','profile':0},'固定D=150B、参考配比时，质量与参数总Loss弹性之差在三种映射情景中的相图。暖色为质量弹性更大，冷色为参数弹性更大，实线为二者相等；斜线是映射截断区。',src('generalized_scaling_parameters.json'),'未计成本，不代表等预算最优方向。截断区按实际映射规则令质量导数为0，而非把边界奇异导数延伸进去。','Q2-30_弹性场.png','单一绿色箭头场 → 三sQ有符号弹性主导相图，主结论由等弹性边界表达，取消依赖坐标缩放的方向箭头。')

@reg('26')
def finite_equivalence_panels():
    a=D.table('quality_parameter_equivalence.csv');ref=D.obj('generalized_scaling_parameters.json')['Q_ref'];fig,axs,pal=facets(2,2,(160,113),sharex=True,margins=(.11,.98,.15,.84),wspace=.22,hspace=.38);keys={.5:'blue_gray',1:'forest_green',1.5:'ochre'}
    for r,dq in enumerate([.05,.1]):
        for col,(field,label) in enumerate([('finite_N_multiplier_minus_1','参数相对增幅 / %'),('finite_D_multiplier_minus_1','数据相对增幅 / %')]):
            ax=axs[r,col]
            for sq,key in keys.items():
                g=a[np.isclose(a.delta_Q,dq)&np.isclose(a.s_Q,sq)].sort_values('Q_A');ax.plot(g.Q_A,100*g[field],color=pal.line[key],marker={.5:'o',1:'s',1.5:'D'}[sq],ls={.5:'--',1:'-',1.5:'-.'}[sq],mfc=SOFT[key],mec='white',mew=.7,lw=1.5,ms=5)
            ax.axvline(ref,color=pal.line['base'],ls='--',lw=.7);ax.set(xlim=(.37,.83),xticks=[.4,ref,.8],xticklabels=['0.4','Qref','0.8'],ylabel=label);P.facet_title(ax,f'ΔQ = {dq:.2f}',fontsize=8.5);P.light_grid(ax)
    fig.supxlabel('基线质量 Q',y=.025,fontsize=9);legend(fig,[lhandle(pal,key,f'sQ = {sq:g}',ls={.5:'--',1:'-',1.5:'-.'}[sq],marker={.5:'o',1:'s',1.5:'D'}[sq]) for sq,key in keys.items()]+[lhandle(pal,'base','Qref','--')],4)
    main=a[np.isclose(a.delta_Q,.1)&np.isclose(a.s_Q,1)&np.isclose(a.Q_A,ref)].iloc[0]
    save(fig,'26','质量提升有限等价量',{'rows':len(a),'main_parameter_percent':float(100*main.finite_N_multiplier_minus_1),'main_data_percent':float(100*main.finite_D_multiplier_minus_1),'all_root_status_N':a.root_status_N.value_counts().to_dict()},'18个有限资源等价解按提质幅度及资源类型分面。点为实际求根结果，连线仅按三个基线质量档连接；sQ=1.5的高质量端因映射截断出现回落。',src('quality_parameter_equivalence.csv'),'参考情景ΔQ=0.10等价于原质量下增加参数37.30%或数据56.95%。这些不是成本节省率；不插值生成额外根解。','Q2-34_等效倍数曲线.png','18点投在近同一斜线 → 幅度×资源的四分面，显式显示基线质量、映射强度和截断回落。')

@reg('27')
def physical_gate_facets():
    a=D.table('generalized_scaling_predictions.csv.gz');star=a.query("p_scenario=='p_star_sensitivity'");rows=sorted(set(zip(star.N_params_B,star.D_tokens_B,star.Q_A)));fig,axs,pal=facets(1,3,(160,136),sharex=True,sharey=True,margins=(.20,.88,.17,.87),wspace=.13);v=star.prediction-star.irreducible_loss_E;norm=TwoSlopeNorm(vmin=float(v.min()),vcenter=0,vmax=float(v.max()));counts={}
    for ax,sq in zip(axs[0],[.5,1,1.5]):
        g=star[np.isclose(star.s_Q,sq)].copy();g['gap']=g.prediction-g.irreducible_loss_E;piv=g.pivot(index=['N_params_B','D_tokens_B','Q_A'],columns='mixture_strength_profile',values='gap').reindex(rows);mat=piv.to_numpy();im=ax.imshow(mat,aspect='auto',cmap=pal.cmap_div(soft=True),norm=norm,interpolation='nearest');yy,xx=np.where(mat<0);ax.scatter(xx,yy,s=10,c=pal.ink);ax.set(xticks=range(4),xticklabels=['0','0.25','0.5','1'],yticks=range(27),yticklabels=[f'{n:g} / {d:g} / {q:.2f}' for n,d,q in rows]);ax.tick_params(labelsize=7.5);P.facet_title(ax,f'sQ = {sq:g}',fontsize=8.5)
        for split in [8.5,17.5]:ax.axhline(split,c='white',lw=2)
        counts[str(sq)]=int((mat<0).sum())
    axs[0,0].set_ylabel('N / D / Q（N、D：十亿）');fig.supxlabel('配比迁移强度 profile',y=.055,fontsize=9);cax=fig.add_axes([.915,.22,.017,.58]);cb=fig.colorbar(im,cax=cax);cb.outline.set_visible(False);cb.set_label('预测 Loss − E')
    legend(fig,[Line2D([],[],ls='',marker='o',ms=3,c=pal.ink,label='近优配比：低于不可约下界'),Patch(fc=SOFT['leaf_green'],label='参考配比：0 / 324 违规')],2)
    save(fig,'27','配比接口下界缺口',{'p_star_violations_by_sQ':counts,'total_violations':int((a.prediction<a.irreducible_loss_E).sum()),'p_ref_rows':324,'p_star_rows':324,'gap_range':[float(v.min()),float(v.max())],'profile_to_lambda0':float(D.raw('B7').val_loss.quantile(.75)-D.raw('B7').val_loss.quantile(.25))},'近优配比敏感性网格按sQ分面，黑点为预测Loss低于E的违规状态。参考配比全部324个状态无违规，因此其重复色块省略，并以顶部汇总注明；三个分面保留全部324个近优配比状态。',src('generalized_scaling_predictions.csv.gz','mixture_physical_gate.csv'),'总计52个违规，联合配比接口被阻断；profile是无量纲情景倍数，λ0=profile×0.482225，不可将profile当作已识别迁移参数。','Q2-37_迁移情景热图.png','648格冗长列层级 → sQ三分面保留完整风险网格，无风险参考配比以严格计数汇总，避免重复信息。')
