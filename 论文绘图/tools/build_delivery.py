"""Generate captions, index, and an offline gallery without touching plots."""
from pathlib import Path
import sys,json,hashlib,html
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config as C

INTERPRET={
 'Q3-01':'五个离散上下文档位来自实际架构记录，因此后续敏感性分析有数据依据。',
 'Q3-02':'三种成本在高质量区域的增长速度不同，决定质量投资的边际取舍；不将题设函数视为观测规律。',
 'Q3-03':'上下文越长，注意力占训练与注意力合计的份额越高；达到50%时两类开销相等。',
 'Q3-04':'严格域存在解析的最低预算；处于不可行区的情景不能通过换优化器变成可行。',
 'Q3-05':'在同一预算下，三类成本分别产生内点、质量下界和质量上界最优；连续地形是冻结模型的评估。',
 'Q3-06':'谷底宽度反映近优配置的容许范围，不能从一个中心点推断唯一的工程方案。',
 'Q3-07':'资源配置随预算联合变化，质量约束状态改变时路径几何也会改变。淡框只给出N-D支持，未覆盖质量维度。',
 'Q3-08':'提质份额先升后降，份额下降并不意味着绝对提质投入下降。',
 'Q3-10':'四个面板显示参数、数据、质量和提质份额的联动；窄条件区间没有人为加宽。',
 'Q3-11':'长上下文对低预算损失的惩罚更明显，具体幅度应按成本类型与预算分开比较。',
 'Q3-12':'模型上的提质收益依赖成本形状与预算，重合点代表当前设定下无提质收益。',
 'Q3-14':'受约束的质量最优解存在边界点质量，平滑密度容易掩盖这种决策结构。',
 'Q3-15':'N与D的不确定性具有配对关系，不能把两个边际区间当成独立联合域。',
 'Q3-17':'数值定位宽度与参数传播区间是两种不同误差来源，必须分别阅读。',
 'Q3-18':'质量启动、内点投入和饱和形成可识别的预算阶段；研究区间起点已饱和不等于没有转移。',
 'Q3-20':'启动和饱和预算在同一抽样中联动，离散重复格点来自登记的局部求解格网。',
 'Q3-21':'提质成本和映射斜率的假设会改变最优Loss；敏感性结果不作因果解释。',
 'Q3-23':'固定领域配比下，最优Token总量按比例分配；领域份额并未随预算迁移。',
 'Q3-25':'有数值但求解失败的记录不能与成功解并列作为最优方案。',
 'Q3-26':'质量内点遵守边际收益平衡；到达上界后，收益比偏离1属于约束效应。',
 'Q4-01':'证据缺失相互交叠；图展示交集结构，不将各项缺失当成可加流失量。',
 'Q4-02':'算力与权重证据稀疏，限制了可用于解释前沿的样本覆盖。',
 'Q4-03':'模型类型的分布差异存在混杂，不能从组间分位差直接估计后训练收益。',
 'Q4-04':'能力与算力存在条件关联，但正式支持样本只有41条，应保留前沿区间及测量证据边界。',
 'Q4-06':'大多数BBH聚合能够精确复现，仍有少数尾部偏差需要审计。',
 'Q4-07':'聚合差异整体接近零但存在少数离群点；零膨胀误差不能自动视为正态。',
 'Q4-08':'随机猜测下界修正会改变任务尺度，原始分与校正分不可混用。',
 'Q4-09':'综合均值隐藏家族与任务的结构差异；该指纹未调整规模，不是架构优劣因果排名。',
 'Q4-10':'Loss到下游能力的换算具有明显散布，高低可比性数据需要分层读图。',
 'Q4-11':'各任务的最佳复杂度不同，弱桥接任务可能仍由常数模型胜出。',
 'Q4-12':'跨家族误差具有系统差异，平均MAE不能揭示全部迁移风险。',
 'Q4-13':'实际覆盖低于名义水平，区间校准存在局限。',
 'Q4-14':'相同上游Loss附近可有相近的映射结果；桥接范围外的点应保留其证据等级。',
 'Q4-15':'情景落入桥接支持范围并不能消除上游质量维度的外推。',
 'Q4-16':'桥接误差对最终不确定性具有重要影响，三类区间应并列而非相加。',
 'Q4-18':'规模关联贡献为正；非规模残余在强正则下接近零且区间跨零，不等于技术无贡献。',
 'Q4-20':'历史方向对家族与来源敏感，不能把旧记录与现代前沿直接拼成同口径时序。',
 'Q4-21':'预测使用的趋势端点高于截止月原始月度值，两者应分别展示。',
 'Q4-22':'放缓情景降低预测中心，但区间仍宽；结果是从2025-03起点作出的条件外推。',
 'Q4-23':'配对比较保留情景共享的不确定性；减速代价不能由独立区间端点相减得到。',
}

def sig(x):
    if not pd.notna(x):return '未定义'
    if x==0:return '0'
    return f'{x:.4g}'
def summary(meta):
    stats=meta.get('stats',{});table=stats.get('table')
    if not table or not Path(table).exists():return '具体数值与选择条件见同名JSON。'
    d=pd.read_csv(table)
    preferred=['N_B','D_B','Q_A','share_train','share_quality','share_attention','delta_loss','profile_delta_loss',
      'cost_ratio','minimum_budget','frontier','absolute_error','difference','center','cv_mae','forecast_score','loss_of_score','year_coef','trend']
    parts=[]
    for col in preferred:
        if col in d and pd.api.types.is_numeric_dtype(d[col]):
            v=d[col].dropna()
            if len(v):parts.append(f'{col}：{sig(v.min())}–{sig(v.max())}')
        if len(parts)>=3:break
    return f'随图数据表{len(d):,}行。'+('；'.join(parts)+'。' if parts else '分组条件与逐点数值见随图数据。')

def main():
    records=[]
    for q in C.TOP20:
        metas={m['id']:m for p in (C.OUTPUT/q).glob('Q*.json') if (m:=json.loads(p.read_text())).get('id')}
        for number in C.TOP20[q]:
            ident=q.upper()+f'-{number:02d}'
            if ident not in metas:continue
            m=metas[ident];p=Path(m['png']);sha=hashlib.sha256(p.read_bytes()).hexdigest();rp=ROOT/'reviews'/f'{ident}.json';r=json.loads(rp.read_text()) if rp.exists() else {};m['review_status']='通过' if r.get('status')=='pass' and r.get('png_sha256')==sha and r.get('viewed') is True and r.get('visual_revision')=='reference_v2' else '待审或待复审';m['review']=r;m['summary']=summary(m);records.append(m)
        if q in ['q3','q4']:
            lines=[f'# 问题{q[-1]} 图表说明（Top 20）','', '每张独立导出300dpi PNG与嵌入字体PDF。宋体中文、Times New Roman英文与数字；颜色统一为柔绿森林。下列数值从当前随图CSV计算，解释文字置于图外。','']
            for m in [x for x in records if x['question']==q]:
                n=m['name'];sz=m['size_mm'];typ=m['id'];figtype=next((x['name'] for x in json.loads((ROOT/'config/figure_catalog.json').read_text()) if x['id']==typ),n)
                lines += [f'## {n}',f'- **文件与尺寸**：[PNG](output/{q}/{n}.png) · [PDF](output/{q}/{n}.pdf)；设计尺寸 {sz[0]} × {sz[1]} mm。',f'- **图型与编码**：{figtype}。实际符号、区间和参考线见图例；完整编码口径见图注。',f'- **数据来源**：'+ '；'.join(f'`{str(Path(x).relative_to(C.REPO)) if Path(x).is_relative_to(C.REPO) else x}`' for x in m['sources'])+'。',f'- **数据摘要**：{m["summary"]}',f'- **解读与正文写法**：{INTERPRET.get(typ,m["caption"])}',f'- **论文图注**：{m["caption"]}',f'- **审阅**：{m["review_status"]}；以 `reviews/{typ}.json` 与图片SHA-256一致为准。','']
            (ROOT/f'图表说明_{q.upper()}.md').write_text('\n'.join(lines))
    lines=['# Top 20 论文插图索引','',f'计划四问各20张，当前已生成{len(records)}/80张；每张包含PNG、PDF、数据与图注。独立审阅通过{sum(x["review_status"]=="通过" for x in records)}张，状态与图片SHA-256绑定。','', '[离线图册](图册.html) · [统一风格配置](config/style.json)','']
    for q in C.TOP20:
        lines += [f'## {q.upper()}',f'[逐图说明](图表说明_{q.upper()}.md) · [20张缩略总览](previews/{q}_top20/contact_{q}_01.png)','', '|优先级|图号与名称|文件|独立审阅|','|---:|---|---|---|']
        for rank,m in enumerate([x for x in records if x['question']==q],1):
            n=m['name'];lines.append(f'|{rank}|{n}|[PNG](output/{q}/{n}.png) · [PDF](output/{q}/{n}.pdf)|{m["review_status"]}|')
        lines.append('')
    (ROOT/'图表说明.md').write_text('\n'.join(lines))
    cards=[]
    for m in records:
        q=m['question'];n=m['name'];src=f'output/{q}/{n}.png';pdf=f'output/{q}/{n}.pdf';search=html.escape(n+' '+m['caption'],quote=True)
        cards.append(f'<article data-q="{q}" data-search="{search}"><a class="image" href="{src}" target="_blank"><img src="{src}" loading="lazy" alt="{html.escape(n)}"></a><div class="meta"><span>{q.upper()} · {m["review_status"]}</span><h2>{html.escape(n)}</h2><div class="links"><a href="{src}" target="_blank">PNG</a><a href="{pdf}" target="_blank">PDF</a><a href="output/{q}/{n}.json" target="_blank">数据与口径</a></div><details><summary>图注</summary><p>{html.escape(m["caption"])}</p></details></div></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论文图册 · 柔绿森林</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f1;color:#303735;font-family:"Times New Roman","Songti SC",serif}header{padding:42px 5vw 24px;border-bottom:1px solid #d9e2d7;background:#fff}h1{font-size:30px;font-weight:500;margin:0 0 10px}header p{color:#69756e;font-size:15px;line-height:1.7;margin:0}nav{padding:18px 5vw;background:#f3f6f1;display:flex;gap:10px;flex-wrap:wrap;align-items:center;position:sticky;top:0;z-index:2;border-bottom:1px solid #e1e7de}button,input{font:inherit;border:1px solid #d1dccc;border-radius:5px;padding:9px 17px;background:white;color:#405445}button{cursor:pointer}button.active{background:#52745d;color:white;border-color:#52745d}input{margin-left:auto;min-width:230px}main{padding:26px 5vw 50px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}article{background:white;border:1px solid #dfe6db;min-width:0}.image{height:290px;display:flex;align-items:center;justify-content:center;padding:18px;background:#fff}.image img{max-width:100%;max-height:100%;object-fit:contain}.meta{padding:15px 20px 20px;border-top:1px solid #edf0ea}.meta span{font-size:12px;color:#718471}h2{font-size:17px;font-weight:500;line-height:1.6;margin:5px 0 10px}.links{display:flex;gap:16px}a{color:#4a4f7e;text-decoration:none}a:hover{text-decoration:underline}details{font-size:13px;color:#667265;margin-top:12px;line-height:1.7}summary{cursor:pointer}#count{color:#6a776b;font-size:14px}article[hidden]{display:none}@media(max-width:700px){header{padding-top:25px}main{grid-template-columns:1fr;padding:16px}input{width:100%;margin:0}.image{height:280px}}
</style><header><h1>论文图册 · 柔绿森林</h1><p>四问 Top 20 · 宋体 / Times New Roman · PNG 300 dpi + PDF<br>点击图片查看原图；图注与数据口径保留在图外。</p></header><nav><button class="active" data-filter="all">全部</button><button data-filter="q1">问题一</button><button data-filter="q2">问题二</button><button data-filter="q3">问题三</button><button data-filter="q4">问题四</button><span id="count"></span><input id="search" placeholder="搜索图号、名称或图注" aria-label="搜索图册"></nav><main>'''+''.join(cards)+'''</main><script>
let filter='all';const cards=[...document.querySelectorAll('article')];function update(){let n=0;const value=document.querySelector('#search').value.toLowerCase();cards.forEach(c=>{const show=(filter==='all'||c.dataset.q===filter)&&c.dataset.search.toLowerCase().includes(value);c.hidden=!show;if(show)n++});document.querySelector('#count').textContent=n+' 张'}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));update()});document.querySelector('#search').oninput=update;update();</script></html>'''
    page=page.replace('四问 Top 20 ·',f'参考重设计 V2 · 已生成 {len(records)}/80 张 ·')
    (ROOT/'图册.html').write_text(page)
    (ROOT/'reviews/technical').mkdir(parents=True,exist_ok=True)
    (ROOT/'reviews/technical/gallery_manifest.json').write_text(json.dumps([{'id':m['id'],'name':m['name'],'question':m['question'],'review_status':m['review_status']} for m in records],ensure_ascii=False,indent=2))
    print(f'Built index, offline gallery and Q3/Q4 captions: {len(records)} figures')
if __name__=='__main__':main()
