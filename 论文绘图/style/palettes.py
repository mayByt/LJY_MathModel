"""User's soft forest palette, with stable semantic colors and smooth ramps."""
import json
from pathlib import Path
from matplotlib.colors import to_rgb, to_hex
from style.theme import SCHEMES

cfg=json.loads((Path(__file__).resolve().parents[1]/'config/style.json').read_text())
fill=dict(cfg['palette']);line=dict(cfg['line'])
aliases={'N':'blue_gray','D':'forest_green','Q':'leaf_green','train':'blue_gray','quality':'leaf_green','attention':'ochre',
         'base':'slate_teal','base2':'olive_gray','neg':'indigo','pos':'ochre','neu':'slate_teal',
         'ours':'indigo','alt':'forest_green','c1':'leaf_green','c2':'blue_gray','c3':'ochre'}
for a,b in aliases.items():fill[a]=fill[b];line[a]=line[b]
def tint(color,white=.55):
    return to_hex([(1-white)*v+white for v in to_rgb(color)])
SOFT={k:tint(v,.48) for k,v in fill.items()}
SOFT.update({fill[k]:v for k,v in list(SOFT.items())})
DOMAIN_KEYS={'arxiv':'indigo','book':'ochre','github':'blue_gray','stackexchange':'forest_green','commoncrawl':'leaf_green','c4':'slate_teal','wikipedia':'olive_gray'}
Q_DOMAIN={k:fill[v] for k,v in DOMAIN_KEYS.items()}
Q_DOMAIN_LINE={k:line[v] for k,v in DOMAIN_KEYS.items()}
COST={'exponential':line['forest_green'],'power':line['indigo'],'logarithmic':line['ochre']}
COST_STYLE={'exponential':('-','o'),'power':('--','s'),'logarithmic':('-.','^')}
questions={'default':{'accent':line['forest_green'],'seq':[tint(fill['light_green'],.78),fill['light_green'],fill['forest_green'],fill['indigo']]}}
for q,s in cfg['questions'].items():
    questions[q]={'accent':line[s['primary']],'seq':[tint(s['sequential'][0],.80),*s['sequential']]}
SCHEMES['forest']={'name':'forest','fill':fill,'line':line,'questions':questions,
 'cycle':['forest_green','indigo','ochre','blue_gray','leaf_green'],
 'div':[fill['indigo'],tint(fill['blue_gray'],.38),'#FCFCF8',tint(fill['ochre'],.48),fill['ochre']],
 'ink':'#303735','grid':'#E6EBE5','ref':'#A9B3B0','extra':{
 'band_alpha':.28,'soft':SOFT,
 'seq_soft':['#F6F9F2',tint(fill['light_green'],.35),fill['light_green'],fill['leaf_green'],fill['forest_green']],
 'div_soft':[tint(fill['indigo'],.25),tint(fill['blue_gray'],.30),'#FCFCF8',tint(fill['ochre'],.55),fill['ochre']]}}
