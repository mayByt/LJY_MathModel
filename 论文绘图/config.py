from pathlib import Path
import json, os
PLOT = Path(__file__).resolve().parent
REPO = PLOT.parent
OUTPUT = PLOT / 'output'
CACHE = PLOT / 'cache'
RAW = Path(os.environ.get('PAPER_FIGURE_RAW', REPO.parent / '中文题目/F题/real_attachments'))
RESULTS = {k: REPO / v for k,v in json.loads((PLOT/'config/data_sources.json').read_text()).items() if k in ('q1','q2','q3','q4')}
TOP20 = {
 'q1':[2,8,9,10,11,13,17,19,20,23,25,27,28,5,6,7,15,16,18,22],
 'q2':[1,4,10,14,19,23,26,27,3,6,8,9,12,13,15,16,18,21,24,25],
 'q3':[3,5,7,8,10,11,12,14,15,17,18,21,23,25,26,1,2,4,6,20],
 'q4':[4,6,10,16,18,22,1,2,3,7,8,9,11,12,13,14,15,20,21,23],
}
