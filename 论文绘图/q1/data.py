from pathlib import Path
from functools import lru_cache
import itertools,json
import numpy as np,pandas as pd
import config as C
R=C.RESULTS['q1']; A=C.RAW/'A_data_value'; CACHE=C.CACHE/'q1_quality'
DOMAIN={'arxiv':'学术论文','book':'图书','c4':'C4网页','commoncrawl':'通用网页','github':'代码','stackexchange':'问答社区','wikipedia':'百科','freelaw':'法律文书','nih_exporter':'科研项目','pubmed_central':'医学全文','wikipedia_en':'百科','dm_mathematics':'数学推理','philpapers':'哲学论文','enron_emails':'商务邮件','gutenberg_pg_19':'文学图书','pile_cc':'通用网页','ubuntu_irc':'技术对话','europarl':'议会记录','hackernews':'科技社区','pubmed_abstracts':'医学摘要','uspto_backgrounds':'专利背景'}
QUALITY_DOMAINS=['c4','commoncrawl','arxiv','stackexchange','book','wikipedia','github']
BLOCKS={'education':'教育与领域价值','expression':'表达与整洁','reasoning':'推理与信息','noise':'噪声与重复','structure':'结构充分性'}
FIELDS={'fineweb_edu':'教育价值','qurater':'综合评审','dsir_books':'图书相似度','dsir_wiki':'百科相似度','dsir_math':'数学相似度','fluency_en':'流畅度','modernbert_readability':'可读性','modernbert_cleanliness':'整洁度','rps_lines_ending_with_terminal_punctution_mark':'句末标点','modernbert_reasoning':'推理性','modernbert_professionalism':'专业性','rps_doc_unigram_entropy':'词元熵','rps_doc_frac_unique_words':'词汇多样性','ad_en':'低广告','rps_doc_frac_no_alph_words':'字母词完整性','rps_doc_frac_chars_top_2gram':'低二元重复','rps_doc_frac_chars_top_3gram':'低三元重复','rps_lines_uppercase_letter_fraction':'大写规范性','rps_lines_numerical_chars_fraction':'数字适度性','rps_doc_word_count':'词数适度性','rps_doc_num_sentences':'句数适度性','rps_doc_mean_word_length':'词长适度性'}
@lru_cache(None)
def csv(n):return pd.read_csv(R/n)
@lru_cache(None)
def bridge():return dict(np.load(R/'problem2_bridge_input.npz',allow_pickle=False))
@lru_cache(None)
def quality():
 check=json.loads((CACHE/'verification.json').read_text())
 if not check['matches_at_1e-6']:raise ValueError('Reconstructed quality tables did not match frozen results')
 return pd.read_csv(CACHE/'quality_sample.csv.gz')
def features(p):
 p=np.asarray(p);pairs=list(itertools.combinations(range(p.shape[-1]),2));return np.concatenate([p,np.stack([p[...,i]*p[...,j] for i,j in pairs],axis=-1)],axis=-1)
def objective(p):
 b=bridge();coef=(b['coefficient_point']/b['loss_iqr'][:,None]).mean(0);return features(p)@coef
