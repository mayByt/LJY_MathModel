"""Check the current Q1 artifacts against independent review hashes."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
from PIL import Image
import config as C

def main():
 out=C.OUTPUT/'q1';expected={f'Q1-{n:02d}' for n in C.TOP20['q1']};rows=[]
 for p in sorted(out.glob('Q1-*.json')):
  m=json.loads(p.read_text());png=Path(m['png']);pdf=Path(m['pdf']);rpath=C.PLOT/'reviews'/f"{m['id']}.json";rv=json.loads(rpath.read_text()) if rpath.exists() else {};digest=hashlib.sha256(png.read_bytes()).hexdigest()
  with Image.open(png) as image:dpi=image.info.get('dpi',[0,0]);pixels=list(image.size)
  row={'id':m['id'],'png_sha256':digest,'metadata_matches_png':m['png_sha256']==digest,'independent_review_matches':rv.get('png_sha256')==digest and rv.get('status')=='pass' and rv.get('viewed') is True,'automatic_issues':m['automatic_issues'],'dpi':list(dpi),'pixels':pixels,'pdf_present':pdf.is_file(),'sources_present':all(Path(x).exists() for x in m['sources'])}
  row['pass']=row['metadata_matches_png'] and row['independent_review_matches'] and not row['automatic_issues'] and row['pdf_present'] and row['sources_present'] and all(abs(d-300)<.1 for d in dpi)
  rows.append(row)
 ids={r['id'] for r in rows};report={'question':'q1','checked_at':datetime.now(timezone.utc).isoformat(),'expected':sorted(expected),'count':len(rows),'ids_exact':ids==expected and len(rows)==len(expected),'passed':sum(r['pass'] for r in rows),'rows':rows};report['all_pass']=report['ids_exact'] and all(r['pass'] for r in rows);(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:report[k] for k in ['count','ids_exact','passed','all_pass']},ensure_ascii=False))
 if not report['all_pass']:raise SystemExit(1)
if __name__=='__main__':main()
