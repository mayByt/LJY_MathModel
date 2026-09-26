"""Verify existing Q2 artifacts without redrawing or touching frozen inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import subprocess
from PIL import Image
import config as C
from q2.run import ORDER

def validate():
    out=C.OUTPUT/'q2'
    records=[json.loads(p.read_text()) for p in sorted(out.glob('Q2-*.json'))]
    checks=[]
    for rec in records:
        png,pdf=Path(rec['png']),Path(rec['pdf'])
        image=Image.open(png)
        dpi=image.info.get('dpi', (0,0))
        font_text=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        fonts=[]
        for line in font_text.splitlines()[2:]:
            parts=line.split()
            if parts:
                # The final five columns are emb, sub, uni, object-number, generation.
                fonts.append({'name':parts[0].split('+')[-1], 'embedded':parts[-5]=='yes'})
        review_path=C.PLOT/'reviews'/f"{rec['id']}.json"
        review=json.loads(review_path.read_text()) if review_path.exists() else {}
        actual_hash=hashlib.sha256(png.read_bytes()).hexdigest()
        checks.append({'id':rec['id'],'png':str(png),'pdf':str(pdf),'dpi':dpi,
            'dpi_300':all(abs(v-300)<.01 for v in dpi),
            'hash_matches_metadata':actual_hash==rec['png_sha256'],
            'fonts':fonts,'all_fonts_embedded':all(f['embedded'] for f in fonts),
            'automatic_issues':rec['automatic_issues'],
            'current_review_pass':review.get('status')=='pass' and review.get('png_sha256')==actual_hash})
    ids_match={x['id'] for x in checks}=={'Q2-'+x for x in ORDER}
    assets_ok=ids_match and all(x['dpi_300'] and x['hash_matches_metadata'] and x['all_fonts_embedded'] and not x['automatic_issues'] for x in checks)
    reviews_ok=all(x['current_review_pass'] for x in checks)
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'expected_count':len(ORDER),'actual_count':len(checks),'ids_match':ids_match,'assets_pass':assets_ok,'reviews_pass':reviews_ok,'figures':checks}
    (out/'delivery_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False))
    return result
if __name__=='__main__': validate()
