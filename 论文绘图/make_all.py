"""Rebuild selected publication figures from read-only inputs."""
import argparse,importlib,json,hashlib,sys
from pathlib import Path
import config as C

def png_hashes():
    return {p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.OUTPUT.glob('q*/Q*.png')}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('questions',nargs='*')
    parser.add_argument('--id',help='One candidate ID, e.g. Q3-07')
    args=parser.parse_args()
    if any(q not in C.TOP20 for q in args.questions):parser.error('questions must be q1, q2, q3 or q4')
    before=png_hashes()
    questions=args.questions or list(C.TOP20)
    if args.id:questions=[args.id.split('-')[0].lower()]
    for q in questions:
        module=importlib.import_module(q+'.run')
        nums=[int(args.id.split('-')[1])] if args.id else C.TOP20[q]
        for num in nums:
            key=f'{num:02d}' if q=='q2' else num
            module.FIGS[key]()
    after=png_hashes()
    diff={k:{'before':before.get(k),'after':v} for k,v in after.items() if before.get(k)!=v}
    report={'recreated':sum(1 if args.id else len(C.TOP20[q]) for q in questions),'changed_images':diff,
            'note':'Changed hashes need independent visual review; existing matching review hashes remain valid.'}
    target=C.PLOT/'reviews/technical/rebuild_check.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"Rebuilt {report['recreated']}; changed PNG hashes: {len(diff)}")
if __name__=='__main__':main()
