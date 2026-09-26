"""Run one registered figure: python -m q2.run Q2-01."""
import argparse
from q2.redesign import FIGS
ORDER=['01','04','10','14','19','23','26','27','03','06','08','09','12','13','15','16','18','21','24','25']
def main():
 p=argparse.ArgumentParser();p.add_argument('ids',nargs='*');args=p.parse_args()
 keys=[x.removeprefix('Q2-').zfill(2) for x in args.ids] or ORDER
 for k in keys:
  if k not in FIGS: raise SystemExit(f'Unknown Q2 ID: {k}')
  FIGS[k]()
if __name__=='__main__': main()
