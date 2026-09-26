import sys
from .redesign_core import FIGS as CORE
FIGS=dict(CORE)
try:
    from .redesign_evidence import FIGS as EVIDENCE
    FIGS.update(EVIDENCE)
except ModuleNotFoundError as e:
    if e.name!='q4.redesign_evidence':raise

def main():
    args=sys.argv[1:]
    if not args:raise SystemExit('指定单张图号，例如Q4-04')
    for arg in args:FIGS[int(arg.split('-')[-1])]()
if __name__=='__main__':main()
